#!/usr/bin/env python3
"""Length-filtered CREMP COV/MAT scorer with one persistent global process pool.

Preserves the released COV/MAT protocol (RDKit GetBestRMS, ratio=2) without
per-molecule pool creation. Accepts both dict-style records and PyG Data objects.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import datamol as dm
import numpy as np
import pandas as pd
import torch
from loguru import logger as log
from rdkit import Chem
from rdkit.Chem import rdMolAlign as MA
from tqdm import tqdm

from cycpepflow.commons import load_pkl, save_pkl
from cycpepflow.commons.covmat import (
    calc_performance_stats,
    print_covmat_results,
    set_rdmol_positions,
)

_GLOBAL_TRUE_MOLS = None
_GLOBAL_GEN_MOLS = None
_GLOBAL_USE_FF = False
_GLOBAL_USE_ALIGNMOL = False


def get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def set_field(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def canonicalize_mol_noh_unmapped(mol: Chem.Mol) -> str:
    mol = Chem.Mol(Chem.RemoveHs(mol))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def canonical_noh_smiles_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse manifest SMILES: {smiles[:160]}")
    return canonicalize_mol_noh_unmapped(mol)


def canonical_noh_smiles_from_record(d: Any) -> str:
    try:
        rdmol = get_field(d, "rdmol")
    except Exception:
        rdmol = None
    mol = Chem.Mol(rdmol) if rdmol is not None else Chem.MolFromSmiles(str(get_field(d, "smiles")), sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse packed record SMILES: {str(get_field(d, 'smiles'))[:160]}")
    return canonicalize_mol_noh_unmapped(mol)


def load_manifest_filter(path: Path, num_monomers: int):
    keep = {}
    total_rows = 0
    counts: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            n = int(row["num_monomers"])
            counts[n] = counts.get(n, 0) + 1
            if n != num_monomers:
                continue
            canon = canonical_noh_smiles_from_smiles(row["smiles"])
            if canon in keep:
                raise RuntimeError(f"Duplicate canonical manifest SMILES for filter: {canon}")
            keep[canon] = row
    return keep, total_rows, counts


def merge_and_filter_parts(parts, manifest: Path, num_monomers: int, limit_filtered_molecules: int = 0):
    keep_meta, manifest_total, manifest_counts = load_manifest_filter(manifest, num_monomers)
    packed = []
    part_rows = []
    unmatched_examples = []
    for part in parts:
        p = Path(part)
        data = load_pkl(str(p))
        kept_here = 0
        for d in data:
            canon = canonical_noh_smiles_from_record(d)
            meta = keep_meta.get(canon)
            if meta is None:
                if len(unmatched_examples) < 5:
                    unmatched_examples.append(str(get_field(d, "smiles"))[:160])
                continue
            set_field(d, "manifest_sequence", meta.get("sequence"))
            set_field(d, "manifest_num_monomers", int(meta.get("num_monomers")))
            set_field(d, "manifest_split_index", int(meta.get("split_index")))
            set_field(d, "manifest_archive_pickle_basename", meta.get("archive_pickle_basename"))
            packed.append(d)
            kept_here += 1
        part_rows.append({"part": str(p), "n_molecules_raw": len(data), "n_molecules_kept": kept_here})

    packed.sort(key=lambda d: (int(get_field(d, "manifest_split_index")), str(get_field(d, "smiles"))))
    if limit_filtered_molecules:
        packed = packed[:limit_filtered_molecules]

    expected = len(keep_meta)
    if not limit_filtered_molecules and len(packed) != expected:
        raise RuntimeError(
            f"Filtered molecule count mismatch: kept {len(packed)} but manifest has {expected} "
            f"num_monomers={num_monomers} rows; manifest_counts={manifest_counts}; examples unmatched={unmatched_examples}"
        )
    if not packed:
        raise RuntimeError(
            f"No molecules kept for num_monomers={num_monomers}; manifest_counts={manifest_counts}; examples unmatched={unmatched_examples}"
        )
    return packed, manifest_total, expected, part_rows, manifest_counts


def _init_global_pool(true_mols, gen_mols, use_force_field=False, use_alignmol=False):
    global _GLOBAL_TRUE_MOLS, _GLOBAL_GEN_MOLS, _GLOBAL_USE_FF, _GLOBAL_USE_ALIGNMOL
    _GLOBAL_TRUE_MOLS = true_mols
    _GLOBAL_GEN_MOLS = gen_mols
    _GLOBAL_USE_FF = use_force_field
    _GLOBAL_USE_ALIGNMOL = use_alignmol


def _global_ref_chunk_worker(task):
    smiles, start, stop = task
    true_list = _GLOBAL_TRUE_MOLS[smiles]
    gen_list = _GLOBAL_GEN_MOLS[smiles]
    rows = []
    for i_true in range(start, stop):
        ref = true_list[i_true]
        rmsd_vals = []
        for gen in gen_list:
            try:
                if _GLOBAL_USE_ALIGNMOL:
                    rmsd_vals.append(MA.AlignMol(gen, ref))
                else:
                    rmsd_vals.append(MA.GetBestRMS(gen, ref))
            except Exception:  # noqa: BLE001 - preserve nan-on-failure behavior
                rmsd_vals.append(np.nan)
        rows.append((i_true, rmsd_vals))
    return smiles, rows


def compute_covmat_global_pool(
    packed_data_list,
    num_workers: int,
    thresholds: np.ndarray,
    ratio: int = 2,
    use_force_field: bool = False,
    use_alignmol: bool = False,
    filter_disconnected: bool = True,
    ref_chunk_size: int = 1,
    target_pairs_per_task: int = 0,
    pool_chunksize: int = 1,
):
    t0 = time.time()
    compute_covmat_global_pool.target_pairs_per_task = int(target_pairs_per_task or 0)
    compute_covmat_global_pool.pool_chunksize = int(pool_chunksize or 1)
    rmsd_results = {}
    true_mols = {}
    gen_mols = {}

    for data in tqdm(packed_data_list, desc="Prepare RDKit mols", total=len(packed_data_list)):
        try:
            pos_gen = get_field(data, "pos_gen")
            pos_ref = get_field(data, "pos_ref")
        except Exception:
            log.info("skipping due to missing pos_gen or pos_ref")
            continue
        smiles = str(get_field(data, "smiles"))
        if filter_disconnected and ("." in smiles):
            log.info("skipping due to disconnected molecule")
            continue

        if isinstance(pos_gen, torch.Tensor):
            pos_gen = pos_gen.cpu().numpy()
        if isinstance(pos_ref, torch.Tensor):
            pos_ref = pos_ref.cpu().numpy()

        num_atoms = pos_gen.shape[1]
        mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
        pos_ref = pos_ref.reshape(-1, num_atoms, 3)
        pos_gen = pos_gen.reshape(-1, num_atoms, 3)

        num_true = pos_ref.shape[0]
        num_gen = num_true * ratio
        if pos_gen.shape[0] < num_gen:
            log.info("skipping due to insufficient number of generated conformers")
            continue
        pos_gen = pos_gen[:num_gen]
        set_field(data, "pos_ref", pos_ref)
        set_field(data, "pos_gen", pos_gen)

        # The reference implementation removes hydrogens on every single
        # reference/generated comparison. For CREMP 5-mers this is ~1.34B calls,
        # so repeated Chem.RemoveHs dominates avoidable overhead. Pre-remove H
        # once per conformer and still call RDKit's GetBestRMS on the heavy-atom
        # molecules, preserving symmetry-aware GetBestRMS semantics exactly.
        true_mols[smiles] = [Chem.RemoveHs(set_rdmol_positions(mol, pos_ref[i])) for i in range(num_true)]
        gen_mols[smiles] = [Chem.RemoveHs(set_rdmol_positions(mol, pos_gen[i])) for i in range(num_gen)]
        rmsd_results[smiles] = {
            "n_true": num_true,
            "n_model": num_gen,
            "rmsd": np.nan * np.ones((num_true, num_gen)),
        }

    smiles_order = list(rmsd_results.keys())
    tasks = []
    task_pair_estimates = []
    ref_chunk_size = max(1, int(ref_chunk_size))
    target_pairs_per_task = int(getattr(compute_covmat_global_pool, "target_pairs_per_task", 0) or 0)
    for smiles in smiles_order:
        n_true = int(rmsd_results[smiles]["n_true"])
        n_gen = int(rmsd_results[smiles]["n_model"])
        if target_pairs_per_task > 0:
            # Adaptive preNoH scheduling: fixed ref_chunk_size creates many tiny
            # jobs for 4-/5-mers with small n_gen, causing Python Pool IPC to
            # dominate and leaving cores idle.  Choose refs/task to target a
            # roughly constant number of GetBestRMS calls while preserving the
            # exact same RMSD matrix values.
            chunk = max(1, int(target_pairs_per_task // max(1, n_gen)))
        else:
            chunk = ref_chunk_size
        for start in range(0, n_true, chunk):
            stop = min(start + chunk, n_true)
            tasks.append((smiles, start, stop))
            task_pair_estimates.append((stop - start) * n_gen)
    # Longest tasks first reduces tail effects when molecules differ greatly in
    # number of generated conformers; each task still writes back to its own
    # molecule slice, so results are order-independent.
    if task_pair_estimates:
        tasks = [t for _, t in sorted(zip(task_pair_estimates, tasks), key=lambda x: x[0], reverse=True)]
    n_ref_total = sum(int(rmsd_results[s]["n_true"]) for s in smiles_order)
    pool_chunksize = max(1, int(getattr(compute_covmat_global_pool, "pool_chunksize", 1) or 1))
    print(
        f"GLOBALPOOL prepare_seconds={time.time()-t0:.3f} molecules={len(smiles_order)} "
        f"ref_total={n_ref_total} tasks={len(tasks)} workers={num_workers} "
        f"ref_chunk_size={ref_chunk_size} target_pairs_per_task={target_pairs_per_task} "
        f"pool_chunksize={pool_chunksize}",
        flush=True,
    )

    if tasks:
        ctx = get_context("fork")
        with ctx.Pool(
            int(num_workers),
            initializer=_init_global_pool,
            initargs=(true_mols, gen_mols, use_force_field, use_alignmol),
        ) as pool:
            for smiles, rows in tqdm(
                pool.imap_unordered(_global_ref_chunk_worker, tasks, chunksize=pool_chunksize),
                total=len(tasks),
                desc=f"CovMat adaptive preNoH workers={num_workers}",
            ):
                arr = rmsd_results[smiles]["rmsd"]
                for i_true, rmsd_vals in rows:
                    arr[i_true] = rmsd_vals

    stats = [calc_performance_stats(rmsd_results[smiles]["rmsd"], thresholds) for smiles in smiles_order]
    coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)
    results = {
        "CoverageR": np.array(coverage_recall),
        "MatchingR": np.array(amr_recall),
        "thresholds": thresholds,
        "CoverageP": np.array(coverage_precision),
        "MatchingP": np.array(amr_precision),
    }
    return results, rmsd_results


def write_outputs(args, packed, manifest_total, expected, manifest_counts, part_rows, results, rmsd_results, elapsed_seconds: float):
    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    merged_path = outdir / "generated_files.pkl"
    save_pkl(str(merged_path), packed)

    with open(metrics_dir / "input_molecule_counts.csv", "w", newline="") as f:
        fieldnames = ["manifest_split_index", "sequence", "num_monomers", "smiles", "n_ref", "n_gen", "n_atoms"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in packed:
            pos_ref = get_field(d, "pos_ref")
            pos_gen = get_field(d, "pos_gen")
            w.writerow({
                "manifest_split_index": int(get_field(d, "manifest_split_index")),
                "sequence": get_field(d, "manifest_sequence"),
                "num_monomers": int(get_field(d, "manifest_num_monomers")),
                "smiles": get_field(d, "smiles"),
                "n_ref": int(pos_ref.reshape(-1, pos_ref.shape[-2], 3).shape[0]),
                "n_gen": int(pos_gen.reshape(-1, pos_gen.shape[-2], 3).shape[0]),
                "n_atoms": int(pos_gen.shape[-2]),
            })

    cov_df, mat = print_covmat_results(results)
    cov_curve_path = metrics_dir / "cremp_coverage_curve.csv"
    cov_df.to_csv(cov_curve_path, index=False)

    mol_rows = []
    by_smiles = {str(get_field(d, "smiles")): d for d in packed}
    for smiles, rr in rmsd_results.items():
        rmsd = rr["rmsd"]
        d = by_smiles.get(smiles)
        mol_rows.append({
            "manifest_split_index": int(get_field(d, "manifest_split_index")) if d is not None else -1,
            "sequence": get_field(d, "manifest_sequence") if d is not None else "",
            "num_monomers": int(get_field(d, "manifest_num_monomers")) if d is not None else args.num_monomers,
            "smiles": smiles,
            "n_true": int(rr["n_true"]),
            "n_model": int(rr["n_model"]),
            "cov_recall_at_threshold": float(np.mean(np.nanmin(rmsd, axis=1) < args.threshold)),
            "cov_precision_at_threshold": float(np.mean(np.nanmin(rmsd, axis=0) < args.threshold)),
            "mat_recall": float(np.mean(np.nanmin(rmsd, axis=1))),
            "mat_precision": float(np.mean(np.nanmin(rmsd, axis=0))),
        })
    per_mol_metrics_path = metrics_dir / "cremp_per_molecule_covmat.csv"
    pd.DataFrame(mol_rows).sort_values(["manifest_split_index", "smiles"]).to_csv(per_mol_metrics_path, index=False)

    row = cov_df.iloc[(cov_df["Threshold"] - args.threshold).abs().argsort()[:1]].iloc[0].to_dict()
    metric = {k: (float(v) * 100.0 if k.startswith("COV-") else float(v)) for k, v in row.items() if k != "Threshold"}
    metric.update({k: float(v) for k, v in mat.items()})
    metric["Threshold"] = float(row["Threshold"])
    summary = {
        "dataset": "ringer_cremp_top30_testall",
        "split": "test",
        "filter": {
            "manifest": str(Path(args.manifest).resolve()),
            "num_monomers": args.num_monomers,
            "manifest_rows_total": manifest_total,
            "manifest_counts": manifest_counts,
            "manifest_rows_matching_filter": expected,
            "limit_filtered_molecules": args.limit_filtered_molecules,
        },
        "generated_files": str(merged_path),
        "n_molecules_input": len(packed),
        "n_reference_conformers": int(sum(get_field(d, "pos_ref").shape[0] for d in packed)),
        "n_generated_conformers": int(sum(get_field(d, "pos_gen").shape[0] for d in packed)),
        "threshold_A": args.threshold,
        "num_workers": args.num_workers,
        "engine": "global_ref_chunk_pool_adaptive_preNoH",
        "ref_chunk_size": args.ref_chunk_size,
        "target_pairs_per_task": args.target_pairs_per_task,
        "pool_chunksize": args.pool_chunksize,
        "score_shard_id": int(getattr(args, "score_shard_id", 0)),
        "num_score_shards": int(getattr(args, "num_score_shards", 1)),
        "full_n_molecules_input_before_score_sharding": int(getattr(args, "full_packed_count", len(packed))),
        "elapsed_seconds": elapsed_seconds,
        "metrics": metric,
        "parts": part_rows,
        "coverage_curve_csv": str(cov_curve_path),
        "per_molecule_covmat_csv": str(per_mol_metrics_path),
        "note": f"COV/MAT computed with RDKit GetBestRMS at threshold {args.threshold} on only test-set cyclic peptides with num_monomers={args.num_monomers}; global process-pool scheduler preserves the released ratio=2 protocol.",
    }
    summary_path = metrics_dir / "cremp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    pd.DataFrame([metric]).to_csv(metrics_dir / "cremp_metrics.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--num-monomers", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--num_workers", type=int, default=96)
    ap.add_argument("--use_alignmol", action="store_true")
    ap.add_argument("--limit-filtered-molecules", type=int, default=0)
    ap.add_argument("--ref-chunk-size", type=int, default=int(os.environ.get("CYCPEPFLOW_COVMAT_REF_CHUNK_SIZE", "1")))
    ap.add_argument("--target-pairs-per-task", type=int, default=int(os.environ.get("CYCPEPFLOW_COVMAT_TARGET_PAIRS_PER_TASK", "0")))
    ap.add_argument("--pool-chunksize", type=int, default=int(os.environ.get("CYCPEPFLOW_COVMAT_POOL_CHUNKSIZE", "1")))
    ap.add_argument("--score-shard-id", type=int, default=int(os.environ.get("CYCPEPFLOW_SCORE_SHARD_ID", "0")))
    ap.add_argument("--num-score-shards", type=int, default=int(os.environ.get("CYCPEPFLOW_NUM_SCORE_SHARDS", "1")))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    packed, manifest_total, expected, part_rows, manifest_counts = merge_and_filter_parts(
        args.parts, Path(args.manifest), args.num_monomers, args.limit_filtered_molecules
    )
    args.num_score_shards = max(1, int(args.num_score_shards))
    args.score_shard_id = int(args.score_shard_id)
    if args.score_shard_id < 0 or args.score_shard_id >= args.num_score_shards:
        raise SystemExit(f"invalid score shard {args.score_shard_id}/{args.num_score_shards}")
    full_packed_count = len(packed)
    if args.num_score_shards > 1:
        packed = [d for i, d in enumerate(packed) if i % args.num_score_shards == args.score_shard_id]
        if not packed:
            raise RuntimeError(f"score shard {args.score_shard_id}/{args.num_score_shards} is empty from {full_packed_count} molecules")
        print(f"SCORE_SHARD_FILTER shard={args.score_shard_id}/{args.num_score_shards} full_molecules={full_packed_count} shard_molecules={len(packed)}", flush=True)
    args.full_packed_count = full_packed_count
    thresholds = np.arange(0.05, 3.05, 0.05)
    results, rmsd_results = compute_covmat_global_pool(
        packed,
        num_workers=args.num_workers,
        thresholds=thresholds,
        ratio=2,
        use_alignmol=args.use_alignmol,
        ref_chunk_size=args.ref_chunk_size,
        target_pairs_per_task=args.target_pairs_per_task,
        pool_chunksize=args.pool_chunksize,
    )
    write_outputs(args, packed, manifest_total, expected, manifest_counts, part_rows, results, rmsd_results, time.time() - t0)


if __name__ == "__main__":
    main()
