#!/usr/bin/env python3
"""Generate ETFlow conformer samples for one test-set shard.

This is a shardable variant of ETFlow scripts/eval.py that accepts an explicit
molecule-level processed data_dir/partition.  It is intended for COV/MAT: each
output Data has pos_ref = all reference conformers and pos_gen = 2x as many
model samples, matching etflow.commons.covmat.CovMatEvaluator(ratio=2).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Make the repository importable when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT,):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpy as np
import torch
import yaml
from loguru import logger as log
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from etflow.commons import save_pkl
from etflow.data import EuclideanDataset
from etflow.models import BaseFlow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True, help="Processed root containing <partition>/test/*.pt")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--out", required=True, help="Output shard pkl")
    ap.add_argument("--summary", required=True, help="Output shard summary json")
    ap.add_argument("--shard_id", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--max_molecules", type=int, default=None, help="Optional per-shard molecule cap for smoke tests")
    ap.add_argument("--max_ref_confs", type=int, default=None, help="Optional reference-conformer cap for fast smoke tests; production runs leave unset")
    ap.add_argument("--manifest", default=None, help="Optional manifest CSV used to restrict generation by split_index/num_monomers")
    ap.add_argument("--num-monomers", type=int, default=None, help="When --manifest is set, keep only rows with this num_monomers value")
    ap.add_argument("--batch_size", type=int, default=None, help="Generation batch size override; defaults to config eval_args.batch_size")
    ap.add_argument(
        "--network-amp",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Inference-only AMP around the TorchMD network forward; ODE state/output remain fp32.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stp-chirality-repair",
        action="store_true",
        help="After sampling each molecule, apply deterministic STP center-reflection chirality projection before saving.",
    )
    ap.add_argument("--stp-repair-max-passes", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(args.seed + args.shard_id)
    np.random.seed(args.seed + args.shard_id)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Shard {args.shard_id}/{args.num_shards}: using device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    cfg_test_dataset_args = dict(
        (cfg.get("datamodule_args", {}).get("test_dataset_args", {}) or {})
    )
    # Generation must preserve molecule-level indexing; only propagate the
    # chiral-tag node_attr ablation flag from training configs.
    generation_dataset_args = {}
    if "include_chirality_in_node_attr" in cfg_test_dataset_args:
        generation_dataset_args["include_chirality_in_node_attr"] = bool(
            cfg_test_dataset_args["include_chirality_in_node_attr"]
        )
    dataset = EuclideanDataset(
        data_dir=args.data_dir,
        partition=args.partition,
        split="test",
        **generation_dataset_args,
    )
    log.info(f"Generation dataset args={generation_dataset_args}")
    all_indices = list(range(len(dataset)))
    filter_manifest_rows = None
    filter_num_monomers_counts = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        filter_manifest_rows = list(csv.DictReader(manifest_path.open()))
        if args.num_monomers is not None:
            filter_manifest_rows = [
                r for r in filter_manifest_rows
                if int(r.get("num_monomers", -1)) == int(args.num_monomers)
            ]
        # ETFlow processed test files are named with the manifest split_index as
        # their six-digit prefix (e.g. 000880_*.pt).  Build the map from the
        # actual dataset files instead of assuming sorted list position.
        prefix_to_dataset_pos = {}
        for pos, data_file in enumerate(dataset.data_files):
            try:
                prefix_to_dataset_pos[int(Path(data_file).name.split("_", 1)[0])] = pos
            except Exception as exc:  # pragma: no cover - defensive artifact check
                raise ValueError(f"Cannot parse split_index prefix from {data_file}") from exc
        missing = []
        filtered_indices = []
        for row in filter_manifest_rows:
            split_index = int(row.get("split_index", row.get("source_row_0based", -1)))
            if split_index not in prefix_to_dataset_pos:
                missing.append(split_index)
                continue
            filtered_indices.append(prefix_to_dataset_pos[split_index])
        if missing:
            raise RuntimeError(f"Manifest rows missing from processed dataset: first={missing[:10]} n={len(missing)}")
        all_indices = filtered_indices
        from collections import Counter
        filter_num_monomers_counts = dict(Counter(int(r["num_monomers"]) for r in filter_manifest_rows))
        log.info(
            "Manifest filter enabled: manifest={} num_monomers={} selected={} counts={}",
            manifest_path,
            args.num_monomers,
            len(all_indices),
            filter_num_monomers_counts,
        )
    indices = all_indices[args.shard_id :: args.num_shards]
    if args.max_molecules is not None:
        indices = indices[: args.max_molecules]
    log.info(
        f"Dataset size={len(dataset)}; filtered indices={len(all_indices)}; "
        f"shard indices={len(indices)}; first={indices[:5]}"
    )

    if cfg["model"] != "BaseFlow":
        raise ValueError(f"Unsupported model class in release config: {cfg['model']}")
    model = BaseFlow(**cfg["model_args"])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model: BaseFlow = model.to(device)
    model.network_amp = args.network_amp
    model.eval()
    log.info(f"Network-only AMP mode={args.network_amp}; model output/state kept fp32")

    eval_args = dict(cfg.get("eval_args", {}))
    max_batch_size = args.batch_size or int(eval_args.get("batch_size", 32))
    sampler_args = dict(eval_args.get("sampler_args", {}))
    log.info(f"Generation batch_size={max_batch_size}; sampler_args={sampler_args}")

    generated = []
    times = []
    molecule_rows = []
    repair_rows = []
    repair_totals = {
        "strict_before": 0,
        "strict_after": 0,
        "center_match_before": 0,
        "center_match_after": 0,
        "center_total": 0,
        "reflections": 0,
        "degenerate_reflections": 0,
        "not_converged_conformers": 0,
    }
    repair_record_fn = None
    if args.stp_chirality_repair:
        from repair_etflow_stp_chirality import repair_record as repair_record_fn  # noqa: PLC0415

        log.info("STP chirality repair enabled: center-reflect projection, max_passes={}", args.stp_repair_max_passes)
    for idx in tqdm(indices, desc=f"shard{args.shard_id}"):
        data = dataset[idx]
        source = torch.load(dataset.data_files[idx])
        pos_ref = source.pos.cpu().numpy()
        if args.max_ref_confs is not None:
            pos_ref = pos_ref[: args.max_ref_confs]
        n_true = int(pos_ref.shape[0])
        n_atoms = int(pos_ref.shape[1])
        n_gen = 2 * n_true
        smiles = data.smiles
        log.info(f"idx={idx} n_true={n_true} n_gen={n_gen} n_atoms={n_atoms} smiles={smiles}")
        pos_gen_chunks = []
        for batch_start in range(0, n_gen, max_batch_size):
            bs = min(max_batch_size, n_gen - batch_start)
            batched_data = Batch.from_data_list([data] * bs)
            z = batched_data["atomic_numbers"].to(device)
            edge_index = batched_data["edge_index"].to(device)
            edge_attr = batched_data.get("edge_attr", None)
            edge_attr = edge_attr.to(device) if edge_attr is not None else None
            batch = batched_data["batch"].to(device)
            node_attr = batched_data["node_attr"].to(device)
            chiral_index = batched_data["chiral_index"].to(device)
            chiral_nbr_index = batched_data["chiral_nbr_index"].to(device)
            chiral_tag = batched_data["chiral_tag"].to(device)
            tb = time.time()
            with torch.no_grad():
                pos = model.sample(
                    z,
                    edge_index,
                    batch,
                    node_attr=node_attr,
                    edge_attr=edge_attr,
                    chiral_index=chiral_index,
                    chiral_nbr_index=chiral_nbr_index,
                    chiral_tag=chiral_tag,
                    smiles=[smiles] * bs,
                    **sampler_args,
                )
            times.append((time.time() - tb) / bs)
            pos = pos.view(bs, -1, 3).cpu().detach().numpy()
            pos_gen_chunks.append(pos)
        pos_gen = np.concatenate(pos_gen_chunks, axis=0)
        record = Data(smiles=smiles, pos_ref=pos_ref, rdmol=data.mol, pos_gen=pos_gen)
        repair_summary = None
        if repair_record_fn is not None:
            _, record, repair_summary = repair_record_fn((int(idx), record, 1e-7, args.stp_repair_max_passes))
            repair_rows.append(repair_summary)
            for key in repair_totals:
                repair_totals[key] += int(repair_summary.get(key, 0))
        generated.append(record)
        molecule_row = {
            "idx": int(idx),
            "file": str(dataset.data_files[idx]),
            "smiles": smiles,
            "n_true": n_true,
            "n_gen": int(pos_gen.shape[0]),
            "n_atoms": n_atoms,
            "seconds_per_generated_conformer_mean_so_far": float(np.mean(times)) if times else None,
        }
        if repair_summary is not None:
            molecule_row.update({
                "stp_repair_strict_before": int(repair_summary["strict_before"]),
                "stp_repair_strict_after": int(repair_summary["strict_after"]),
                "stp_repair_center_match_before": int(repair_summary["center_match_before"]),
                "stp_repair_center_match_after": int(repair_summary["center_match_after"]),
                "stp_repair_center_total": int(repair_summary["center_total"]),
                "stp_repair_reflections": int(repair_summary["reflections"]),
            })
        molecule_rows.append(molecule_row)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(str(out), generated)
    payload = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_global_step": int(ckpt.get("global_step", -1)),
        "data_dir": str(Path(args.data_dir).resolve()),
        "partition": args.partition,
        "split": "test",
        "manifest_filter": str(Path(args.manifest).resolve()) if args.manifest else None,
        "num_monomers_filter": int(args.num_monomers) if args.num_monomers is not None else None,
        "n_filtered_molecules_total": int(len(all_indices)),
        "filter_num_monomers_counts": filter_num_monomers_counts,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_dataset_molecules": int(len(dataset)),
        "n_shard_molecules": int(len(generated)),
        "n_reference_conformers": int(sum(r["n_true"] for r in molecule_rows)),
        "n_generated_conformers": int(sum(r["n_gen"] for r in molecule_rows)),
        "batch_size": int(max_batch_size),
        "network_amp": args.network_amp,
        "generation_dataset_args": generation_dataset_args,
        "include_chirality_in_node_attr": bool(
            getattr(dataset, "include_chirality_in_node_attr", True)
        ),
        "seconds_per_generated_conformer_mean": float(np.mean(times)) if times else None,
        "elapsed_seconds": float(time.time() - t0),
        "max_ref_confs": int(args.max_ref_confs) if args.max_ref_confs is not None else None,
        "stp_chirality_repair_enabled": bool(args.stp_chirality_repair),
        "stp_repair_max_passes": int(args.stp_repair_max_passes),
        "stp_repair_totals": repair_totals if args.stp_chirality_repair else None,
        "stp_repair_strict_before_rate": (
            repair_totals["strict_before"] / sum(r["n_gen"] for r in molecule_rows)
            if args.stp_chirality_repair and molecule_rows else None
        ),
        "stp_repair_strict_after_rate": (
            repair_totals["strict_after"] / sum(r["n_gen"] for r in molecule_rows)
            if args.stp_chirality_repair and molecule_rows else None
        ),
        "stp_repair_center_before_rate": (
            repair_totals["center_match_before"] / repair_totals["center_total"]
            if args.stp_chirality_repair and repair_totals["center_total"] else None
        ),
        "stp_repair_center_after_rate": (
            repair_totals["center_match_after"] / repair_totals["center_total"]
            if args.stp_chirality_repair and repair_totals["center_total"] else None
        ),
        "molecules": molecule_rows,
        "stp_repair_molecules": repair_rows if args.stp_chirality_repair else None,
        "out": str(out),
    }
    summary = Path(args.summary); summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
