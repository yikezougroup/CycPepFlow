#!/usr/bin/env python3
"""Coordinate-geometry (STP/oriented-volume) chirality evaluator for ETFlow generated_files.pkl.

Main metric follows Dizhou's preferred convention: use source-specified tetrahedral
chiral centers from the input SMILES, calibrate each center's expected handedness
from reference conformer coordinates, then compare generated conformer coordinates
by scalar triple product (oriented volume) at the same atom-indexed center.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

CHI_NAMES = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: "CHI_UNSPECIFIED",
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: "CHI_TETRAHEDRAL_CW",
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: "CHI_TETRAHEDRAL_CCW",
}


def as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mol_from_ordered_smiles(smiles: str, expected_atoms: int | None = None) -> Chem.Mol:
    """Build a mol while preserving the atom order used by ETFlow's atom-mapped SMILES."""
    errors = []
    for mode in ("datamol_ordered", "rdkit"):
        try:
            if mode == "datamol_ordered":
                import datamol as dm  # type: ignore

                mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
                if mol is None:
                    raise ValueError("datamol returned None")
                mol = Chem.Mol(mol)
            else:
                mol = Chem.MolFromSmiles(smiles, sanitize=True)
                if mol is None:
                    raise ValueError("RDKit returned None")
                mol = Chem.Mol(mol)
            if expected_atoms is not None and mol.GetNumAtoms() != expected_atoms:
                raise ValueError(
                    f"atom-count mismatch: mol has {mol.GetNumAtoms()} atoms, coordinates have {expected_atoms}"
                )
            return mol
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{mode}: {exc}")
    raise ValueError(f"failed to build ordered mol for {smiles[:160]} ; " + " | ".join(errors))


def source_chiral_centers(mol: Chem.Mol) -> list[dict[str, Any]]:
    centers: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag in (Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW):
            nbrs = sorted(n.GetIdx() for n in atom.GetNeighbors())
            if len(nbrs) < 3:
                continue
            # Deterministic atom-indexed tuple. The expected sign is calibrated from references,
            # so CW/CCW's RDKit neighbor-order convention is not used directly.
            centers.append(
                {
                    "center_idx": atom.GetIdx(),
                    "neighbor_tuple": tuple(nbrs[:3]),
                    "source_chiral_tag": CHI_NAMES.get(tag, str(int(tag))),
                    "atom_map_num": atom.GetAtomMapNum(),
                    "atomic_num": atom.GetAtomicNum(),
                }
            )
    return centers


def stp_sign(xyz: np.ndarray, center_idx: int, nbrs: tuple[int, int, int], eps: float) -> int:
    c = xyz[center_idx]
    v1 = xyz[nbrs[0]] - c
    v2 = xyz[nbrs[1]] - c
    v3 = xyz[nbrs[2]] - c
    val = float(np.dot(np.cross(v1, v2), v3))
    if val > eps:
        return 1
    if val < -eps:
        return -1
    return 0


def modal_nonzero_sign(signs: list[int]) -> tuple[int, dict[str, int]]:
    counts = Counter(signs)
    pos = counts.get(1, 0)
    neg = counts.get(-1, 0)
    if pos > neg:
        return 1, {str(k): int(v) for k, v in sorted(counts.items())}
    if neg > pos:
        return -1, {str(k): int(v) for k, v in sorted(counts.items())}
    return 0, {str(k): int(v) for k, v in sorted(counts.items())}


def record_field(d: Any, name: str) -> Any:
    """Read a generated_files.pkl record field from PyG Data objects or dict records."""
    if isinstance(d, dict):
        return d[name]
    return getattr(d, name)


def eval_one(args: tuple[int, Any, float]) -> dict[str, Any]:
    mol_idx, d, eps = args
    smiles = str(record_field(d, "smiles"))
    pos_ref = as_numpy(record_field(d, "pos_ref"))
    pos_gen = as_numpy(record_field(d, "pos_gen"))
    n_atoms = int(pos_ref.shape[1])
    mol = mol_from_ordered_smiles(smiles, expected_atoms=n_atoms)
    centers = source_chiral_centers(mol)

    usable_centers: list[dict[str, Any]] = []
    skipped_centers: list[dict[str, Any]] = []
    for center in centers:
        signs = [stp_sign(xyz, center["center_idx"], center["neighbor_tuple"], eps) for xyz in pos_ref]
        expected, sign_counts = modal_nonzero_sign(signs)
        c2 = dict(center)
        c2["reference_sign_counts"] = sign_counts
        c2["expected_sign"] = expected
        if expected == 0:
            skipped_centers.append(c2)
        else:
            usable_centers.append(c2)

    rows = []
    status_counts: Counter[str] = Counter()
    center_match_count = 0
    center_total = 0
    n_strict_ok = 0
    per_center_fail_counts: Counter[str] = Counter()
    for gen_rank, xyz in enumerate(pos_gen, start=1):
        signs = {}
        mismatches = {}
        matches = 0
        for center in usable_centers:
            idx = int(center["center_idx"])
            sign = stp_sign(xyz, idx, center["neighbor_tuple"], eps)
            signs[str(idx)] = sign
            expected = int(center["expected_sign"])
            if sign == expected:
                matches += 1
            else:
                mismatches[str(idx)] = {
                    "expected_sign": expected,
                    "generated_sign": sign,
                    "neighbor_tuple": list(center["neighbor_tuple"]),
                    "source_chiral_tag": center["source_chiral_tag"],
                }
                per_center_fail_counts[str(idx)] += 1
        strict_ok = bool(usable_centers) and matches == len(usable_centers)
        if strict_ok:
            status = "strict_ok"
            n_strict_ok += 1
        elif not usable_centers:
            status = "no_usable_source_centers"
        elif matches == 0:
            status = "all_source_centers_mismatched"
        else:
            status = "partial_source_center_mismatch"
        status_counts[status] += 1
        center_match_count += matches
        center_total += len(usable_centers)
        rows.append(
            {
                "molecule_index": mol_idx,
                "conformer_rank_for_molecule": gen_rank,
                "status": status,
                "strict_ok": strict_ok,
                "n_source_centers": len(centers),
                "n_usable_source_centers": len(usable_centers),
                "n_matching_centers": matches,
                "generated_signs_by_center": json.dumps(signs, sort_keys=True),
                "mismatched_centers": json.dumps(mismatches, sort_keys=True),
                "smiles": smiles,
            }
        )

    return {
        "mol_idx": mol_idx,
        "smiles": smiles,
        "n_ref_confs": int(pos_ref.shape[0]),
        "n_gen_confs": int(pos_gen.shape[0]),
        "n_atoms": n_atoms,
        "n_source_centers": len(centers),
        "n_usable_source_centers": len(usable_centers),
        "usable_centers": usable_centers,
        "skipped_centers": skipped_centers,
        "rows": rows,
        "status_counts": dict(status_counts),
        "strict_conformer_pass_count": n_strict_ok,
        "center_level_match_count": center_match_count,
        "center_level_total": center_total,
        "any_strict": n_strict_ok > 0,
        "per_center_fail_counts": dict(per_center_fail_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated-pkl", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--molecule-summary-json", required=True)
    ap.add_argument("--num-workers", type=int, default=24)
    ap.add_argument("--max-molecules", type=int, default=0, help="0 means all")
    ap.add_argument("--eps", type=float, default=1e-7)
    args = ap.parse_args()

    with open(args.generated_pkl, "rb") as fh:
        data = pickle.load(fh)
    if args.max_molecules and args.max_molecules > 0:
        data = data[: args.max_molecules]

    jobs = [(i, d, args.eps) for i, d in enumerate(data)]
    if args.num_workers > 1 and len(jobs) > 1:
        with Pool(args.num_workers) as pool:
            results = list(pool.imap_unordered(eval_one, jobs))
    else:
        results = [eval_one(j) for j in jobs]
    results.sort(key=lambda r: r["mol_idx"])

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "generated_record",
        "molecule_index",
        "conformer_rank_for_molecule",
        "status",
        "strict_ok",
        "n_source_centers",
        "n_usable_source_centers",
        "n_matching_centers",
        "generated_signs_by_center",
        "mismatched_centers",
        "smiles",
    ]

    status_counts: Counter[str] = Counter()
    center_match = 0
    center_total = 0
    n_generated = 0
    n_chiral_generated = 0
    n_chiral_strict = 0
    n_source_centers_total = 0
    n_usable_source_centers_total = 0
    chiral_molecules = set()
    usable_chiral_molecules = set()
    strict_molecules = set()

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        rec = 0
        for res in results:
            status_counts.update(res["status_counts"])
            center_match += int(res["center_level_match_count"])
            center_total += int(res["center_level_total"])
            n_source_centers_total += int(res["n_source_centers"])
            n_usable_source_centers_total += int(res["n_usable_source_centers"])
            if res["n_source_centers"] > 0:
                chiral_molecules.add(res["mol_idx"])
            if res["n_usable_source_centers"] > 0:
                usable_chiral_molecules.add(res["mol_idx"])
                if res["any_strict"]:
                    strict_molecules.add(res["mol_idx"])
            for row in res["rows"]:
                row = dict(row)
                row["generated_record"] = rec
                rec += 1
                n_generated += 1
                if row["n_usable_source_centers"] > 0:
                    n_chiral_generated += 1
                    if row["strict_ok"] is True:
                        n_chiral_strict += 1
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    mol_summary = [
        {
            "molecule_index": r["mol_idx"],
            "smiles": r["smiles"],
            "n_atoms": r["n_atoms"],
            "n_ref_confs": r["n_ref_confs"],
            "n_gen_confs": r["n_gen_confs"],
            "n_source_centers": r["n_source_centers"],
            "n_usable_source_centers": r["n_usable_source_centers"],
            "usable_centers": r["usable_centers"],
            "skipped_centers": r["skipped_centers"],
            "status_counts": r["status_counts"],
            "strict_conformer_pass_count": r["strict_conformer_pass_count"],
            "center_level_match_count": r["center_level_match_count"],
            "center_level_total": r["center_level_total"],
        }
        for r in results
    ]
    mol_json = Path(args.molecule_summary_json)
    mol_json.parent.mkdir(parents=True, exist_ok=True)
    mol_json.write_text(json.dumps(mol_summary, indent=2, sort_keys=True))

    summary = {
        "generated_pkl": args.generated_pkl,
        "output_csv": str(out_csv),
        "molecule_summary_json": str(mol_json),
        "n_reference_molecules": len(results),
        "n_generated_records": n_generated,
        "n_confs_per_generated_molecule_min": min((r["n_gen_confs"] for r in results), default=0),
        "n_confs_per_generated_molecule_max": max((r["n_gen_confs"] for r in results), default=0),
        "source_chiral_molecule_count": len(chiral_molecules),
        "usable_source_chiral_molecule_count": len(usable_chiral_molecules),
        "source_chiral_center_count_total_by_molecule": n_source_centers_total,
        "usable_source_chiral_center_count_total_by_molecule": n_usable_source_centers_total,
        "chiral_generated_records": n_chiral_generated,
        "strict_conformer_pass_count": n_chiral_strict,
        "strict_conformer_pass_rate": n_chiral_strict / n_chiral_generated if n_chiral_generated else 0.0,
        "strict_molecule_any_pass_count": len(strict_molecules),
        "strict_molecule_any_pass_rate": len(strict_molecules) / len(usable_chiral_molecules) if usable_chiral_molecules else 0.0,
        "center_level_match_count": center_match,
        "center_level_total": center_total,
        "center_level_match_rate": center_match / center_total if center_total else 0.0,
        "status_counts": dict(status_counts),
        "eps": args.eps,
        "method": (
            "Coordinate scalar-triple-product/oriented-volume chirality. Source-specified tetrahedral centers are "
            "atoms with RDKit CHI_TETRAHEDRAL_CW/CCW in the ETFlow atom-mapped SMILES. Neighbor tuple is the "
            "sorted first-three atom indices. Expected handedness is the modal nonzero STP sign over reference "
            "conformers. A generated conformer is strict-correct if all usable source-specified centers match expected sign."
        ),
    }
    out_json = Path(args.summary_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
