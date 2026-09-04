#!/usr/bin/env python3
"""Aggregate 4/5/6-mer outputs and compare them with the paper table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VARIANTS = {
    "cycpepflow_b": "CycPepFlow-B",
    "cycpepflow_apg_b": "CycPepFlow-APG-B",
    "cycpepflow_l": "CycPepFlow-L",
    "cycpepflow_apg_l": "CycPepFlow-APG-L",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, default=Path("metrics"))
    parser.add_argument("--expected", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--expected-stereo", type=Path, default=Path("results/model_scale_stereo.csv"))
    parser.add_argument("--output", type=Path, default=Path("metrics/reproduced_variant_results.json"))
    parser.add_argument("--cov-tolerance-pp", type=float, default=0.10)
    parser.add_argument("--mat-tolerance-a", type=float, default=0.002)
    parser.add_argument("--stp-tolerance-pp", type=float, default=0.10)
    args = parser.parse_args()

    expected = pd.read_csv(args.expected).set_index("method")
    expected_stereo = pd.read_csv(args.expected_stereo).set_index("model")
    results = []
    failures = []
    for stem, display in VARIANTS.items():
        frames = []
        strict_ok = 0
        strict_den = 0
        counts = {"molecules": 0, "reference_conformers": 0, "generated_conformers": 0}
        for nmer in (4, 5, 6):
            base = args.metrics_root / stem / f"{nmer}mer"
            cov_path = base / "metrics" / "cremp_per_molecule_covmat.csv"
            summary_path = base / "metrics" / "cremp_summary.json"
            stereo_path = base / "stereochemistry_summary.json"
            frame = pd.read_csv(cov_path)
            if frame["manifest_split_index"].duplicated().any():
                raise RuntimeError(f"duplicate split indices in {cov_path}")
            frames.append(frame)
            summary = json.loads(summary_path.read_text())
            stereo = json.loads(stereo_path.read_text())
            counts["molecules"] += int(summary["n_molecules_input"])
            counts["reference_conformers"] += int(summary["n_reference_conformers"])
            counts["generated_conformers"] += int(summary["n_generated_conformers"])
            strict_ok += int(stereo["strict_conformer_pass_count"])
            strict_den += int(stereo["chiral_generated_records"])

        pooled = pd.concat(frames, ignore_index=True)
        if len(pooled) != 1000 or pooled["manifest_split_index"].nunique() != 1000:
            raise RuntimeError(f"{display}: expected 1,000 unique test molecules, got {len(pooled)}")
        row = {
            "Model": display,
            "COV-R": 100.0 * pooled["cov_recall_at_threshold"].mean(),
            "COV-P": 100.0 * pooled["cov_precision_at_threshold"].mean(),
            "MAT-R": pooled["mat_recall"].mean(),
            "MAT-P": pooled["mat_precision"].mean(),
            "STP": 100.0 * strict_ok / strict_den,
            **counts,
        }
        exp = expected.loc[display]
        exp_stereo = expected_stereo.loc[display]
        deltas = {
            "COV-R": row["COV-R"] - float(exp["COV_R_pct"]),
            "COV-P": row["COV-P"] - float(exp["COV_P_pct"]),
            "MAT-R": row["MAT-R"] - float(exp["MAT_R_A"]),
            "MAT-P": row["MAT-P"] - float(exp["MAT_P_A"]),
            "STP": row["STP"] - float(exp_stereo["strict_STP_pct"]),
        }
        within = (
            abs(deltas["COV-R"]) <= args.cov_tolerance_pp
            and abs(deltas["COV-P"]) <= args.cov_tolerance_pp
            and abs(deltas["MAT-R"]) <= args.mat_tolerance_a
            and abs(deltas["MAT-P"]) <= args.mat_tolerance_a
            and abs(deltas["STP"]) <= args.stp_tolerance_pp
        )
        row["delta_from_paper"] = deltas
        row["within_tolerance"] = within
        results.append(row)
        if not within:
            failures.append(display)

    payload = {"status": "ok" if not failures else "mismatch", "results": results, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
