#!/usr/bin/env python3
"""Static release audit: required files, privacy-sensitive paths, and exclusions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REQUIRED = {
    "LICENSE", "LICENSES/ET-Flow-MIT.txt", "NOTICE", "README.md",
    "pyproject.toml", "environment.yml",
    "cycpepflow/__init__.py", "cycpepflow/data/__init__.py",
    "cycpepflow/data/dataset.py", "cycpepflow/models/model.py",
    "configs/cycpepflow_b.yaml", "configs/cycpepflow_apg_b.yaml",
    "configs/cycpepflow_l.yaml", "configs/cycpepflow_apg_l.yaml",
    "checkpoints/checkpoint_manifest.json", "checkpoints/SHA256SUMS",
    "benchmark/splits/ringer_cremp_combined_manifest.csv",
    "benchmark/splits/test_manifest.csv",
    "benchmark/splits/split_summary.json",
    "results/main_results.csv", "results/per_size_results.csv",
    "results/model_scale_stereo.csv", "scripts/aggregate_results.py",
    "scripts/generate_cremp.py", "scripts/score_covmat.py",
    "scripts/score_stereochemistry.py", "scripts/convert_cremp.py",
}
FORBIDDEN_CONTENT = re.compile(
    "/" + "scratch/|/" + "public/home/|/" + "home/wdzwjc|"
    + "wang" + "aiting|wu" + "dizhou|"
    + r"ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".csv", ".slurm"}
CODE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".slurm"}
LEGACY_PACKAGE = "et" + "flow"
LEGACY_ENV_PREFIX = "ET" + "FLOW_"
LEGACY_NAMESPACE = re.compile(
    rf"\b{LEGACY_PACKAGE}\b|\b{LEGACY_ENV_PREFIX}",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    failures = []

    def excluded(path: Path) -> bool:
        return ".git" in path.parts or "__pycache__" in path.parts

    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not excluded(p)}
    missing = sorted(REQUIRED - present)
    if missing:
        failures.append({"missing_required": missing})

    forbidden_names = sorted(
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and p.name.lower() == "train.py" and not excluded(p)
    )
    if forbidden_names:
        failures.append({"forbidden_train_py": forbidden_names})

    if (root / LEGACY_PACKAGE).exists():
        failures.append({"legacy_package_directory": LEGACY_PACKAGE})

    legacy_namespace_files = []
    for p in root.rglob("*"):
        if (
            not p.is_file()
            or excluded(p)
            or p.suffix.lower() not in CODE_SUFFIXES
        ):
            continue
        if LEGACY_NAMESPACE.search(p.read_text(errors="replace")):
            legacy_namespace_files.append(str(p.relative_to(root)))
    if legacy_namespace_files:
        failures.append({"legacy_namespace_content": sorted(legacy_namespace_files)})

    path_leaks = []
    for p in root.rglob("*"):
        if not p.is_file() or excluded(p) or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = p.read_text(errors="replace")
        if FORBIDDEN_CONTENT.search(text):
            path_leaks.append(str(p.relative_to(root)))
    if path_leaks:
        failures.append({"forbidden_content": sorted(path_leaks)})

    license_text = (root / "LICENSE").read_text(errors="replace") if (root / "LICENSE").exists() else ""
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        failures.append({"license": "LICENSE is not Apache-2.0"})

    result = {
        "status": "failed" if failures else "ok",
        "root": str(root),
        "file_count": len(present),
        "train_py_count": len(forbidden_names),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
