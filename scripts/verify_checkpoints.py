#!/usr/bin/env python3
"""Strictly instantiate and load all four released CycPepFlow checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.models import BaseFlow  # noqa: E402

VARIANTS = {
    "CycPepFlow-B": ("cycpepflow_b.yaml", "cycpepflow-b.ckpt", 18_303_745),
    "CycPepFlow-APG-B": ("cycpepflow_apg_b.yaml", "cycpepflow-apg-b.ckpt", 18_309_185),
    "CycPepFlow-L": ("cycpepflow_l.yaml", "cycpepflow-l.ckpt", 43_422_219),
    "CycPepFlow-APG-L": ("cycpepflow_apg_l.yaml", "cycpepflow-apg-l.ckpt", 43_430_379),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    manifest_path = args.checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    by_file = {x["release_filename"]: x for x in manifest["checkpoints"]}
    results = []

    torch.set_num_threads(1)
    for name, (config_name, checkpoint_name, expected_params) in VARIANTS.items():
        cfg = yaml.safe_load((args.config_dir / config_name).read_text())
        model = BaseFlow(**cfg["model_args"])
        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count != expected_params:
            raise RuntimeError(f"{name}: {parameter_count=} != {expected_params=}")

        ckpt_path = args.checkpoint_dir / checkpoint_name
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if not args.skip_sha256:
            actual_sha = sha256(ckpt_path)
            expected_sha = by_file[checkpoint_name]["release_checkpoint_sha256"]
            if actual_sha != expected_sha:
                raise RuntimeError(f"{name}: SHA256 mismatch")
        else:
            actual_sha = None

        results.append({
            "variant": name,
            "checkpoint": str(ckpt_path),
            "parameter_count": parameter_count,
            "state_dict_tensors": len(checkpoint["state_dict"]),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "sha256": actual_sha,
            "strict_load": True,
        })
        del checkpoint, model

    print(json.dumps({"status": "ok", "checkpoints": results}, indent=2))


if __name__ == "__main__":
    main()
