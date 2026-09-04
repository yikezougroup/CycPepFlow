#!/usr/bin/env python3
"""Download the four private CycPepFlow release assets and verify SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = "yikezougroup/CycPepFlow"
TAG = "v0.1.1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    if shutil.which("gh") is None:
        raise SystemExit("GitHub CLI `gh` is required: https://cli.github.com/")
    auth = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if auth.returncode:
        raise SystemExit("Authenticate first with `gh auth login` (private repository).")

    args.dir.mkdir(parents=True, exist_ok=True)
    command = [
        "gh", "release", "download", args.tag,
        "--repo", args.repo,
        "--dir", str(args.dir),
        "--pattern", "*.ckpt",
        "--clobber",
    ]
    subprocess.run(command, check=True)

    manifest_path = args.dir / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing committed manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for row in manifest["checkpoints"]:
        path = args.dir / row["release_filename"]
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        actual = sha256(path)
        if actual != row["release_checkpoint_sha256"]:
            failures.append(f"SHA256 mismatch for {path}: {actual}")
        else:
            print(f"OK  {path}  {actual}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
