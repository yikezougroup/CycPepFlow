# CycPepFlow v0.1.1 — private checkpoint release

## Unreleased — inference-core cleanup

- Replaced the training-framework base with a checkpoint-compatible plain PyTorch model.
- Removed unused training, upstream-download, xTB, duplicate sampling/scoring,
  obsolete I/O, and unused output-head/distance code.
- Removed direct Lightning, TorchMetrics, Pydantic, and fsspec requirements.
- Reused static hop/APG topology during integration while preserving dynamic radius edges.
- Consolidated record featurization to one SMILES parse and fixed stale reference
  conformers when distinct records share the same SMILES.
- Preserved original configuration files, checkpoint assets, scientific scoring,
  Apache license, and the required upstream MIT attribution.
- Reduced the Python API to the documented inference workflow; see README migration notes.

The published v0.1.1 assets and tag are not changed by this unreleased cleanup.

## Published v0.1.1

This private pre-release contains the four validation-selected checkpoints used in the CycPepFlow paper:

- CycPepFlow-B
- CycPepFlow-APG-B
- CycPepFlow-L
- CycPepFlow-APG-L

Each asset is an inference-only Lightning checkpoint with a tensor-identical model `state_dict`; optimizer, scheduler, callback, and trainer-loop state has been removed. SHA-256 digests, selected epochs/steps, validation losses, exact parameter counts, and source-checkpoint digests are recorded in `checkpoints/checkpoint_manifest.json`.

The repository includes exact inference/evaluation configs, the official-split identity manifest, generation and COV/MAT/strict-STP evaluation scripts, expected unrounded paper values, SLURM templates, and a detailed reproduction guide.

The training entry point (`train.py`) is intentionally not distributed. The public Python namespace is `cycpepflow`; the release also includes a purpose-built, inference-only loader for converted cyclic-peptide records. CycPepFlow modifications are licensed under Apache-2.0; adapted ET-Flow source retains its MIT notice.

## Changes from v0.1.0

- Renamed the installed source package and all operational imports from `etflow` to `cycpepflow`.
- Added the missing inference-only processed-record loader required by `scripts/generate_cremp.py`.
- Changed converted `.pt` output to a restricted-load-compatible tensor/primitive format; legacy pickle-backed records require an explicit trust opt-in.
- Added namespace/package checks to the static release audit and expanded loader regression tests.
- Removed the nonfunctional optional STP-repair CLI path; released inference remains explicitly unmodified.
- Kept the four checkpoint tensors and their SHA-256 digests unchanged.
