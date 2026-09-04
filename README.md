# CycPepFlow

**Topology-Aware Equivariant Flow Matching for Cyclic Peptide Conformer Ensembles**

This private pre-release contains the inference and evaluation code, exact CREMP/RINGER split manifest, paper-value tables, and the four validation-selected checkpoints used for the CycPepFlow paper.

> **Release scope.** This repository reproduces checkpoint inference and the reported held-out-test metrics. The training entry point (`train.py`) is intentionally **not distributed**, and optimizer/scheduler/trainer state has been removed from the release checkpoints. These checkpoints cannot be used to resume training.

## Released variants

All variants use 20 local equivariant-attention layers, scalar gated global attention in every layer, covalent + exact 2-hop + exact 3-hop + dynamic nonbonded radius edges, ProductLite residuals at layers 5/10/15/20, formal-charge and chirality conditioning, and signed-volume chirality loss during training. APG adds a head-specific all-pairs shortest-path bias to global attention.

| Variant | Width | Heads | APG | Parameters | Selected epoch | Global step | Validation loss |
|---|---:|---:|:---:|---:|---:|---:|---:|
| CycPepFlow-B | 192 | 8 | No | 18,303,745 | 144 | 71,775 | 1.1103653908 |
| CycPepFlow-APG-B | 192 | 8 | Yes | 18,309,185 | 144 | 71,775 | 1.1119488478 |
| CycPepFlow-L | 300 | 12 | No | 43,422,219 | 144 | 71,775 | 1.0934225321 |
| CycPepFlow-APG-L | 300 | 12 | Yes | 43,430,379 | 185 | 92,070 | 1.0600241423 |

Selection was performed independently for each run using the minimum finite epoch-level `val/loss` among the retained top-five Lightning checkpoints. The selected epoch/global-step metadata and both original/release SHA-256 digests are recorded in [`checkpoints/checkpoint_manifest.json`](checkpoints/checkpoint_manifest.json).

## Paper reproduction targets

Official 1,000-molecule CREMP test split; all 877,898 retained reference conformers; exactly `2K` generated conformers for a molecule with `K` references; symmetry-aware heavy-atom RMSD; COV threshold `δ = 0.75 Å`; metrics averaged uniformly over molecules; strict STP pooled over generated conformers with usable source-specified stereocenters.

| Variant | COV-R ↑ | COV-P ↑ | COV-F1 ↑ | MAT-R ↓ | MAT-P ↓ | MAT-F1 ↓ | Strict STP ↑ | Center STP ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CycPepFlow-B | 74.95% | 77.18% | 76.05% | 0.531 Å | 0.461 Å | 0.494 Å | 99.632% | 99.914% |
| CycPepFlow-APG-B | 73.81% | 77.99% | 75.85% | 0.543 Å | 0.449 Å | 0.492 Å | 99.648% | 99.926% |
| CycPepFlow-L | 74.16% | 78.21% | **76.13%** | 0.539 Å | 0.441 Å | 0.485 Å | 99.643% | 99.925% |
| CycPepFlow-APG-L | 72.65% | **79.78%** | 76.05% | 0.555 Å | **0.419 Å** | **0.477 Å** | **99.744%** | **99.942%** |

Unrounded pooled values are in [`results/main_results.csv`](results/main_results.csv), exact length-stratified values are in [`results/per_size_results.csv`](results/per_size_results.csv), and model/stereochemistry values are in [`results/model_scale_stereo.csv`](results/model_scale_stereo.csv).

## Repository layout

```text
benchmark/splits/    exact seed-6489 train/validation/test identity manifest
checkpoints/         checkpoint manifest and SHA-256 checksums (weights are release assets)
configs/             four inference/evaluation configurations
cycpepflow/          CycPepFlow model, inference data loader, and utilities
results/             unrounded paper values used as reproduction targets
scripts/             data conversion, generation, scoring, aggregation, and audits
slurm/               resource-matched generation and scoring templates
```

## 1. Installation

The paper runs were executed on Linux with Python 3.12.13, PyTorch 2.4.1, CUDA 12.1, PyTorch Geometric 2.6.1, `torch-cluster` 1.6.3, Lightning 2.6.1, NumPy 1.26.4, RDKit 2026.03.1, and datamol 0.12.5. The network forward used BF16 on NVIDIA RTX 4090 D GPUs; ODE state and saved coordinates remained FP32.

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate cycpepflow
python -m pip install -e .
```

If your solver cannot obtain the PyG CUDA extensions from Conda, install PyTorch 2.4.1/CUDA 12.1 first and then use the matching PyG wheel index:

```bash
python -m pip install torch-geometric==2.6.1
python -m pip install torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install -e .
```

Verify imports and the static release contract:

```bash
python -c "import torch, torch_geometric, torch_cluster, rdkit, cycpepflow; print(torch.__version__)"
python scripts/audit_release.py
```

## 2. Download and verify the four checkpoints

The weights are GitHub Release assets, not Git blobs. Because the repository is private, authenticate with an account that has repository access:

```bash
gh auth login                       # omit when already authenticated
python scripts/download_checkpoints.py
python scripts/verify_checkpoints.py
```

The verification script reconstructs every architecture from its release YAML, checks the exact parameter count, performs a strict `state_dict` load, and verifies SHA-256. Expected files:

```text
checkpoints/cycpepflow-b.ckpt
checkpoints/cycpepflow-apg-b.ckpt
checkpoints/cycpepflow-l.ckpt
checkpoints/cycpepflow-apg-l.ckpt
```

## 3. Prepare the exact CREMP/RINGER split

The split manifest in this repository contains 31,678 training, 3,520 validation, and 1,000 held-out test molecules, stratified by peptide length with seed 6489. Training/validation retain at most 30 conformers per molecule; test retains all references. The held-out test set contains 533 four-mers, 347 five-mers, and 120 six-mers.

Download the official corrected CREMP pickle archive from Zenodo record [8010582](https://zenodo.org/records/8010582):

- archive: `pickle.tar.gz`
- compressed size: 28.9 GB; approximately 32 GB extracted
- MD5: `925d058e9d96942e5aca55b12480efc3`

Use an existing local/HPC copy when available. For a new download, use a resumable direct transfer and verify the published checksum:

```bash
mkdir -p downloads
wget -c 'https://zenodo.org/records/8010582/files/pickle.tar.gz?download=1' \
  -O downloads/pickle.tar.gz
printf '%s  %s\n' 925d058e9d96942e5aca55b12480efc3 downloads/pickle.tar.gz | md5sum -c -
```

Convert directly from the archive (no extraction required):

```bash
mkdir -p data/processed/ringer_cremp_top30_testall data/reports
python scripts/convert_cremp.py \
  --archive downloads/pickle.tar.gz \
  --manifest benchmark/splits/ringer_cremp_combined_manifest.csv \
  --out-root data/processed/ringer_cremp_top30_testall \
  --max-confs 30 \
  --all-confs-splits test \
  --summary-json data/reports/conversion_summary.json \
  --failures-csv data/reports/conversion_failures.csv
```

Alternatively, pass `--pickle-dir /path/to/extracted/pickle` instead of `--archive`. Before generation, verify that `data/processed/ringer_cremp_top30_testall/test/` contains all 1,000 test records and that the conversion summary reports 877,898 retained test conformers.

> **Serialized-data safety.** The official CREMP input is Python-pickle data, so only use the checksum-verified Zenodo archive above. CycPepFlow v0.1.1 writes restricted-load-compatible tensor/primitive `.pt` records. If you reuse trusted records created by v0.1.0, generation requires `--allow-unsafe-legacy-records`; this enables pickle execution and must never be used with untrusted files. Re-conversion is safer.

## 4. One-molecule smoke generation

This checks an actual checkpoint forward pass, 50-step ODE integration, output serialization, COV/MAT plumbing, and strict stereochemistry evaluation. The cap is only for a fast smoke and must not be used for the paper benchmark.

```bash
mkdir -p generated/smoke metrics/smoke
python scripts/generate_cremp.py \
  --config configs/cycpepflow_b.yaml \
  --checkpoint checkpoints/cycpepflow-b.ckpt \
  --data_dir data/processed \
  --partition ringer_cremp_top30_testall \
  --manifest benchmark/splits/test_manifest.csv \
  --num-monomers 4 \
  --shard_id 0 --num_shards 1 \
  --max_molecules 1 --max_ref_confs 1 \
  --batch_size 2 --network-amp bf16 --seed 42 \
  --out generated/smoke/shard_0.pkl \
  --summary generated/smoke/shard_0.summary.json

python scripts/score_covmat.py \
  --parts generated/smoke/shard_0.pkl \
  --outdir metrics/smoke \
  --manifest benchmark/splits/test_manifest.csv \
  --num-monomers 4 --threshold 0.75 --num_workers 1 \
  --limit-filtered-molecules 1

python scripts/score_stereochemistry.py \
  --generated-pkl metrics/smoke/generated_files.pkl \
  --output-csv metrics/smoke/stereochemistry.csv \
  --summary-json metrics/smoke/stereochemistry_summary.json \
  --molecule-summary-json metrics/smoke/stereochemistry_by_molecule.json \
  --num-workers 1
```

## 5. Full paper benchmark

### Exact shared inference contract

- sampler: deterministic Euler ODE
- integration steps: 50
- prior: zero-center-of-mass harmonic, `α = 1.0`
- flow path noise: `σ = 0.1`
- generation seed: 42
- generated samples: exactly `2K` for `K` references per molecule
- network precision: BF16; ODE state/output: FP32
- checkpoint-specific generation batch: 192 for B/APG-B, 96 for L/APG-L
- no parity/mirror flip
- no force-field relaxation
- no STP repair

The included SLURM templates use one GPU plus 4 CPUs/16 GB for each generation shard and 64 CPUs/256 GB for each scoring job, preserving approximately 4 GB per requested CPU core. They use normal QoS and a maximum two-day time limit.

Create the log directory **before** submission because SLURM opens output paths before the script body runs:

```bash
mkdir -p logs generated metrics
variants=(cycpepflow_b cycpepflow_apg_b cycpepflow_l cycpepflow_apg_l)
for variant in "${variants[@]}"; do
  for nmer in 4 5 6; do
    gen_id=$(sbatch --parsable \
      --export=ALL,VARIANT="$variant",NMER="$nmer" \
      slurm/generate_array.slurm)
    sbatch --dependency="afterok:${gen_id}" \
      --export=ALL,VARIANT="$variant",NMER="$nmer" \
      slurm/score_length.slurm
  done
done
```

Do not aggregate until all 12 scoring jobs complete successfully. Then reproduce the pooled 1,000-molecule table and compare it with the bundled unrounded targets:

```bash
python scripts/aggregate_results.py
```

The default acceptance window is ±0.10 percentage points for COV/strict STP and ±0.002 Å for MAT. The exact software stack and matched NVIDIA hardware should normally agree more closely; different GPU architectures, CUDA kernels, or BF16 numerics can move conformers close to the 0.75 Å threshold.

## Metric definitions

For molecule-specific symmetry-aware heavy-atom RMSD matrix `D(r,g)`:

- **COV-R:** fraction of references with `min_g D(r,g) < 0.75 Å`.
- **COV-P:** fraction of generated conformers with `min_r D(r,g) < 0.75 Å`.
- **MAT-R:** mean over references of `min_g D(r,g)`.
- **MAT-P:** mean over generated conformers of `min_r D(r,g)`.
- **F1:** harmonic mean of the unrounded aggregate recall and precision (MAT-F1 is retained for direct comparison with RINGER and must be read beside MAT-R/P).
- **Strict STP:** a generated conformer passes only if every usable source-specified tetrahedral center matches the modal nonzero signed-volume orientation of the references.
- **Center STP:** pooled agreement across individual evaluated stereocenters.

COV/MAT are first computed per molecule and then averaged uniformly across all 1,000 molecules. Strict STP excludes achiral molecules and centers without a usable ordered-neighbor reference sign from its denominator.

## Reproducibility and provenance

- The four checkpoint pointers were read from each run's `selected_best_checkpoint.txt` and cross-checked against epoch-level Lightning validation metrics.
- Release checkpoints preserve model tensors exactly; only optimizer, LR-scheduler, callback, and trainer-loop state was removed.
- `checkpoint_manifest.json` records original and release SHA-256 values, selected epoch/step, validation loss, parameter count, and strict-load verification.
- The exact split identity—not just split sizes—is committed in `benchmark/splits/ringer_cremp_combined_manifest.csv`.
- The generation and scoring programs are the paper-run implementations with internal absolute paths replaced by repository-relative paths.
- Reported samples use no post-hoc parity operation and no force-field relaxation.

## Intentionally not included

- `train.py` or any training entry point
- optimizer/scheduler/trainer state required to resume training
- private experiment logs, credentials, internal absolute paths, or cluster-specific run directories
- the externally licensed 28.9 GB CREMP archive

## License and third-party notices

CycPepFlow additions and modifications are licensed under the [Apache License 2.0](LICENSE). Source adapted from ET-Flow remains accompanied by its original MIT notice in [`LICENSES/ET-Flow-MIT.txt`](LICENSES/ET-Flow-MIT.txt); see [`NOTICE`](NOTICE). CREMP data are distributed separately under the license stated by the Zenodo record and are not redistributed here.

## Citation

The paper is currently under review. Please cite:

> *CycPepFlow: Topology-Aware Equivariant Flow Matching for Cyclic Peptide Conformer Ensembles.* Under review, 2026.

This section will be replaced with the final BibTeX/DOI after publication.
