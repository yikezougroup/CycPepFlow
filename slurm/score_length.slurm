#!/bin/bash
#SBATCH -J cpf_score
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/score_%j.out
#SBATCH -e logs/score_%j.err

set -euo pipefail

: "${VARIANT:?set VARIANT to a release config stem}"
: "${NMER:?set NMER to 4, 5, or 6}"
: "${OUT_ROOT:=generated}"
: "${METRICS_ROOT:=metrics}"
: "${NUM_WORKERS:=${SLURM_CPUS_PER_TASK:-64}}"

mkdir -p logs "$METRICS_ROOT/$VARIANT/${NMER}mer"
shopt -s nullglob
PARTS=("$OUT_ROOT/$VARIANT/${NMER}mer"/shard_*.pkl)
if [[ "${#PARTS[@]}" -ne 8 ]]; then
  echo "Expected 8 generated shards; found ${#PARTS[@]}" >&2
  exit 2
fi

python scripts/score_covmat.py \
  --parts "${PARTS[@]}" \
  --outdir "$METRICS_ROOT/$VARIANT/${NMER}mer" \
  --manifest benchmark/splits/ringer_cremp_combined_manifest.csv \
  --num-monomers "$NMER" \
  --threshold 0.75 \
  --num_workers "$NUM_WORKERS"

python scripts/score_stereochemistry.py \
  --generated-pkl "$METRICS_ROOT/$VARIANT/${NMER}mer/generated_files.pkl" \
  --output-csv "$METRICS_ROOT/$VARIANT/${NMER}mer/stereochemistry.csv" \
  --summary-json "$METRICS_ROOT/$VARIANT/${NMER}mer/stereochemistry_summary.json" \
  --molecule-summary-json "$METRICS_ROOT/$VARIANT/${NMER}mer/stereochemistry_by_molecule.json" \
  --num-workers "$NUM_WORKERS"
