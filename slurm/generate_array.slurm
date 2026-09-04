#!/bin/bash
#SBATCH -J cpf_gen
#SBATCH -p gpu_cpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH -a 0-7%8
#SBATCH -o logs/generate_%A_%a.out
#SBATCH -e logs/generate_%A_%a.err

set -euo pipefail

: "${VARIANT:?set VARIANT to cycpepflow_b, cycpepflow_apg_b, cycpepflow_l, or cycpepflow_apg_l}"
: "${NMER:?set NMER to 4, 5, or 6}"
: "${DATA_DIR:=data/processed}"
: "${PARTITION:=ringer_cremp_top30_full456_test6}"
: "${OUT_ROOT:=generated}"
: "${NETWORK_AMP:=bf16}"
: "${NUM_SHARDS:=8}"
: "${SEED:=42}"

case "$VARIANT" in
  cycpepflow_b|cycpepflow_apg_b) BATCH_SIZE="${BATCH_SIZE:-192}" ;;
  cycpepflow_l|cycpepflow_apg_l) BATCH_SIZE="${BATCH_SIZE:-96}" ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

mkdir -p logs "$OUT_ROOT/$VARIANT/${NMER}mer"
python scripts/generate_cremp.py \
  --config "configs/${VARIANT}.yaml" \
  --checkpoint "checkpoints/${VARIANT//_/-}.ckpt" \
  --data_dir "$DATA_DIR" \
  --partition "$PARTITION" \
  --manifest benchmark/splits/test_manifest.csv \
  --num-monomers "$NMER" \
  --shard_id "$SLURM_ARRAY_TASK_ID" \
  --num_shards "$NUM_SHARDS" \
  --batch_size "$BATCH_SIZE" \
  --network-amp "$NETWORK_AMP" \
  --seed "$SEED" \
  --out "$OUT_ROOT/$VARIANT/${NMER}mer/shard_${SLURM_ARRAY_TASK_ID}.pkl" \
  --summary "$OUT_ROOT/$VARIANT/${NMER}mer/shard_${SLURM_ARRAY_TASK_ID}.summary.json"
