#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Cell Reports FMT donor-similarity classification setup:
# random 72%/8%/20% train/validation/test splits, repeated 10 times.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DATA_DIR="${DATA_DIR:-data/downstreams/FMT_cellreports}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/FMT_cellreports/community_similarity_random}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs_grid/stage2/20260505_094404}"

DATA_FILE="${DATA_DIR}/community_all.pkl"
NTV3_EMBEDDING="${DATA_DIR}/community_all.cellreports_embeddings.h5"
SPLIT_ROOT="${DATA_DIR}/similarity_random_splits"

STAGE2_SCALES=(${STAGE2_SCALES:-scale_small scale_medium scale_large scale_xlarge})
SPLITS=(${SPLITS:-0 1 2 3 4 5 6 7 8 9})
SEED_OFFSET="${SEED_OFFSET:-11}"
DEVICES=(${DEVICES:-0})
MAX_EPOCHS="${MAX_EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PRECISION="${PRECISION:-bf16}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-best}"
SEQUENCE_MODE="${SEQUENCE_MODE:-sparse}"

for required in "${DATA_FILE}" "${NTV3_EMBEDDING}" "${SPLIT_ROOT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path not found: ${required}" >&2
    echo "Run first: ${PYTHON_BIN} scripts/downstreams/FMT_cellreports/prepare_similarity_random_splits.py --data_dir ${DATA_DIR}" >&2
    exit 1
  fi
done

for model_scale in "${STAGE2_SCALES[@]}"; do
  config_path="${STAGE2_ROOT}/${model_scale}/config.json"
  checkpoint_path="${STAGE2_ROOT}/${model_scale}/checkpoints/last.ckpt"
  if [[ ! -f "${config_path}" || ! -f "${checkpoint_path}" ]]; then
    echo "[$(date '+%F %T')] Skipping ${model_scale}: missing config or last checkpoint"
    continue
  fi
  for split in "${SPLITS[@]}"; do
    split_padded="$(printf '%02d' "${split}")"
    split_dir="${SPLIT_ROOT}/split_${split_padded}"
    train_meta="${split_dir}/train_meta.csv"
    val_meta="${split_dir}/val_meta.csv"
    test_meta="${split_dir}/test_meta.csv"
    [[ -f "${train_meta}" && -f "${val_meta}" && -f "${test_meta}" ]] || continue
    seed="$((SEED_OFFSET + split))"
    run_name="${model_scale}_cellreports_similarity_random_split${split_padded}_seed${seed}"
    run_dir="${OUTPUT_ROOT}/${model_scale}/split_${split_padded}"
    mkdir -p "${run_dir}"
    echo "[$(date '+%F %T')] Running ${run_name}"
    "${PYTHON_BIN}" -u src/training/FMTCommunityClassification.py \
      --train_data_file "${DATA_FILE}" \
      --train_metadata_file "${train_meta}" \
      --val_metadata_file "${val_meta}" \
      --test_data_file "${DATA_FILE}" \
      --test_metadata_file "${test_meta}" \
      --ntv3_embedding_path "${NTV3_EMBEDDING}" \
      --test_ntv3_embedding_path "${NTV3_EMBEDDING}" \
      --config "${config_path}" \
      --max_epochs "${MAX_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --num_workers "${NUM_WORKERS}" \
      --prefetch_factor "${PREFETCH_FACTOR}" \
      --devices "${DEVICES[@]}" \
      --precision "${PRECISION}" \
      --output_dir "${run_dir}" \
      --freeze_backbone \
      --pretrained_checkpoint "${checkpoint_path}" \
      --experiment_name "${run_name}" \
      --prediction_file "${run_dir}/test_predictions.csv" \
      --eval_checkpoint "${EVAL_CHECKPOINT}" \
      --class_names similarity_increased similarity_decreased \
      --sequence_mode "${SEQUENCE_MODE}" \
      --seed "${seed}" \
      2>&1 | tee "${run_dir}/train.log"
  done
done

echo "[$(date '+%F %T')] Finished Cell Reports donor-similarity random split runs"
