#!/usr/bin/env bash
set -euo pipefail

# Train FMT response classification models for each stage-2 MGM2 scale and fold.
#
# Run from the repository root:
#   DEVICES="0 5" STAGE2_SCALES="scale_small" FOLDS="0 1 2 3 4" bash scripts/downstreams/FMT/train_fmt_classification_all_scales.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"

DATA_DIR="data/downstreams/FMT"
OUTPUT_ROOT="outputs_downstreams/FMT/community_classification"
STAGE2_ROOT="outputs_grid/stage2/20260505_094404"

TRAIN_DATA_FILE="${DATA_DIR}/community_train.pkl"
TEST_DATA_FILE="${DATA_DIR}/community_test.pkl"
TRAIN_METADATA_FILE="${DATA_DIR}/response_train_meta_split.csv"
TEST_METADATA_FILE="${DATA_DIR}/response_test_meta_split.csv"
TRAIN_NTV3_EMBEDDING="${DATA_DIR}/community_train.resolved_embeddings.h5"
TEST_NTV3_EMBEDDING="${DATA_DIR}/community_test.resolved_embeddings.h5"

STAGE2_SCALES=(${STAGE2_SCALES:-scale_small scale_medium scale_large scale_xlarge})
SEEDS=(${SEEDS:-11})
FOLDS=(${FOLDS:-0 1 2 3 4})
NUM_FOLDS="${NUM_FOLDS:-5}"

MAX_EPOCHS="${MAX_EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DEVICES=(${DEVICES:-0 5})
PRECISION="${PRECISION:-bf16}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-best}"

mkdir -p "${OUTPUT_ROOT}"

for required in "${TRAIN_DATA_FILE}" "${TEST_DATA_FILE}" "${TRAIN_METADATA_FILE}" "${TEST_METADATA_FILE}" "${TRAIN_NTV3_EMBEDDING}" "${TEST_NTV3_EMBEDDING}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required file not found: ${required}" >&2
        echo "Run first: bash scripts/downstreams/FMT/build_mgm2_fmt_dataset.sh" >&2
        exit 1
    fi
done

for model_scale in "${STAGE2_SCALES[@]}"; do
    config_path="${STAGE2_ROOT}/${model_scale}/config.json"
    checkpoint_path="${STAGE2_ROOT}/${model_scale}/checkpoints/last.ckpt"
    if [[ ! -f "${config_path}" ]]; then
        echo "[$(date '+%F %T')] Skipping ${model_scale}: config not found at ${config_path}"
        continue
    fi
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "[$(date '+%F %T')] Skipping ${model_scale}: checkpoint not found at ${checkpoint_path}"
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        for fold in "${FOLDS[@]}"; do
            run_name="${model_scale}_fmt_response_seed${seed}_fold${fold}"
            run_dir="${OUTPUT_ROOT}/${model_scale}/seed_${seed}/fold_${fold}"
            prediction_file="${run_dir}/test_predictions.csv"
            log_file="${run_dir}/train.log"
            mkdir -p "${run_dir}"

            echo "[$(date '+%F %T')] Running ${run_name} with ${checkpoint_path}"
            "${PYTHON_BIN}" -u src/training/FMTCommunityClassification.py \
                --train_data_file "${TRAIN_DATA_FILE}" \
                --train_metadata_file "${TRAIN_METADATA_FILE}" \
                --test_data_file "${TEST_DATA_FILE}" \
                --test_metadata_file "${TEST_METADATA_FILE}" \
                --ntv3_embedding_path "${TRAIN_NTV3_EMBEDDING}" \
                --test_ntv3_embedding_path "${TEST_NTV3_EMBEDDING}" \
                --config "${config_path}" \
                --fold_index "${fold}" \
                --num_folds "${NUM_FOLDS}" \
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
                --prediction_file "${prediction_file}" \
                --eval_checkpoint "${EVAL_CHECKPOINT}" \
                --seed "${seed}" \
                2>&1 | tee "${log_file}"
        done
    done
done

echo "[$(date '+%F %T')] Finished FMT response classification runs for available stage-2 scales"
