#!/usr/bin/env bash
set -euo pipefail

# Train the fully connected FMT response classification baseline for 5 folds.
#
# Run from the repository root:
#   DEVICE="cuda:0" FOLDS="0 1 2 3 4" bash scripts/downstreams/FMT/train_mlp_classification_baseline.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"

DATA_DIR="data/downstreams/FMT"
OUTPUT_ROOT="outputs_downstreams/FMT/mlp_classification_baseline"

TRAIN_DATA_FILE="${DATA_DIR}/community_train.pkl"
TEST_DATA_FILE="${DATA_DIR}/community_test.pkl"
TRAIN_METADATA_FILE="${DATA_DIR}/response_train_meta_split.csv"
TEST_METADATA_FILE="${DATA_DIR}/response_test_meta_split.csv"
TRAIN_NTV3_EMBEDDING="${DATA_DIR}/community_train.resolved_embeddings.h5"
TEST_NTV3_EMBEDDING="${DATA_DIR}/community_test.resolved_embeddings.h5"

SEEDS=(${SEEDS:-11})
FOLDS=(${FOLDS:-0 1 2 3 4})
NUM_FOLDS="${NUM_FOLDS:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.2}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PATIENCE="${PATIENCE:-30}"
DEVICE="${DEVICE:-auto}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"

mkdir -p "${OUTPUT_ROOT}"

for required in "${TRAIN_DATA_FILE}" "${TEST_DATA_FILE}" "${TRAIN_METADATA_FILE}" "${TEST_METADATA_FILE}" "${TRAIN_NTV3_EMBEDDING}" "${TEST_NTV3_EMBEDDING}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required file not found: ${required}" >&2
        echo "Run first: bash scripts/downstreams/FMT/build_mgm2_fmt_dataset.sh" >&2
        exit 1
    fi
done

for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/seed_${seed}/fold_${fold}"
        log_file="${run_dir}/train.log"
        mkdir -p "${run_dir}"
        echo "[$(date '+%F %T')] Running FMT MLP response baseline seed=${seed} fold=${fold}"
        "${PYTHON_BIN}" -u scripts/downstreams/FMT/train_mlp_classification_baseline.py \
            --train_data_file "${TRAIN_DATA_FILE}" \
            --train_metadata_file "${TRAIN_METADATA_FILE}" \
            --test_data_file "${TEST_DATA_FILE}" \
            --test_metadata_file "${TEST_METADATA_FILE}" \
            --ntv3_embedding_path "${TRAIN_NTV3_EMBEDDING}" \
            --test_ntv3_embedding_path "${TEST_NTV3_EMBEDDING}" \
            --fold_index "${fold}" \
            --num_folds "${NUM_FOLDS}" \
            --max_epochs "${MAX_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --hidden_size "${HIDDEN_SIZE}" \
            --num_layers "${NUM_LAYERS}" \
            --dropout "${DROPOUT}" \
            --learning_rate "${LEARNING_RATE}" \
            --weight_decay "${WEIGHT_DECAY}" \
            --patience "${PATIENCE}" \
            --class_weight "${CLASS_WEIGHT}" \
            --device "${DEVICE}" \
            --output_dir "${run_dir}" \
            --seed "${seed}" \
            2>&1 | tee "${log_file}"
    done
done

echo "[$(date '+%F %T')] Finished FMT MLP response classification baseline"
