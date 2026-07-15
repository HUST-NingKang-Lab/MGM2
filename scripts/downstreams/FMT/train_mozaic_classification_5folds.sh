#!/usr/bin/env bash
set -euo pipefail

# Train MOZAIC reproduction with standard internal/outer 5-fold FMT response CV.
#
# Run from the repository root:
#   DEVICE="cuda:0" FOLDS="0 1 2 3 4" bash scripts/downstreams/FMT/train_mozaic_classification_5folds.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"

DATA_DIR="${DATA_DIR:-data/downstreams/FMT_5folds}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/FMT_5folds/mozaic_classification}"

DATA_FILE="${DATA_FILE:-${DATA_DIR}/community_all.pkl}"
METADATA_FILE="${METADATA_FILE:-${DATA_DIR}/response_all_meta.csv}"
NTV3_EMBEDDING="${NTV3_EMBEDDING:-${DATA_DIR}/community_all.motu_embeddings.h5}"

SEEDS=(${SEEDS:-0})
FOLDS=(${FOLDS:-0 1 2 3 4})
NUM_FOLDS="${NUM_FOLDS:-5}"
VAL_RATIO="${VAL_RATIO:-0.1}"
LABEL_COLUMN="${LABEL_COLUMN:-indication}"

MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUTPUT_ROOT}"

for required in "${DATA_FILE}" "${METADATA_FILE}" "${NTV3_EMBEDDING}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required file not found: ${required}" >&2
        echo "Run first: bash scripts/downstreams/FMT/build_mgm2_fmt_5fold_dataset.sh" >&2
        exit 1
    fi
done

for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/seed_${seed}/fold_${fold}"
        log_file="${run_dir}/train.log"
        mkdir -p "${run_dir}"
        echo "[$(date '+%F %T')] Running FMT MOZAIC response 5-fold seed=${seed} fold=${fold}/${NUM_FOLDS}"
        "${PYTHON_BIN}" -u scripts/downstreams/FMT/train_mozaic_classification_5folds.py \
            --data_file "${DATA_FILE}" \
            --metadata_file "${METADATA_FILE}" \
            --ntv3_embedding_path "${NTV3_EMBEDDING}" \
            --output_dir "${run_dir}" \
            --fold_index "${fold}" \
            --num_folds "${NUM_FOLDS}" \
            --val_ratio "${VAL_RATIO}" \
            --label_column "${LABEL_COLUMN}" \
            --max_epochs "${MAX_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LEARNING_RATE}" \
            --weight_decay "${WEIGHT_DECAY}" \
            --seed "${seed}" \
            --device "${DEVICE}" \
            2>&1 | tee "${log_file}"
    done
done

echo "[$(date '+%F %T')] Finished FMT MOZAIC 5-fold response classification runs"
