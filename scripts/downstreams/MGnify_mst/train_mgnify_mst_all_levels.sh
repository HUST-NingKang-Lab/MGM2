#!/usr/bin/env bash
set -euo pipefail

# Train and predict MGnify MST community classifiers for each stage-2 MGM2 scale
# and each hierarchy level. Each level is repeated with five random seeds.
#
# Run from the repository root:
#   bash scripts/downstreams/MGnify_mst/train_mgnify_mst_all_levels.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="data/downstreams/MGnify_mst"
OUTPUT_ROOT="outputs_downstreams/MGnify_mst"
STAGE2_ROOT="outputs_grid/stage2/20260505_094404"

TRAIN_DATA_FILE="${DATA_DIR}/train.pkl"
TEST_DATA_FILE="${DATA_DIR}/test.pkl"
TRAIN_METADATA_FILE="${DATA_DIR}/train_meta_split.csv"
TEST_METADATA_FILE="${DATA_DIR}/test_meta_split.csv"
TRAIN_NTV3_EMBEDDING="${DATA_DIR}/train.resolved_embeddings.h5"
TEST_NTV3_EMBEDDING="${DATA_DIR}/test.resolved_embeddings.h5"

SAMPLE_ID_COLUMN="SampleID"
STAGE2_SCALES=(scale_small scale_medium scale_large scale_xlarge)
LEVELS=(env_level_1 env_level_2 env_level_3 env_level_4 env_level_5)
SEEDS=(11 22 33 44 55)

MAX_EPOCHS=100
BATCH_SIZE=2048
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
VAL_RATIO=0.1
NUM_WORKERS=1
POOLING_MODE="cls"
DEVICES=(1 2 3 4)

mkdir -p "${OUTPUT_ROOT}"

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

    for label_column in "${LEVELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_name="${model_scale}_${label_column}_seed${seed}"
            run_dir="${OUTPUT_ROOT}/${model_scale}/${label_column}/seed_${seed}"
            prediction_file="${run_dir}/test_predictions.csv"
            log_file="${run_dir}/train.log"
            mkdir -p "${run_dir}"

            echo "[$(date '+%F %T')] Running ${run_name} with ${checkpoint_path}"
            python -u src/training/CommunityClassification.py \
                --train_data_file "${TRAIN_DATA_FILE}" \
                --train_metadata_file "${TRAIN_METADATA_FILE}" \
                --test_data_file "${TEST_DATA_FILE}" \
                --test_metadata_file "${TEST_METADATA_FILE}" \
                --ntv3_embedding_path "${TRAIN_NTV3_EMBEDDING}" \
                --test_ntv3_embedding_path "${TEST_NTV3_EMBEDDING}" \
                --config "${config_path}" \
                --sample_id_column "${SAMPLE_ID_COLUMN}" \
                --label_column "${label_column}" \
                --val_ratio "${VAL_RATIO}" \
                --max_epochs "${MAX_EPOCHS}" \
                --batch_size "${BATCH_SIZE}" \
                --learning_rate "${LEARNING_RATE}" \
                --weight_decay "${WEIGHT_DECAY}" \
                --num_workers "${NUM_WORKERS}" \
                --pooling_mode "${POOLING_MODE}" \
                --devices "${DEVICES[@]}" \
                --output_dir "${run_dir}" \
                --freeze_backbone \
                --pretrained_checkpoint "${checkpoint_path}" \
                --experiment_name "${run_name}" \
                --prediction_file "${prediction_file}" \
                2>&1 | tee "${log_file}"
        done
    done
done

echo "[$(date '+%F %T')] Finished MGnify MST hierarchy classification runs for available stage-2 scales"
