#!/usr/bin/env bash
set -euo pipefail

# Train MGM2 cross-attention drug perturbation models for each stage-2 MGM2 scale.
# Each split is repeated with the listed random seeds.
#
# Run from the repository root:
#   bash scripts/downstreams/drug_perturbation/train_mgm2_scales.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/zhanghaohong/anaconda3/envs/mwm/bin/python"

DATA_DIR="/data9/zhanghaohong/databases/MicrobeDrugPerturbation/data"
OUTPUT_ROOT="outputs_downstreams/drug_perturbation/mgm2_scales"
STAGE2_ROOT="outputs_grid/stage2/20260505_094404"
NTV3_EMBEDDING_PATH="ntv3_embeddings/otus_99_embeddings.h5"
SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"

STAGE2_SCALES=(scale_small scale_medium scale_large scale_xlarge)
SPLIT_MODES=(pair)
SEEDS=(42)

MAX_EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=1e-3
VAL_FRACTION=0.15
MAX_OTUS=768
NUM_WORKERS=4
NUM_DRUG_TOKENS=4
ACCELERATOR="auto"
PRECISION="bf16"
DEVICES=(0)

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

    for split_mode in "${SPLIT_MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_name="${model_scale}_${split_mode}_seed${seed}"
            run_dir="${OUTPUT_ROOT}/${model_scale}/${split_mode}/seed_${seed}"
            log_file="${run_dir}/train.log"
            mkdir -p "${run_dir}"

            echo "[$(date '+%F %T')] Running ${run_name} with ${checkpoint_path}"
            "${PYTHON_BIN}" -u src/training/DrugPerturbation.py \
                --data_dir "${DATA_DIR}" \
                --ntv3_embedding_path "${NTV3_EMBEDDING_PATH}" \
                --config "${config_path}" \
                --pretrained_checkpoint "${checkpoint_path}" \
                --output_dir "${run_dir}" \
                --experiment_name "${run_name}" \
                --split_mode "${split_mode}" \
                --val_fraction "${VAL_FRACTION}" \
                --max_otus "${MAX_OTUS}" \
                --batch_size "${BATCH_SIZE}" \
                --num_workers "${NUM_WORKERS}" \
                --max_epochs "${MAX_EPOCHS}" \
                --learning_rate "${LEARNING_RATE}" \
                --num_drug_tokens "${NUM_DRUG_TOKENS}" \
                --devices "${DEVICES[@]}" \
                --accelerator "${ACCELERATOR}" \
                --precision "${PRECISION}" \
                --seed "${seed}" \
                --freeze_backbone \
                2>&1 | tee "${log_file}"

            "${PYTHON_BIN}" scripts/downstreams/drug_perturbation/summarize_mgm2_scale_results.py \
                --run_dir "${run_dir}" \
                --summary_csv "${SUMMARY_CSV}" \
                --model_scale "${model_scale}" \
                --split_mode "${split_mode}" \
                --seed "${seed}" \
                --run_name "${run_name}"
        done
    done
done

echo "[$(date '+%F %T')] Finished MGM2 drug perturbation scale runs for available stage-2 scales"
echo "[$(date '+%F %T')] Summary written to ${SUMMARY_CSV}"
