#!/usr/bin/env bash
set -euo pipefail

# Extract frozen MGM2 backbone caches once per scale, then train cached
# cross-attention heads for drug perturbation leave-one-group evaluation.
#
# Run from the repository root:
#   bash scripts/downstreams/drug_perturbation/train_mgm2_cached_scales.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/zhanghaohong/anaconda3/envs/mwm/bin/python"

PREPROCESSED_DATA="data/downstreams/drug_perturbation/drug_perturbation_top1024.pkl"
CACHE_ROOT="data/downstreams/drug_perturbation/cache"
OUTPUT_ROOT="outputs_downstreams/drug_perturbation/mgm2_cached_scales"
STAGE2_ROOT="outputs_grid/stage2/20260505_094404"

STAGE2_SCALES=(scale_small scale_medium scale_large scale_xlarge)
SPLIT_MODES=(pair)
SEEDS=(42)

# DEVICE: single value or space-separated list. Examples:
#   DEVICE=7            -> single GPU cuda:7
#   DEVICE="0 1"        -> two GPUs cuda:0,cuda:1 via torchrun
#   DEVICE=cpu          -> CPU only
DEVICE=${DEVICE:-"0 1"}

read -ra GPU_IDS <<< "${DEVICE}"
if [ "${DEVICE}" = "cpu" ]; then
    EXTRACT_DEVICE="cpu"
    N_GPUS=0
else
    EXTRACT_DEVICE="cuda:${GPU_IDS[0]}"
    N_GPUS=${#GPU_IDS[@]}
fi
EXTRACT_BATCH_SIZE=64
TRAIN_BATCH_SIZE=128
MAX_EPOCHS=200
MAX_FOLDS=0
VAL_FRACTION=0.15
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
PATIENCE=8
EXTRACT_NUM_WORKERS=4
TRAIN_NUM_WORKERS=0
PREFETCH_FACTOR=2
NUM_DRUG_TOKENS=4
PRECISION="bf16"
STORAGE_DTYPE="float16"
FORCE_REBUILD_CACHE=0

mkdir -p "${CACHE_ROOT}" "${OUTPUT_ROOT}"

if [[ ! -f "${PREPROCESSED_DATA}" ]]; then
    echo "Preprocessed data not found: ${PREPROCESSED_DATA}" >&2
    echo "Run: bash scripts/downstreams/drug_perturbation/build_mgm2_dataset.sh" >&2
    exit 1
fi

for model_scale in "${STAGE2_SCALES[@]}"; do
    config_path="${STAGE2_ROOT}/${model_scale}/config.json"
    checkpoint_path="${STAGE2_ROOT}/${model_scale}/checkpoints/last.ckpt"
    cache_path="${CACHE_ROOT}/mgm2_backbone_cache_${model_scale}_top1024.pt"
    cache_log="${CACHE_ROOT}/extract_${model_scale}.log"

    if [[ ! -f "${config_path}" ]]; then
        echo "[$(date '+%F %T')] Skipping ${model_scale}: config not found at ${config_path}"
        continue
    fi
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "[$(date '+%F %T')] Skipping ${model_scale}: checkpoint not found at ${checkpoint_path}"
        continue
    fi

    if [[ "${FORCE_REBUILD_CACHE}" == "1" || ! -f "${cache_path}" ]]; then
        echo "[$(date '+%F %T')] Extracting frozen backbone cache for ${model_scale} -> ${cache_path}"
        "${PYTHON_BIN}" -u scripts/downstreams/drug_perturbation/extract_mgm2_backbone_cache.py \
            --preprocessed_data "${PREPROCESSED_DATA}" \
            --config "${config_path}" \
            --pretrained_checkpoint "${checkpoint_path}" \
            --output "${cache_path}" \
            --batch_size "${EXTRACT_BATCH_SIZE}" \
            --num_workers "${EXTRACT_NUM_WORKERS}" \
            --prefetch_factor "${PREFETCH_FACTOR}" \
            --pin_memory \
            --device "${EXTRACT_DEVICE}" \
            --num_drug_tokens "${NUM_DRUG_TOKENS}" \
            --storage_dtype "${STORAGE_DTYPE}" \
            2>&1 | tee "${cache_log}"
    else
        echo "[$(date '+%F %T')] Reusing existing cache for ${model_scale}: ${cache_path}"
    fi

    for split_mode in "${SPLIT_MODES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_name="${model_scale}_${split_mode}_seed${seed}"
            run_dir="${OUTPUT_ROOT}/${model_scale}/${split_mode}/seed_${seed}"
            log_file="${run_dir}/train_cached_head.log"
            mkdir -p "${run_dir}"

            echo "[$(date '+%F %T')] Training cached head ${run_name}"
            TRAIN_CMD=("--cache" "${cache_path}"
                "--output_dir" "${run_dir}"
                "--split_mode" "${split_mode}"
                "--max_folds" "${MAX_FOLDS}"
                "--val_fraction" "${VAL_FRACTION}"
                "--batch_size" "${TRAIN_BATCH_SIZE}"
                "--num_workers" "${TRAIN_NUM_WORKERS}"
                "--prefetch_factor" "${PREFETCH_FACTOR}"
                "--pin_memory"
                "--max_epochs" "${MAX_EPOCHS}"
                "--patience" "${PATIENCE}"
                "--learning_rate" "${LEARNING_RATE}"
                "--weight_decay" "${WEIGHT_DECAY}"
                "--precision" "${PRECISION}"
                "--seed" "${seed}")

            if [ "$N_GPUS" -gt 1 ]; then
                CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")" torchrun --nproc_per_node="$N_GPUS" \
                    scripts/downstreams/drug_perturbation/train_mgm2_cached_head.py \
                    "${TRAIN_CMD[@]}" 2>&1 | tee "${log_file}"
            elif [ "${DEVICE}" = "cpu" ]; then
                "${PYTHON_BIN}" -u scripts/downstreams/drug_perturbation/train_mgm2_cached_head.py \
                    "${TRAIN_CMD[@]}" 2>&1 | tee "${log_file}"
            else
                "${PYTHON_BIN}" -u scripts/downstreams/drug_perturbation/train_mgm2_cached_head.py \
                    --gpu "${GPU_IDS[0]}" \
                    "${TRAIN_CMD[@]}" 2>&1 | tee "${log_file}"
            fi
        done
    done
done

echo "[$(date '+%F %T')] Finished cached MGM2 drug perturbation scale runs"
echo "[$(date '+%F %T')] Caches: ${CACHE_ROOT}"
echo "[$(date '+%F %T')] Outputs: ${OUTPUT_ROOT}/<scale>/<split>/seed_<seed>/"
