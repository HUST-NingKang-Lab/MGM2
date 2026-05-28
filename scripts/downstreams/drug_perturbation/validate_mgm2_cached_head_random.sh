#!/usr/bin/env bash
set -euo pipefail

# Validate the best quick-tune MGM2 configurations on a random fold subset
# before launching full leave-one-pair evaluation.
#
# Run from the repository root:
#   DEVICE=7 SCALE=scale_large MAX_FOLDS=100 FOLD_SUBSET_SEED=20260527 \
#     bash scripts/downstreams/drug_perturbation/validate_mgm2_cached_head_random.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/zhanghaohong/anaconda3/envs/mwm/bin/python"

SCALE="${SCALE:-scale_large}"
SEED="${SEED:-42}"
MAX_FOLDS="${MAX_FOLDS:-100}"
FOLD_SUBSET_SEED="${FOLD_SUBSET_SEED:-20260527}"
DEVICE="${DEVICE:-7}"

CACHE_PATH="data/downstreams/drug_perturbation/cache/mgm2_backbone_cache_${SCALE}_top1024.pt"
OUTPUT_ROOT="outputs_downstreams/drug_perturbation/mgm2_random_validation/${SCALE}/folds_${MAX_FOLDS}_seed_${FOLD_SUBSET_SEED}"

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-120}"
PATIENCE="${PATIENCE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
PRECISION="${PRECISION:-bf16}"

if [[ ! -f "${CACHE_PATH}" ]]; then
    echo "Cache not found: ${CACHE_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

run_variant() {
    local name="$1"
    shift
    local run_dir="${OUTPUT_ROOT}/${name}/seed_${SEED}"
    local log_file="${run_dir}/train.log"
    mkdir -p "${run_dir}"
    echo "[$(date '+%F %T')] Validating ${SCALE}/${name} random_folds=${MAX_FOLDS} fold_seed=${FOLD_SUBSET_SEED}"
    "${PYTHON_BIN}" -u scripts/downstreams/drug_perturbation/train_mgm2_cached_head.py \
        --cache "${CACHE_PATH}" \
        --output_dir "${run_dir}" \
        --split_mode pair \
        --max_folds "${MAX_FOLDS}" \
        --fold_selection random \
        --fold_subset_seed "${FOLD_SUBSET_SEED}" \
        --val_fraction "${VAL_FRACTION}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers 0 \
        --no-pin_memory \
        --max_epochs "${MAX_EPOCHS}" \
        --patience "${PATIENCE}" \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --precision "${PRECISION}" \
        --gpu "${DEVICE}" \
        --seed "${SEED}" \
        "$@" 2>&1 | tee "${log_file}"
}

COMMON_ARGS=(--target_space log1p_raw --standardize_delta)

# Best direction/Bray/CLR compromise from round3.
run_variant "log1p_std_cos010_clr025" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10 \
    --clr_mse_weight 0.25

# Best Bray/delta-L1 conservative variant from round3.
run_variant "log1p_std_shrink050_cos010" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10 \
    --delta_shrinkage 0.50

# Best direction + moderate Bray variant from round3.
run_variant "log1p_std_blend010_cos010" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10 \
    --blend_no_effect 0.10

"${PYTHON_BIN}" scripts/downstreams/drug_perturbation/summarize_quick_tune.py \
    --quick_root "${OUTPUT_ROOT}" \
    --mlp_dir outputs_downstreams/drug_perturbation/mlp_pair_top1024_full \
    --output "${OUTPUT_ROOT}/random_validation_comparison.csv"

echo "[$(date '+%F %T')] Random validation finished"
echo "Outputs: ${OUTPUT_ROOT}"
