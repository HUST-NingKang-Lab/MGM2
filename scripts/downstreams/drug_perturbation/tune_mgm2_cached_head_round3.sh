#!/usr/bin/env bash
set -euo pipefail

# Third-round quick tuning. Round2 showed that log1p_raw + cosine supervision
# beats MLP on perturbation direction but hurts CLR. This sweep lowers cosine
# pressure and optionally adds a small CLR loss.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/zhanghaohong/anaconda3/envs/mwm/bin/python"

SCALE="${SCALE:-scale_large}"
SEED="${SEED:-42}"
MAX_FOLDS="${MAX_FOLDS:-30}"
DEVICE="${DEVICE:-7}"

CACHE_PATH="data/downstreams/drug_perturbation/cache/mgm2_backbone_cache_${SCALE}_top1024.pt"
OUTPUT_ROOT="outputs_downstreams/drug_perturbation/mgm2_quick_tune_round3/${SCALE}"

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
    echo "[$(date '+%F %T')] Running round3 ${SCALE}/${name} max_folds=${MAX_FOLDS}"
    "${PYTHON_BIN}" -u scripts/downstreams/drug_perturbation/train_mgm2_cached_head.py \
        --cache "${CACHE_PATH}" \
        --output_dir "${run_dir}" \
        --split_mode pair \
        --max_folds "${MAX_FOLDS}" \
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

run_variant "log1p_std_cos005" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.05

run_variant "log1p_std_cos010" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10

run_variant "log1p_std_cos015" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.15

run_variant "log1p_std_shrink075_cos010" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10 \
    --delta_shrinkage 0.75

run_variant "log1p_std_shrink050_cos010" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10 \
    --delta_shrinkage 0.50

run_variant "log1p_std_blend010_cos010" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10 \
    --blend_no_effect 0.10

run_variant "log1p_std_cos010_clr010" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10 \
    --clr_mse_weight 0.10

run_variant "log1p_std_cos010_clr025" \
    "${COMMON_ARGS[@]}" \
    --delta_cosine_weight 0.10 \
    --clr_mse_weight 0.25

"${PYTHON_BIN}" scripts/downstreams/drug_perturbation/summarize_quick_tune.py \
    --quick_root "${OUTPUT_ROOT}" \
    --mlp_dir outputs_downstreams/drug_perturbation/mlp_pair_top1024_full \
    --output "${OUTPUT_ROOT}/quick_tune_comparison.csv"

echo "[$(date '+%F %T')] Round3 quick tune finished"
echo "Outputs: ${OUTPUT_ROOT}"
