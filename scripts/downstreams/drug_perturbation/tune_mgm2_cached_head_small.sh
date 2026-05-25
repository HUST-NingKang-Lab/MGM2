#!/usr/bin/env bash
set -euo pipefail

# Small-scale MGM2 perturbation head tuning. This intentionally evaluates only
# the first N pair folds so weak configurations can be rejected before full LOO.
#
# Run from the repository root:
#   DEVICE=7 bash scripts/downstreams/drug_perturbation/tune_mgm2_cached_head_small.sh
#
# Useful overrides:
#   SCALE=scale_large MAX_FOLDS=30 DEVICE=7 ...

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/zhanghaohong/anaconda3/envs/mwm/bin/python"

SCALE="${SCALE:-scale_large}"
SEED="${SEED:-42}"
MAX_FOLDS="${MAX_FOLDS:-30}"
DEVICE="${DEVICE:-7}"

CACHE_ROOT="data/downstreams/drug_perturbation/cache"
CACHE_PATH="${CACHE_ROOT}/mgm2_backbone_cache_${SCALE}_top1024.pt"
OUTPUT_ROOT="outputs_downstreams/drug_perturbation/mgm2_quick_tune/${SCALE}"

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-120}"
PATIENCE="${PATIENCE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
PRECISION="${PRECISION:-bf16}"

if [[ ! -f "${CACHE_PATH}" ]]; then
    echo "Cache not found: ${CACHE_PATH}" >&2
    echo "Build it first with scripts/downstreams/drug_perturbation/train_mgm2_cached_scales.sh or extract_mgm2_backbone_cache.py" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

run_variant() {
    local name="$1"
    shift
    local run_dir="${OUTPUT_ROOT}/${name}/seed_${SEED}"
    local log_file="${run_dir}/train.log"
    mkdir -p "${run_dir}"
    echo "[$(date '+%F %T')] Running ${SCALE}/${name} max_folds=${MAX_FOLDS}"
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

# Baseline with only the raw-space zero fix. This isolates whether floor-to-1e-4
# was the main cause of poor Bray-Curtis.
run_variant "zero_floor" \
    --raw_zero_threshold 0.00010001

# MLP-like target scaling plus zero floor.
run_variant "std_zero_floor" \
    --standardize_delta \
    --raw_zero_threshold 0.00010001

# Conservative perturbation: reduce over-shooting while keeping direction.
run_variant "std_zero_shrink050" \
    --standardize_delta \
    --delta_shrinkage 0.50 \
    --raw_zero_threshold 0.00010001

run_variant "std_zero_shrink075" \
    --standardize_delta \
    --delta_shrinkage 0.75 \
    --raw_zero_threshold 0.00010001

# Add raw-space auxiliary terms to reward final-evaluation behavior.
run_variant "std_zero_multiloss" \
    --standardize_delta \
    --raw_zero_threshold 0.00010001 \
    --raw_after_mse_weight 10.0 \
    --magnitude_weight 0.25

# Blend toward no-effect to test whether a calibrated conservative prediction
# can recover Bray while retaining some perturbation signal.
run_variant "std_zero_blend025" \
    --standardize_delta \
    --raw_zero_threshold 0.00010001 \
    --blend_no_effect 0.25

"${PYTHON_BIN}" scripts/downstreams/drug_perturbation/summarize_quick_tune.py \
    --quick_root "${OUTPUT_ROOT}" \
    --mlp_dir outputs_downstreams/drug_perturbation/mlp_pair_top1024_full \
    --output "${OUTPUT_ROOT}/quick_tune_comparison.csv"

echo "[$(date '+%F %T')] Quick tune finished"
echo "Outputs: ${OUTPUT_ROOT}"
