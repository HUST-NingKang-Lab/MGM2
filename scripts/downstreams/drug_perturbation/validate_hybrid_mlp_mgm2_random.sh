#!/usr/bin/env bash
set -euo pipefail

# Validate whether frozen MGM2 embeddings improve the existing MLP baseline on
# the same random fold subset used for MGM2 head validation.
#
# Run from the repository root:
#   DEVICE=7 SCALE=scale_large MAX_FOLDS=100 FOLD_SUBSET_SEED=20260527 \
#     bash scripts/downstreams/drug_perturbation/validate_hybrid_mlp_mgm2_random.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/home/zhanghaohong/anaconda3/envs/mwm/bin/python"

SCALE="${SCALE:-scale_large}"
SEED="${SEED:-42}"
MAX_FOLDS="${MAX_FOLDS:-100}"
FOLD_SUBSET_SEED="${FOLD_SUBSET_SEED:-20260527}"
DEVICE="${DEVICE:-7}"

CACHE_PATH="data/downstreams/drug_perturbation/cache/mgm2_backbone_cache_${SCALE}_top1024.pt"
OUTPUT_ROOT="outputs_downstreams/drug_perturbation/hybrid_mlp_mgm2_random_validation/${SCALE}/folds_${MAX_FOLDS}_seed_${FOLD_SUBSET_SEED}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"

if [[ ! -f "${CACHE_PATH}" ]]; then
    echo "Cache not found: ${CACHE_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

run_variant() {
    local mode="$1"
    local run_dir="${OUTPUT_ROOT}/hybrid_${mode}/seed_${SEED}"
    local log_file="${run_dir}/train.log"
    mkdir -p "${run_dir}"
    echo "[$(date '+%F %T')] Validating hybrid MLP+MGM2 ${SCALE}/${mode} random_folds=${MAX_FOLDS} fold_seed=${FOLD_SUBSET_SEED}"
    "${PYTHON_BIN}" -u scripts/downstreams/drug_perturbation/train_mlp_baseline.py \
        --split_modes pair \
        --top_k_otus 1024 \
        --max_folds "${MAX_FOLDS}" \
        --fold_selection random \
        --fold_subset_seed "${FOLD_SUBSET_SEED}" \
        --hidden_size 256 \
        --dropout 0.1 \
        --batch_size "${BATCH_SIZE}" \
        --max_epochs "${MAX_EPOCHS}" \
        --patience "${PATIENCE}" \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --val_fraction "${VAL_FRACTION}" \
        --seed "${SEED}" \
        --transform log1p \
        --mgm2_cache "${CACHE_PATH}" \
        --mgm2_feature_mode "${mode}" \
        --no_mean_baselines \
        --output_dir "${run_dir}" \
        --device "cuda:${DEVICE}" \
        2>&1 | tee "${log_file}"
}

run_variant "cls"
run_variant "mean"
run_variant "cls_mean"

"${PYTHON_BIN}" scripts/downstreams/drug_perturbation/summarize_quick_tune.py \
    --quick_root "${OUTPUT_ROOT}" \
    --mlp_dir outputs_downstreams/drug_perturbation/mlp_pair_top1024_full \
    --output "${OUTPUT_ROOT}/hybrid_validation_comparison.csv"

echo "[$(date '+%F %T')] Hybrid random validation finished"
echo "Outputs: ${OUTPUT_ROOT}"
