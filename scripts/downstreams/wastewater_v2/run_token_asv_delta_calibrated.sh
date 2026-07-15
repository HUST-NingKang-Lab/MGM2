#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
CACHE_ROOT="${CACHE_ROOT:-data/downstreams/wastewater_v2/embeddings}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/wastewater_v2/token_asv_delta_transformer_optimized/calibrated}"
SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
DEVICE="${DEVICE:-cuda:7}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-120}"
PATIENCE="${PATIENCE:-16}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
TOKEN_BOTTLENECK_DIM="${TOKEN_BOTTLENECK_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-2}"
ASV_LAYERS="${ASV_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.2}"

LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.2}"
LOSS_CLR_DELTA_WEIGHT="${LOSS_CLR_DELTA_WEIGHT:-0.02}"
LOSS_CORR_WEIGHT="${LOSS_CORR_WEIGHT:-0.01}"
LOSS_DIRECTION_WEIGHT="${LOSS_DIRECTION_WEIGHT:-0.005}"
LOSS_DELTA_L2_WEIGHT="${LOSS_DELTA_L2_WEIGHT:-1e-4}"
LOSS_WEIGHTED_BRAY_WEIGHT="${LOSS_WEIGHTED_BRAY_WEIGHT:-0.0}"
LOSS_WEIGHTED_MSE_WEIGHT="${LOSS_WEIGHTED_MSE_WEIGHT:-0.0}"
LOSS_ASV_CORR_WEIGHT="${LOSS_ASV_CORR_WEIGHT:-0.0}"

SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
FORCE="${FORCE:-0}"

mkdir -p "${OUTPUT_ROOT}"

save_predictions_arg=""
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  save_predictions_arg="--save_predictions"
fi

for scale in ${SCALES}; do
  cache="${CACHE_ROOT}/mgm2_${scale}_top200_tokens.pkl"
  out="${OUTPUT_ROOT}/${scale}"
  if [[ ! -f "${cache}" ]]; then
    echo "[error] missing token cache for ${scale}: ${cache}" >&2
    exit 1
  fi
  if [[ "${FORCE}" != "1" && -s "${out}/summary.csv" && -s "${out}/predictions.csv.gz" ]]; then
    echo "[skip] ${scale}: summary and predictions exist"
    continue
  fi

  echo "[run] calibrated V2 token ASV-delta Transformer ${scale}"
  "${PYTHON}" scripts/downstreams/wastewater_v2/train_token_asv_delta_transformer.py \
    --dataset "${DATASET}" \
    --token_cache "${cache}" \
    --output_dir "${out}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --token_bottleneck_dim "${TOKEN_BOTTLENECK_DIM}" \
    --num_heads "${NUM_HEADS}" \
    --temporal_layers "${TEMPORAL_LAYERS}" \
    --asv_layers "${ASV_LAYERS}" \
    --dropout "${DROPOUT}" \
    --loss_mse_weight "${LOSS_MSE_WEIGHT}" \
    --loss_clr_delta_weight "${LOSS_CLR_DELTA_WEIGHT}" \
    --loss_corr_weight "${LOSS_CORR_WEIGHT}" \
    --loss_direction_weight "${LOSS_DIRECTION_WEIGHT}" \
    --loss_delta_l2_weight "${LOSS_DELTA_L2_WEIGHT}" \
    --loss_weighted_bray_weight "${LOSS_WEIGHTED_BRAY_WEIGHT}" \
    --loss_weighted_mse_weight "${LOSS_WEIGHTED_MSE_WEIGHT}" \
    --loss_asv_corr_weight "${LOSS_ASV_CORR_WEIGHT}" \
    ${save_predictions_arg}
done

echo "Done. Results under ${OUTPUT_ROOT}"
