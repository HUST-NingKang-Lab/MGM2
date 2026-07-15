#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
CACHE_ROOT="${CACHE_ROOT:-data/downstreams/wastewater_v2/embeddings}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_downstreams/wastewater_v2/clr_delta_transformer}"
SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
DEVICE="${DEVICE:-cuda:7}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-160}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.1}"
LOSS_CLR_DELTA_WEIGHT="${LOSS_CLR_DELTA_WEIGHT:-0.05}"
LOSS_CORR_WEIGHT="${LOSS_CORR_WEIGHT:-0.05}"
LOSS_DIRECTION_WEIGHT="${LOSS_DIRECTION_WEIGHT:-0.02}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
FORCE="${FORCE:-0}"

mkdir -p "${OUTPUT_ROOT}"

save_predictions_arg=""
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  save_predictions_arg="--save_predictions"
fi

for scale in ${SCALES}; do
  cache="${CACHE_ROOT}/mgm2_${scale}_top200_cls.pkl"
  out="${OUTPUT_ROOT}/${scale}"
  if [[ ! -f "${cache}" ]]; then
    echo "[error] missing cache for ${scale}: ${cache}" >&2
    exit 1
  fi
  if [[ "${FORCE}" != "1" && -s "${out}/summary.csv" ]]; then
    echo "[skip] ${scale}: ${out}/summary.csv exists"
    continue
  fi

  echo "[run] V2 CLR-delta Transformer ${scale}"
  "${PYTHON}" scripts/downstreams/wastewater_v2/train_clr_delta_transformer.py \
    --dataset "${DATASET}" \
    --embedding_cache "${cache}" \
    --output_dir "${out}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_heads "${NUM_HEADS}" \
    --num_layers "${NUM_LAYERS}" \
    --dropout "${DROPOUT}" \
    --loss_mse_weight "${LOSS_MSE_WEIGHT}" \
    --loss_clr_delta_weight "${LOSS_CLR_DELTA_WEIGHT}" \
    --loss_corr_weight "${LOSS_CORR_WEIGHT}" \
    --loss_direction_weight "${LOSS_DIRECTION_WEIGHT}" \
    ${save_predictions_arg}
done

echo "Done. Results under ${OUTPUT_ROOT}"
