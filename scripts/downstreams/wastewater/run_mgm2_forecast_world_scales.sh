#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
DATA_ROOT="${DATA_ROOT:-/data9/zhanghaohong/databases/wastewater/MC-prediction/data_organized}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs_grid/stage2/20260505_094404}"
DEVICE="${DEVICE:-cuda:7}"

CACHE_DIR="${CACHE_DIR:-data/downstreams/wastewater/cache}"
FORECAST_ROOT="${FORECAST_ROOT:-outputs_downstreams/wastewater/mgm2_token_cross_attention}"
WORLD_ROOT="${WORLD_ROOT:-outputs_downstreams/wastewater/mgm2_world_model_0.5mse}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
FORECAST_BATCH_SIZE="${FORECAST_BATCH_SIZE:-4}"
WORLD_BATCH_SIZE="${WORLD_BATCH_SIZE:-8}"

FORECAST_MAX_EPOCHS="${FORECAST_MAX_EPOCHS:-80}"
FORECAST_PATIENCE="${FORECAST_PATIENCE:-12}"
WORLD_MAX_EPOCHS="${WORLD_MAX_EPOCHS:-80}"
WORLD_PATIENCE="${WORLD_PATIENCE:-10}"
JEPA_PROBE_EPOCHS="${JEPA_PROBE_EPOCHS:-30}"
JEPA_PROBE_PATIENCE="${JEPA_PROBE_PATIENCE:-6}"

SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
FORCE="${FORCE:-0}"

save_predictions_args=()
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  save_predictions_args=(--save_predictions)
fi

run_or_skip() {
  local done_file="$1"
  local description="$2"
  shift 2
  if [[ "${FORCE}" != "1" && -s "${done_file}" ]]; then
    echo "[skip] ${description}: ${done_file} exists"
    return 0
  fi
  echo "[run] ${description}"
  "$@"
}

for scale in ${SCALES}; do
  config="${STAGE2_ROOT}/${scale}/config.json"
  checkpoint="${STAGE2_ROOT}/${scale}/checkpoints/last.ckpt"
  cache="${CACHE_DIR}/mgm2_${scale}_top200_tokens.pkl"
  forecast_dir="${FORECAST_ROOT}/${scale}_top200"
  mae_dir="${WORLD_ROOT}/mae_${scale}_top200"
  delta_mae_dir="${WORLD_ROOT}/delta_mae_${scale}_top200"
  jepa_dir="${WORLD_ROOT}/jepa_${scale}_top200"

  if [[ ! -f "${config}" ]]; then
    echo "[error] missing config for ${scale}: ${config}" >&2
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] missing checkpoint for ${scale}: ${checkpoint}" >&2
    exit 1
  fi

  mkdir -p "${CACHE_DIR}" "${forecast_dir}" "${mae_dir}" "${delta_mae_dir}" "${jepa_dir}"

  run_or_skip "${cache}" "${scale} token cache" \
    bash scripts/downstreams/wastewater/extract_mgm2_token_cache.sh \
      --dataset "${DATASET}" \
      --data_root "${DATA_ROOT}" \
      --stage2_root "${STAGE2_ROOT}" \
      --scale "${scale}" \
      --device "${DEVICE}" \
      --batch_size "${CACHE_BATCH_SIZE}" \
      --output "${cache}" \
      2>&1 | tee "${CACHE_DIR}/mgm2_${scale}_top200_tokens.log"

  run_or_skip "${forecast_dir}/summary.csv" "${scale} token cross-attention forecast" \
    bash scripts/downstreams/wastewater/train_mgm2_token_forecaster.sh \
      --dataset "${DATASET}" \
      --cache "${cache}" \
      --output_dir "${forecast_dir}" \
      --device "${DEVICE}" \
      --batch_size "${FORECAST_BATCH_SIZE}" \
      --max_epochs "${FORECAST_MAX_EPOCHS}" \
      --patience "${FORECAST_PATIENCE}" \
      "${save_predictions_args[@]}" \
      2>&1 | tee "${forecast_dir}/train.log"

  run_or_skip "${mae_dir}/summary.csv" "${scale} MAE world model" \
    bash scripts/downstreams/wastewater/train_mgm2_world_model.sh \
      --dataset "${DATASET}" \
      --cache "${cache}" \
      --mode mae \
      --output_dir "${mae_dir}" \
      --device "${DEVICE}" \
      --batch_size "${WORLD_BATCH_SIZE}" \
      --max_epochs "${WORLD_MAX_EPOCHS}" \
      --patience "${WORLD_PATIENCE}" \
      "${save_predictions_args[@]}" \
      2>&1 | tee "${mae_dir}/train.log"

  run_or_skip "${jepa_dir}/summary.csv" "${scale} JEPA world model" \
    bash scripts/downstreams/wastewater/train_mgm2_world_model.sh \
      --dataset "${DATASET}" \
      --cache "${cache}" \
      --mode jepa \
      --output_dir "${jepa_dir}" \
      --device "${DEVICE}" \
      --batch_size "${WORLD_BATCH_SIZE}" \
      --max_epochs "${WORLD_MAX_EPOCHS}" \
      --patience "${WORLD_PATIENCE}" \
      --probe_epochs "${JEPA_PROBE_EPOCHS}" \
      --probe_patience "${JEPA_PROBE_PATIENCE}" \
      "${save_predictions_args[@]}" \
      2>&1 | tee "${jepa_dir}/train.log"
done

echo "Done. Results:"
find "${FORECAST_ROOT}" "${WORLD_ROOT}" -maxdepth 2 -name summary.csv | sort
