#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
DATA_ROOT="${DATA_ROOT:-/data9/zhanghaohong/databases/wastewater/MC-prediction/data_organized}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs_grid/stage2/20260505_094404}"
MODEL_PATH="${MODEL_PATH:-/data7/zhanghaohong/transformers_model/InstaDeepAI/NTv3_650M_pre}"
FALLBACK_EMBEDDING="${FALLBACK_EMBEDDING:-ntv3_embeddings/otus_99_taxonomy_fallback.h5}"
ASV_NTV3_EMBEDDING="${ASV_NTV3_EMBEDDING:-data/downstreams/wastewater/cache_ntv3_asv/wastewater_asv_ntv3_embeddings.h5}"

DEVICE="${DEVICE:-cuda:7}"
GPU_IDS="${GPU_IDS:-7}"
CACHE_DIR="${CACHE_DIR:-data/downstreams/wastewater/cache_ntv3_asv}"
WORLD_ROOT="${WORLD_ROOT:-outputs_downstreams/wastewater/mgm2_world_model_ntv3_asv}"

EMBED_FORCE="${EMBED_FORCE:-0}"
CACHE_FORCE="${CACHE_FORCE:-0}"
FORCE="${FORCE:-1}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
WORLD_BATCH_SIZE="${WORLD_BATCH_SIZE:-8}"
WORLD_MAX_EPOCHS="${WORLD_MAX_EPOCHS:-80}"
WORLD_PATIENCE="${WORLD_PATIENCE:-10}"
JEPA_PROBE_EPOCHS="${JEPA_PROBE_EPOCHS:-30}"
JEPA_PROBE_PATIENCE="${JEPA_PROBE_PATIENCE:-6}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

save_predictions_args=()
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  save_predictions_args=(--save_predictions)
fi

run_or_skip() {
  local force_flag="$1"
  local done_file="$2"
  local description="$3"
  shift 3
  if [[ "${force_flag}" != "1" && -s "${done_file}" ]]; then
    echo "[skip] ${description}: ${done_file} exists"
    return 0
  fi
  echo "[run] ${description}"
  "$@"
}

mkdir -p "${CACHE_DIR}" "${WORLD_ROOT}"

run_or_skip "${EMBED_FORCE}" "${ASV_NTV3_EMBEDDING}" "wastewater ASV NTv3 embeddings" \
  python scripts/downstreams/wastewater/build_asv_ntv3_embeddings.py \
    --data_root "${DATA_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --output_path "${ASV_NTV3_EMBEDDING}" \
    --gpu_ids "${GPU_IDS}" \
    2>&1 | tee "${CACHE_DIR}/build_asv_ntv3_embeddings.log"

for scale in ${SCALES}; do
  config="${STAGE2_ROOT}/${scale}/config.json"
  checkpoint="${STAGE2_ROOT}/${scale}/checkpoints/last.ckpt"
  cache="${CACHE_DIR}/mgm2_${scale}_top200_tokens.pkl"
  mae_dir="${WORLD_ROOT}/mae_${scale}_top200"
  jepa_dir="${WORLD_ROOT}/jepa_${scale}_top200"

  if [[ ! -f "${config}" ]]; then
    echo "[error] missing config for ${scale}: ${config}" >&2
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] missing checkpoint for ${scale}: ${checkpoint}" >&2
    exit 1
  fi

  mkdir -p "${mae_dir}" "${jepa_dir}"

  run_or_skip "${CACHE_FORCE}" "${cache}" "${scale} token cache with ASV NTv3 embeddings" \
    bash scripts/downstreams/wastewater/extract_mgm2_token_cache.sh \
      --dataset "${DATASET}" \
      --data_root "${DATA_ROOT}" \
      --stage2_root "${STAGE2_ROOT}" \
      --scale "${scale}" \
      --ntv3_embedding "${ASV_NTV3_EMBEDDING}" \
      --fallback_embedding "${FALLBACK_EMBEDDING}" \
      --device "${DEVICE}" \
      --batch_size "${CACHE_BATCH_SIZE}" \
      --output "${cache}" \
      2>&1 | tee "${CACHE_DIR}/mgm2_${scale}_top200_tokens.log"

  run_or_skip "${FORCE}" "${mae_dir}/summary.csv" "${scale} MAE world model with ASV NTv3 embeddings" \
    bash scripts/downstreams/wastewater/train_mgm2_world_model.sh \
      --dataset "${DATASET}" \
      --cache "${cache}" \
      --mode mae \
      --output_dir "${mae_dir}" \
      --device "${DEVICE}" \
      --batch_size "${WORLD_BATCH_SIZE}" \
      --max_epochs "${WORLD_MAX_EPOCHS}" \
      --patience "${WORLD_PATIENCE}" \
      --abundance_head logratio_residual \
      --query_mode temporal_attention \
      --residual_scale 1.0 \
      --loss_mse_weight 0.0 \
      --bc_loss_mode normalized \
      "${save_predictions_args[@]}" \
      2>&1 | tee "${mae_dir}/train.log"

  run_or_skip "${FORCE}" "${jepa_dir}/summary.csv" "${scale} JEPA world model with ASV NTv3 embeddings" \
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
      --abundance_head logratio_residual \
      --query_mode temporal_attention \
      --residual_scale 1.0 \
      --loss_mse_weight 0.0 \
      --bc_loss_mode normalized \
      "${save_predictions_args[@]}" \
      2>&1 | tee "${jepa_dir}/train.log"
done

echo "Done. Results:"
find "${WORLD_ROOT}" -maxdepth 2 -name summary.csv | sort
