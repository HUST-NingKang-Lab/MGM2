#!/usr/bin/env bash
set -euo pipefail

# Stage-2 scaling pretraining with Qwen3 alignment targets loaded but zero-weighted.
# This launches all scale presets into a separate output root so it does not
# interfere with an existing stage-2 run.
#
# Example:
#   bash scripts/grid_search_pretrain_stage2_qwen0.sh

BASE_CONFIG="outputs_grid/stage1/20260426_230440/run_005_lr5e-5_wd0.01_mask0.15_reg0.3_qw0.1/config.json"
DATA_FILE="data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.768.pkl"
NTV3_EMBEDDING_PATH="ntv3_embeddings/otus_99_embeddings.h5"
QWEN3_EMBEDDING_PATH="qwen3_embeddings/microbeatlas_keywords_emb.h5"
OUTPUT_ROOT="outputs_grid/stage2_qwen0"
RUN_GROUP="$(date +%Y%m%d_%H%M%S)"
MAX_EPOCHS=50
NUM_WORKERS=4
DEVICES=(1 2 3 4)
EXPERIMENT_PREFIX="stage2_qwen0"

PRESETS=(
  "scale_small|224|6|4|896|1024"
  "scale_medium|320|6|5|1280|512"
  "scale_large|384|8|6|1536|512"
  "scale_xlarge|512|10|8|2048|256"
)

for required in "${BASE_CONFIG}" "${DATA_FILE}" "${NTV3_EMBEDDING_PATH}" "${QWEN3_EMBEDDING_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file not found: ${required}" >&2
    exit 1
  fi
done

STAGE_ROOT="${OUTPUT_ROOT}/${RUN_GROUP}"
SUMMARY_CSV="${STAGE_ROOT}/summary.csv"
mkdir -p "${STAGE_ROOT}"

latest_metrics_csv() {
  local run_dir="$1"
  find "${run_dir}/logs" -path '*/version_*/metrics.csv' -type f 2>/dev/null | sort -V | tail -n 1
}

summarize_run() {
  local run_name="$1"
  local run_dir="$2"
  local config_path="$3"
  local metrics_csv
  metrics_csv="$(latest_metrics_csv "${run_dir}")"
  if [[ -z "${metrics_csv}" || ! -f "${metrics_csv}" ]]; then
    echo "Could not find metrics.csv for ${run_name} under ${run_dir}/logs" >&2
    exit 1
  fi

  python scripts/summarize_grid_results.py \
    --metrics_csv "${metrics_csv}" \
    --summary_csv "${SUMMARY_CSV}" \
    --run_name "${run_name}" \
    --output_dir "${run_dir}" \
    --config_json "${config_path}" \
    --checkpoint_dir "${run_dir}/checkpoints"
}

for preset in "${PRESETS[@]}"; do
  IFS='|' read -r preset_name hidden_size num_hidden_layers num_attention_heads intermediate_size batch_size <<< "${preset}"
  run_name="${EXPERIMENT_PREFIX}_${preset_name}"
  run_dir="${STAGE_ROOT}/${preset_name}"
  config_path="${run_dir}/config.json"
  resume_ckpt="${run_dir}/checkpoints/last.ckpt"

  mkdir -p "${run_dir}"

  if [[ ! -f "${config_path}" ]]; then
    python scripts/build_grid_config.py \
      --base "${BASE_CONFIG}" \
      --output "${config_path}" \
      --set model.hidden_size="${hidden_size}" \
      --set model.num_hidden_layers="${num_hidden_layers}" \
      --set model.num_attention_heads="${num_attention_heads}" \
      --set model.intermediate_size="${intermediate_size}" \
      --set qwen3_alignment.enabled=true \
      --set qwen3_alignment.embedding_path="${QWEN3_EMBEDDING_PATH}" \
      --set qwen3_alignment.loss_weight=0.0
  fi

  pretrain_cmd=(
    python -u src/training/Pretrain.py
    --data_file "${DATA_FILE}"
    --ntv3_embedding_path "${NTV3_EMBEDDING_PATH}"
    --enable_qwen3_alignment
    --qwen3_embedding_path "${QWEN3_EMBEDDING_PATH}"
    --config "${config_path}"
    --max_epochs "${MAX_EPOCHS}"
    --batch_size "${batch_size}"
    --num_workers "${NUM_WORKERS}"
    --output_dir "${run_dir}"
    --experiment_name "${run_name}"
    --devices "${DEVICES[@]}"
  )
  if [[ -f "${resume_ckpt}" ]]; then
    pretrain_cmd+=(--resume "${resume_ckpt}")
    echo "Resuming ${preset_name} from ${resume_ckpt}"
  else
    echo "Starting ${preset_name} with qwen3_alignment.loss_weight=0.0"
  fi

  "${pretrain_cmd[@]}" 2>&1 | tee -a "${run_dir}/train.log"
  summarize_run "${preset_name}" "${run_dir}" "${config_path}"
done

echo "Stage-2 Qwen0 summary written to ${SUMMARY_CSV}"
