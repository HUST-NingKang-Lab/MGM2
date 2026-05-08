#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted stage-2 scaling search at outputs_grid/stage2/20260505_094404.
# Existing rows in summary.csv are skipped. scale_small and scale_medium are treated
# as completed and skipped; resume starts from scale_large.
#
# Example:
#   bash scripts/resume_grid_search_pretrain_stage2_20260505_094404.sh

BASE_CONFIG="outputs_grid/stage1/20260426_230440/run_005_lr5e-5_wd0.01_mask0.15_reg0.3_qw0.1/config.json"
REUSE_SMALL_FROM_BASE_CONFIG=true
DATA_FILE="data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.768.pkl"
NTV3_EMBEDDING_PATH="ntv3_embeddings/otus_99_embeddings.h5"
QWEN3_EMBEDDING_PATH="qwen3_embeddings/microbeatlas_keywords_emb.h5"
STAGE_ROOT="outputs_grid/stage2/20260505_094404"
SUMMARY_CSV="${STAGE_ROOT}/summary.csv"
MAX_EPOCHS=50
NUM_WORKERS=4
DEVICES=(1 2 3 4)
EXPERIMENT_PREFIX="stage2_scale"
RESUME_FROM_PRESET="scale_large"

PRESETS=(
  "scale_small|224|6|4|896|1024"
  "scale_medium|320|6|5|1280|512"
  "scale_large|384|8|6|1536|512"
  "scale_xlarge|512|10|8|2048|512"
)

for required in "${BASE_CONFIG}" "${DATA_FILE}" "${NTV3_EMBEDDING_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file not found: ${required}" >&2
    exit 1
  fi
done

if [[ ! -d "${STAGE_ROOT}" ]]; then
  echo "Stage root not found: ${STAGE_ROOT}" >&2
  exit 1
fi

QWEN3_PRETRAIN_ARGS=()
if [[ -n "${QWEN3_EMBEDDING_PATH}" ]]; then
  if [[ ! -f "${QWEN3_EMBEDDING_PATH}" ]]; then
    echo "Qwen3 embedding file not found: ${QWEN3_EMBEDDING_PATH}" >&2
    exit 1
  fi
  QWEN3_PRETRAIN_ARGS=(--enable_qwen3_alignment --qwen3_embedding_path "${QWEN3_EMBEDDING_PATH}")
fi

summary_has_run() {
  local run_name="$1"
  [[ -f "${SUMMARY_CSV}" ]] && awk -F, -v run="${run_name}" 'NR > 1 && $1 == run {found=1} END {exit found ? 0 : 1}' "${SUMMARY_CSV}"
}

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

resume_started=false
for preset in "${PRESETS[@]}"; do
  IFS='|' read -r preset_name hidden_size num_hidden_layers num_attention_heads intermediate_size batch_size <<< "${preset}"
  run_name="${EXPERIMENT_PREFIX}_${preset_name}"
  run_dir="${STAGE_ROOT}/${preset_name}"
  config_path="${run_dir}/config.json"
  resume_ckpt="${run_dir}/checkpoints/last.ckpt"

  if [[ "${preset_name}" == "${RESUME_FROM_PRESET}" ]]; then
    resume_started=true
  fi
  if [[ "${resume_started}" != true ]]; then
    echo "Skipping ${preset_name}: before resume preset ${RESUME_FROM_PRESET}"
    continue
  fi
  if summary_has_run "${preset_name}"; then
    echo "Skipping ${preset_name}: already present in ${SUMMARY_CSV}"
    continue
  fi

  mkdir -p "${run_dir}"

  if [[ "${preset_name}" == "scale_small" && "${REUSE_SMALL_FROM_BASE_CONFIG}" == true ]]; then
    base_run_dir="$(dirname "${BASE_CONFIG}")"
    echo "Reusing small model result from ${base_run_dir}"
    cp "${BASE_CONFIG}" "${config_path}"
    if [[ -d "${base_run_dir}/checkpoints" ]]; then
      mkdir -p "${run_dir}/checkpoints"
      cp -a "${base_run_dir}/checkpoints/." "${run_dir}/checkpoints/"
    fi
    if [[ -d "${base_run_dir}/logs" ]]; then
      mkdir -p "${run_dir}/logs"
      cp -a "${base_run_dir}/logs/." "${run_dir}/logs/"
    fi
    if [[ -f "${base_run_dir}/train.log" ]]; then
      cp "${base_run_dir}/train.log" "${run_dir}/train.log"
    fi
    summarize_run "${preset_name}" "${run_dir}" "${config_path}"
    continue
  fi

  if [[ ! -f "${config_path}" ]]; then
    python scripts/build_grid_config.py \
      --base "${BASE_CONFIG}" \
      --output "${config_path}" \
      --set model.hidden_size="${hidden_size}" \
      --set model.num_hidden_layers="${num_hidden_layers}" \
      --set model.num_attention_heads="${num_attention_heads}" \
      --set model.intermediate_size="${intermediate_size}"
  fi

  pretrain_cmd=(
    python -u src/training/Pretrain.py
    --data_file "${DATA_FILE}"
    --ntv3_embedding_path "${NTV3_EMBEDDING_PATH}"
    --config "${config_path}"
    --max_epochs "${MAX_EPOCHS}"
    --batch_size "${batch_size}"
    --num_workers "${NUM_WORKERS}"
    --output_dir "${run_dir}"
    --experiment_name "${run_name}"
    --devices "${DEVICES[@]}"
  )
  if (( ${#QWEN3_PRETRAIN_ARGS[@]} )); then
    pretrain_cmd+=("${QWEN3_PRETRAIN_ARGS[@]}")
  fi
  if [[ -f "${resume_ckpt}" ]]; then
    pretrain_cmd+=(--resume "${resume_ckpt}")
    echo "Resuming ${preset_name} from ${resume_ckpt}"
  else
    echo "Starting ${preset_name} from scratch"
  fi

  "${pretrain_cmd[@]}" 2>&1 | tee -a "${run_dir}/train.log"
  summarize_run "${preset_name}" "${run_dir}" "${config_path}"
done

echo "Stage-2 resume complete. Summary written to ${SUMMARY_CSV}"
