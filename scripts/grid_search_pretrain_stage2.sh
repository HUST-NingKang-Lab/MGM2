#!/usr/bin/env bash
set -euo pipefail

# Stage-2 scaling search for MGM2 pretraining.
# Edit the fixed settings in this header block before launching the script.
# Uses curated parameter-budget presets instead of a full Cartesian product.
#
# Example:
#   bash scripts/grid_search_pretrain_stage2.sh

BASE_CONFIG="outputs_grid/stage1/REPLACE_WITH_BEST_RUN/config.json"
DATA_FILE="data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.pkl"
NTV3_EMBEDDING_PATH="ntv3_embeddings/otus_99_embeddings.h5"
QWEN3_EMBEDDING_PATH="qwen3_embeddings/microbeatlas_keywords_emb.h5"
OUTPUT_ROOT="outputs_grid/stage2"
RUN_GROUP="$(date +%Y%m%d_%H%M%S)"
MAX_EPOCHS=50
BATCH_SIZE=32
NUM_WORKERS=4
DEVICES=(6 7)
EXPERIMENT_PREFIX="stage2_scale"

PRESETS=(
  "scale_small|224|6|4|896"
  "scale_medium|320|6|5|1280"
  "scale_large|384|8|6|1536"
  "scale_xlarge|512|10|8|2048"
)

for required in "$BASE_CONFIG" "$DATA_FILE" "$NTV3_EMBEDDING_PATH"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
done

STAGE_ROOT="${OUTPUT_ROOT}/${RUN_GROUP}"
SUMMARY_CSV="${STAGE_ROOT}/summary.csv"
mkdir -p "${STAGE_ROOT}"

QWEN3_PRETRAIN_ARGS=()
if [[ -n "${QWEN3_EMBEDDING_PATH}" ]]; then
  if [[ ! -f "${QWEN3_EMBEDDING_PATH}" ]]; then
    echo "Qwen3 embedding file not found: ${QWEN3_EMBEDDING_PATH}" >&2
    exit 1
  fi
  QWEN3_PRETRAIN_ARGS=(--enable_qwen3_alignment --qwen3_embedding_path "${QWEN3_EMBEDDING_PATH}")
fi

for preset in "${PRESETS[@]}"; do
  IFS='|' read -r preset_name hidden_size num_hidden_layers num_attention_heads intermediate_size <<< "${preset}"
  run_name="${EXPERIMENT_PREFIX}_${preset_name}"
  run_dir="${STAGE_ROOT}/${preset_name}"
  config_path="${run_dir}/config.json"
  metrics_csv="${run_dir}/logs/${run_name}/version_0/metrics.csv"

  mkdir -p "${run_dir}"
  python scripts/build_grid_config.py     --base "${BASE_CONFIG}"     --output "${config_path}"     --set model.hidden_size="${hidden_size}"     --set model.num_hidden_layers="${num_hidden_layers}"     --set model.num_attention_heads="${num_attention_heads}"     --set model.intermediate_size="${intermediate_size}"

  pretrain_cmd=(
    python src/training/Pretrain.py
    --data_file "${DATA_FILE}"
    --ntv3_embedding_path "${NTV3_EMBEDDING_PATH}"
    --config "${config_path}"
    --max_epochs "${MAX_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --output_dir "${run_dir}"
    --experiment_name "${run_name}"
    --devices "${DEVICES[@]}"
  )
  if (( ${#QWEN3_PRETRAIN_ARGS[@]} )); then
    pretrain_cmd+=("${QWEN3_PRETRAIN_ARGS[@]}")
  fi
  "${pretrain_cmd[@]}" > "${run_dir}/train.log" 2>&1

  python scripts/summarize_grid_results.py     --metrics_csv "${metrics_csv}"     --summary_csv "${SUMMARY_CSV}"     --run_name "${preset_name}"     --output_dir "${run_dir}"     --config_json "${config_path}"     --checkpoint_dir "${run_dir}/checkpoints"
done

echo "Stage-2 summary written to ${SUMMARY_CSV}"
