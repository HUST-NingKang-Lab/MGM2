#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ACTIVATIONS="${ACTIVATIONS:-data/sae/pretrain_xlarge_sample200k/mgm2_scale_xlarge_last_hidden_otu_activations.h5}"
TOKEN_TYPE="${TOKEN_TYPE:-otu}"
DICT_SIZE="${DICT_SIZE:-4096}"
TOP_K_GRID="${TOP_K_GRID:-8 16 32 64 128}"
AUX_PENALTY="${AUX_PENALTY:-0.03125}"
AUX_K="${AUX_K:-512}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
STEPS="${STEPS:-61035}"
DEVICE="${DEVICE:-cuda:6}"
LEARNING_RATE="${LEARNING_RATE:-7e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.0}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
PRECISION="${PRECISION:-bf16-mixed}"
DEAD_FEATURE_THRESHOLD="${DEAD_FEATURE_THRESHOLD:-5}"
INACTIVE_COUNTER_MODE="${INACTIVE_COUNTER_MODE:-forwards}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
EARLY_STOPPING="${EARLY_STOPPING:-0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-5}"
EARLY_STOPPING_MIN_STEPS="${EARLY_STOPPING_MIN_STEPS:-20000}"
PRELOAD_TO_MEMORY="${PRELOAD_TO_MEMORY:-1}"
PRELOAD_DTYPE="${PRELOAD_DTYPE:-float16}"
PRELOAD_CHUNK_ROWS="${PRELOAD_CHUNK_ROWS:-262144}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_sae/mgm2_xlarge_otu_sample200k_batchtopk_grid_d${DICT_SIZE}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${OUTPUT_ROOT}"

for top_k in ${TOP_K_GRID}; do
  aux_label="${AUX_PENALTY//./p}"
  output_dir="${OUTPUT_ROOT}/k${top_k}_aux${aux_label}"
  log_path="${output_dir}/run.log"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_dir}/metrics.csv" ]]; then
    echo "Skipping existing run: ${output_dir}"
    continue
  fi

  mkdir -p "${output_dir}"
  echo "Running sample200k OTU BatchTopK SAE: top_k=${top_k} aux_penalty=${AUX_PENALTY} output=${output_dir}"
  (
    export ACTIVATIONS
    export TOKEN_TYPE
    export DICT_SIZE
    export TOP_K="${top_k}"
    export AUX_K
    export AUX_PENALTY
    export BATCH_SIZE
    export STEPS
    export DEVICE
    export LEARNING_RATE
    export WEIGHT_DECAY
    export ADAM_BETA1
    export ADAM_BETA2
    export WARMUP_STEPS
    export PRECISION
    export DEAD_FEATURE_THRESHOLD
    export INACTIVE_COUNTER_MODE
    export EVAL_EVERY
    export EVAL_BATCHES
    export SAVE_EVERY
    export EARLY_STOPPING
    export EARLY_STOPPING_PATIENCE
    export EARLY_STOPPING_MIN_DELTA
    export EARLY_STOPPING_MIN_STEPS
    export PRELOAD_TO_MEMORY
    export PRELOAD_DTYPE
    export PRELOAD_CHUNK_ROWS
    export OUTPUT_DIR="${output_dir}"
    bash scripts/sae/train_mgm2_xlarge_otu_batchtopk_sae.sh
  ) 2>&1 | tee "${log_path}"
done
