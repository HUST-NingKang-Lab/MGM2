#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ACTIVATIONS="${ACTIVATIONS:-data/sae/pretrain_xlarge_sample50k/mgm2_scale_xlarge_last_hidden_activations.h5}"
TOKEN_TYPE="${TOKEN_TYPE:-otu}"
DICT_SIZE="${DICT_SIZE:-}"
TOP_K="${TOP_K:-32}"
AUX_K="${AUX_K:-16}"
AUX_PENALTY="${AUX_PENALTY:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
STEPS="${STEPS:-20000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
PRECISION="${PRECISION:-float32}"
DEAD_FEATURE_THRESHOLD="${DEAD_FEATURE_THRESHOLD:-10000000}"
INACTIVE_COUNTER_MODE="${INACTIVE_COUNTER_MODE:-tokens}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-50}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
EARLY_STOPPING="${EARLY_STOPPING:-0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-5}"
EARLY_STOPPING_MIN_STEPS="${EARLY_STOPPING_MIN_STEPS:-0}"
DEVICE="${DEVICE:-cuda:7}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PRELOAD_TO_MEMORY="${PRELOAD_TO_MEMORY:-1}"
PRELOAD_DTYPE="${PRELOAD_DTYPE:-float16}"
PRELOAD_CHUNK_ROWS="${PRELOAD_CHUNK_ROWS:-262144}"

cmd=(
  /home/zhanghaohong/anaconda3/envs/mwm/bin/python
  scripts/sae/train_batchtopk_sae.py
  --activations "${ACTIVATIONS}"
  --token_type "${TOKEN_TYPE}"
  --top_k "${TOP_K}"
  --aux_k "${AUX_K}"
  --aux_penalty "${AUX_PENALTY}"
  --batch_size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --adam_beta1 "${ADAM_BETA1}"
  --adam_beta2 "${ADAM_BETA2}"
  --warmup_steps "${WARMUP_STEPS}"
  --precision "${PRECISION}"
  --dead_feature_threshold "${DEAD_FEATURE_THRESHOLD}"
  --inactive_counter_mode "${INACTIVE_COUNTER_MODE}"
  --eval_every "${EVAL_EVERY}"
  --eval_batches "${EVAL_BATCHES}"
  --save_every "${SAVE_EVERY}"
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}"
  --early_stopping_min_delta "${EARLY_STOPPING_MIN_DELTA}"
  --early_stopping_min_steps "${EARLY_STOPPING_MIN_STEPS}"
  --device "${DEVICE}"
  --preload_dtype "${PRELOAD_DTYPE}"
  --preload_chunk_rows "${PRELOAD_CHUNK_ROWS}"
)

if [[ "${PRELOAD_TO_MEMORY}" == "1" ]]; then
  cmd+=(--preload_to_memory)
fi
if [[ "${EARLY_STOPPING}" == "1" ]]; then
  cmd+=(--early_stopping)
fi
if [[ -n "${DICT_SIZE}" ]]; then
  cmd+=(--dict_size "${DICT_SIZE}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output_dir "${OUTPUT_DIR}")
fi
cmd+=("$@")

"${cmd[@]}"
