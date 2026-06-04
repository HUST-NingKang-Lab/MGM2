#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_FILE="${DATA_FILE:-data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.pkl}"
MANIFEST_DIR="${MANIFEST_DIR:-data/sae/pretrain_xlarge_sample50k}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs_grid/stage2/20260505_094404}"
SCALE="${SCALE:-scale_xlarge}"
NTV3_EMBEDDING="${NTV3_EMBEDDING:-ntv3_embeddings/otus_99_embeddings.h5}"
OUTPUT="${OUTPUT:-${MANIFEST_DIR}/mgm2_${SCALE}_last_hidden_activations.h5}"
DEVICE="${DEVICE:-cuda:6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-4096}"
DTYPE="${DTYPE:-float16}"
TOKEN_TYPE="${TOKEN_TYPE:-all}"
COMPRESSION="${COMPRESSION:-lzf}"
RESUME="${RESUME:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

resume_args=()
if [[ "${RESUME}" == "1" ]]; then
  resume_args=(--resume)
fi

/home/zhanghaohong/anaconda3/envs/mwm/bin/python scripts/sae/extract_mgm2_sae_activations.py \
  --data_file "${DATA_FILE}" \
  --manifest_dir "${MANIFEST_DIR}" \
  --stage2_root "${STAGE2_ROOT}" \
  --scale "${SCALE}" \
  --ntv3_embedding "${NTV3_EMBEDDING}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_batch_tokens "${MAX_BATCH_TOKENS}" \
  --dtype "${DTYPE}" \
  --token_type "${TOKEN_TYPE}" \
  --compression "${COMPRESSION}" \
  "${resume_args[@]}" \
  "$@"
