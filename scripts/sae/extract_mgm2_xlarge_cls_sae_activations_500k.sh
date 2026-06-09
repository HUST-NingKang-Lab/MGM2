#!/usr/bin/env bash
set -euo pipefail

MANIFEST_DIR="${MANIFEST_DIR:-data/sae/pretrain_xlarge_sample500k}"
TOKEN_TYPE="${TOKEN_TYPE:-cls}"
OUTPUT="${OUTPUT:-${MANIFEST_DIR}/mgm2_scale_xlarge_last_hidden_cls_activations.h5}"
DEVICE="${DEVICE:-cuda:6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-4096}"
DTYPE="${DTYPE:-float16}"
COMPRESSION="${COMPRESSION:-lzf}"
RESUME="${RESUME:-0}"

MANIFEST_DIR="${MANIFEST_DIR}" \
TOKEN_TYPE="${TOKEN_TYPE}" \
OUTPUT="${OUTPUT}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS}" \
DTYPE="${DTYPE}" \
COMPRESSION="${COMPRESSION}" \
RESUME="${RESUME}" \
bash scripts/sae/extract_mgm2_xlarge_sae_activations.sh "$@"
