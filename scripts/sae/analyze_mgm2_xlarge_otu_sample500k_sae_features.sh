#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

GRID_DIR="${GRID_DIR:-outputs_sae/mgm2_xlarge_otu_sample500k_batchtopk_grid_d4096}"
ACTIVATIONS="${ACTIVATIONS:-data/sae/pretrain_xlarge_sample500k/mgm2_scale_xlarge_last_hidden_otu_activations.h5}"
SAMPLES_CSV="${SAMPLES_CSV:-data/sae/pretrain_xlarge_sample500k/samples.csv}"
TOP_K_VALUES="${TOP_K_VALUES:-8 16 32 64}"
SPLITS="${SPLITS:-val}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
CHUNK_ROWS="${CHUNK_ROWS:-262144}"
MAX_TOKENS="${MAX_TOKENS:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
if [[ "${SHARD_COUNT}" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-outputs_sae_analysis/otu_sample500k_feature_analysis}"
  FIGURE_DIR="${FIGURE_DIR:-figures/sae/otu_sample500k_feature_analysis}"
  VERIFY_OUTPUTS="${VERIFY_OUTPUTS:-1}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-outputs_sae_analysis/otu_sample500k_feature_analysis_shards/shard_${SHARD_INDEX}_of_${SHARD_COUNT}}"
  FIGURE_DIR="${FIGURE_DIR:-figures/sae/otu_sample500k_feature_analysis_shards/shard_${SHARD_INDEX}_of_${SHARD_COUNT}}"
  VERIFY_OUTPUTS="${VERIFY_OUTPUTS:-0}"
fi
INDEX_CACHE="${INDEX_CACHE:-outputs_sae_analysis/otu_sample500k_feature_analysis/selected_indices_val.npy}"
STATS_DIR="${STATS_DIR:-${OUTPUT_DIR}/raw_stats}"
DEVICE="${DEVICE:-cuda:6}"
EMBEDDING_METHOD="${EMBEDDING_METHOD:-auto}"
SEED="${SEED:-42}"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python scripts/sae/analyze_mgm2_xlarge_otu_sae_features.py \
  --grid-dir "${GRID_DIR}" \
  --activations "${ACTIVATIONS}" \
  --samples-csv "${SAMPLES_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --figure-dir "${FIGURE_DIR}" \
  --top-k ${TOP_K_VALUES} \
  --splits ${SPLITS} \
  --batch-size "${BATCH_SIZE}" \
  --chunk-rows "${CHUNK_ROWS}" \
  --max-tokens "${MAX_TOKENS}" \
  --index-cache "${INDEX_CACHE}" \
  --shard-count "${SHARD_COUNT}" \
  --shard-index "${SHARD_INDEX}" \
  --stats-dir "${STATS_DIR}" \
  --device "${DEVICE}" \
  --embedding-method "${EMBEDDING_METHOD}" \
  --seed "${SEED}" \
  "$@"

if [[ "${VERIFY_OUTPUTS}" == "1" ]]; then
  /home/zhanghaohong/anaconda3/envs/mwm/bin/python scripts/sae/verify_mgm2_xlarge_otu_sae_feature_analysis.py \
    --analysis-dir "${OUTPUT_DIR}" \
    --figure-dir "${FIGURE_DIR}"
fi
