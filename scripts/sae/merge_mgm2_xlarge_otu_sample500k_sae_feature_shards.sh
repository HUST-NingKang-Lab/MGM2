#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SHARD_ROOT="${SHARD_ROOT:-outputs_sae_analysis/otu_sample500k_feature_analysis_shards}"
GRID_DIR="${GRID_DIR:-outputs_sae/mgm2_xlarge_otu_sample500k_batchtopk_grid_d4096}"
SAMPLES_CSV="${SAMPLES_CSV:-data/sae/pretrain_xlarge_sample500k/samples.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_sae_analysis/otu_sample500k_feature_analysis}"
FIGURE_DIR="${FIGURE_DIR:-figures/sae/otu_sample500k_feature_analysis}"
TOP_K_VALUES="${TOP_K_VALUES:-8 16 32 64}"
EMBEDDING_METHOD="${EMBEDDING_METHOD:-auto}"
SEED="${SEED:-42}"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python scripts/sae/merge_mgm2_xlarge_otu_sae_feature_shards.py \
  --shard-root "${SHARD_ROOT}" \
  --grid-dir "${GRID_DIR}" \
  --samples-csv "${SAMPLES_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --figure-dir "${FIGURE_DIR}" \
  --top-k ${TOP_K_VALUES} \
  --embedding-method "${EMBEDDING_METHOD}" \
  --seed "${SEED}" \
  "$@"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python scripts/sae/verify_mgm2_xlarge_otu_sae_feature_analysis.py \
  --analysis-dir "${OUTPUT_DIR}" \
  --figure-dir "${FIGURE_DIR}"
