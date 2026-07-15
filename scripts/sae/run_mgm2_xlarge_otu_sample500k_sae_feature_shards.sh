#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SHARD_COUNT="${SHARD_COUNT:-4}"
DEVICE="${DEVICE:-cuda:6}"
SHARD_ROOT="${SHARD_ROOT:-outputs_sae_analysis/otu_sample500k_feature_analysis_shards}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/sae/otu_sample500k_feature_analysis_shards}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_sae_analysis/otu_sample500k_feature_analysis}"
FIGURE_DIR="${FIGURE_DIR:-figures/sae/otu_sample500k_feature_analysis}"
TOP_K_VALUES="${TOP_K_VALUES:-8 16 32 64}"
SKIP_COMPLETE_SHARDS="${SKIP_COMPLETE_SHARDS:-1}"

mkdir -p "${SHARD_ROOT}" "${FIGURE_ROOT}"

expected_raw_stats=(
  "k8_aux0p03125.npz"
  "k16_aux0p03125.npz"
  "k32_aux0p03125.npz"
  "k64_aux0p03125.npz"
)

for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_output="${SHARD_ROOT}/shard_${shard_index}_of_${SHARD_COUNT}"
  shard_figures="${FIGURE_ROOT}/shard_${shard_index}_of_${SHARD_COUNT}"
  complete=1
  for filename in "${expected_raw_stats[@]}"; do
    if [[ ! -s "${shard_output}/raw_stats/${filename}" ]]; then
      complete=0
      break
    fi
  done
  if [[ "${SKIP_COMPLETE_SHARDS}" == "1" && "${complete}" == "1" ]]; then
    echo "Skipping complete shard ${shard_index}/${SHARD_COUNT}: ${shard_output}"
    continue
  fi

  echo "Running SAE feature-analysis shard ${shard_index}/${SHARD_COUNT} on ${DEVICE}"
  DEVICE="${DEVICE}" \
  SHARD_COUNT="${SHARD_COUNT}" \
  SHARD_INDEX="${shard_index}" \
  OUTPUT_DIR="${shard_output}" \
  FIGURE_DIR="${shard_figures}" \
  TOP_K_VALUES="${TOP_K_VALUES}" \
  VERIFY_OUTPUTS=0 \
  bash scripts/sae/analyze_mgm2_xlarge_otu_sample500k_sae_features.sh
done

echo "Merging ${SHARD_COUNT} shards into ${OUTPUT_DIR}"
SHARD_ROOT="${SHARD_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FIGURE_DIR="${FIGURE_DIR}" \
TOP_K_VALUES="${TOP_K_VALUES}" \
bash scripts/sae/merge_mgm2_xlarge_otu_sample500k_sae_feature_shards.sh
