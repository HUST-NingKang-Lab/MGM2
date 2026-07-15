#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/zhanghaohong/anaconda3/envs/mwm/bin/python}"
DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
DATA_ROOT="${DATA_ROOT:-/home/zhanghaohong/data6/databases/wastewater/MC-prediction/data_organized}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs_grid/stage2/20260505_094404}"
ASV_NTV3_EMBEDDING="${ASV_NTV3_EMBEDDING:-data/downstreams/wastewater/cache_ntv3_asv/wastewater_asv_ntv3_embeddings.h5}"
FALLBACK_EMBEDDING="${FALLBACK_EMBEDDING:-ntv3_embeddings/otus_99_taxonomy_fallback.h5}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/downstreams/wastewater_v2/embeddings}"
SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
DEVICE="${DEVICE:-cuda:7}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FORCE="${FORCE:-0}"
ALLOW_MISSING_TAXONOMY="${ALLOW_MISSING_TAXONOMY:-0}"

mkdir -p "${OUTPUT_ROOT}"

missing_taxonomy_arg=""
if [[ "${ALLOW_MISSING_TAXONOMY}" == "1" ]]; then
  missing_taxonomy_arg="--allow_missing_taxonomy"
fi

echo "Token cache extraction settings:"
echo "  dataset: ${DATASET}"
echo "  output:  ${OUTPUT_ROOT}"
echo "  scales:  ${SCALES}"
echo "  device:  ${DEVICE}"
echo "  batch:   ${BATCH_SIZE}"

for scale in ${SCALES}; do
  config="${STAGE2_ROOT}/${scale}/config.json"
  checkpoint="${STAGE2_ROOT}/${scale}/checkpoints/last.ckpt"
  output="${OUTPUT_ROOT}/mgm2_${scale}_top200_tokens.pkl"
  log="${OUTPUT_ROOT}/mgm2_${scale}_top200_tokens.log"

  if [[ ! -f "${config}" ]]; then
    echo "[error] missing config for ${scale}: ${config}" >&2
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] missing checkpoint for ${scale}: ${checkpoint}" >&2
    exit 1
  fi
  if [[ "${FORCE}" != "1" && -s "${output}" ]]; then
    echo "[skip] ${scale}: ${output} exists"
    continue
  fi

  echo "[run] ${scale} MGM2 ASV token states"
  "${PYTHON}" scripts/downstreams/wastewater/extract_mgm2_token_cache.py \
    --dataset "${DATASET}" \
    --data_root "${DATA_ROOT}" \
    --stage2_root "${STAGE2_ROOT}" \
    --scale "${scale}" \
    --ntv3_embedding "${ASV_NTV3_EMBEDDING}" \
    --fallback_embedding "${FALLBACK_EMBEDDING}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --output "${output}" \
    ${missing_taxonomy_arg} \
    2>&1 | tee "${log}"
done

echo "Done. Token cache files:"
find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name 'mgm2_*_top200_tokens.pkl' | sort
