#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl}"
DATA_ROOT="${DATA_ROOT:-/home/zhanghaohong/data6/databases/wastewater/MC-prediction/data_organized}"
STAGE2_ROOT="${STAGE2_ROOT:-outputs_grid/stage2/20260505_094404}"
ASV_NTV3_EMBEDDING="${ASV_NTV3_EMBEDDING:-data/downstreams/wastewater/cache_ntv3_asv/wastewater_asv_ntv3_embeddings.h5}"
FALLBACK_EMBEDDING="${FALLBACK_EMBEDDING:-ntv3_embeddings/otus_99_taxonomy_fallback.h5}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/downstreams/wastewater_v2/embeddings}"
SCALES="${SCALES:-scale_small scale_medium scale_large scale_xlarge}"
KINDS="${KINDS:-cls}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FORCE="${FORCE:-0}"
ALLOW_MISSING_TAXONOMY="${ALLOW_MISSING_TAXONOMY:-0}"
PYTHON="${PYTHON:-python}"

mkdir -p "${OUTPUT_ROOT}"

missing_taxonomy_arg=""
if [[ "${ALLOW_MISSING_TAXONOMY}" == "1" ]]; then
  missing_taxonomy_arg="--allow_missing_taxonomy"
fi

run_or_skip() {
  local output="$1"
  local description="$2"
  shift 2
  if [[ "${FORCE}" != "1" && -s "${output}" ]]; then
    echo "[skip] ${description}: ${output} exists"
    return 0
  fi
  echo "[run] ${description}"
  "$@"
}

for scale in ${SCALES}; do
  config="${STAGE2_ROOT}/${scale}/config.json"
  checkpoint="${STAGE2_ROOT}/${scale}/checkpoints/last.ckpt"
  if [[ ! -f "${config}" ]]; then
    echo "[error] missing config for ${scale}: ${config}" >&2
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] missing checkpoint for ${scale}: ${checkpoint}" >&2
    exit 1
  fi

  for kind in ${KINDS}; do
    case "${kind}" in
      cls)
        output="${OUTPUT_ROOT}/mgm2_${scale}_top200_cls.pkl"
        log="${OUTPUT_ROOT}/mgm2_${scale}_top200_cls.log"
        run_or_skip "${output}" "${scale} CLS sample embeddings" \
          "${PYTHON}" scripts/downstreams/wastewater/extract_mgm2_cls_cache.py \
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
        ;;
      token|tokens)
        output="${OUTPUT_ROOT}/mgm2_${scale}_top200_tokens.pkl"
        log="${OUTPUT_ROOT}/mgm2_${scale}_top200_tokens.log"
        run_or_skip "${output}" "${scale} ASV token embeddings" \
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
        ;;
      *)
        echo "[error] unknown KINDS entry: ${kind}; expected cls or token" >&2
        exit 1
        ;;
    esac
  done
done

echo "Done. Cache files under ${OUTPUT_ROOT}:"
find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name 'mgm2_*_top200_*.pkl' | sort
