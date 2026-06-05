#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/sae_interpretation/otu_xlarge_k64_aux0p03_global.json}
DEVICE=${DEVICE:-cpu}
NUM_THREADS=${NUM_THREADS:-16}
FEATURE_LOOKUP_BATCH_SIZE=${FEATURE_LOOKUP_BATCH_SIZE:-256}

cmd=(
  /home/zhanghaohong/anaconda3/envs/mwm/bin/python
  scripts/sae_interpretation/run_otu_feature_interpretation_pipeline.py
  --config "${CONFIG}"
  --all-features
  --device "${DEVICE}"
  --num-threads "${NUM_THREADS}"
  --feature-lookup-batch-size "${FEATURE_LOOKUP_BATCH_SIZE}"
)

if [[ -n "${MAX_CHUNKS:-}" ]]; then
  cmd+=(--max-chunks "${MAX_CHUNKS}")
fi

"${cmd[@]}"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python \
  scripts/sae_interpretation/summarize_feature_evidence_catalog.py \
  --config "${CONFIG}"
