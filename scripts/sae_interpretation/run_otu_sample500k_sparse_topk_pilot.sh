#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_pilot.json}
DEVICE=${DEVICE:-cuda:6}
NUM_THREADS=${NUM_THREADS:-16}
FEATURE_IDS=${FEATURE_IDS:-21,2650,2658,2659,2664,2670,2671}

cmd=(
  /home/zhanghaohong/anaconda3/envs/mwm/bin/python
  scripts/sae_interpretation/run_otu_feature_interpretation_pipeline.py
  --config "${CONFIG}"
  --feature-ids "${FEATURE_IDS}"
  --device "${DEVICE}"
  --num-threads "${NUM_THREADS}"
)

if [[ -n "${MAX_CHUNKS:-}" ]]; then
  cmd+=(--max-chunks "${MAX_CHUNKS}")
fi

"${cmd[@]}"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python \
  scripts/sae_interpretation/summarize_feature_evidence_catalog.py \
  --config "${CONFIG}"
