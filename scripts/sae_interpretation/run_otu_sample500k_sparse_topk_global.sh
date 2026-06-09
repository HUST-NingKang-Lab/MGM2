#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global.json}
DEVICE=${DEVICE:-cuda:6}
NUM_THREADS=${NUM_THREADS:-16}
CANDIDATE_N=${CANDIDATE_N:-500}
CURATED_FEATURE_IDS=${CURATED_FEATURE_IDS:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global}
FEATURE_SCORES=${FEATURE_SCORES:-${OUTPUT_ROOT}/feature_scores.csv}
SELECTED_FEATURES=${SELECTED_FEATURES:-${OUTPUT_ROOT}/selected_interpretable_feature_ids.json}

score_cmd=(
  /home/zhanghaohong/anaconda3/envs/mwm/bin/python
  scripts/sae_interpretation/score_sae_features.py
  --config "${CONFIG}"
  --output-csv "${FEATURE_SCORES}"
  --device "${DEVICE}"
  --num-threads "${NUM_THREADS}"
)

if [[ -n "${MAX_CHUNKS:-}" ]]; then
  score_cmd+=(--max-chunks "${MAX_CHUNKS}" --overwrite)
fi

"${score_cmd[@]}"

/home/zhanghaohong/anaconda3/envs/mwm/bin/python \
  scripts/sae_interpretation/select_interpretable_features.py \
  --config "${CONFIG}" \
  --feature-scores "${FEATURE_SCORES}" \
  --output-json "${SELECTED_FEATURES}" \
  --total-n "${CANDIDATE_N}" \
  --curated-feature-ids "${CURATED_FEATURE_IDS}"

cmd=(
  /home/zhanghaohong/anaconda3/envs/mwm/bin/python
  scripts/sae_interpretation/run_otu_feature_interpretation_pipeline.py
  --config "${CONFIG}"
  --feature-ids-file "${SELECTED_FEATURES}"
  --feature-scores "${FEATURE_SCORES}"
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
