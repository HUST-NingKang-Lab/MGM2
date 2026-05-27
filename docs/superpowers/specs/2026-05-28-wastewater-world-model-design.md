# Wastewater MGM2 World Model Design

Date: 2026-05-28

## Goal

Build a microbiome world model for wastewater time-series forecasting on the
organized MC-prediction dataset. The model should reuse MGM2's ASV/OTU token
encoder and compare supervised forecasting, MAE-style reconstruction, and
JEPA-style latent prediction under one shared data and evaluation protocol.

The first implementation should prioritize small-scale, trustworthy experiments
over full leave-plant or all-plant sweeps. The success criterion for the first
stage is to beat simple temporal baselines and a small MLP/TCN-style forecaster
on held-out future samples before expanding to all 24 plants.

## Data Context

Primary data path:

`/home/zhanghaohong/data6/databases/wastewater/MC-prediction/data_organized`

The dataset contains 24 Danish wastewater treatment plants, 4,676 samples, and
6,023 unique ASVs. Each plant is split chronologically into 80% train, 5% val,
and 15% test. The organized `combined` directory contains Plant-prefixed ASV
features such as `Aalborg E|ASV1`, while each plant directory contains
plant-local ASV tables.

Important constraints:

- Values are relative abundance percentages in `[0, 100]`; MGM2 inputs should
  convert them to `[0, 1]` before applying the existing abundance preprocessor.
- About 90.2% of unique ASVs have FASTA sequences. Missing sequences need the
  existing taxonomy/file fallback embedding path.
- Sampling intervals are irregular, so date deltas and seasonal covariates
  should be included as optional temporal features.
- The `combined` split has overlapping global date ranges because the split is
  chronological within each plant. Evaluation must respect plant-local temporal
  order.

## Existing Baseline To Match

The MC-prediction reference implementation uses a fixed-dimensional time-series
formulation:

- Input window: 10 historical samples.
- Prediction horizon: 10 future samples.
- Core model: graph temporal convolution with a GraphicalLasso-derived support
  matrix, plus an LSTM baseline.
- Loss: Bray-Curtis dissimilarity, with MSE/MAE as auxiliary metrics.
- Taxa are modeled in selected clusters/groups rather than always predicting
  every ASV globally.

MGM2 should not copy the fixed-vector representation directly. It should use
ASV identity embeddings and variable-length community token sets, then decode
future abundance over a plant-specific ASV universe.

## Shared Data Protocol

Represent each sample as a community token set:

- `plant_id`
- `sample_id`
- `date`
- optional metadata: week, year, season, temperature, SVI/DSVI, plant config
- ASV tokens with raw relative abundance in `[0, 1]`
- NTv3 or fallback embedding for each ASV

Training examples are sliding windows inside each plant:

- context: samples `t-9 ... t`
- target: samples `t+1 ... t+10`
- stride: 1
- train targets must lie in the plant-local train period.
- validation targets must lie in the plant-local validation period.
- test targets must lie in the plant-local test period.
- context samples may come from earlier splits when forecasting validation or
  test targets. This preserves chronological causality and avoids dropping most
  validation/test windows in plants with short 5% validation splits.

Target ASV universe should be plant-specific for the first version. This avoids
forcing every plant to predict the full `Plant|ASV` combined matrix and keeps
the decoder small enough for quick experiments.

Two target-universe modes should be supported:

- Closed-world evaluation: predict the plant-local ASV universe from the
  organized table. This matches the fixed-feature setup used by the reference
  time-series baselines.
- Train-observed evaluation: predict only ASVs observed in the train period and
  report unseen target abundance as an explicit residual/unseen mass. This is
  stricter and more realistic for deployment.

## Evaluation Protocol

All model variants must produce predicted relative abundance tables on the
original abundance scale. Metrics are computed after inverse transform and
renormalization per predicted sample.

Core metrics:

- Bray-Curtis dissimilarity over future samples.
- MAE and RMSE on relative abundance.
- Spearman correlation over nonzero or top-abundance taxa.
- Top-k dominant taxa overlap for `k in {10, 20, 50}`.
- Multi-step degradation curves across horizons 1 through 10.
- Per-plant macro averages and sample-weighted micro averages should both be
  reported, so large plants do not hide failures on smaller plants.

Metadata policy:

- Calendar features such as horizon index, month, week, and time delta are
  allowed for future targets because they are known at forecast time.
- Measured covariates such as temperature, SVI, and DSVI should be context-only
  by default unless a separate future-covariate forecasting or oracle setting
  is explicitly enabled.

Baselines:

- Persistence: future equals the latest context sample.
- Historical mean: future equals mean of the context window.
- Seasonal or rolling mean if enough history exists.
- Small MLP/TCN forecaster on fixed plant-local abundance vectors.
- MC-prediction LSTM/GCN results where available or reproducible.

## Model Option A: Frozen MGM2 Temporal Forecaster

This is the recommended first implementation.

For each sample, encode the community with MGM2 and use the CLS vector as the
sample-level community state. A temporal model then predicts future community
states from the previous 10 states.

Data flow:

`sample tokens -> frozen MGM2 -> CLS_t`

`CLS_{t-9:t} + time/meta embeddings -> Temporal Transformer or TCN`

`future latent states + ASV query embeddings -> abundance decoder`

Training objective:

- Bray-Curtis on decoded relative abundance.
- Optional MSE on MGM2-preprocessed log abundance.
- Optional bin classification for high-level abundance structure.

Advantages:

- Fastest and lowest-risk path.
- Reuses pretrained MGM2 without immediately fine-tuning a large backbone.
- Gives an interpretable benchmark for whether MGM2 community states contain
  useful temporal information.

Main risk:

- CLS may compress away ASV-specific signals. If this happens, move to
  token-level cross-attention decoding.

## Model Option B: MGM2-MAE

Use a masked future reconstruction objective. The model receives observed past
samples and masked future sample tokens, then reconstructs future ASV abundance.

Masking strategy:

- Past samples are visible.
- Future abundance values are masked.
- ASV identity embeddings remain visible, so the model knows which organism it
  is predicting.
- Optionally mask blocks of taxa or complete future time steps.

Training objective:

- MGM2 abundance bin classification on masked future tokens.
- Regression on high-abundance future tokens.
- Bray-Curtis loss on decoded future abundance.

Advantages:

- Closest to the current MGM2 pretraining objective.
- Directly optimizes abundance reconstruction.
- Easier to evaluate than JEPA because output abundance is native.

Main risk:

- Can overfit to smoothed or mean-like predictions if zero-heavy taxa dominate
  the loss. Use high-abundance weighting and dominant-taxa metrics.

## Model Option C: MGM2-JEPA

Use latent future prediction rather than direct abundance reconstruction.

Architecture:

- Context encoder: past 10 samples.
- Target encoder: true future 10 samples.
- Predictor: maps context latent to future latent.
- Target encoder can be an EMA copy of the context encoder.
- A lightweight abundance decoder is trained for evaluation and optional
  supervised fine-tuning.

Training objective:

- Latent prediction loss between predicted future states and target future
  states.
- Stop-gradient or EMA target branch to avoid collapse.
- Optional variance/covariance regularization if needed.

Advantages:

- Better aligned with a "world model" framing.
- May learn smoother community dynamics and be less sensitive to noisy sparse
  abundance targets.
- Can be evaluated both as representation learning and as abundance forecasting.

Main risk:

- Latent loss alone is not enough. The model must be compared through the same
  abundance-scale forecasting metrics as all other models.

## Recommended Build Order

1. Build the wastewater window dataset and cache MGM2-ready sample objects.
2. Run persistence, historical mean, and a small MLP/TCN fixed-vector baseline.
3. Implement Frozen MGM2 Temporal Forecaster with frozen backbone.
4. Add token-level cross-attention decoder if CLS-only underperforms.
5. Implement MGM2-MAE using future abundance masking.
6. Implement MGM2-JEPA once the shared decoder and metrics are stable.
7. Expand from selected high-sample plants to all 24 plants.

Initial plant subset:

- Aalborg W
- Aalborg E
- Randers
- Ribe
- Fredericia

These plants have relatively more samples and enough test windows for early
model comparison.

## Implementation Boundaries

New code should live under a wastewater-specific downstream module, following
the existing organization style:

- `src/data/` for reusable dataset/window builders if generally useful.
- `src/models/` for MGM2 temporal, MAE, and JEPA modules.
- `src/training/` for Lightning training modules.
- `scripts/downstreams/wastewater/` for build, train, evaluate, and plot entry
  points.

Do not modify drug perturbation scripts for this work.

## Open Decisions

- Whether the first decoder predicts all plant-local ASVs or only a top-K ASV
  subset plus an "other" bucket.
- Whether metadata covariates should be mandatory or optional in the first run.
- Whether MGM2 is frozen for all early experiments or lightly fine-tuned after
  the temporal head stabilizes.
- Whether MC-prediction GCN should be rerun locally or used only as a reported
  external baseline in the first phase.
