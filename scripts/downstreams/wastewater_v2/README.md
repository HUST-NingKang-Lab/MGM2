# Wastewater V2 Forecasting Workspace

This folder is a clean workspace for rethinking wastewater ASV forecasting from
the data and representation level. It intentionally does not depend on the
previous wastewater world-model implementations.

## Problem Reassessment

The previous MGM2 world-model results were better than GRU, TCN, MicroProphet,
and context mean on aggregate metrics, but the ASV-level trend plots show that
the model still often fails to capture real abundance changes. The key issue is
not only architecture scale. The task itself is difficult because:

- Most ASVs are sparse, low-abundance, and noisy.
- Bray-Curtis and MAE reward good global composition but can hide poor
  per-ASV temporal trend recovery.
- Persistence/context-like behavior is strong, so models can look reasonable
  without learning real changes.
- Samples are irregularly spaced, and target horizons can span different
  calendar gaps.
- The output is compositional, so increasing one ASV changes the rest.

Therefore V2 should treat this as **compositional temporal change modeling**,
not just future abundance reconstruction.

## Fixed Representation Cache

First build frozen MGM2 embeddings for every unique wastewater sample. The cache
is a reusable input layer for all V2 models.

Default cache root:

```text
data/downstreams/wastewater_v2/embeddings/
```

Recommended command:

```bash
PYTHON=/home/zhanghaohong/anaconda3/envs/mwm/bin/python \
bash scripts/downstreams/wastewater_v2/build_mgm2_embeddings.sh
```

Useful overrides:

```bash
PYTHON=/home/zhanghaohong/anaconda3/envs/mwm/bin/python \
SCALES="scale_medium scale_xlarge" \
KINDS="cls" \
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/build_mgm2_embeddings.sh
```

Set `KINDS="token"` or `KINDS="cls token"` only when token-level ASV states are needed.

Dedicated token-cache command:

```bash
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/build_mgm2_token_cache.sh
```

Each CLS cache contains:

- `sample_keys`: `(plant, sample_id)` keys.
- `sample_to_idx`: map from sample key to row index.
- `cls_embeddings`: `[n_samples, hidden_size]`.
- `asv_universes`: plant-specific ASV order.
- `embedding_source_counts`: NTv3/fallback resolution counts.

Each token cache contains:

- `token_states`: `[n_samples, n_asvs, hidden_size]`.
- The same sample and ASV metadata as the CLS cache.

For the current top-200 plus `__OTHER__` dataset, the expected token-state
shape is `[4676, 201, hidden_size]` for each scale.

## Candidate V2 Models

### 1. CLR Delta Temporal Transformer

Model abundance changes in centered log-ratio space:

```text
context abundance -> clr/logratio sequence
context MGM2 CLS embeddings -> sample-level state sequence
plant embedding + calendar gaps
  -> temporal transformer
  -> future clr delta
  -> inverse transform / closure
```

Primary losses:

- ASV-wise correlation loss or negative Pearson/Spearman on selected variable ASVs.
- Directional change loss on `sign(delta)`.
- Bray-Curtis as a secondary composition-level loss.

This directly targets the failure mode seen in ASV trend plots.

### 2. ASV Factor Dynamics Model

Represent each future abundance change as a low-rank factor model:

```text
sample latent state z_t from CLS trajectory
ASV identity/state from MGM2 token embedding
future delta(asv) = shared temporal factors x ASV loadings
```

This is more appropriate than predicting 201 ASVs independently because ASVs
co-vary through ecological/community factors.

### 3. Retrieval + Residual Forecaster

Use MGM2 CLS trajectories to retrieve similar historical context windows:

```text
nearest historical windows in MGM2 CLS trajectory space
  -> weighted future delta template
  -> small residual network conditioned on current abundance
```

This is a strong sanity baseline. If a complex model cannot beat it on
ASV-level trend correlation, the learned model is probably not extracting useful
dynamics.

### 4. Two-Stage Forecasting

Separate the problem:

1. Predict community-level movement in MGM2 embedding space.
2. Decode that movement into ASV-level clr deltas with an ASV-conditioned head.

This keeps the temporal model focused on lower-dimensional dynamics and moves
ASV specificity to the decoder.

## Evaluation Changes

V2 should report both composition-level and ASV-trend-level metrics:

- Bray-Curtis, MAE, RMSE for comparability.
- Per-ASV Pearson/Spearman over the 10 target steps.
- Directional accuracy for abundance increases/decreases.
- Metrics on variable ASVs only, not all nearly-zero ASVs.
- Per-plant robustness and examples of high-variance ASVs.

The current aggregate score is not enough to claim useful temporal forecasting.

## Suggested First Experiments

1. Build MGM2 embedding caches for `scale_medium` and `scale_xlarge`.
2. Implement a retrieval + residual baseline.
3. Implement CLR Delta Temporal Transformer.
4. Compare against GRU/TCN/context mean on:
   - full-composition metrics;
   - ASV trend correlation;
   - directional change accuracy.

## Implemented V2 Baseline: Plant-Specific CLR-Delta Transformer

The first implemented V2 model is:

```text
context abundance + context MGM2 CLS trajectory + plant id
  -> shared temporal Transformer encoder/decoder
  -> plant-specific CLR-delta head
  -> softmax closure back to relative abundance
```

The plant-specific heads are important because the top-200 ASV columns are
plant-local. Column 17 in two plants is not guaranteed to be the same ASV, so a
single global output head would mix unrelated taxa.

Run all scales:

```bash
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/run_clr_delta_transformer_scales.sh
```

Useful quick test:

```bash
SCALES="scale_small" \
MAX_EPOCHS=3 \
PATIENCE=2 \
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/run_clr_delta_transformer_scales.sh
```

Outputs are written to:

```text
outputs_downstreams/wastewater_v2/clr_delta_transformer/<scale>/
```

Key files:

- `summary.csv`: per-plant test metrics, including ASV change metrics.
- `metrics_by_horizon.csv`: horizon-wise composition metrics.
- `train_history.csv`: training/holdout loss curves.
- `checkpoint.pt`: model weights and run metadata.

The added ASV-change metrics are:

- `abundance_delta_pearson`: Pearson correlation between true and predicted
  abundance changes, flattened over windows, horizons, and ASVs.
- `mean_asv_delta_pearson`: mean per-ASV abundance-change correlation.
- `top_variable_asv_delta_pearson`: mean per-ASV correlation among the most
  variable ASVs in that plant.
- `direction_accuracy`: sign accuracy for non-trivial abundance changes.

## Implemented V2 Model: Token ASV-Delta Transformer

After token caches are available, train the ASV-conditioned model:

```text
context abundance + context MGM2 ASV-token states + plant id
  -> per-ASV temporal Transformer over the context window
  -> horizon-specific ASV interaction Transformer
  -> shared scalar CLR-delta head per ASV
  -> softmax closure back to relative abundance
```

This model uses token states as ASV identity/state information, so it no longer
relies on a plant-specific output vector to memorize local ASV columns.

Run all scales:

```bash
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/run_token_asv_delta_transformer_scales.sh
```

Useful quick test:

```bash
SCALES="scale_small" \
MAX_EPOCHS=3 \
PATIENCE=2 \
BATCH_SIZE=4 \
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/run_token_asv_delta_transformer_scales.sh
```

Summarize results:

```bash
python scripts/downstreams/wastewater_v2/summarize_token_asv_delta_results.py
```

## Optimized Token Loss

The optimized token run keeps the same architecture but changes the training
objective:

- upweights ASVs with higher abundance or stronger future changes;
- adds weighted Bray-Curtis and weighted MSE terms;
- adds a per-ASV CLR-delta correlation loss;
- keeps the original composition-level losses for comparability.

Run the optimized all-scale sweep:

```bash
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/run_token_asv_delta_optimized.sh
```

Quick smoke run:

```bash
SCALES="scale_large" \
MAX_EPOCHS=3 \
PATIENCE=2 \
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/run_token_asv_delta_optimized.sh
```

Summarize optimized results:

```bash
python scripts/downstreams/wastewater_v2/summarize_token_asv_delta_optimized.py
```

The first `change_focus` run improved some ASV-change metrics but hurt
composition metrics. A more conservative profile keeps the original token loss
dominant and adds only light ASV-change supervision:

```bash
DEVICE=cuda:7 \
bash scripts/downstreams/wastewater_v2/run_token_asv_delta_conservative.sh
```

Summarize and compare all token optimization runs:

```bash
python scripts/downstreams/wastewater_v2/summarize_token_asv_delta_conservative.py
python scripts/downstreams/wastewater_v2/compare_token_optimization_runs.py
```
