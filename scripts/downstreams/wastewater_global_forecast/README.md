# Wastewater Global Forecast

This directory contains the no-residual wastewater forecasting experiments.

Design:

- Use the existing wastewater window dataset and token caches.
- Train one global model across all plants.
- Do not use persistence residual heads.
- Do not add metadata or plant covariates.
- Use `BC + MSE_WEIGHT * MSE` for trainable abundance-prediction losses.
- Evaluate `test` split by default.

Main commands:

```bash
conda activate mwm

# Simple non-trainable baselines.
bash scripts/downstreams/wastewater_global_forecast/run_simple_baselines.sh

# Global abundance-only Transformer baseline.
DEVICE=cuda:7 MSE_WEIGHT=0.1 \
  bash scripts/downstreams/wastewater_global_forecast/run_global_abundance_transformer.sh

# MGM2 no-residual softmax models over all scales.
DEVICE=cuda:7 MSE_WEIGHT=0.1 \
  bash scripts/downstreams/wastewater_global_forecast/run_mgm2_no_residual_scales.sh

# Summarize all available outputs.
python scripts/downstreams/wastewater_global_forecast/summarize_global_forecast.py
```

Full run:

```bash
conda activate mwm
DEVICE=cuda:7 MSE_WEIGHT=0.1 \
  bash scripts/downstreams/wastewater_global_forecast/run_all.sh
```

Outputs are written under `outputs_downstreams/wastewater_global_forecast/`.
