# Drug Perturbation Prediction

This workflow predicts a drug-treated microbial community from its matched
control community, a drug representation, and donor context. Evaluation can
hold out a donor-drug pair, an entire drug, or an entire donor.

## Inputs

The raw dataset directory supplies experiment metadata, microbial abundance
tables, and drug embeddings. MGM2 preprocessing aligns matched controls and
treated samples, selects a shared OTU vocabulary, stores raw evaluation
abundances, and attaches fixed OTU embeddings.

Configure the raw-data directory, output directory, Python executable, and NTv3
embedding path at the top of `build_mgm2_dataset.sh`, then run:

```bash
bash scripts/downstreams/drug_perturbation/build_mgm2_dataset.sh
```

The default artifact is:

```text
data/downstreams/drug_perturbation/drug_perturbation_top1024.pkl
```

This shared preprocessed dataset is used by MGM2 and cached-head experiments so
that training targets and restored relative-abundance evaluation are aligned.

## Simple baselines

Mean-delta, donor/drug conditional, nearest-drug, and no-effect baselines are
implemented in `baseline.py`:

```bash
python scripts/downstreams/drug_perturbation/baseline.py \
  --data_dir /path/to/drug_perturbation_data \
  --output_dir outputs_downstreams/drug_perturbation/baseline \
  --split_modes pair drug donor \
  --top_k_otus 1024
```

Train the fully connected baseline with the desired split protocol:

```bash
python scripts/downstreams/drug_perturbation/train_mlp_baseline.py \
  --data_dir /path/to/drug_perturbation_data \
  --split_modes pair \
  --top_k_otus 1024 \
  --output_dir outputs_downstreams/drug_perturbation/mlp_pair_top1024 \
  --device cuda:0
```

## MGM2 cached-backbone workflow

For a frozen MGM2 backbone, precompute token states once per scale and train the
cross-attention perturbation head from those caches:

```bash
DEVICE="0 1" \
bash scripts/downstreams/drug_perturbation/train_mgm2_cached_scales.sh
```

Set `STAGE2_ROOT`, `STAGE2_SCALES`, data paths, training parameters, and output
roots near the top of the launcher for the local installation. `DEVICE` accepts
a single GPU index, a space-separated GPU list for distributed head training,
or `cpu`.

The launcher resolves each pretrained scale as:

```text
${STAGE2_ROOT}/${SCALE}/config.json
${STAGE2_ROOT}/${SCALE}/checkpoints/last.ckpt
```

Backbone caches are written below
`data/downstreams/drug_perturbation/cache/`; trained heads and fold-level
predictions are written below
`outputs_downstreams/drug_perturbation/mgm2_cached_scales/`. Set
`FORCE_REBUILD_CACHE=1` in the launcher when the source checkpoint, dataset, or
cache format changes.

The two stages can also be run directly:

```bash
python scripts/downstreams/drug_perturbation/extract_mgm2_backbone_cache.py \
  --preprocessed_data data/downstreams/drug_perturbation/drug_perturbation_top1024.pkl \
  --config /path/to/stage2/scale_small/config.json \
  --pretrained_checkpoint /path/to/stage2/scale_small/checkpoints/last.ckpt \
  --output data/downstreams/drug_perturbation/cache/scale_small.pt \
  --device cuda:0

python scripts/downstreams/drug_perturbation/train_mgm2_cached_head.py \
  --cache data/downstreams/drug_perturbation/cache/scale_small.pt \
  --output_dir outputs_downstreams/drug_perturbation/mgm2_cached_scales/scale_small/pair/seed_42 \
  --split_mode pair \
  --max_folds 0 \
  --gpu 0
```

Use `--max_folds` with `--fold_selection random` for compact development runs
before launching all leave-one-group folds.

## End-to-end training and outputs

`train_mgm2_scales.sh` provides the end-to-end frozen-backbone alternative when
caching is not required. Both training paths export fold summaries, raw-scale
relative-abundance predictions, and evaluation metrics. Aggregate scale runs
with `summarize_mgm2_scale_results.py`; use `--help` to select the run directory,
summary file, and validation monitor.
