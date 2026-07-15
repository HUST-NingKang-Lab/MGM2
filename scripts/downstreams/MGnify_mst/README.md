# MGnify MST Environment Classification

The MGnify MST workflow evaluates hierarchical environment classification from
microbial community profiles. A separate classifier is trained for each of five
environment hierarchy levels (`env_level_1` through `env_level_5`).

## Inputs

The benchmark uses train and test community matrices together with metadata
tables containing `SampleID` and the five hierarchy labels. The dataset builder
accepts HDF5 community matrices, resolves OTU embeddings through NTv3 and the
taxonomy fallback store, and writes compact MGM2 pickle files plus resolved
embedding HDF5 files.

Before running the launcher, configure its input directory, output directory,
NTv3 paths, HDF5 key, and `TOP_K_OTUS` for the local installation. Then run:

```bash
bash scripts/downstreams/MGnify_mst/build_mgnify_mst_datasets.sh
```

Expected prepared files are:

```text
data/downstreams/MGnify_mst/train.pkl
data/downstreams/MGnify_mst/test.pkl
data/downstreams/MGnify_mst/train.resolved_embeddings.h5
data/downstreams/MGnify_mst/test.resolved_embeddings.h5
data/downstreams/MGnify_mst/train_meta_split.csv
data/downstreams/MGnify_mst/test_meta_split.csv
```

Use `split_mgnify_mst_metadata.py` to construct aligned metadata partitions when
starting from an unsplit metadata table; inspect its command-line options with
`--help` for source and output paths.

## Frozen-backbone training

Set the data, output, stage-2, scale, seed, hierarchy, and device variables near
the top of `train_mgnify_mst_all_levels.sh` for the target installation, then
launch all configured runs:

```bash
bash scripts/downstreams/MGnify_mst/train_mgnify_mst_all_levels.sh
```

The default experiment matrix covers four MGM2 scales, five hierarchy levels,
and five random seeds. Each run freezes the MGM2 backbone, uses CLS pooling, and
loads:

```text
${STAGE2_ROOT}/${SCALE}/config.json
${STAGE2_ROOT}/${SCALE}/checkpoints/last.ckpt
```

Outputs are organized as
`outputs_downstreams/MGnify_mst/<scale>/<level>/seed_<seed>/` and contain model
checkpoints, logs, label mappings, metrics, and `test_predictions.csv`.

## Representations and figures

Extract frozen MGM2 CLS representations and generate scale-wise embedding and
biological-conservation panels:

```bash
python scripts/downstreams/MGnify_mst/plot_mgnify_embedding_umap.py \
  --data_dir data/downstreams/MGnify_mst \
  --mgnify_root outputs_downstreams/MGnify_mst \
  --stage2_root /path/to/stage2 \
  --scales scale_small scale_medium scale_large scale_xlarge \
  --gpu 0 \
  --output_dir figures/mgnify_embedding
```

Extended MGM2 scale, hierarchy, class-frequency, confusion, calibration, and
representation panels are available through:

```bash
python scripts/downstreams/MGnify_mst/plot_mgnify_extended_figures.py \
  --mgnify_root outputs_downstreams/MGnify_mst \
  --stage2_root /path/to/stage2 \
  --output_dir figures/mgnify_extended
```
