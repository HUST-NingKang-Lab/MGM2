# Downstream Tasks

This directory contains reproducible downstream evaluations built on the MGM2
community encoder. Each task keeps dataset construction, model training, and
evaluation scripts close to the task-specific data contract.

## Available workflows

| Task | Input | Prediction target | Documentation |
| --- | --- | --- | --- |
| FMT | Donor and pre-FMT recipient communities | Post-FMT community or clinical response | [FMT](FMT/README.md) |
| Cell Reports FMT | Donor, recipient, and post-FMT multi-kingdom profiles | Post-FMT community, response, or donor-similarity direction | [FMT Cell Reports](FMT_cellreports/README.md) |
| MGnify MST | Microbial community profiles | Hierarchical environment labels | [MGnify MST](MGnify_mst/README.md) |
| Drug perturbation | Control community and drug representation | Post-perturbation community | [Drug perturbation](drug_perturbation/README.md) |

Additional wastewater forecasting workflows are documented in their own
directories.

## Common conventions

Run commands from the repository root and activate the project environment:

```bash
conda activate mwm
```

Prepared datasets are written below `data/downstreams/`, run artifacts below
`outputs_downstreams/`, and publication figures below `figures/`. Raw datasets,
embedding files, pretrained checkpoints, and generated outputs are external
artifacts and are not expected to be stored in Git.

MGM2 scale launchers use `scale_small`, `scale_medium`, `scale_large`, and
`scale_xlarge`. Unless an explicit path is supplied, scale-specific model
configuration and pretrained weights are resolved as:

```text
${STAGE2_ROOT}/${SCALE}/config.json
${STAGE2_ROOT}/${SCALE}/checkpoints/last.ckpt
```

The stage-2 `last.ckpt` initializes the downstream backbone. Downstream scripts
that expose `EVAL_CHECKPOINT` use their validation-selected `best` checkpoint by
default for final prediction. Dataset locations, output roots, selected scales,
seeds, folds, and devices can be set through the launcher variables documented
in each task README.

## Reproducibility

Build each task dataset before training, preserve the generated metadata and
split files, and run all compared methods on the same split definitions. The
Python entry points expose additional options through `--help`:

```bash
python path/to/script.py --help
```
