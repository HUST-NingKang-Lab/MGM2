# Development and release checks

## Repository layout

```text
mgm2/
  models/          Encoder, reconstruction heads and checkpoint loading
  data/            Embedding banks, taxonomy, preprocessing and table validation
  inference/       Reusable table inference and archive export
  interpretability/ Sparse autoencoder and checkpoint loading
  training/        Masking, reconstruction and alignment objectives
  cli.py           Command-line argument handling
  __init__.py      Stable public imports
tests/         CPU unit and interface tests
notebooks/     Executable end-to-end tutorial
examples/      Command-line Python examples
configs/       Model architecture and pretraining configuration
benchmarks/    Aggregate article results
docs/          Installation, model contract and migration guide
.github/       Automated package checks
pyproject.toml Package metadata, dependencies and tool configuration
```

The package lives directly in the repository's `mgm2/` directory. Package
discovery includes only `mgm2` and its subpackages. CI installs the built wheel
and runs tests outside the checkout to verify the installed artifact.
The scientific backbone and preprocessing retain the verified checkpoint
implementation. Existing imports from `mgm2.model`, `mgm2.inputs`, `mgm2.sae`
and the original underscore-prefixed modules are compatibility re-exports;
each implementation has a single source in its domain package.

`models` and `data` provide independent building blocks. `inference` combines
them, and `cli` delegates to that same pipeline. `training` consumes the shared
model defaults without being imported during ordinary model inference. Optional
table, Hub and notebook dependencies are loaded only by the features using them.

The public entry points remain `from mgm2 import MGM2, EmbeddingBank,
prepare_sample, collate_samples, load_sae`. The additional
`mgm2.inference` interface supports model reuse across tables; see the
[inference guide](INFERENCE.md).

## Checks

Install development dependencies as described in [Installation](INSTALLATION.md).
Unit tests use small local fixtures. Release-level checks additionally execute
the tutorial with the matching model bundle and compare exported model outputs
against the training checkpoints.

```bash
python -m pytest
python -m ruff check mgm2 tests examples
python -m ruff format --check mgm2 tests examples
python -m build
python -m twine check dist/*
```

CI builds a source distribution, rebuilds its wheel, installs the wheel, and
runs tests outside the checkout. This checks both the repository and the files
users receive. Model weights, embedding banks, article figures and benchmark
tables remain separate from the runtime wheel. Tests cover legacy imports,
optional-dependency isolation, preprocessing invariants, objective gradients,
CLI/API agreement, batch-size invariance, malformed tables, output provenance
and overwrite protection. Table tests use a locally generated tiny checkpoint
and require the `tables` extra; no download is needed.

## Tutorial execution

```bash
export MODEL_ROOT=/path/to/model-bundle
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=mgm2 \
  --output /tmp/mgm2-getting-started.executed.ipynb \
  notebooks/01_getting_started.ipynb
```

Commit the clean tutorial with empty outputs. Keep executed copies and exported
arrays in the chosen output directory. The demonstration communities have
constructed abundances over real vocabulary entries; they illustrate the API.

## Publication sequence

1. Confirm the release version, package name and code license.
2. Build the wheel and source distribution; run unit, tutorial and model checks.
3. Review the changes, commit message and release notes, then publish the approved
   source revision and build artifacts to GitHub.
4. Publish matching model resources to Hugging Face when they change and record
   the model revision independently from the code revision.

Distribution currently uses GitHub and Hugging Face. Package-check CI validates
artifacts without publishing to PyPI. The runtime package and model bundle have
independent release artifacts, joined by the model configuration format and
checkpoint identifiers. Documentation and internal refactors leave the model
release tag unchanged.

Reference: [Python packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
