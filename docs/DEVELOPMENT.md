# Development and release checks

## Repository layout

```text
src/mgm2/       Model loading, preprocessing, SAE and CLI
tests/         CPU unit and interface tests
notebooks/     Executable end-to-end tutorial
examples/      Command-line Python examples
configs/       Model architecture and pretraining configuration
benchmarks/    Aggregate article results
docs/          Installation, model contract and migration guide
.github/       Automated package checks
pyproject.toml Package metadata, dependencies and tool configuration
```

The `src/` layout makes installation explicit. Tests import the installed
package; they do not add the repository root to the module search path.
The scientific backbone and preprocessing retain the verified checkpoint
implementation. Modules beginning with `_` are internal implementations.

## Checks

Install development dependencies as described in [Installation](INSTALLATION.md).
Unit tests use small local fixtures. Release-level checks additionally execute
the tutorial with the matching model bundle and compare exported model outputs
against the training checkpoints.

```bash
python -m pytest
python -m ruff check src tests examples
python -m build
python -m twine check dist/*
```

CI builds a source distribution, rebuilds its wheel, installs the wheel, and
runs tests outside the checkout. This checks both the repository and the files
users receive. Model weights, embedding banks, article figures and benchmark
tables remain separate from the runtime wheel.

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
3. Publish the candidate to TestPyPI and install that exact artifact in a fresh environment.
4. Publish the approved artifacts to PyPI; publish the matching model revision
   to Hugging Face and record its revision identifier.

Use a protected GitHub release environment and PyPI Trusted Publishing for
publication. CI in this repository performs checks only. The runtime package
and model bundle have independent sizes and release artifacts, joined by the
model configuration format and checkpoint identifiers.

References: [Python packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/),
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
