"""Feature-by-sample table validation; pandas is an optional dependency."""

import csv
from pathlib import Path

import numpy as np


def validate_table(table):
    """Validate identifiers and abundances before starting model inference."""
    if table.empty:
        raise ValueError("Expected a nonempty feature-by-sample table")
    for label, identifiers in (("feature", table.index), ("sample", table.columns)):
        names = [str(value) for value in identifiers]
        if identifiers.isna().any() or any(not name.strip() for name in names):
            raise ValueError(f"Table {label} identifiers must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError(f"Table {label} identifiers must be unique")
    # Check one column at a time, avoiding a second full dense matrix.
    for name in table.columns:
        try:
            values = table[name].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sample {name!r} contains nonnumeric abundances") from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Sample {name!r} abundances must be finite and non-negative")
        total = values.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Sample {name!r} must have a positive finite total abundance")


def read_abundance_table(path):
    """Read CSV without letting pandas silently rename duplicate sample IDs."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("CSV support requires pandas: install 'mgm2-community[tables]'") from exc
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if (
        len(header) < 2
        or any(not name.strip() for name in header[1:])
        or len(set(header[1:])) != len(header) - 1
    ):
        raise ValueError("CSV sample identifiers must be nonempty and unique")
    # Preserve lexical feature identifiers such as 001 and NA.
    table = pd.read_csv(
        path,
        header=0,
        names=list(range(len(header))),
        index_col=0,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    table.index.name = header[0]
    table.columns = header[1:]
    validate_table(table)
    return table
