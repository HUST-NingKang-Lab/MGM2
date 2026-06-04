#!/usr/bin/env python3
"""Summarize no-residual global wastewater forecasting experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["bray_curtis", "mae", "rmse", "spearman", "top50_overlap"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs_downstreams/wastewater_global_forecast"))
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def _weighted(frame: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or []
    rows = []
    groups = [((), frame)] if not keys else frame.groupby(keys, dropna=False)
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(keys, key)}
        weights = pd.to_numeric(group["n_windows"], errors="coerce").fillna(0.0)
        row["n_windows"] = int(weights.sum())
        if "plant" in group.columns and "plant" not in keys:
            row["n_plants"] = int(group["plant"].nunique())
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            mask = values.notna() & (weights > 0)
            row[metric] = float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _load_summary(path: Path, model_group: str, source: str, mode: str = "", scale: str = "") -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["model_group"] = model_group
    frame["source"] = source
    frame["mode"] = mode
    frame["scale"] = scale
    frame["run_dir"] = str(path.parent)
    return frame


def main():
    args = parse_args()
    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    simple = args.root / "simple_baselines_top200" / "summary.csv"
    if simple.exists():
        base = pd.read_csv(simple)
        for model in sorted(base["model"].unique()):
            sub = base[base["model"] == model].copy()
            sub["model_group"] = f"baseline_{model}"
            sub["source"] = "simple_baseline"
            sub["mode"] = model
            sub["scale"] = ""
            sub["run_dir"] = str(simple.parent)
            frames.append(sub)

    abundance = args.root / "abundance_transformer_top200" / "summary.csv"
    if abundance.exists():
        frames.append(_load_summary(abundance, "global_abundance_transformer", "abundance_only", "transformer", "global"))

    mgm_root = args.root / "mgm2_no_residual_ntv3_asv"
    for summary_path in sorted(mgm_root.glob("*_scale_*_top200/summary.csv")):
        run_name = summary_path.parent.name
        mode = run_name.split("_scale_")[0]
        scale = run_name.split("_scale_")[1].replace("_top200", "")
        frames.append(_load_summary(summary_path, f"mgm2_no_residual_{mode}_{scale}", "mgm2_ntv3_asv", mode, scale))

    if not frames:
        raise FileNotFoundError(f"no summary.csv files found under {args.root}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "combined_summary.csv", index=False)
    weighted = _weighted(combined, ["model_group", "source", "mode", "scale"])
    weighted.sort_values("bray_curtis").to_csv(output_dir / "weighted_summary.csv", index=False)

    print(f"Saved combined summary: {output_dir / 'combined_summary.csv'}")
    print(f"Saved weighted summary: {output_dir / 'weighted_summary.csv'}")
    print(weighted.sort_values("bray_curtis").to_string(index=False))


if __name__ == "__main__":
    main()
