#!/usr/bin/env python3
"""Plot kingdom-level activation bars for selected OTU SAE features."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mgm2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mgm2-cache")

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)

KINGDOM_COLORS = {
    "Archaea": "#B07AA1",
    "Bacteria": "#4E79A7",
    "Eukaryota": "#59A14F",
    "Eukaryota[chloroplast]": "#9CD66B",
    "Eukaryota[mitochondrion]": "#86BCB6",
    "unknown": "#8A8A8A",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--feature-id", type=int, default=2664)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metric", choices=["sum", "mean"], default="sum")
    return parser.parse_args()


def _parse_kingdom(taxonomy: object) -> str:
    tax = "" if pd.isna(taxonomy) else str(taxonomy)
    if not tax:
        return "unknown"
    first = tax.split(";")[0].strip()
    match = re.match(r"^[dk]__?(.*)$", first)
    if match:
        value = match.group(1).strip()
        return value or "unknown"
    return first or "unknown"


def _load_short_name(root: Path, feature_id: int) -> str:
    path = root / "codex_outputs" / f"feature_{feature_id:06d}" / "unified_description.json"
    if not path.exists():
        return ""
    import json

    return json.loads(path.read_text()).get("short_name", "")


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    output_dir = args.output_dir or root / "reports" / "selected_feature_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = root / "feature_lookup" / f"feature_{args.feature_id:06d}.events.csv.gz"
    events = pd.read_csv(events_path)
    events["activation"] = pd.to_numeric(events["activation"], errors="coerce").fillna(0.0)
    events["kingdom"] = events["taxonomy"].map(_parse_kingdom)

    grouped = (
        events.groupby("kingdom", dropna=False)
        .agg(count=("activation", "size"), activation_sum=("activation", "sum"), activation_mean=("activation", "mean"))
        .reset_index()
    )
    for kingdom in ["Archaea", "Bacteria"]:
        if kingdom not in set(grouped["kingdom"]):
            grouped = pd.concat(
                [
                    grouped,
                    pd.DataFrame([{"kingdom": kingdom, "count": 0, "activation_sum": 0.0, "activation_mean": 0.0}]),
                ],
                ignore_index=True,
            )
    metric_col = "activation_sum" if args.metric == "sum" else "activation_mean"
    grouped = grouped.sort_values(metric_col, ascending=False)

    labels = grouped["kingdom"].astype(str).tolist()
    values = grouped[metric_col].astype(float).tolist()
    counts = grouped["count"].astype(int).tolist()
    colors = [KINGDOM_COLORS.get(label, "#BAB0AC") for label in labels]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, alpha=0.88)
    ax.set_ylabel("Activation sum among lookup top events" if args.metric == "sum" else "Mean activation")
    short_name = _load_short_name(root, args.feature_id)
    ax.set_title(f"Feature {args.feature_id}: kingdom-level activation\n{short_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    max_value = max(values) if values else 0.0
    for bar, value, count in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(max_value * 0.02, 1e-6),
            f"{value:.2f}\nn={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    stem = output_dir / f"feature_{args.feature_id:06d}_kingdom_activation_{args.metric}"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print({"png": str(png_path), "pdf": str(pdf_path)})


if __name__ == "__main__":
    main()
