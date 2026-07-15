#!/usr/bin/env python3
"""Plot MGnify model parameter count versus average ROC-AUC.

RF is intentionally excluded because tree ensembles do not have a directly
comparable neural parameter count. MGM2 is plotted with Qwen3 alignment disabled,
and the table additionally reports the Qwen3-enabled parameter count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

LEVELS = ["env_level_1", "env_level_2", "env_level_3", "env_level_4", "env_level_5"]
MGM2_SCALES = ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]
MGM2_LABELS = {
    "scale_small": "MGM2-small",
    "scale_medium": "MGM2-medium",
    "scale_large": "MGM2-large",
    "scale_xlarge": "MGM2-xlarge",
}
MODEL_ORDER = ["Expert", "DeepPhylo", "MGM1", "MGM2-small", "MGM2-medium", "MGM2-large", "MGM2-xlarge"]
MODEL_COLORS = {
    "Expert": "#4E7B48",
    "DeepPhylo": "#6D2A1F",
    "MGM1": "#7CB5EC",
    "MGM2-small": "#4363D8",
    "MGM2-medium": "#1A237E",
    "MGM2-large": "#00897B",
    "MGM2-xlarge": "#6A1B9A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MGnify parameter count vs average ROC-AUC table and plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics_json", type=Path, default=Path("figures/mgnify_comparison/all_metrics.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/mgnify_params"))
    parser.add_argument("--stage2_root", type=Path, default=Path("outputs_grid/stage2/20260505_094404"))
    parser.add_argument("--mgm1_root", type=Path, default=Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/MGM/models"))
    parser.add_argument("--expert_model_dir", type=Path, default=Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/expert/models/seed0_model/base"))
    parser.add_argument("--expert_python", type=Path, default=Path("/data4/zhanghaohong/anaconda3/envs/expert/bin/python"))
    parser.add_argument("--deepphylo_embedding", type=Path, default=Path("/home/zhanghaohong/data6/projects/MGM2Benchmarks/MGnify_mst/deepphylo/phy_embedding_raw_hs80.npy"))
    parser.add_argument("--deepphylo_hidden_size", type=int, default=80)
    parser.add_argument("--deepphylo_kernel_size_conv", type=int, default=7)
    return parser.parse_args()


def tensor_numel(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel())
    if isinstance(obj, dict):
        return sum(tensor_numel(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_numel(value) for value in obj)
    return 0


def state_dict_param_count(path: Path) -> int:
    obj = torch.load(path, map_location="cpu")
    return tensor_numel(obj)


def count_mgm1_params(mgm1_root: Path) -> Dict[str, Any]:
    counts = []
    for level_idx in range(1, 6):
        path = mgm1_root / f"finetuned_model_layer{level_idx}_seed11" / "pytorch_model.bin"
        if not path.exists():
            continue
        counts.append(state_dict_param_count(path))
    if not counts:
        path = mgm1_root / "pretrain_scratch" / "pytorch_model.bin"
        counts.append(state_dict_param_count(path))
    return {
        "params_for_plot": float(np.mean(counts)),
        "params_without_qwen3": np.nan,
        "params_with_qwen3": np.nan,
        "params_min": int(np.min(counts)),
        "params_max": int(np.max(counts)),
        "source": "mean full finetuned L1-L5 seed11 checkpoints" if len(counts) > 1 else "full pretrain_scratch checkpoint",
    }


def count_deepphylo_params(embedding_path: Path, hidden_size: int, kernel_size_conv: int) -> Dict[str, Any]:
    phy = np.load(embedding_path, mmap_mode="r")
    input_size = int(phy.shape[0])
    embedding_dim = int(phy.shape[1])
    output_sizes = [4, 35, 65, 97, 78]
    counts = []
    for output_size in output_sizes:
        embedding = (input_size + 1) * embedding_dim
        conv = hidden_size * hidden_size * kernel_size_conv + hidden_size
        batch_norm = 2 * hidden_size
        fc_abundance = input_size * hidden_size + hidden_size
        fc_phy = embedding_dim * hidden_size + hidden_size
        fc_pred = hidden_size * output_size + output_size
        counts.append(embedding + conv + batch_norm + fc_abundance + fc_phy + fc_pred)
    return {
        "params_for_plot": float(np.mean(counts)),
        "params_without_qwen3": np.nan,
        "params_with_qwen3": np.nan,
        "params_min": int(np.min(counts)),
        "params_max": int(np.max(counts)),
        "source": "mean full DeepPhylo L1-L5 model formula including frozen phy embedding",
    }


def count_expert_params(expert_python: Path, expert_model_dir: Path) -> Dict[str, Any]:
    code = f"""
import tensorflow as tf
path = {str(expert_model_dir)!r}
model = tf.saved_model.load(path)
variables = list(model.variables) if hasattr(model, 'variables') else []
print(sum(int(tf.size(v).numpy()) for v in variables))
"""
    result = subprocess.run(
        [str(expert_python), "-c", code],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    count = int(result.stdout.strip().splitlines()[-1])
    return {
        "params_for_plot": float(count),
        "params_without_qwen3": np.nan,
        "params_with_qwen3": np.nan,
        "params_min": count,
        "params_max": count,
        "source": "TensorFlow SavedModel variables, seed0_model/base",
    }


def count_mgm2_params(stage2_root: Path) -> Dict[str, Dict[str, Any]]:
    from src.configs.Config import MGM2Config
    from src.models import MGM2Model

    rows: Dict[str, Dict[str, Any]] = {}
    for scale in MGM2_SCALES:
        config = MGM2Config.load(str(stage2_root / scale / "config.json"))
        kwargs = dict(
            config=config.model.__dict__,
            num_bins=config.abundance.num_bins,
            enable_regression=config.abundance.enable_regression,
            qwen3_projection_dim=config.qwen3_alignment.projection_dim,
            mlp_dropout=config.model.mlp_dropout,
        )
        model_without = MGM2Model(**kwargs, enable_qwen3_alignment=False)
        model_with = MGM2Model(**kwargs, enable_qwen3_alignment=True)
        without_count = sum(parameter.numel() for parameter in model_without.parameters())
        with_count = sum(parameter.numel() for parameter in model_with.parameters())
        rows[MGM2_LABELS[scale]] = {
            "params_for_plot": float(without_count),
            "params_without_qwen3": int(without_count),
            "params_with_qwen3": int(with_count),
            "params_min": int(without_count),
            "params_max": int(without_count),
            "source": f"{scale} config; plot uses Qwen3-disabled MGM2Model",
        }
    return rows


def average_auc(metrics: Dict[str, Any], model_name: str) -> Dict[str, float]:
    values = []
    for level in LEVELS:
        row = metrics.get(model_name, {}).get(level)
        if row and "roc_mean" in row:
            values.append(float(row["roc_mean"]))
    return {
        "avg_auc_l1_l5": float(np.mean(values)) if values else np.nan,
        "auc_levels_used": len(values),
    }


def build_table(args: argparse.Namespace) -> pd.DataFrame:
    with args.metrics_json.open() as handle:
        metrics = json.load(handle)

    param_rows: Dict[str, Dict[str, Any]] = {
        "Expert": count_expert_params(args.expert_python, args.expert_model_dir),
        "DeepPhylo": count_deepphylo_params(args.deepphylo_embedding, args.deepphylo_hidden_size, args.deepphylo_kernel_size_conv),
        "MGM1": count_mgm1_params(args.mgm1_root),
    }
    param_rows.update(count_mgm2_params(args.stage2_root))

    rows = []
    for model in MODEL_ORDER:
        if model not in param_rows:
            continue
        row = {"model": model}
        row.update(param_rows[model])
        row.update(average_auc(metrics, model))
        row["params_for_plot_million"] = row["params_for_plot"] / 1e6
        if not pd.isna(row["params_without_qwen3"]):
            row["params_without_qwen3_million"] = row["params_without_qwen3"] / 1e6
            row["params_with_qwen3_million"] = row["params_with_qwen3"] / 1e6
        else:
            row["params_without_qwen3_million"] = np.nan
            row["params_with_qwen3_million"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_params_vs_auc(table: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for _, row in table.iterrows():
        model = row["model"]
        ax.scatter(
            row["params_for_plot"],
            row["avg_auc_l1_l5"],
            s=95,
            color=MODEL_COLORS.get(model, "#777777"),
            edgecolor="#222222",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            model,
            (row["params_for_plot"], row["avg_auc_l1_l5"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Full parameter count (log scale)")
    ax.set_ylabel("Average ROC-AUC across L1-L5")
    ax.set_title("MGnify performance versus model size")
    ax.grid(axis="both", color="#dddddd", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, output_dir, "mgnify_params_vs_avg_auc")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = build_table(args)
    table.to_csv(args.output_dir / "mgnify_model_parameter_table.csv", index=False)
    plot_params_vs_auc(table, args.output_dir)
    printable = table[
        [
            "model",
            "params_for_plot_million",
            "params_without_qwen3_million",
            "params_with_qwen3_million",
            "avg_auc_l1_l5",
            "auc_levels_used",
            "source",
        ]
    ]
    print(printable.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Done. {args.output_dir}")


if __name__ == "__main__":
    main()
