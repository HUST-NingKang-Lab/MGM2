#!/usr/bin/env python3
"""Plot Stage2 model scaling results (small / medium / large / xlarge)."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "figures/stage2_scale"
os.makedirs(OUTPUT_DIR, exist_ok=True)

models = ["Small", "Medium", "Large", "XLarge"]

arch = {
    "Small":  dict(h=224, L=6, heads=4,  ff=896),
    "Medium": dict(h=320, L=6, heads=5,  ff=1280),
    "Large":  dict(h=384, L=8, heads=6,  ff=1536),
    "XLarge": dict(h=512, L=8, heads=8,  ff=2048),
}
params    = {"Small": 5.2, "Medium": 9.7, "Large": 17.1, "XLarge": 29.4}
batch_szs = {"Small": 1024, "Medium": 512, "Large": 512, "XLarge": 512}
param_labels = {k: f"{v:.1f}M" for k, v in params.items()}

metric_files = {
    "Small":  "outputs_grid/stage2/20260505_094404/scale_small/logs/run_005_lr5e-5_wd0.01_mask0.15_reg0.3_qw0.1/version_1/metrics.csv",
    "Medium": "outputs_grid/stage2/20260505_094404/scale_medium/logs/stage2_scale_scale_medium/version_1/metrics.csv",
    "Large":  "outputs_grid/stage2/20260505_094404/scale_large/logs/stage2_scale_scale_large/version_1/metrics.csv",
    "XLarge": "outputs_grid/stage2/20260505_094404/scale_xlarge/logs/stage2_scale_scale_xlarge/version_1/metrics.csv",
}

# Best val metrics
val_total_loss = [0.8365, 0.6530, 0.6147, 0.4105]
val_bin_acc    = [0.4633, 0.5030, 0.5032, 0.5221]
val_reg_mse    = [0.2722, 0.2567, 0.2492, 0.2323]
val_reg_corr   = [0.6697, 0.6928, 0.7014, 0.7349]
val_qwen_loss  = [0.6037, 0.4385, 0.4016, 0.2116]

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
colors = ["#7CB5EC", "#4363D8", "#1A237E", "#4A148C"]

x = np.arange(len(models))

# ===================================================================
# Figure 1: Bar charts — best metrics
# ===================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes = axes.flatten()

metric_info = [
    (val_total_loss, "Val Total Loss",       "\u2193"),
    (val_bin_acc,    "Val Bin Accuracy",     "\u2191"),
    (val_reg_mse,    "Val Regression MSE",   "\u2193"),
    (val_reg_corr,   "Val Regression Corr",  "\u2191"),
    (val_qwen_loss,  "Val Qwen3 Align Loss", "\u2193"),
]

for ax, (vals, ylabel, arrow) in zip(axes[:5], metric_info):
    ax.bar(x, vals, color=colors, edgecolor=["#333"] * 4, linewidth=1.2, width=0.55)

    for i, v in enumerate(vals):
        offset = max(vals) * 0.03
        va = "bottom" if v < max(vals) * 0.75 else "top"
        y_pos = v + offset if va == "bottom" else v - offset
        ax.text(i, y_pos, f"{v:.4f}", ha="center", va=va, fontsize=9,
                fontweight="bold", color="#222")

    for i, m in enumerate(models):
        arc = f"h={arch[m]['h']} L={arch[m]['L']} d_ff={arch[m]['ff']}"
        ax.text(i, -max(vals) * 0.04, f"{param_labels[m]}",
                ha="center", va="top", fontsize=8, color="#555", fontweight="bold")
        ax.text(i, -max(vals) * 0.08, f"({arc})",
                ha="center", va="top", fontsize=6.5, color="#888")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{ylabel} {arrow}", fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymin = min(vals) * 0.7 if min(vals) > 0 else 0
    ax.set_ylim(bottom=ymin if ymin < 1 else 0)

axes[5].set_visible(False)

fig.suptitle("MGM2 Stage 2 — Model Scaling (NTv3 + Qwen3, 50 epochs, best val)",
             fontsize=16, fontweight="bold", y=1.01)
plt.subplots_adjust(hspace=0.35, wspace=0.25)
for fmt in ["png", "pdf"]:
    fig.savefig(os.path.join(OUTPUT_DIR, f"stage2_scale_bars.{fmt}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {OUTPUT_DIR}/stage2_scale_bars.{{png,pdf}}")

# ===================================================================
# Figure 2: Training curves — all metrics vs samples_seen (batch-size aligned)
# ===================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(22, 13))
axes2 = axes2.flatten()

curve_metrics = [
    ("val/total_loss",             "Val Total Loss",          "\u2193"),
    ("val/bin_loss",               "Val Bin Loss",            "\u2193"),
    ("val/bin_accuracy",           "Val Bin Accuracy",        "\u2191"),
    ("val/regression_mse",         "Val Regression MSE",      "\u2193"),
    ("val/regression_corr",        "Val Regression Corr",     "\u2191"),
    ("val/qwen3_alignment_loss",   "Val Qwen3 Align Loss",    "\u2193"),
]

for ax, (col, ylabel, arrow) in zip(axes2, curve_metrics):
    for m, c in zip(models, colors):
        df = pd.read_csv(metric_files[m])
        curve = df.dropna(subset=[col])[["step", col, "epoch"]].copy()
        bs = batch_szs[m]
        curve["samples"] = (curve["step"] * bs).astype(int)

        # Smooth with sliding window
        curve = curve.sort_values("step")
        window = max(1, len(curve) // 10)
        curve["smoothed"] = curve[col].rolling(window, min_periods=2, center=True).mean()

        ax.plot(curve["samples"], curve["smoothed"], color=c, linewidth=2, alpha=0.9, label=m)
        # Epoch markers (circles at epoch boundaries)
        epoch_ends = curve[["samples", col, "epoch"]].dropna()
        ax.scatter(epoch_ends["samples"], epoch_ends[col],
                   color=c, s=15, alpha=0.4, edgecolors="none", zorder=5)

    ax.set_xlabel("Samples seen (step \u00d7 batch_size)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{ylabel} {arrow}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, title="Scale", title_fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

fig2.suptitle("MGM2 Stage 2 — Val Metric Curves (batch-size aligned, smoothed)",
              fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
for fmt in ["png", "pdf"]:
    fig2.savefig(os.path.join(OUTPUT_DIR, f"stage2_scale_curves.{fmt}"),
                 dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {OUTPUT_DIR}/stage2_scale_curves.{{png,pdf}}")
print("Done.")
