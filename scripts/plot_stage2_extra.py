#!/usr/bin/env python3
"""Extra Stage2 plots: scaling law, loss decomposition, heatmap, efficiency."""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

OUTPUT_DIR = "figures/stage2_scale"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
models       = ["Small", "Medium", "Large", "XLarge"]
params_M     = [5.2, 9.7, 17.1, 29.4]
colors       = ["#7CB5EC", "#4363D8", "#1A237E", "#4A148C"]

# Weighted losses: L = 1.0*bin + 0.3*reg + 0.1*qwen
arch = {
    "Small":  "h=224 L=6 d_ff=896",
    "Medium": "h=320 L=6 d_ff=1280",
    "Large":  "h=384 L=8 d_ff=1536",
    "XLarge": "h=512 L=8 d_ff=2048",
}

# [val_total, val_bin_loss, 0.3*val_reg_loss, 0.1*val_qwen_loss]  (weighted contribs)
losses = {
    "Small":  [0.8365, 0.1512, 0.3*0.0816, 0.1*0.6037],
    "Medium": [0.6530, 0.1375, 0.3*0.0770, 0.1*0.4385],
    "Large":  [0.6147, 0.1383, 0.3*0.0748, 0.1*0.4016],
    "XLarge": [0.4105, 0.1292, 0.3*0.0697, 0.1*0.2116],
}

# Best metrics per model
metrics = {
    "Small":  {"Total Loss":0.8365, "Bin Acc":0.4633, "Bin Loss":0.1512,
               "Reg MSE":0.2722, "Reg Corr":0.6697, "Qwen Loss":0.6037},
    "Medium": {"Total Loss":0.6530, "Bin Acc":0.5030, "Bin Loss":0.1375,
               "Reg MSE":0.2567, "Reg Corr":0.6928, "Qwen Loss":0.4385},
    "Large":  {"Total Loss":0.6147, "Bin Acc":0.5032, "Bin Loss":0.1383,
               "Reg MSE":0.2492, "Reg Corr":0.7014, "Qwen Loss":0.4016},
    "XLarge": {"Total Loss":0.4105, "Bin Acc":0.5221, "Bin Loss":0.1292,
               "Reg MSE":0.2323, "Reg Corr":0.7349, "Qwen Loss":0.2116},
}

# Training curves
batch_szs = {"Small": 1024, "Medium": 512, "Large": 512, "XLarge": 512}
metric_files = {
    "Small":  "outputs_grid/stage2/20260505_094404/scale_small/logs/run_005_lr5e-5_wd0.01_mask0.15_reg0.3_qw0.1/version_1/metrics.csv",
    "Medium": "outputs_grid/stage2/20260505_094404/scale_medium/logs/stage2_scale_scale_medium/version_1/metrics.csv",
    "Large":  "outputs_grid/stage2/20260505_094404/scale_large/logs/stage2_scale_scale_large/version_1/metrics.csv",
    "XLarge": "outputs_grid/stage2/20260505_094404/scale_xlarge/logs/stage2_scale_scale_xlarge/version_1/metrics.csv",
}

x = np.arange(len(models))

# ===================================================================
# Figure 3: Scaling Law (log-log)
# ===================================================================
fig, ax = plt.subplots(figsize=(7, 6))

params_arr = np.array(params_M)
total_losses = np.array([losses[m][0] for m in models])

# Power-law fit: loss = a * N^b
def power_law(N, a, b):
    return a * N ** b

log_p = np.log(params_arr)
log_l = np.log(total_losses)
b, log_a = np.polyfit(log_p, log_l, 1)
a = np.exp(log_a)

# Plot
ax.scatter(params_arr, total_losses, c=colors, s=120, edgecolors="#333", linewidth=1.5, zorder=5)
for m, px, py in zip(models, params_arr, total_losses):
    ax.annotate(m, (px, py), textcoords="offset points", xytext=(8, -4),
                fontsize=10, fontweight="bold", color="#333")

# Fitted curve
N_fit = np.linspace(params_arr[0] * 0.8, params_arr[-1] * 1.2, 100)
ax.plot(N_fit, power_law(N_fit, a, b), color="#D4782B", linewidth=2, linestyle="--",
        label=f"$L = {a:.2f} \\cdot N^{{{b:.3f}}}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Parameters (M)", fontsize=12)
ax.set_ylabel("Val Total Loss", fontsize=12)
ax.set_title("MGM2 Scaling Law\n(val total loss vs model size, log-log)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3)

fig.subplots_adjust(top=0.92, right=0.95)
for fmt in ["png", "pdf"]:
    fig.savefig(os.path.join(OUTPUT_DIR, f"stage2_scaling_law.{fmt}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved scaling_law")

# ===================================================================
# Figure 4: Loss Decomposition Stacked Bar
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 6))

bar_w = 0.5
bottom = np.zeros(len(models))

components = [
    ("Bin Loss      (\u03bb=1.0)",  [losses[m][1] for m in models], "#4E7B48"),
    ("Reg Loss      (\u03bb=0.3)",  [losses[m][2] for m in models], "#3D549D"),
    ("Qwen3 Align   (\u03bb=0.1)",  [losses[m][3] for m in models], "#D4782B"),
]

for label, vals, clr in components:
    bars = ax.bar(x, vals, bar_w, bottom=bottom, color=clr, edgecolor="white", linewidth=0.8, label=label)
    for i, v in enumerate(vals):
        if v > 0.01 * max(total_losses):
            ax.text(i, bottom[i] + v / 2, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
    bottom = bottom + vals

# Total loss label on top
for i, (m, tl) in enumerate(zip(models, total_losses)):
    ax.text(i, tl + max(total_losses) * 0.02, f"{tl:.4f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#222")

for i, m in enumerate(models):
    ax.text(i, -max(total_losses) * 0.03, f"{params_M[i]:.1f}M\n({arch[m]})",
            ha="center", va="top", fontsize=7.5, color="#555", linespacing=1.3)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight="bold")
ax.set_ylabel("Weighted Loss Contribution", fontsize=11)
ax.set_title("MGM2 Stage 2 — Loss Decomposition by Task\n(L = 1.0\u00d7Bin + 0.3\u00d7Reg + 0.1\u00d7Qwen)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(bottom=0)

fig.subplots_adjust(top=0.88, bottom=0.12)
for fmt in ["png", "pdf"]:
    fig.savefig(os.path.join(OUTPUT_DIR, f"stage2_loss_decomp.{fmt}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved loss_decomp")

# ===================================================================
# Figure 5: Relative Improvement Heatmap
# ===================================================================
metric_names = list(metrics["Small"].keys())
metric_dirs = {"Total Loss":"\u2193","Bin Loss":"\u2193","Qwen Loss":"\u2193",
               "Reg MSE":"\u2193","Bin Acc":"\u2191","Reg Corr":"\u2191"}

# % change from Small baseline
baseline = {k: metrics["Small"][k] for k in metric_names}
heat_data = np.zeros((len(metric_names), len(models)))
annot_data = np.empty((len(metric_names), len(models)), dtype=object)

for j, m in enumerate(models):
    for i, mn in enumerate(metric_names):
        val = metrics[m][mn]
        base = baseline[mn]
        # For metrics where lower is better, negate so green always = improvement
        direction = -1 if metric_dirs[mn] == "\u2193" else 1
        pct = (val - base) / base * 100 * direction
        heat_data[i, j] = pct
        annot_data[i, j] = f"{pct:+.1f}%\n({val:.4f})"

fig, ax = plt.subplots(figsize=(9, 7))

# Custom cmap: red (degradation) -> white -> green (improvement)
cmap = LinearSegmentedColormap.from_list("rg", ["#D32F2F", "#F5F5F5", "#2E7D32"])

im = ax.imshow(heat_data, cmap=cmap, aspect="auto", vmin=-60, vmax=60)

for i in range(len(metric_names)):
    for j in range(len(models)):
        val = heat_data[i, j]
        color = "white" if abs(val) > 30 else "#222"
        ax.text(j, i, annot_data[i, j], ha="center", va="center",
                fontsize=8, color=color, fontweight="bold", linespacing=1.1)

ax.set_xticks(range(len(models)))
ax.set_xticklabels([f"{m}\n({p}M)" for m, p in zip(models, params_M)], fontsize=10, fontweight="bold")
ax.set_yticks(range(len(metric_names)))
ax.set_yticklabels([f"{mn} {metric_dirs[mn]}" for mn in metric_names], fontsize=10)
ax.set_title("MGM2 Stage 2 — Relative Change vs Small Baseline\n(green = improvement, red = degradation)",
             fontsize=12, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("% Change (negated for loss metrics)", fontsize=10)

fig.subplots_adjust(left=0.22, right=0.92, top=0.88, bottom=0.08)
for fmt in ["png", "pdf"]:
    fig.savefig(os.path.join(OUTPUT_DIR, f"stage2_improvement_heatmap.{fmt}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved improvement_heatmap")

# ===================================================================
# Figure 6: Training Efficiency — samples to reach target loss
# ===================================================================
fig, ax = plt.subplots(figsize=(9, 6))

target_losses = [0.80, 0.70, 0.60, 0.50, 0.45]
marker_styles = ["o", "s", "D", "^", "v"]

efficiency_data = {m: [] for m in models}

for m, c, bs in zip(models, colors, [batch_szs[m] for m in models]):
    df = pd.read_csv(metric_files[m])
    curve = df.dropna(subset=["val/total_loss"])[["step", "val/total_loss", "epoch"]].copy()
    curve["samples"] = (curve["step"] * bs).astype(int)
    curve = curve.sort_values("step")

    # Smoothed
    window = max(1, len(curve) // 10)
    curve["smoothed"] = curve["val/total_loss"].rolling(window, min_periods=2, center=True).mean()

    ax.plot(curve["samples"], curve["smoothed"], color=c, linewidth=2.5, alpha=0.8, label=m)

    # Find when each target loss is first reached
    for tgt, mk in zip(target_losses, marker_styles):
        reached = curve[curve["smoothed"] <= tgt]
        if len(reached) > 0:
            first = reached.iloc[0]
            samples = first["samples"]
            efficiency_data[m].append((tgt, samples))
            ax.scatter(samples, tgt, color=c, marker=mk, s=80, edgecolors="#333",
                       linewidth=1.2, zorder=6)

ax.set_xlabel("Samples seen (step \u00d7 batch_size)", fontsize=11)
ax.set_ylabel("Val Total Loss (smoothed)", fontsize=11)
ax.set_title("MGM2 Stage 2 — Training Efficiency\n(markers = first epoch reaching each target loss)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, title="Scale", title_fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3)
ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

# Legend: first model lines, then target markers
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
target_handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor="#555",
                         markersize=8, label=f"Target={tgt:.2f}")
                  for tgt, mk in zip(target_losses, marker_styles)]
all_handles = handles + target_handles
all_labels = labels + [f"Target={tgt:.2f}" for tgt in target_losses]
ax.legend(all_handles, all_labels, fontsize=8, ncol=2, loc="upper right")

fig.subplots_adjust(top=0.92, right=0.97)
for fmt in ["png", "pdf"]:
    fig.savefig(os.path.join(OUTPUT_DIR, f"stage2_efficiency.{fmt}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved efficiency")

# Print efficiency table
print("\nSamples to reach target loss:")
print(f"{'Model':>10}", end="")
for tgt in target_losses:
    print(f"  {'loss<='+str(tgt):>12}", end="")
print()
for m in models:
    print(f"{m:>10}", end="")
    for tgt in target_losses:
        found = [s for t, s in efficiency_data[m] if t == tgt]
        if found:
            print(f"  {found[0]:>12,.0f}", end="")
        else:
            print(f"  {'---':>12}", end="")
    print()

print(f"\nDone. All figures saved to {OUTPUT_DIR}/")
