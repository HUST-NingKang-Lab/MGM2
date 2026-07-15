#!/usr/bin/env python3
"""MGnify frozen-backbone embedding UMAP and AvgBIO diagnostics.

The script samples balanced L1 MGnify test communities, extracts CLS embeddings
from frozen stage2 MGM2 backbones at multiple scales, and compares their
biological conservation against a raw-abundance baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

LEVELS = ["env_level_1", "env_level_2", "env_level_3", "env_level_4", "env_level_5"]
LEVEL_LABELS = ["L1", "L2", "L3", "L4", "L5"]
MGM2_SCALES = ["scale_small", "scale_medium", "scale_large", "scale_xlarge"]
SCALE_LABELS = {
    "scale_small": "MGM2-small",
    "scale_medium": "MGM2-medium",
    "scale_large": "MGM2-large",
    "scale_xlarge": "MGM2-xlarge",
}
MODEL_ORDER = ["Raw abundance"] + [SCALE_LABELS[scale] for scale in MGM2_SCALES]
MODEL_COLORS = {
    "Raw abundance": "#777777",
    "MGM2-small": "#4363D8",
    "MGM2-medium": "#1A237E",
    "MGM2-large": "#00897B",
    "MGM2-xlarge": "#6A1B9A",
}
LEVEL_COLORS = {
    "Engineered": "#C44E52",
    "Environmental": "#4C72B0",
    "Host-associated": "#55A868",
}
UNKNOWN_COLOR = "#B0B0B0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot frozen MGM2 CLS UMAPs and AvgBIO metrics for MGnify.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", type=Path, default=Path("figures/mgnify_embedding"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/downstreams/MGnify_mst"))
    parser.add_argument("--mgnify_root", type=Path, default=Path("outputs_downstreams/MGnify_mst"))
    parser.add_argument("--stage2_root", type=Path, default=Path("outputs_grid/stage2/20260505_094404"))
    parser.add_argument("--scales", nargs="+", default=MGM2_SCALES)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--samples_per_l1_class", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--svd_components", type=int, default=50)
    parser.add_argument("--silhouette_sample_size", type=int, default=2500)
    parser.add_argument("--force_extract", action="store_true", help="Recompute CLS embeddings even if cached pkl files exist")
    parser.add_argument("--skip_umap", action="store_true")
    parser.add_argument("--skip_avgbio", action="store_true")
    return parser.parse_args()


def normalize_label(label) -> Optional[str]:
    if pd.isna(label):
        return None
    value = str(label).strip()
    if not value:
        return None
    return value.replace("root:", "")


def is_known_label(label) -> bool:
    value = normalize_label(label)
    return value is not None and value.lower() != "unknown"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, **kwargs)


def savefig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def select_balanced_l1_samples(test_meta: pd.DataFrame, per_class: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    selected: List[str] = []
    labels = [label for label in ["Engineered", "Environmental", "Host-associated"] if label in set(test_meta["env_level_1"])]
    for label in labels:
        ids = test_meta.loc[test_meta["env_level_1"] == label, "SampleID"].astype(str).to_numpy()
        if len(ids) == 0:
            continue
        n = min(per_class, len(ids))
        selected.extend(rng.choice(ids, size=n, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def shared_known_labels(train_meta: pd.DataFrame, test_meta: pd.DataFrame, level: str) -> List[str]:
    train = {normalize_label(value) for value in train_meta[level] if is_known_label(value)}
    test = {normalize_label(value) for value in test_meta[level] if is_known_label(value)}
    return sorted(train & test)


def load_stage2_backbone(model, checkpoint_path: Path) -> None:
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    if any(key.startswith("model.backbone.") for key in state_dict):
        backbone_state = {
            key[len("model.backbone."):]: value
            for key, value in state_dict.items()
            if key.startswith("model.backbone.")
        }
    elif any(key.startswith("backbone.") for key in state_dict):
        backbone_state = {
            key[len("backbone."):]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
    else:
        backbone_state = state_dict
    model.backbone.load_state_dict(backbone_state, strict=True)


def extract_scale_embeddings(args: argparse.Namespace, scale: str, sample_ids: List[str], test_meta: pd.DataFrame) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader, Subset

    from src.configs.Config import MGM2Config
    from src.data.CommunityClassificationDataset import (
        create_community_classification_dataloaders,
        create_community_classification_test_loader,
    )
    from src.data.Dataset import BatchCollatorMGM2
    from src.models import MGM2Model

    stage2_dir = args.stage2_root / scale
    ckpt = stage2_dir / "checkpoints" / "last.ckpt"
    config_path = stage2_dir / "config.json"
    label_path = args.mgnify_root / scale / "env_level_1" / f"seed_{args.seed}" / "label_to_id.json"
    if not ckpt.exists() or not config_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing checkpoint/config/label map for {scale}")

    config = MGM2Config.load(str(config_path))
    with label_path.open() as handle:
        label_to_id = {str(label): int(idx) for label, idx in json.load(handle).items()}

    train_loader, _, _ = create_community_classification_dataloaders(
        data_path=args.data_dir / "train.pkl",
        metadata_path=args.data_dir / "train_meta_split.csv",
        ntv3_embedding_path=args.data_dir / "train.resolved_embeddings.h5",
        abundance_bin_config=config.abundance.__dict__,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_id_column="SampleID",
        label_column="env_level_1",
        val_ratio=0.1,
        seed=config.seed,
    )
    test_loader_full = create_community_classification_test_loader(
        data_path=args.data_dir / "test.pkl",
        metadata_path=args.data_dir / "test_meta_split.csv",
        ntv3_embedding_path=args.data_dir / "test.resolved_embeddings.h5",
        abundance_bin_config=config.abundance.__dict__,
        label_to_id=label_to_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_id_column="SampleID",
        label_column="env_level_1",
        abundance_preprocessor=train_loader.dataset.abundance_preprocessor,
        bin_manager=train_loader.dataset.bin_manager,
    )

    normalized_to_index = {
        test_loader_full.dataset._normalize_sample_id(sample_id): idx
        for idx, sample_id in enumerate(test_loader_full.dataset.data["samples"])
    }
    indices = [normalized_to_index[str(sample_id).strip()] for sample_id in sample_ids if str(sample_id).strip() in normalized_to_index]
    subset = Subset(test_loader_full.dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=BatchCollatorMGM2(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = MGM2Model(
        config=config.model.__dict__,
        num_bins=config.abundance.num_bins,
        enable_regression=config.abundance.enable_regression,
        enable_qwen3_alignment=config.qwen3_alignment.enabled,
        qwen3_projection_dim=config.qwen3_alignment.projection_dim,
        mlp_dropout=config.model.mlp_dropout,
    )
    load_stage2_backbone(model, ckpt)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    model.to(device)
    model.eval()

    labels_by_sample = test_meta.set_index("SampleID")["env_level_1"].map(normalize_label).to_dict()
    embeddings = []
    ordered_ids = []
    with torch.no_grad():
        for batch in loader:
            model_batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=model_batch["input_ids"],
                    embeddings=model_batch["embeddings"],
                    abundance_values=model_batch["abundance_values"],
                    attention_mask=model_batch.get("attention_mask"),
                )
            embeddings.append(outputs["cls_hidden"].detach().float().cpu().numpy())
            ordered_ids.extend([str(sample_id) for sample_id in batch["sample_id"]])

    matrix = np.concatenate(embeddings, axis=0)
    emb_cols = [f"dim_{idx}" for idx in range(matrix.shape[1])]
    df = pd.DataFrame(matrix, columns=emb_cols)
    df.insert(0, "SampleID", ordered_ids)
    df.insert(1, "label", [labels_by_sample.get(sample_id, "unknown") for sample_id in ordered_ids])
    return df


def load_raw_abundance_embedding(args: argparse.Namespace, sample_ids: List[str]) -> pd.DataFrame:
    from scipy import sparse
    from src.data.utils import load_preprocessed_data

    data = load_preprocessed_data(str(args.data_dir / "test.pkl"))
    sample_lookup = {str(sample_id).strip(): sample_id for sample_id in data["samples"]}
    otu_dim = len(data.get("otu_id_to_idx", {}))
    if otu_dim <= 0:
        max_id = 0
        for sample_id in sample_ids:
            raw_id = sample_lookup.get(str(sample_id).strip())
            if raw_id is None:
                continue
            ids = np.asarray(data["otu_data"][raw_id]["input_ids"], dtype=np.int64)
            ids = ids[ids >= 0]
            if ids.size:
                max_id = max(max_id, int(ids.max()))
        otu_dim = max_id + 1

    rows = []
    cols = []
    values = []
    kept_ids = []
    for row_idx, sample_id in enumerate(sample_ids):
        raw_id = sample_lookup.get(str(sample_id).strip())
        if raw_id is None:
            continue
        sample_data = data["otu_data"][raw_id]
        ids = np.asarray(sample_data["input_ids"], dtype=np.int64)
        abundance = np.asarray(sample_data["abundance_values"], dtype=np.float32)
        mask = (ids >= 0) & (ids < otu_dim) & np.isfinite(abundance)
        rows.extend([len(kept_ids)] * int(mask.sum()))
        cols.extend(ids[mask].astype(int).tolist())
        values.extend(abundance[mask].astype(np.float32).tolist())
        kept_ids.append(str(sample_id))

    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(kept_ids), otu_dim), dtype=np.float32)
    matrix.data = np.log1p(matrix.data)
    n_components = min(args.svd_components, max(2, min(matrix.shape) - 1))
    reduced = TruncatedSVD(n_components=n_components, random_state=args.seed).fit_transform(matrix)
    reduced = normalize(reduced)
    emb_cols = [f"dim_{idx}" for idx in range(reduced.shape[1])]
    df = pd.DataFrame(reduced, columns=emb_cols)
    df.insert(0, "SampleID", kept_ids)
    return df


def embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [col for col in df.columns if col.startswith("dim_")]
    return df[cols].to_numpy(np.float32)


def cached_embedding_path(args: argparse.Namespace, scale: str) -> Path:
    return args.output_dir / f"mgnify_{scale}_cls_embeddings.pkl"


def load_or_extract_scale_embeddings(
    args: argparse.Namespace,
    scale: str,
    sample_ids: List[str],
    test_meta: pd.DataFrame,
) -> pd.DataFrame:
    cache_path = cached_embedding_path(args, scale)
    model_name = SCALE_LABELS.get(scale, scale)
    if cache_path.exists() and not args.force_extract:
        print(f"Loading cached {model_name}: {cache_path}", flush=True)
        embeddings = pd.read_pickle(cache_path)
        if "SampleID" not in embeddings.columns:
            raise ValueError(f"Cached embedding file lacks SampleID column: {cache_path}")
        if not any(col.startswith("dim_") for col in embeddings.columns):
            raise ValueError(f"Cached embedding file lacks dim_* columns: {cache_path}")
        embeddings["SampleID"] = embeddings["SampleID"].astype(str).str.strip()
        wanted = pd.DataFrame({"SampleID": [str(sample_id).strip() for sample_id in sample_ids]})
        embeddings = wanted.merge(embeddings, on="SampleID", how="inner")
        if embeddings.empty:
            raise ValueError(f"Cached embedding file has no overlap with selected samples: {cache_path}")
        if len(embeddings) < len(sample_ids):
            print(
                f"  cache covers {len(embeddings)}/{len(sample_ids)} selected samples; "
                "continuing with cached overlap",
                flush=True,
            )
        return embeddings

    print(f"Extracting {model_name}", flush=True)
    embeddings = extract_scale_embeddings(args, scale, sample_ids, test_meta)
    embeddings.to_pickle(cache_path)
    return embeddings


def label_color_map(labels: Iterable[str]) -> Dict[str, str]:
    labels = [label for label in labels if label != "unknown"]
    base_colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(cmap_name)
        base_colors.extend([mpl.colors.to_hex(cmap(i)) for i in range(cmap.N)])
    color_map = {label: base_colors[idx % len(base_colors)] for idx, label in enumerate(labels)}
    color_map["unknown"] = UNKNOWN_COLOR
    return color_map


def level_labels_for_samples(sample_ids: Iterable[str], test_meta: pd.DataFrame, level: str) -> pd.Series:
    meta = test_meta.set_index("SampleID")
    labels = []
    for sample_id in sample_ids:
        label = None
        if sample_id in meta.index:
            raw_label = meta.loc[sample_id, level]
            if isinstance(raw_label, pd.Series):
                raw_label = raw_label.iloc[0]
            label = normalize_label(raw_label)
        if label is None or str(label).lower() == "unknown":
            label = "unknown"
        labels.append(label)
    return pd.Series(labels)


def plot_umap_panels(args: argparse.Namespace, embeddings_by_model: Dict[str, pd.DataFrame], test_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, df in embeddings_by_model.items():
        matrix = embedding_matrix(df)
        pca = PCA(n_components=min(args.svd_components, matrix.shape[1]), random_state=args.seed)
        reduced = pca.fit_transform(matrix)
        coords = reduced[:, :2]
        method = "PCA"
        try:
            import umap

            reducer = umap.UMAP(n_neighbors=30, min_dist=0.08, metric="cosine", random_state=args.seed)
            coords = reducer.fit_transform(reduced)
            method = "UMAP"
        except Exception as exc:
            print(f"{model_name}: UMAP unavailable, using PCA ({exc})", flush=True)
        coord_df = pd.DataFrame(
            {
                "SampleID": df["SampleID"].values,
                "model": model_name,
                "method": method,
                "x": coords[:, 0],
                "y": coords[:, 1],
            }
        )
        for level in LEVELS:
            coord_df[level] = level_labels_for_samples(coord_df["SampleID"].astype(str), test_meta, level).values
        rows.append(coord_df)
    coords_all = pd.concat(rows, ignore_index=True)
    coords_all.to_csv(args.output_dir / "mgnify_scale_embedding_umap_coordinates.csv", index=False)

    models = [SCALE_LABELS[scale] for scale in args.scales if SCALE_LABELS[scale] in embeddings_by_model]
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))

    for level, level_label in zip(LEVELS, LEVEL_LABELS):
        level_values = coords_all[level].fillna("unknown").map(lambda value: normalize_label(value) or "unknown")
        counts = level_values.value_counts()
        ordered_labels = [label for label in counts.index.tolist() if label != "unknown"]
        if "unknown" in counts.index:
            ordered_labels.append("unknown")
        colors = label_color_map(ordered_labels)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols + 2.8, 5.8 * nrows), squeeze=False)
        for ax, model_name in zip(axes.ravel(), models):
            sub = coords_all[coords_all["model"] == model_name].copy()
            sub["_plot_label"] = sub[level].fillna("unknown").map(lambda value: normalize_label(value) or "unknown")
            unknown = sub[sub["_plot_label"] == "unknown"]
            if not unknown.empty:
                ax.scatter(
                    unknown["x"],
                    unknown["y"],
                    s=7,
                    alpha=0.35,
                    color=UNKNOWN_COLOR,
                    linewidths=0,
                    zorder=1,
                )
            for label in ordered_labels:
                if label == "unknown":
                    continue
                label_sub = sub[sub["_plot_label"] == label]
                if label_sub.empty:
                    continue
                ax.scatter(
                    label_sub["x"],
                    label_sub["y"],
                    s=7,
                    alpha=0.62,
                    color=colors[label],
                    linewidths=0,
                    zorder=2,
                )
            method = sub["method"].iloc[0] if not sub.empty else "UMAP"
            ax.set_title(model_name)
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes.ravel()[len(models):]:
            ax.axis("off")

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=f"{label} ({int(counts.get(label, 0))})",
                markerfacecolor=colors.get(label, UNKNOWN_COLOR),
                markeredgewidth=0,
                markersize=5,
            )
            for label in ordered_labels
        ]
        fig.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(0.84, 0.5),
            title=f"{level_label} labels",
            title_fontsize=8,
        )
        fig.suptitle(f"Frozen MGM2 CLS embeddings colored by MGnify {level_label}", y=1.01, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.83, 0.98])
        savefig(fig, args.output_dir, f"main_mgnify_scale_embedding_umap_{level_label.lower()}")
    return coords_all


def compute_avgbio_for_matrix(
    matrix: np.ndarray,
    labels: Iterable,
    seed: int,
    silhouette_sample_size: int,
) -> Dict[str, float]:
    labels = np.asarray([normalize_label(label) for label in labels], dtype=object)
    known = np.array([is_known_label(label) for label in labels], dtype=bool)
    labels = labels[known]
    matrix = matrix[known]
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2 or matrix.shape[0] < len(unique_labels):
        return {"ari_cell": np.nan, "nmi_cell": np.nan, "asw_cell": np.nan, "avg_bio": np.nan, "n_samples": int(matrix.shape[0]), "n_labels": len(unique_labels)}

    matrix = normalize(matrix.astype(np.float32))
    cluster_labels = KMeans(n_clusters=len(unique_labels), n_init=20, random_state=seed).fit_predict(matrix)
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels, average_method="arithmetic")
    if len(labels) > silhouette_sample_size:
        sample_size = silhouette_sample_size
    else:
        sample_size = None
    try:
        asw_raw = silhouette_score(matrix, labels, metric="cosine", sample_size=sample_size, random_state=seed)
        asw = (float(asw_raw) + 1.0) / 2.0
    except ValueError:
        asw = np.nan
    avg_bio = float(np.nanmean([ari, nmi, asw]))
    return {
        "ari_cell": float(ari),
        "nmi_cell": float(nmi),
        "asw_cell": float(asw),
        "avg_bio": avg_bio,
        "n_samples": int(matrix.shape[0]),
        "n_labels": len(unique_labels),
    }


def plot_avgbio(args: argparse.Namespace, representations: Dict[str, pd.DataFrame], test_meta: pd.DataFrame) -> pd.DataFrame:
    meta = test_meta.set_index("SampleID")
    rows = []
    for model_name, df in representations.items():
        matrix = embedding_matrix(df)
        sample_ids = df["SampleID"].astype(str).tolist()
        for level, level_label in zip(LEVELS, LEVEL_LABELS):
            labels = meta.loc[sample_ids, level].values
            metrics = compute_avgbio_for_matrix(matrix, labels, args.seed, args.silhouette_sample_size)
            metrics.update({"model": model_name, "level": level, "level_label": level_label})
            rows.append(metrics)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.output_dir / "mgnify_embedding_avgbio.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(LEVELS))
    models = [model for model in MODEL_ORDER if model in representations]
    width = 0.82 / max(len(models), 1)
    for idx, model_name in enumerate(models):
        sub = scores[scores["model"] == model_name].set_index("level").reindex(LEVELS)
        offset = (idx - len(models) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            sub["avg_bio"].to_numpy(float),
            width,
            color=MODEL_COLORS.get(model_name, "#777777"),
            edgecolor="#333333",
            linewidth=0.45,
            label=model_name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylabel("AvgBIO")
    ax.set_ylim(0, 1.02)
    ax.set_title("Biological conservation of raw abundance and frozen MGM2 CLS embeddings")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, args.output_dir, "main_mgnify_embedding_avgbio")
    return scores


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_meta = read_csv(args.data_dir / "test_meta_split.csv", dtype={"SampleID": str})
    train_meta = read_csv(args.data_dir / "train_meta_split.csv", dtype={"SampleID": str})
    test_meta["SampleID"] = test_meta["SampleID"].astype(str).str.strip()
    train_meta["SampleID"] = train_meta["SampleID"].astype(str).str.strip()

    sample_ids = select_balanced_l1_samples(test_meta, args.samples_per_l1_class, args.seed)
    pd.DataFrame({"SampleID": sample_ids}).to_csv(args.output_dir / "mgnify_embedding_sample_ids.csv", index=False)
    print(f"Selected {len(sample_ids)} balanced L1 samples", flush=True)

    embeddings_by_scale: Dict[str, pd.DataFrame] = {}
    for scale in args.scales:
        model_name = SCALE_LABELS.get(scale, scale)
        embeddings = load_or_extract_scale_embeddings(args, scale, sample_ids, test_meta)
        embeddings_by_scale[model_name] = embeddings

    raw_embedding = load_raw_abundance_embedding(args, sample_ids)
    raw_embedding = raw_embedding.merge(test_meta[["SampleID", "env_level_1"]], on="SampleID", how="left")
    raw_embedding = raw_embedding.rename(columns={"env_level_1": "label"})
    raw_embedding["label"] = raw_embedding["label"].map(normalize_label)

    if not args.skip_umap:
        plot_umap_panels(args, embeddings_by_scale, test_meta)
    if not args.skip_avgbio:
        representations = {"Raw abundance": raw_embedding}
        representations.update(embeddings_by_scale)
        plot_avgbio(args, representations, test_meta)
    print(f"Done. {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
