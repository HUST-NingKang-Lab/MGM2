"""Load BatchTopK SAE checkpoints and compute feature activations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

from src.models.BatchTopKSAE import BatchTopKSAE


def load_sae_from_checkpoint(path: Path, device: str = "cpu") -> BatchTopKSAE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    state = ckpt["model_state_dict"]
    input_dim = int(args.get("dict_size", 0))  # placeholder for legacy fallbacks
    if "W_enc" in state:
        input_dim = int(state["W_enc"].shape[0])
        dict_size = int(state["W_enc"].shape[1])
    else:
        raise KeyError("checkpoint model_state_dict missing W_enc")
    model = BatchTopKSAE(
        input_dim=input_dim,
        dict_size=dict_size,
        top_k=int(args.get("top_k", 1)),
        aux_k=int(args.get("aux_k", max(1, int(args.get("top_k", 1)) // 2))),
        aux_penalty=float(args.get("aux_penalty", 0.0)),
        input_unit_norm=not bool(args.get("no_input_unit_norm", False)),
    )
    model.load_state_dict(state)
    model.eval()
    return model.to(torch.device(device))


def select_feature_ids(
    checkpoint: Path,
    feature_ids: str | None = None,
    top_n_features: int | None = None,
    feature_scores_path: Path | None = None,
    feature_selection_metric: str = "activation_sum",
    all_features: bool = False,
) -> List[int]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    dict_size = int(state["W_enc"].shape[1])
    if feature_ids:
        selected = []
        for item in feature_ids.split(","):
            item = item.strip()
            if item:
                selected.append(int(item))
        return selected
    if all_features:
        return list(range(dict_size))
    if top_n_features:
        if feature_scores_path is None:
            raise ValueError("top-n feature selection requires feature scores; run score_sae_features.py first")
        scores = pd.read_csv(feature_scores_path)
        if "feature_id" not in scores.columns:
            raise ValueError(f"feature scores missing feature_id column: {feature_scores_path}")
        if feature_selection_metric not in scores.columns:
            raise ValueError(f"feature scores missing metric {feature_selection_metric!r}: {feature_scores_path}")
        scores = scores[scores["feature_id"].between(0, dict_size - 1)].copy()
        scores = scores.sort_values([feature_selection_metric, "feature_id"], ascending=[False, True])
        return [int(idx) for idx in scores["feature_id"].head(int(top_n_features)).tolist()]
    raise ValueError("provide --feature-ids, --top-n-features, or --all-features")


@torch.no_grad()
def dense_feature_activations(model: BatchTopKSAE, x: np.ndarray, feature_ids: Iterable[int], device: str = "cpu") -> np.ndarray:
    feature_ids = list(feature_ids)
    tensor = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    processed, _ = model.preprocess(tensor)
    ids = torch.as_tensor(feature_ids, dtype=torch.long, device=processed.device)
    acts = torch.relu((processed - model.b_dec) @ model.W_enc[:, ids])
    return acts.detach().cpu().numpy()


@torch.no_grad()
def sparse_topk_feature_activations(model: BatchTopKSAE, x: np.ndarray, feature_ids: Iterable[int], device: str = "cpu") -> np.ndarray:
    """Return BatchTopK-gated feature activations for selected features.

    BatchTopK gating is global over the full SAE dictionary for the current
    batch. Therefore this cannot be computed feature-by-feature without
    changing which activations survive the top-k gate.
    """

    feature_ids = list(feature_ids)
    tensor = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    processed, _ = model.preprocess(tensor)
    all_sparse = model.encode(processed, update_inactive=False)
    ids = torch.as_tensor(feature_ids, dtype=torch.long, device=all_sparse.device)
    return all_sparse[:, ids].detach().cpu().numpy()


def feature_activations(
    model: BatchTopKSAE,
    x: np.ndarray,
    feature_ids: Iterable[int],
    device: str = "cpu",
    activation_mode: str = "dense",
) -> np.ndarray:
    if activation_mode == "dense":
        return dense_feature_activations(model, x, feature_ids, device=device)
    if activation_mode == "sparse_topk":
        return sparse_topk_feature_activations(model, x, feature_ids, device=device)
    raise ValueError(f"unknown activation_mode: {activation_mode!r}")
