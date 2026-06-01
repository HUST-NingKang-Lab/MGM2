"""Metrics for wastewater community forecasting."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def normalize_abundance(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize abundance over the last axis while preserving all-zero rows."""
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    totals = arr.sum(axis=-1, keepdims=True)
    return np.divide(arr, totals, out=np.zeros_like(arr), where=totals > eps)


def bray_curtis_distance(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Bray-Curtis dissimilarity for one abundance vector pair."""
    true = normalize_abundance(np.asarray(y_true, dtype=np.float32))
    pred = normalize_abundance(np.asarray(y_pred, dtype=np.float32))
    shared = np.minimum(true, pred).sum()
    denom = true.sum() + pred.sum()
    if denom <= eps:
        return 0.0
    return float(1.0 - (2.0 * shared / (denom + eps)))


def topk_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Fractional overlap between true and predicted top-k taxa."""
    true = np.asarray(y_true, dtype=np.float32)
    pred = np.asarray(y_pred, dtype=np.float32)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("topk_overlap expects 1D vectors")
    if len(true) != len(pred):
        raise ValueError("true and predicted vectors must have the same length")
    if len(true) == 0:
        return 0.0
    k = max(1, min(int(k), len(true)))
    true_top = set(np.argpartition(-true, kth=k - 1)[:k].tolist())
    pred_top = set(np.argpartition(-pred, kth=k - 1)[:k].tolist())
    return float(len(true_top & pred_top) / k)


def _spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan")

    corr = spearmanr(y_true, y_pred).correlation
    if corr is None or np.isnan(corr):
        return float("nan")
    return float(corr)


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k_values: Sequence[int] = (10, 20, 50),
) -> Dict[str, float]:
    """Compute aggregate forecast metrics for arrays shaped [..., horizon, taxa]."""
    true = normalize_abundance(np.asarray(y_true, dtype=np.float32))
    pred = normalize_abundance(np.asarray(y_pred, dtype=np.float32))
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")

    flat_true = true.reshape(-1, true.shape[-1])
    flat_pred = pred.reshape(-1, pred.shape[-1])
    diff = flat_pred - flat_true
    metrics = {
        "bray_curtis": float(np.mean([bray_curtis_distance(t, p) for t, p in zip(flat_true, flat_pred)])),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "spearman": _spearman_corr(flat_true.ravel(), flat_pred.ravel()),
    }
    for k in top_k_values:
        metrics[f"top{k}_overlap"] = float(np.mean([topk_overlap(t, p, k) for t, p in zip(flat_true, flat_pred)]))
    return metrics


def aggregate_forecast_metrics(
    records: Iterable[Dict[str, np.ndarray]],
    top_k_values: Sequence[int] = (10, 20, 50),
) -> pd.DataFrame:
    """Aggregate metrics per horizon from records with y_true/y_pred arrays."""
    rows: List[Dict[str, float]] = []
    for record in records:
        true = normalize_abundance(record["y_true"])
        pred = normalize_abundance(record["y_pred"])
        if true.shape != pred.shape:
            raise ValueError(f"shape mismatch for record {record.get('plant')}: {true.shape} vs {pred.shape}")
        if true.ndim == 2:
            true = true[None, :, :]
            pred = pred[None, :, :]
        if true.ndim != 3:
            raise ValueError("records must contain arrays shaped [windows, horizon, taxa] or [horizon, taxa]")
        for horizon_idx in range(true.shape[1]):
            metrics = compute_forecast_metrics(
                true[:, horizon_idx : horizon_idx + 1, :],
                pred[:, horizon_idx : horizon_idx + 1, :],
                top_k_values=top_k_values,
            )
            rows.append(
                {
                    "plant": record.get("plant", "all"),
                    "split": record.get("split", "unknown"),
                    "horizon": horizon_idx + 1,
                    "n_windows": int(true.shape[0]),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
