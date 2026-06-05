"""Streaming lookup-table construction for SAE features."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .sae_features import dense_feature_activations, sparse_topk_feature_activations


@dataclass
class FeatureHeap:
    max_events: int
    heap: List[tuple] = field(default_factory=list)
    count: int = 0
    positive_count: int = 0
    activation_sum: float = 0.0
    activation_max: float = 0.0
    heap_item_count: int = 0

    def update_stats(self, activations: np.ndarray) -> None:
        activations = np.asarray(activations, dtype=np.float64)
        self.count += int(activations.size)
        self.positive_count += int((activations > 0).sum())
        self.activation_sum += float(activations.sum())
        if activations.size:
            self.activation_max = max(self.activation_max, float(activations.max()))

    def add_candidates(self, activations: np.ndarray, rows: pd.DataFrame, candidate_indices: np.ndarray) -> None:
        for idx in candidate_indices:
            row = rows.iloc[int(idx)]
            value = float(activations[int(idx)])
            record = row.to_dict()
            record["activation"] = value
            self._push(value, record)

    def _push(self, value: float, record: dict) -> None:
        self.heap_item_count += 1
        value = float(value)
        item = (value, self.heap_item_count, record)
        if len(self.heap) < self.max_events:
            heapq.heappush(self.heap, item)
        elif value > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def records(self) -> List[dict]:
        return [item[2] for item in sorted(self.heap, key=lambda x: x[0], reverse=True)]

    def stats(self) -> dict:
        positive_fraction = float(self.positive_count / max(1, self.count))
        return {
            "count": int(self.count),
            "positive_count": int(self.positive_count),
            "positive_fraction": positive_fraction,
            "activation_sum": float(self.activation_sum),
            "activation_mean": float(self.activation_sum / max(1, self.count)),
            "mean_positive_activation": float(self.activation_sum / max(1, self.positive_count)),
            "activation_max": float(self.activation_max),
        }


def build_lookup_tables(
    activations_path,
    tokens_path,
    samples: pd.DataFrame,
    taxonomy_map: Dict[str, dict],
    model,
    feature_ids: Iterable[int],
    splits: set[str],
    max_events_per_feature: int,
    chunk_size: int,
    feature_batch_size: int,
    device: str,
    activation_mode: str = "dense",
    max_chunks: int | None = None,
    show_progress: bool = True,
) -> Dict[int, dict]:
    feature_ids = [int(fid) for fid in feature_ids]
    heaps = {fid: FeatureHeap(max_events=max_events_per_feature) for fid in feature_ids}
    sample_cols = [col for col in ["sample_id", "project", "institution", "keywords", "nonzero_otu_count"] if col in samples.columns]
    sample_info = samples[sample_cols].drop_duplicates("sample_id")

    with h5py.File(activations_path, "r") as handle:
        h5_activations = handle["activations"]
        reader = pd.read_csv(tokens_path, chunksize=chunk_size)
        total_chunks = math.ceil(int(h5_activations.shape[0]) / max(1, int(chunk_size)))
        if max_chunks is not None:
            total_chunks = min(total_chunks, int(max_chunks))
        progress = tqdm(reader, total=total_chunks, desc="feature_lookup", unit="chunk", disable=not show_progress)
        offset = 0
        for chunk_idx, token_chunk in enumerate(progress):
            if max_chunks is not None and chunk_idx >= max_chunks:
                break
            end = offset + len(token_chunk)
            mask = (token_chunk["token_type"] == "otu") & (token_chunk["split"].isin(splits))
            selected = token_chunk.loc[mask].copy()
            if selected.empty:
                offset = end
                progress.set_postfix(tokens=end, selected=0)
                continue
            selected_positions = np.flatnonzero(mask.to_numpy())
            x = h5_activations[offset:end][selected_positions].astype(np.float32)
            selected = selected.merge(sample_info, on="sample_id", how="left", suffixes=("", "_sample"))
            selected["taxonomy"] = selected["otu_id"].map(lambda otu: taxonomy_map.get(str(otu), {}).get("taxonomy", ""))
            selected["rep_species"] = selected["otu_id"].map(lambda otu: taxonomy_map.get(str(otu), {}).get("rep_species", ""))
            if activation_mode == "sparse_topk":
                sparse_acts = sparse_topk_feature_activations(model, x, feature_ids, device=device)
                for feature_col, feature_id in enumerate(feature_ids):
                    feature_acts = sparse_acts[:, feature_col]
                    heaps[feature_id].update_stats(feature_acts)
                    positive_indices = np.flatnonzero(feature_acts > 0)
                    if len(positive_indices) == 0:
                        continue
                    n_candidates = min(heaps[feature_id].max_events, len(positive_indices))
                    if n_candidates == len(positive_indices):
                        candidate_indices = positive_indices
                    else:
                        local = np.argpartition(feature_acts[positive_indices], -n_candidates)[-n_candidates:]
                        candidate_indices = positive_indices[local]
                    heaps[feature_id].add_candidates(feature_acts, selected, candidate_indices)
            elif activation_mode == "dense":
                for feature_start in range(0, len(feature_ids), int(feature_batch_size)):
                    feature_batch = feature_ids[feature_start : feature_start + int(feature_batch_size)]
                    acts = dense_feature_activations(model, x, feature_batch, device=device)
                    for feature_col, feature_id in enumerate(feature_batch):
                        feature_acts = acts[:, feature_col]
                        heaps[feature_id].update_stats(feature_acts)
                        n_candidates = min(heaps[feature_id].max_events, len(feature_acts))
                        if n_candidates == len(feature_acts):
                            candidate_indices = np.arange(len(feature_acts), dtype=np.int64)
                        else:
                            candidate_indices = np.argpartition(feature_acts, -n_candidates)[-n_candidates:]
                        heaps[feature_id].add_candidates(feature_acts, selected, candidate_indices)
            else:
                raise ValueError(f"unknown activation_mode: {activation_mode!r}")
            offset = end
            progress.set_postfix(tokens=end, selected=len(selected), features=len(feature_ids), mode=activation_mode)

    return {fid: {"stats": heaps[fid].stats(), "events": heaps[fid].records()} for fid in feature_ids}
