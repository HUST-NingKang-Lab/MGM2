"""Streaming lookup-table construction for SAE features."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .activation_alignment import infer_activation_token_type
from .sae_features import dense_feature_activations, sparse_topk_feature_events


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

    def add_sparse_candidates(self, rows: pd.DataFrame, row_indices: np.ndarray, values: np.ndarray) -> None:
        for idx, value in zip(row_indices, values):
            row = rows.iloc[int(idx)]
            record = row.to_dict()
            record["activation"] = float(value)
            self._push(float(value), record)

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
    activation_token_type: str = "auto",
    max_chunks: int | None = None,
    show_progress: bool = True,
) -> Dict[int, dict]:
    feature_ids = [int(fid) for fid in feature_ids]
    heaps = {fid: FeatureHeap(max_events=max_events_per_feature) for fid in feature_ids}
    sample_cols = [col for col in ["sample_id", "project", "institution", "keywords", "nonzero_otu_count"] if col in samples.columns]
    sample_info = samples[sample_cols].drop_duplicates("sample_id")
    dict_size = int(model.W_enc.shape[1])
    selected_feature_mask = np.zeros(dict_size, dtype=bool)
    selected_feature_mask[feature_ids] = True

    activations_path = Path(activations_path)
    activation_token_type = infer_activation_token_type(activations_path, activation_token_type)
    with h5py.File(activations_path, "r") as handle:
        h5_activations = handle["activations"]
        reader = pd.read_csv(tokens_path, chunksize=chunk_size)
        total_chunks = math.ceil(int(h5_activations.shape[0]) / max(1, int(chunk_size)))
        if max_chunks is not None:
            total_chunks = min(total_chunks, int(max_chunks))
        progress = tqdm(reader, total=total_chunks, desc="feature_lookup", unit="chunk", disable=not show_progress)
        offset = 0
        activation_offset = 0
        for chunk_idx, token_chunk in enumerate(progress):
            if max_chunks is not None and chunk_idx >= max_chunks:
                break
            end = offset + len(token_chunk)
            token_type_mask = (token_chunk["token_type"] == "otu").to_numpy()
            split_mask = token_chunk["split"].isin(splits).to_numpy()
            mask = token_type_mask & split_mask
            selected = token_chunk.loc[mask].copy()
            if selected.empty:
                if activation_token_type == "otu":
                    activation_offset += int(token_type_mask.sum())
                offset = end
                progress.set_postfix(tokens=end, selected=0)
                continue
            if activation_token_type == "all":
                selected_positions = np.flatnonzero(mask)
                x = h5_activations[offset:end][selected_positions].astype(np.float32)
            elif activation_token_type == "otu":
                within_otu_positions = np.cumsum(token_type_mask, dtype=np.int64) - 1
                selected_positions = activation_offset + within_otu_positions[mask]
                x = h5_activations[selected_positions].astype(np.float32)
                activation_offset += int(token_type_mask.sum())
            else:
                raise ValueError(f"unknown activation_token_type: {activation_token_type!r}")
            selected = selected.merge(sample_info, on="sample_id", how="left", suffixes=("", "_sample"))
            selected["taxonomy"] = selected["otu_id"].map(lambda otu: taxonomy_map.get(str(otu), {}).get("taxonomy", ""))
            selected["rep_species"] = selected["otu_id"].map(lambda otu: taxonomy_map.get(str(otu), {}).get("rep_species", ""))
            if activation_mode == "sparse_topk":
                sparse_rows, sparse_cols, sparse_values = sparse_topk_feature_events(model, x, device=device)
                if len(sparse_values):
                    keep = selected_feature_mask[sparse_cols]
                    sparse_rows = sparse_rows[keep]
                    sparse_cols = sparse_cols[keep]
                    sparse_values = sparse_values[keep]
                for heap in heaps.values():
                    heap.count += int(len(selected))
                if len(sparse_values):
                    positive_count = np.bincount(sparse_cols, minlength=dict_size)
                    activation_sum = np.bincount(sparse_cols, weights=sparse_values, minlength=dict_size)
                    activation_max = np.zeros(dict_size, dtype=np.float32)
                    np.maximum.at(activation_max, sparse_cols, sparse_values)
                    active_features = np.flatnonzero(positive_count)
                    for feature_id in active_features:
                        if not selected_feature_mask[feature_id]:
                            continue
                        heap = heaps[int(feature_id)]
                        heap.positive_count += int(positive_count[feature_id])
                        heap.activation_sum += float(activation_sum[feature_id])
                        heap.activation_max = max(heap.activation_max, float(activation_max[feature_id]))

                    order = np.argsort(sparse_cols, kind="mergesort")
                    sorted_cols = sparse_cols[order]
                    boundaries = np.flatnonzero(np.r_[True, sorted_cols[1:] != sorted_cols[:-1], True])
                    for start_idx, stop_idx in zip(boundaries[:-1], boundaries[1:]):
                        feature_id = int(sorted_cols[start_idx])
                        if not selected_feature_mask[feature_id]:
                            continue
                        heap = heaps[feature_id]
                        group_order = order[start_idx:stop_idx]
                        group_values = sparse_values[group_order]
                        if len(heap.heap) >= heap.max_events:
                            candidate_mask = group_values > heap.heap[0][0]
                            if not candidate_mask.any():
                                continue
                            group_order = group_order[candidate_mask]
                            group_values = group_values[candidate_mask]
                        elif len(group_values) > heap.max_events:
                            local = np.argpartition(group_values, -heap.max_events)[-heap.max_events:]
                            group_order = group_order[local]
                            group_values = group_values[local]
                        heap.add_sparse_candidates(selected, sparse_rows[group_order], group_values)
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
            progress.set_postfix(
                tokens=end,
                selected=len(selected),
                features=len(feature_ids),
                mode=activation_mode,
                h5_tokens=activation_token_type,
            )

    return {fid: {"stats": heaps[fid].stats(), "events": heaps[fid].records()} for fid in feature_ids}
