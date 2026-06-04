"""Streaming feature scoring for SAE interpretation priority selection."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


@torch.no_grad()
def score_sae_features(
    activations_path: Path,
    tokens_path: Path,
    model,
    splits: set[str],
    chunk_size: int,
    feature_batch_size: int,
    device: str,
    activation_mode: str = "dense",
    max_chunks: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Score all SAE features over selected OTU tokens.

    This intentionally avoids ``num_tokens_since_fired`` because that checkpoint
    counter only says how recently a feature fired during training.
    """

    dict_size = int(model.W_enc.shape[1])
    count = 0
    positive_count = np.zeros(dict_size, dtype=np.int64)
    activation_sum = np.zeros(dict_size, dtype=np.float64)
    activation_max = np.zeros(dict_size, dtype=np.float64)
    token_type_count = 0

    with h5py.File(activations_path, "r") as handle:
        h5_activations = handle["activations"]
        reader = pd.read_csv(tokens_path, chunksize=chunk_size)
        total_chunks = math.ceil(int(h5_activations.shape[0]) / max(1, int(chunk_size)))
        if max_chunks is not None:
            total_chunks = min(total_chunks, int(max_chunks))
        progress = tqdm(reader, total=total_chunks, desc="feature_score", unit="chunk", disable=not show_progress)
        offset = 0
        for chunk_idx, token_chunk in enumerate(progress):
            if max_chunks is not None and chunk_idx >= max_chunks:
                break
            end = offset + len(token_chunk)
            mask = (token_chunk["token_type"] == "otu") & (token_chunk["split"].isin(splits))
            selected_positions = np.flatnonzero(mask.to_numpy())
            token_type_count += int((token_chunk["token_type"] == "otu").sum())
            if len(selected_positions) == 0:
                offset = end
                progress.set_postfix(tokens=end, selected=0)
                continue

            x = h5_activations[offset:end][selected_positions].astype(np.float32)
            tensor = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
            processed, _ = model.preprocess(tensor)
            centered = processed - model.b_dec
            count += int(centered.shape[0])

            if activation_mode == "dense":
                for start in range(0, dict_size, feature_batch_size):
                    stop = min(dict_size, start + feature_batch_size)
                    acts = torch.relu(centered @ model.W_enc[:, start:stop]).detach().cpu().numpy()
                    positive_count[start:stop] += (acts > 0).sum(axis=0).astype(np.int64)
                    activation_sum[start:stop] += acts.sum(axis=0).astype(np.float64)
                    activation_max[start:stop] = np.maximum(activation_max[start:stop], acts.max(axis=0).astype(np.float64))
            elif activation_mode == "sparse_topk":
                acts = model._batch_topk(torch.relu(centered @ model.W_enc), model.top_k).detach().cpu().numpy()
                positive_count += (acts > 0).sum(axis=0).astype(np.int64)
                activation_sum += acts.sum(axis=0).astype(np.float64)
                activation_max = np.maximum(activation_max, acts.max(axis=0).astype(np.float64))
            else:
                raise ValueError(f"unknown activation_mode: {activation_mode!r}")
            offset = end
            progress.set_postfix(tokens=end, selected=len(selected_positions), mode=activation_mode)

    feature_ids = np.arange(dict_size, dtype=np.int64)
    mean_positive = activation_sum / np.maximum(positive_count, 1)
    activation_mean = activation_sum / max(count, 1)
    positive_fraction = positive_count / max(count, 1)
    scores = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "selected_token_count": int(count),
            "otu_token_count_seen": int(token_type_count),
            "positive_count": positive_count,
            "positive_fraction": positive_fraction,
            "activation_sum": activation_sum,
            "activation_mean": activation_mean,
            "mean_positive_activation": mean_positive,
            "activation_max": activation_max,
        }
    )
    scores["interpretability_score"] = scores["activation_sum"]
    scores["activation_mode"] = activation_mode
    return scores
