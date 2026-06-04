#!/usr/bin/env python3
"""Extract frozen MGM2 ASV-token hidden states for wastewater samples."""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.downstreams.wastewater.extract_mgm2_cls_cache import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DATASET,
    DEFAULT_FALLBACK,
    DEFAULT_NTV3,
    DEFAULT_STAGE2_ROOT,
    _collect_samples,
    _load_backbone,
    _load_fallback,
    _load_ntv3,
    _load_taxonomy,
    _resolve_plant_embeddings,
)
from src.data.Preprocessing import AbundancePreprocessor
from src.data.WastewaterDataset import load_wastewater_windows


DEFAULT_OUTPUT = Path("data/downstreams/wastewater/cache/mgm2_scale_small_top200_tokens.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Wastewater windows pickle")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT, help="Organized wastewater data root")
    parser.add_argument("--stage2_root", type=Path, default=DEFAULT_STAGE2_ROOT, help="Stage-2 grid root")
    parser.add_argument("--scale", type=str, default="scale_small", help="MGM2 scale directory name")
    parser.add_argument("--config", type=Path, default=None, help="Explicit MGM2 config path")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit MGM2 checkpoint path")
    parser.add_argument("--ntv3_embedding", type=Path, default=DEFAULT_NTV3, help="NTv3 embedding HDF5")
    parser.add_argument("--fallback_embedding", type=Path, default=DEFAULT_FALLBACK, help="Taxonomy fallback HDF5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output cache pickle")
    parser.add_argument("--plants", nargs="*", default=None, help="Optional plant subset")
    parser.add_argument("--batch_size", type=int, default=4, help="Sample batch size per plant")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--allow_missing_taxonomy", action="store_true", help="Allow missing taxonomy files")
    return parser.parse_args()


def _encode_token_batch(backbone, taxon_embeddings, abundances, preprocessor, device) -> np.ndarray:
    batch_size, num_asvs = abundances.shape
    cls_embeddings = np.zeros((batch_size, 1, taxon_embeddings.shape[1]), dtype=np.float32)
    repeated_embeddings = np.repeat(taxon_embeddings[None, :, :], batch_size, axis=0)
    embeddings = np.concatenate([cls_embeddings, repeated_embeddings], axis=1)
    processed = np.stack([preprocessor.transform(row.astype(np.float32)) for row in abundances])
    abundance_values = np.concatenate([np.zeros((batch_size, 1), dtype=np.float32), processed], axis=1)
    input_ids = np.zeros((batch_size, num_asvs + 1), dtype=np.int64)
    input_ids[:, 0] = -1
    attention_mask = np.ones((batch_size, num_asvs + 1), dtype=np.int64)

    with torch.no_grad():
        outputs = backbone(
            input_ids=torch.from_numpy(input_ids).to(device),
            embeddings=torch.from_numpy(embeddings).to(device),
            abundance_values=torch.from_numpy(abundance_values).to(device),
            attention_mask=torch.from_numpy(attention_mask).to(device),
        )
    return outputs["last_hidden_state"][:, 1:, :].detach().cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    config_path = args.config or (args.stage2_root / args.scale / "config.json")
    checkpoint_path = args.checkpoint or (args.stage2_root / args.scale / "checkpoints" / "last.ckpt")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = load_wastewater_windows(args.dataset)
    sample_map = _collect_samples(dataset, args.plants)
    if not sample_map:
        raise ValueError("no samples found for requested plants")

    backbone, config = _load_backbone(config_path, checkpoint_path, device)
    preprocessor = AbundancePreprocessor(
        min_abundance=config.abundance.min_abundance,
        output_min=config.abundance.output_min,
        output_max=config.abundance.output_max,
        normalize=config.abundance.normalize,
    )
    ntv3_index, ntv3_embeddings = _load_ntv3(args.ntv3_embedding)
    fallback_map, global_embedding = _load_fallback(args.fallback_embedding)

    by_plant = defaultdict(list)
    for key, sample in sample_map.items():
        by_plant[sample["plant"]].append((key, sample))

    sample_keys = []
    token_states = []
    source_counts = {}
    asv_universes = {}
    for plant, items in tqdm(sorted(by_plant.items()), desc="plants"):
        asv_ids = list(items[0][1]["asv_ids"])
        taxonomy = _load_taxonomy(args.data_root, plant, allow_missing=args.allow_missing_taxonomy)
        taxon_embeddings, counts = _resolve_plant_embeddings(
            asv_ids,
            taxonomy,
            ntv3_index,
            ntv3_embeddings,
            fallback_map,
            global_embedding,
            plant=plant,
        )
        source_counts[plant] = counts
        asv_universes[plant] = asv_ids
        for start in tqdm(range(0, len(items), args.batch_size), desc=plant, leave=False):
            batch_items = items[start : start + args.batch_size]
            batch_abundance = np.stack([sample["abundance"] for _, sample in batch_items]).astype(np.float32)
            batch_tokens = _encode_token_batch(backbone, taxon_embeddings, batch_abundance, preprocessor, device)
            for (key, _), states in zip(batch_items, batch_tokens):
                sample_keys.append(key)
                token_states.append(states)

    cache = {
        "version": 1,
        "kind": "mgm2_token_states",
        "dataset": str(args.dataset),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "scale": args.scale,
        "hidden_size": int(config.model.hidden_size),
        "sample_keys": sample_keys,
        "sample_to_idx": {key: idx for idx, key in enumerate(sample_keys)},
        "token_states": np.stack(token_states).astype(np.float32),
        "asv_universes": asv_universes,
        "embedding_source_counts": source_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved MGM2 token cache: {args.output}")
    print(f"Samples: {len(sample_keys)}")
    print(f"Token states: {cache['token_states'].shape}")
    print(f"Plants: {len(by_plant)}")
    print(f"Embedding source counts: {source_counts}")


if __name__ == "__main__":
    main()
