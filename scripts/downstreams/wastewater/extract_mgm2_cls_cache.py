#!/usr/bin/env python3
"""Extract frozen MGM2 CLS embeddings for wastewater samples."""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.configs.Config import MGM2Config
from src.data.Preprocessing import AbundancePreprocessor
from src.data.WastewaterDataset import load_wastewater_windows
from src.models.BaseModel import BaseModel


DEFAULT_DATASET = Path("data/downstreams/wastewater/wastewater_windows_ctx10_h10.pkl")
DEFAULT_STAGE2_ROOT = Path("outputs_grid/stage2/20260505_094404")
DEFAULT_DATA_ROOT = Path("/home/zhanghaohong/data6/databases/wastewater/MC-prediction/data_organized")
DEFAULT_NTV3 = Path("ntv3_embeddings/otus_99_embeddings.h5")
DEFAULT_FALLBACK = Path("ntv3_embeddings/otus_99_taxonomy_fallback.h5")
DEFAULT_OUTPUT = Path("data/downstreams/wastewater/cache/mgm2_scale_small_cls.pkl")

RANKS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
RANK_KEY = {
    "Kingdom": "kingdom",
    "Phylum": "phylum",
    "Class": "class",
    "Order": "order",
    "Family": "family",
    "Genus": "genus",
    "Species": "species",
}


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
    parser.add_argument("--batch_size", type=int, default=8, help="Sample batch size per plant")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument(
        "--allow_missing_taxonomy",
        action="store_true",
        help="Allow missing plant taxonomy files and fall back to global embeddings. Off by default.",
    )
    return parser.parse_args()


def _decode_h5_strings(values) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def _clean_taxon(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "unclassified"}:
        return ""
    if "__" in text:
        text = text.split("__", 1)[1].strip()
    return text


def _taxonomy_keys(row: pd.Series) -> List[str]:
    keys = []
    path = []
    for rank in RANKS:
        name = _clean_taxon(row.get(rank, ""))
        if not name:
            continue
        path.append(name)
        keys.append(f"{RANK_KEY[rank]}|{';'.join(path)}")
    return keys


def _load_ntv3(path: Path):
    with h5py.File(path, "r") as handle:
        ids = _decode_h5_strings(handle["otu_ids"][:])
        embeddings = handle["embeddings"][:].astype(np.float32)
    return {otu_id: idx for idx, otu_id in enumerate(ids)}, embeddings


def _load_fallback(path: Path):
    with h5py.File(path, "r") as handle:
        keys = _decode_h5_strings(handle["taxon_keys"][:])
        embeddings = handle["taxon_embeddings"][:].astype(np.float32)
        global_embedding = handle["global_embedding"][:].astype(np.float32)
    return {key: embeddings[idx] for idx, key in enumerate(keys)}, global_embedding


def _load_taxonomy(data_root: Path, plant: str, allow_missing: bool = False) -> pd.DataFrame:
    frames = []
    searched_paths = []
    for split in ("train", "val", "test"):
        path = data_root / plant / split / "taxonomy.csv"
        searched_paths.append(path)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        if not allow_missing:
            paths = ", ".join(str(path) for path in searched_paths)
            raise FileNotFoundError(f"no taxonomy files found for plant {plant!r}; searched: {paths}")
        return pd.DataFrame(columns=["OTU", *RANKS])
    taxonomy = pd.concat(frames, ignore_index=True)
    taxonomy["OTU"] = taxonomy["OTU"].astype(str)
    return taxonomy.drop_duplicates("OTU", keep="first").set_index("OTU", drop=False)


def _resolve_plant_embeddings(
    asv_ids: Sequence[str],
    taxonomy: pd.DataFrame,
    ntv3_index: Dict[str, int],
    ntv3_embeddings: np.ndarray,
    fallback_map: Dict[str, np.ndarray],
    global_embedding: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    resolved = []
    sources = Counter()
    for asv_id in asv_ids:
        asv_id = str(asv_id)
        if asv_id in ntv3_index:
            resolved.append(ntv3_embeddings[ntv3_index[asv_id]])
            sources["ntv3"] += 1
            continue
        embedding = None
        if asv_id in taxonomy.index:
            for key in reversed(_taxonomy_keys(taxonomy.loc[asv_id])):
                if key in fallback_map:
                    embedding = fallback_map[key]
                    sources[f"fallback:{key.split('|', 1)[0]}"] += 1
                    break
        if embedding is None:
            embedding = global_embedding
            sources["fallback:global"] += 1
        resolved.append(embedding)
    return np.stack(resolved).astype(np.float32), dict(sources)


def _load_backbone(config_path: Path, checkpoint_path: Path, device: torch.device) -> Tuple[BaseModel, MGM2Config]:
    config = MGM2Config.load(str(config_path))
    model_cfg = config.model
    backbone = BaseModel(
        vocab_size=model_cfg.vocab_size,
        embedding_dim=model_cfg.embedding_dim,
        hidden_size=model_cfg.hidden_size,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        intermediate_size=model_cfg.intermediate_size,
        hidden_dropout_prob=model_cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=model_cfg.attention_probs_dropout_prob,
        layer_norm_eps=model_cfg.layer_norm_eps,
        max_position_embeddings=model_cfg.max_position_embeddings,
        pad_token_id=model_cfg.pad_token_id,
        cls_token_id=model_cfg.cls_token_id,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    backbone_state = {
        key[len("model.backbone.") :]: value
        for key, value in state_dict.items()
        if key.startswith("model.backbone.")
    }
    if not backbone_state:
        raise ValueError(f"checkpoint has no model.backbone.* keys: {checkpoint_path}")
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.to(device)
    backbone.eval()
    return backbone, config


def _collect_samples(dataset: Dict, plants: Sequence[str] | None):
    selected = set(plants) if plants else None
    samples = {}
    for window in dataset["windows"]:
        plant = window["plant"]
        if selected is not None and plant not in selected:
            continue
        asv_ids = tuple(window["asv_ids"])
        for sample_id, abundance in zip(window["context_sample_ids"], window["context_abundance"]):
            samples.setdefault((plant, sample_id), {"plant": plant, "sample_id": sample_id, "asv_ids": asv_ids, "abundance": abundance})
        for sample_id, abundance in zip(window["target_sample_ids"], window["target_abundance"]):
            samples.setdefault((plant, sample_id), {"plant": plant, "sample_id": sample_id, "asv_ids": asv_ids, "abundance": abundance})
    return samples


def _encode_batch(
    backbone: BaseModel,
    taxon_embeddings: np.ndarray,
    abundances: np.ndarray,
    preprocessor: AbundancePreprocessor,
    device: torch.device,
) -> np.ndarray:
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
    return outputs["last_hidden_state"][:, 0, :].detach().cpu().numpy().astype(np.float32)


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
    cls_embeddings = []
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
        )
        source_counts[plant] = counts
        asv_universes[plant] = asv_ids
        for start in tqdm(range(0, len(items), args.batch_size), desc=plant, leave=False):
            batch_items = items[start : start + args.batch_size]
            batch_abundance = np.stack([sample["abundance"] for _, sample in batch_items]).astype(np.float32)
            batch_cls = _encode_batch(backbone, taxon_embeddings, batch_abundance, preprocessor, device)
            for (key, _), cls in zip(batch_items, batch_cls):
                sample_keys.append(key)
                cls_embeddings.append(cls)

    cache = {
        "version": 1,
        "dataset": str(args.dataset),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "scale": args.scale,
        "hidden_size": int(config.model.hidden_size),
        "sample_keys": sample_keys,
        "sample_to_idx": {key: idx for idx, key in enumerate(sample_keys)},
        "cls_embeddings": np.stack(cls_embeddings).astype(np.float32),
        "asv_universes": asv_universes,
        "embedding_source_counts": source_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved MGM2 CLS cache: {args.output}")
    print(f"Samples: {len(sample_keys)}")
    print(f"Hidden size: {cache['hidden_size']}")
    print(f"Plants: {len(by_plant)}")
    print(f"Embedding source counts: {source_counts}")


if __name__ == "__main__":
    main()
