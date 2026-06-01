#!/usr/bin/env python3
"""Build direct NTv3-derived embeddings for Cell Reports feature ids."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellreports_utils import feature_genus, parse_feature_id
import build_taxonomy_ntv3_fallback as fb


KINGDOM_ALIASES = {
    "bacteria": {"bacteria"},
    "archaea": {"archaea"},
    "fungi": {"fungi", "eukaryota"},
}


def clean(value: str) -> str:
    return str(value).strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--features", type=Path, default=Path("data/downstreams/FMT_cellreports/cellreports_features.tsv"))
    parser.add_argument("--ntv3_embedding", type=Path, default=Path("ntv3_embeddings/otus_99_embeddings.h5"))
    parser.add_argument("--otu_info", type=Path, default=Path("/data9/zhanghaohong/databases/microbeatals/otus.info.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/downstreams/FMT_cellreports/community_all.cellreports_embeddings.h5"))
    parser.add_argument("--summary", type=Path, default=Path("data/downstreams/FMT_cellreports/community_all.cellreports_embeddings.summary.tsv"))
    parser.add_argument("--chunk_size", type=int, default=200000)
    return parser.parse_args()


def load_ntv3(path: Path) -> Tuple[np.ndarray, List[str]]:
    with h5py.File(path, "r") as handle:
        embeddings = handle["embeddings"][:].astype(np.float32)
        otu_ids = fb.decode_h5_strings(handle["otu_ids"][:])
    return embeddings, otu_ids


def taxonomy_kingdom(taxonomy: str) -> str:
    parts = [part.strip() for part in str(taxonomy).split(";") if str(part).strip()]
    return clean(parts[0]) if parts else ""


def species_name(taxonomy: str, rep_species: str) -> str:
    rep = str(rep_species).split(",", 1)[0].strip()
    if rep and rep.lower() not in {"nan", "none", "unknown", "unclassified"}:
        return rep
    parts = [part.strip() for part in str(taxonomy).split(";") if str(part).strip()]
    return parts[-1] if parts else ""


def build_maps(args: argparse.Namespace, embeddings: np.ndarray, otu_ids: List[str]):
    wanted = set(otu_ids)
    otu_to_idx = {otu_id: idx for idx, otu_id in enumerate(otu_ids)}
    species_map: Dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    genus_map: Dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    kingdom_map: Dict[str, list[np.ndarray]] = defaultdict(list)

    reader = pd.read_csv(
        args.otu_info,
        sep="\t",
        chunksize=args.chunk_size,
        usecols=lambda col: col in {"OTU", "Tax", "RepSpecies"},
        low_memory=False,
    )
    for chunk in reader:
        canonical = chunk["OTU"].map(fb.normalize_otu_id)
        keep = canonical.isin(wanted)
        if not keep.any():
            continue
        kept = chunk.loc[keep].copy()
        kept_canonical = canonical.loc[keep].astype(str).to_numpy()
        tax_values = kept["Tax"].fillna("").astype(str).to_numpy()
        species_values = kept["RepSpecies"].fillna("").astype(str).to_numpy() if "RepSpecies" in kept.columns else np.full(len(kept), "", dtype=object)
        for otu_id, taxonomy, rep_species in zip(kept_canonical, tax_values, species_values):
            vector = embeddings[otu_to_idx[otu_id]]
            kingdom = taxonomy_kingdom(taxonomy)
            species = species_name(taxonomy, rep_species)
            genus = feature_genus(species)
            if species:
                species_map[(kingdom, clean(species))].append(vector)
                species_map[("", clean(species))].append(vector)
            if genus:
                genus_map[(kingdom, clean(genus))].append(vector)
                genus_map[("", clean(genus))].append(vector)
            if kingdom:
                kingdom_map[kingdom].append(vector)
    global_embedding = embeddings.mean(axis=0).astype(np.float32)
    return species_map, genus_map, kingdom_map, global_embedding


def mean_vector(values: list[np.ndarray]) -> np.ndarray:
    return np.stack(values, axis=0).mean(axis=0).astype(np.float32)


def resolve_feature(kingdom: str, name: str, species_map, genus_map, kingdom_map, global_embedding):
    expected = KINGDOM_ALIASES.get(kingdom, {kingdom})
    species_key = clean(name)
    genus_key = clean(feature_genus(name))
    for tax_kingdom in expected:
        values = species_map.get((tax_kingdom, species_key))
        if values:
            return mean_vector(values), f"species:{tax_kingdom}", len(values)
    values = species_map.get(("", species_key))
    if values:
        return mean_vector(values), "species:any", len(values)
    for tax_kingdom in expected:
        values = genus_map.get((tax_kingdom, genus_key))
        if values:
            return mean_vector(values), f"genus:{tax_kingdom}", len(values)
    values = genus_map.get(("", genus_key))
    if values:
        return mean_vector(values), "genus:any", len(values)
    for tax_kingdom in expected:
        values = kingdom_map.get(tax_kingdom)
        if values:
            return mean_vector(values), f"kingdom:{tax_kingdom}", len(values)
    return global_embedding.astype(np.float32), "global", 0


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features, sep="\t")
    if "feature_id" not in features.columns:
        raise KeyError(f"{args.features} must contain feature_id")
    embeddings, otu_ids = load_ntv3(args.ntv3_embedding)
    species_map, genus_map, kingdom_map, global_embedding = build_maps(args, embeddings, otu_ids)
    rows = []
    vectors = []
    for feature_id in features["feature_id"].astype(str):
        kingdom, name = parse_feature_id(feature_id)
        vector, source, count = resolve_feature(kingdom, name, species_map, genus_map, kingdom_map, global_embedding)
        vectors.append(vector)
        rows.append({"feature_id": feature_id, "kingdom": kingdom, "feature_name": name, "embedding_source": source, "ntv3_otu_count": count})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("embeddings", data=np.stack(vectors).astype(np.float32), compression="gzip")
        handle.create_dataset("otu_ids", data=np.asarray(features["feature_id"].astype(str).tolist(), dtype=object), dtype=string_dtype)
        handle.attrs["source_ntv3_embedding"] = str(args.ntv3_embedding)
        handle.attrs["source_otu_info"] = str(args.otu_info)
        handle.attrs["resolution"] = "cellreports_species_genus_direct"
    summary = pd.DataFrame(rows)
    summary.to_csv(args.summary, sep="\t", index=False)
    print(json.dumps({"n_features": int(len(summary)), "source_counts": summary["embedding_source"].value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
