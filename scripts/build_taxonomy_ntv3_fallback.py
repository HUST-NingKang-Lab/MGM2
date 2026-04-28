#!/usr/bin/env python3
"""Build taxonomy-level NTv3 fallback embeddings from OTU metadata.

Example:
    python scripts/build_taxonomy_ntv3_fallback.py \
        --otu_info /data6/zhanghaohong/databases/microbeatals/otus.info.tsv \
        --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5 \
        --output ntv3_embeddings/otus_99_taxonomy_fallback.h5 \
        --summary_output ntv3_embeddings/otus_99_taxonomy_fallback_summary.tsv
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
DEFAULT_MIN_COUNTS = {
    "kingdom": 100,
    "phylum": 100,
    "class": 50,
    "order": 20,
    "family": 10,
    "genus": 5,
    "species": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build taxonomy mean NTv3 embeddings for OTUs without representative sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/build_taxonomy_ntv3_fallback.py "
            "--otu_info /data6/zhanghaohong/databases/microbeatals/otus.info.tsv "
            "--ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5 "
            "--output ntv3_embeddings/otus_99_taxonomy_fallback.h5"
        ),
    )
    parser.add_argument(
        "--otu_info",
        type=Path,
        default=Path("/data6/zhanghaohong/databases/microbeatals/otus.info.tsv"),
        help="MicrobeAtlas OTU metadata table with OTU and taxonomy columns",
    )
    parser.add_argument(
        "--ntv3_embedding",
        type=Path,
        default=Path("ntv3_embeddings/otus_99_embeddings.h5"),
        help="Existing NTv3 embedding HDF5 with datasets 'embeddings' and 'otu_ids'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ntv3_embeddings/otus_99_taxonomy_fallback.h5"),
        help="Output HDF5 for taxonomy fallback embeddings",
    )
    parser.add_argument(
        "--summary_output",
        type=Path,
        default=None,
        help="Optional TSV summary path; defaults to <output>.summary.tsv",
    )
    parser.add_argument(
        "--taxonomy_column",
        default="Tax",
        help="Column containing semicolon-separated taxonomy",
    )
    parser.add_argument(
        "--otu_column",
        default="OTU",
        help="Column containing OTU IDs; entries may include semicolon-separated 90/96/97/98/99 IDs",
    )
    parser.add_argument(
        "--species_column",
        default="RepSpecies",
        help="Optional species-level column appended after genus when present",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        default=list(DEFAULT_RANKS),
        help="Rank names matching taxonomy levels from broad to specific; MicrobeAtlas Tax level 1 is named kingdom by default",
    )
    parser.add_argument(
        "--min_count",
        action="append",
        default=[],
        metavar="RANK=N",
        help="Minimum NTv3-backed OTUs required to keep a taxon prototype; can be repeated",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200000,
        help="Rows per chunk when scanning the large OTU info table",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Debug option: read at most this many metadata rows",
    )
    parser.add_argument(
        "--normalize_to_ntv3_median_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rescale taxon mean vectors to the median norm of raw NTv3 embeddings",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def parse_min_counts(overrides: Sequence[str], ranks: Sequence[str]) -> Dict[str, int]:
    min_counts = {rank: int(DEFAULT_MIN_COUNTS.get(rank, 1)) for rank in ranks}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--min_count must use RANK=N format, got: {item}")
        rank, value = item.split("=", 1)
        rank = rank.strip()
        if rank not in min_counts:
            raise ValueError(f"Unknown rank in --min_count: {rank}")
        min_counts[rank] = int(value)
    return min_counts


def normalize_otu_id(value) -> Optional[str]:
    """Extract the canonical 99_* OTU id used by otus_99_embeddings.h5."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    for part in text.split(";"):
        part = part.strip()
        if part.startswith("99_"):
            return part
    if "99_" in text:
        return "99_" + text.split("99_", 1)[1].split(";", 1)[0].strip()
    return text


def clean_taxon(value) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "unclassified"}:
        return ""
    return text


def parse_taxonomy(taxonomy, ranks: Sequence[str], species: Optional[str] = None) -> List[Tuple[str, str]]:
    """Return cumulative taxonomy paths: [('domain', 'Bacteria'), ('phylum', 'Bacteria;Firmicutes'), ...]."""
    if pd.isna(taxonomy):
        levels: List[str] = []
    else:
        levels = [clean_taxon(part) for part in str(taxonomy).split(";")]
        levels = [part for part in levels if part]

    species_name = clean_taxon(species)
    if species_name and len(levels) < len(ranks):
        levels.append(species_name.split(",", 1)[0].strip())

    paths: List[Tuple[str, str]] = []
    prefix: List[str] = []
    for rank, level in zip(ranks, levels):
        prefix.append(level)
        paths.append((rank, ";".join(prefix)))
    return paths


def taxon_key(rank: str, path: str) -> str:
    return f"{rank}|{path}"


def decode_h5_strings(values) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def load_ntv3_index(path: Path) -> Tuple[np.ndarray, List[str]]:
    logger.info("Loading NTv3 embeddings from %s", path)
    with h5py.File(path, "r") as handle:
        if "embeddings" not in handle or "otu_ids" not in handle:
            raise KeyError("NTv3 HDF5 must contain 'embeddings' and 'otu_ids' datasets")
        embeddings = handle["embeddings"][:].astype(np.float32)
        otu_ids = decode_h5_strings(handle["otu_ids"][:])
    logger.info("Loaded %s embeddings with dim %s", embeddings.shape[0], embeddings.shape[1])
    return embeddings, otu_ids


def read_taxonomy_for_ntv3_ids(
    otu_info_path: Path,
    wanted_ids: set,
    otu_column: str,
    taxonomy_column: str,
    species_column: str,
    chunk_size: int,
    max_rows: Optional[int] = None,
) -> Dict[str, Tuple[str, str]]:
    """Read only taxonomy rows whose canonical OTU id exists in the NTv3 file."""
    logger.info("Scanning OTU metadata from %s", otu_info_path)
    taxonomy_by_id: Dict[str, Tuple[str, str]] = {}
    usecols = [otu_column, taxonomy_column]
    if species_column:
        usecols.append(species_column)

    rows_seen = 0
    reader = pd.read_csv(
        otu_info_path,
        sep="\t",
        usecols=lambda col: col in set(usecols),
        chunksize=chunk_size,
        nrows=max_rows,
        low_memory=False,
    )
    for chunk in reader:
        rows_seen += len(chunk)
        missing = [col for col in [otu_column, taxonomy_column] if col not in chunk.columns]
        if missing:
            raise KeyError(f"Missing required column(s) in {otu_info_path}: {missing}")

        canonical_ids = chunk[otu_column].map(normalize_otu_id)
        keep_mask = canonical_ids.isin(wanted_ids)
        if keep_mask.any():
            kept = chunk.loc[keep_mask].copy()
            kept["_canonical_otu_id"] = canonical_ids.loc[keep_mask].values
            species_values = kept[species_column] if species_column in kept.columns else [""] * len(kept)
            for otu_id, taxonomy, species in zip(
                kept["_canonical_otu_id"],
                kept[taxonomy_column],
                species_values,
            ):
                if otu_id in taxonomy_by_id:
                    continue
                taxonomy_by_id[otu_id] = (taxonomy, species)

        logger.info(
            "Scanned %s rows, matched %s / %s NTv3 OTUs",
            rows_seen,
            len(taxonomy_by_id),
            len(wanted_ids),
        )
    return taxonomy_by_id


def build_taxon_sums(
    embeddings: np.ndarray,
    otu_ids: Sequence[str],
    taxonomy_by_id: Dict[str, Tuple[str, str]],
    ranks: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, Tuple[str, str]], int]:
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = defaultdict(int)
    metadata: Dict[str, Tuple[str, str]] = {}
    matched_otu_count = 0

    for idx, otu_id in enumerate(otu_ids):
        taxonomy_pair = taxonomy_by_id.get(otu_id)
        if taxonomy_pair is None:
            continue
        taxonomy, species = taxonomy_pair
        paths = parse_taxonomy(taxonomy, ranks, species)
        if not paths:
            continue
        matched_otu_count += 1
        vector = embeddings[idx]
        for rank, path in paths:
            key = taxon_key(rank, path)
            if key not in sums:
                sums[key] = np.zeros(embeddings.shape[1], dtype=np.float64)
                metadata[key] = (rank, path)
            sums[key] += vector
            counts[key] += 1

    return sums, counts, metadata, matched_otu_count


def finalize_taxon_embeddings(
    sums: Dict[str, np.ndarray],
    counts: Dict[str, int],
    metadata: Dict[str, Tuple[str, str]],
    min_counts: Dict[str, int],
    target_norm: Optional[float],
) -> Tuple[np.ndarray, List[str], List[str], List[str], np.ndarray]:
    rows = []
    for key, vector_sum in sums.items():
        rank, path = metadata[key]
        count = counts[key]
        if count < min_counts.get(rank, 1):
            continue
        mean_vector = (vector_sum / float(count)).astype(np.float32)
        if target_norm is not None:
            norm = float(np.linalg.norm(mean_vector))
            if norm > 0:
                mean_vector = mean_vector * (target_norm / norm)
        rows.append((rank, path, key, count, mean_vector))

    rows.sort(key=lambda item: (list(DEFAULT_RANKS).index(item[0]) if item[0] in DEFAULT_RANKS else 999, item[1]))
    if not rows:
        raise ValueError("No taxonomy fallback embeddings passed the min_count filters")

    ranks = [row[0] for row in rows]
    paths = [row[1] for row in rows]
    keys = [row[2] for row in rows]
    counts_array = np.asarray([row[3] for row in rows], dtype=np.int64)
    embeddings = np.stack([row[4] for row in rows]).astype(np.float32)
    return embeddings, keys, ranks, paths, counts_array


def write_output(
    output_path: Path,
    embeddings: np.ndarray,
    keys: Sequence[str],
    ranks: Sequence[str],
    paths: Sequence[str],
    counts: np.ndarray,
    global_embedding: np.ndarray,
    attrs: Dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("taxon_embeddings", data=embeddings, compression="gzip")
        handle.create_dataset("taxon_keys", data=np.asarray(keys, dtype=object), dtype=string_dtype)
        handle.create_dataset("taxon_ranks", data=np.asarray(ranks, dtype=object), dtype=string_dtype)
        handle.create_dataset("taxon_paths", data=np.asarray(paths, dtype=object), dtype=string_dtype)
        handle.create_dataset("taxon_counts", data=counts)
        handle.create_dataset("global_embedding", data=global_embedding.astype(np.float32))
        for key, value in attrs.items():
            handle.attrs[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)


def write_summary(summary_path: Path, keys: Sequence[str], ranks: Sequence[str], paths: Sequence[str], counts: np.ndarray):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "taxon_key": keys,
            "rank": ranks,
            "taxon_path": paths,
            "ntv3_otu_count": counts.astype(int),
        }
    )
    frame.to_csv(summary_path, sep="\t", index=False)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    summary_output = args.summary_output or args.output.with_suffix(".summary.tsv")

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Pass --overwrite to replace it.")
    if summary_output.exists() and not args.overwrite:
        raise FileExistsError(f"Summary output exists: {summary_output}. Pass --overwrite to replace it.")

    min_counts = parse_min_counts(args.min_count, args.ranks)
    embeddings, otu_ids = load_ntv3_index(args.ntv3_embedding)
    wanted_ids = set(otu_ids)
    taxonomy_by_id = read_taxonomy_for_ntv3_ids(
        otu_info_path=args.otu_info,
        wanted_ids=wanted_ids,
        otu_column=args.otu_column,
        taxonomy_column=args.taxonomy_column,
        species_column=args.species_column,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )

    sums, counts, metadata, matched_otu_count = build_taxon_sums(embeddings, otu_ids, taxonomy_by_id, args.ranks)
    raw_norms = np.linalg.norm(embeddings, axis=1)
    target_norm = float(np.median(raw_norms)) if args.normalize_to_ntv3_median_norm else None
    taxon_embeddings, keys, ranks, paths, count_array = finalize_taxon_embeddings(
        sums=sums,
        counts=counts,
        metadata=metadata,
        min_counts=min_counts,
        target_norm=target_norm,
    )
    global_embedding = embeddings.mean(axis=0)
    if target_norm is not None:
        norm = float(np.linalg.norm(global_embedding))
        if norm > 0:
            global_embedding = global_embedding * (target_norm / norm)

    attrs = {
        "source_otu_info": str(args.otu_info),
        "source_ntv3_embedding": str(args.ntv3_embedding),
        "taxonomy_column": args.taxonomy_column,
        "otu_column": args.otu_column,
        "species_column": args.species_column,
        "ranks": list(args.ranks),
        "min_counts": min_counts,
        "normalize_to_ntv3_median_norm": bool(args.normalize_to_ntv3_median_norm),
        "target_norm": float(target_norm) if target_norm is not None else "",
        "ntv3_otu_count": int(len(otu_ids)),
        "matched_otu_count": int(matched_otu_count),
        "matched_otu_fraction": float(matched_otu_count / max(1, len(otu_ids))),
        "taxon_embedding_count": int(len(keys)),
    }
    write_output(args.output, taxon_embeddings, keys, ranks, paths, count_array, global_embedding, attrs)
    write_summary(summary_output, keys, ranks, paths, count_array)

    rank_summary = pd.Series(ranks).value_counts().reindex(args.ranks).fillna(0).astype(int)
    logger.info("Saved taxonomy fallback embeddings to %s", args.output)
    logger.info("Saved summary table to %s", summary_output)
    logger.info("Matched OTUs: %s / %s (%.2f%%)", matched_otu_count, len(otu_ids), 100 * attrs["matched_otu_fraction"])
    logger.info("Taxon prototypes by rank: %s", rank_summary.to_dict())


if __name__ == "__main__":
    main()
