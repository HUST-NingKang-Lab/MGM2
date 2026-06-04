"""Data loading helpers for SAE interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .taxonomy import canonical_otu_id, taxonomy_specificity


def load_sample_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "sample_id",
        "split",
        "quota_group",
        "primary_env",
        "env_first2",
        "technology",
        "project",
        "institution",
        "keywords",
        "nonzero_otu_count",
    ]
    return df[[col for col in keep if col in df.columns]].drop_duplicates("sample_id")


def load_taxonomy_map(path: Path, wanted_otu_ids: Iterable[str] | None = None, chunksize: int = 200_000) -> Dict[str, dict]:
    wanted = set(wanted_otu_ids) if wanted_otu_ids is not None else None
    taxonomy: Dict[str, dict] = {}
    usecols = ["OTU", "Tax", "RepSpecies", "Taxaname", "OrigTax"]
    for chunk in pd.read_csv(path, sep="\t", usecols=lambda col: col in usecols, chunksize=chunksize):
        if "OTU" not in chunk:
            continue
        chunk["_canonical_otu_id"] = chunk["OTU"].map(canonical_otu_id)
        if wanted is not None:
            chunk = chunk[chunk["_canonical_otu_id"].isin(wanted)]
        for row in chunk.to_dict(orient="records"):
            otu_id = row["_canonical_otu_id"]
            tax = row.get("Tax", "")
            taxonomy[otu_id] = {
                "taxonomy": "" if pd.isna(tax) else str(tax),
                "rep_species": "" if pd.isna(row.get("RepSpecies", "")) else str(row.get("RepSpecies", "")),
                "taxaname": "" if pd.isna(row.get("Taxaname", "")) else str(row.get("Taxaname", "")),
                "orig_tax": "" if pd.isna(row.get("OrigTax", "")) else str(row.get("OrigTax", "")),
            }
        if wanted is not None and len(taxonomy) >= len(wanted):
            break
    return taxonomy


def load_allinfo_taxonomy_map(
    path: Path,
    wanted_otu_ids: Iterable[str] | None = None,
    chunksize: int = 200_000,
) -> Dict[str, dict]:
    """Load taxonomy from MicrobeAtlas otu99 allinfo.

    The file is headerless. Column 0 contains multi-rank OTU aliases, column 8
    contains GTDB-style taxonomy, and column 14 contains an older taxonomy.
    """
    wanted = set(wanted_otu_ids) if wanted_otu_ids is not None else None
    taxonomy: Dict[str, dict] = {}
    usecols = [0, 8, 14]
    for chunk in pd.read_csv(path, sep="\t", header=None, usecols=usecols, chunksize=chunksize):
        chunk["_canonical_otu_id"] = chunk[0].map(canonical_otu_id)
        if wanted is not None:
            chunk = chunk[chunk["_canonical_otu_id"].isin(wanted)]
        for row in chunk.to_dict(orient="records"):
            otu_id = row["_canonical_otu_id"]
            gtdb = "" if pd.isna(row.get(8, "")) else str(row.get(8, ""))
            legacy = "" if pd.isna(row.get(14, "")) else str(row.get(14, ""))
            if taxonomy_specificity(gtdb) >= taxonomy_specificity(legacy):
                tax = gtdb
                source = "otu99_allinfo_gtdb"
            else:
                tax = legacy
                source = "otu99_allinfo_legacy"
            taxonomy[otu_id] = {
                "taxonomy": tax,
                "taxonomy_source": source,
                "allinfo_gtdb_taxonomy": gtdb,
                "allinfo_legacy_taxonomy": legacy,
            }
        if wanted is not None and len(taxonomy) >= len(wanted):
            break
    return taxonomy


def load_combined_taxonomy_map(
    allinfo_path: Path | None,
    fallback_path: Path | None,
    wanted_otu_ids: Iterable[str] | None = None,
    chunksize: int = 200_000,
) -> Dict[str, dict]:
    """Load taxonomy using otu99 allinfo first, then fill gaps with fallback OTU info."""
    combined: Dict[str, dict] = {}
    if allinfo_path is not None and Path(allinfo_path).exists():
        combined.update(load_allinfo_taxonomy_map(Path(allinfo_path), wanted_otu_ids, chunksize=chunksize))
    fallback = {}
    if fallback_path is not None and Path(fallback_path).exists():
        fallback = load_taxonomy_map(Path(fallback_path), wanted_otu_ids, chunksize=chunksize)
    for otu_id, info in fallback.items():
        current = combined.get(otu_id)
        if current is None or taxonomy_specificity(info.get("taxonomy", "")) > taxonomy_specificity(current.get("taxonomy", "")):
            merged = dict(info)
            merged["taxonomy_source"] = "otus_info_tsv"
            combined[otu_id] = merged
        else:
            current.setdefault("fallback_taxonomy", info.get("taxonomy", ""))
            current.setdefault("rep_species", info.get("rep_species", ""))
            current.setdefault("taxaname", info.get("taxaname", ""))
            current.setdefault("orig_tax", info.get("orig_tax", ""))
    return combined


def collect_wanted_otu_ids(tokens_path: Path, splits: set[str], chunksize: int = 500_000) -> set[str]:
    wanted = set()
    usecols = ["split", "token_type", "otu_id"]
    for chunk in pd.read_csv(tokens_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[(chunk["token_type"] == "otu") & (chunk["split"].isin(splits))]
        wanted.update(chunk["otu_id"].dropna().astype(str).unique().tolist())
    return wanted
