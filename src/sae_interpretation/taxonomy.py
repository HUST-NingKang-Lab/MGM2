"""Taxonomy helpers for MicrobeAtlas OTU identifiers."""

from __future__ import annotations

import re
from typing import Dict


RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def canonical_otu_id(value) -> str:
    """Return the canonical 99-level OTU id when a multi-rank alias is provided."""
    text = "" if value is None else str(value)
    matches = re.findall(r"99_[A-Za-z0-9_.-]+", text)
    if matches:
        return matches[-1]
    return text.strip()


def clean_taxon(value) -> str:
    text = "" if value is None else str(value).strip()
    if "__" in text and len(text) > 3:
        text = text.split("__", 1)[1]
    text = text.strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return "unknown"
    return text


def taxonomy_specificity(taxonomy) -> int:
    """Count non-empty, non-unknown taxonomic ranks."""
    parsed = parse_taxonomy(taxonomy)
    return sum(1 for value in parsed.values() if value and value != "unknown")


def parse_taxonomy(taxonomy) -> Dict[str, str]:
    parts = [clean_taxon(part) for part in str(taxonomy or "").split(";") if str(part).strip()]
    parsed = {}
    for idx, rank in enumerate(RANKS):
        parsed[rank] = parts[idx] if idx < len(parts) else "unknown"
    return parsed


def most_specific_taxon(row: Dict[str, str]) -> str:
    for rank in reversed(RANKS):
        value = row.get(rank, "unknown")
        if value and value != "unknown":
            return f"{rank}:{value}"
    return "unknown"
