#!/usr/bin/env python3
"""Utilities for Cell Reports FMT downstream data."""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
KINGDOM_FILES = {
    "bacteria": "bacteria_combined_summary.xlsx",
    "archaea": "archaea_combined_summary.xlsx",
    "fungi": "fungi_combined_summary.xlsx",
}
RESPONSE_LABELS = {0: "non-responder", 1: "responder", 2: "missing"}


def normalize_sample_id(value) -> str:
    text = str(value).strip()
    text = re.sub(r"_(?:combined|R\d+|S\d+)$", "", text, flags=re.I)
    return re.sub(r"_\d+$", "", text)


def sample_prefix(sample_id: str) -> str:
    match = re.match(r"^([A-Za-z]+)", str(sample_id))
    return match.group(1) if match else "unknown"


def make_cohort_label(disease_code, pre_sample: str, post_sample: str, donor_sample: str) -> str:
    triplet = "_".join([sample_prefix(pre_sample), sample_prefix(post_sample), sample_prefix(donor_sample)])
    return f"disease_{int(disease_code)}__{triplet}"


def assign_consecutive_disease_cohorts(disease_codes: Iterable[object]) -> List[str]:
    """Assign cohort labels by run-length segments of DiseasesClass values."""
    labels: List[str] = []
    previous = None
    cohort_index = 0
    for value in disease_codes:
        disease_code = int(float(value))
        if previous is None or disease_code != previous:
            cohort_index += 1
            previous = disease_code
        labels.append(f"cohort_{cohort_index:03d}__disease_{disease_code}")
    return labels


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def clean_feature_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    return text


def make_feature_id(kingdom: str, feature_name: str) -> str:
    return f"cellreports|{str(kingdom).lower()}|{clean_feature_name(feature_name)}"


def parse_feature_id(feature_id: str) -> Tuple[str, str]:
    parts = str(feature_id).split("|", 2)
    if len(parts) != 3 or parts[0] != "cellreports":
        raise ValueError(f"Not a Cell Reports feature id: {feature_id}")
    return parts[1], parts[2]


def feature_genus(feature_name: str) -> str:
    text = clean_feature_name(feature_name)
    text = re.sub(r"^uncultured\s+", "", text, flags=re.I)
    parts = text.split()
    if not parts:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]", "", parts[0])


def read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    values: List[str] = []
    with zf.open("xl/sharedStrings.xml") as handle:
        for _, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag == f"{NS}si":
                texts = []
                for child in elem.iter(f"{NS}t"):
                    texts.append(child.text or "")
                values.append(html.unescape("".join(texts)))
                elem.clear()
    return values


def cell_col_index(cell_ref: str) -> int:
    letters = re.match(r"^([A-Z]+)", cell_ref).group(1)
    idx = 0
    for char in letters:
        idx = idx * 26 + ord(char) - ord("A") + 1
    return idx - 1


def xlsx_rows(path: Path) -> Iterator[List[object]]:
    """Yield rows from the first worksheet without requiring openpyxl."""
    with zipfile.ZipFile(path) as zf:
        shared = read_shared_strings(zf)
        with zf.open("xl/worksheets/sheet1.xml") as handle:
            for _, row in ET.iterparse(handle, events=("end",)):
                if row.tag != f"{NS}row":
                    continue
                cells: Dict[int, object] = {}
                max_col = -1
                for cell in row.findall(f"{NS}c"):
                    ref = cell.attrib.get("r", "A1")
                    col_idx = cell_col_index(ref)
                    max_col = max(max_col, col_idx)
                    cell_type = cell.attrib.get("t", "")
                    value: object = ""
                    if cell_type == "inlineStr":
                        texts = [node.text or "" for node in cell.iter(f"{NS}t")]
                        value = html.unescape("".join(texts))
                    else:
                        node = cell.find(f"{NS}v")
                        if node is not None and node.text is not None:
                            raw = node.text
                            if cell_type == "s":
                                value = shared[int(raw)]
                            else:
                                try:
                                    value = float(raw)
                                except ValueError:
                                    value = html.unescape(raw)
                    cells[col_idx] = value
                if max_col >= 0:
                    yield [cells.get(idx, "") for idx in range(max_col + 1)]
                row.clear()


def xlsx_first_column_values(path: Path, skip_header: bool = True) -> List[str]:
    """Fast path for sample-id coverage: extract only A-column inline strings."""
    values: List[str] = []
    pattern = re.compile(r'<c[^>]*r="A(\d+)"[^>]*>(.*?)</c>', flags=re.S)
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("xl/worksheets/sheet1.xml").decode("utf-8", errors="replace")
    for match in pattern.finditer(xml):
        if skip_header and match.group(1) == "1":
            continue
        body = match.group(2)
        text_match = re.search(r"<t[^>]*>(.*?)</t>", body, flags=re.S)
        value_match = re.search(r"<v>(.*?)</v>", body, flags=re.S)
        value = text_match.group(1) if text_match else (value_match.group(1) if value_match else "")
        if value:
            values.append(html.unescape(value))
    return values


def read_xlsx_table(path: Path) -> pd.DataFrame:
    rows = list(xlsx_rows(path))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    header = [str(value) for value in rows[0]]
    return pd.DataFrame(rows[1:], columns=header)


def load_metadata(path: Path) -> pd.DataFrame:
    frame = read_xlsx_table(path)
    frame = frame.rename(
        columns={
            frame.columns[0]: "response_code",
            frame.columns[1]: "recipient_sample_id",
            frame.columns[2]: "post_sample_id",
            frame.columns[3]: "donor_sample_id",
            frame.columns[4]: "disease_code",
            frame.columns[5]: "gender_code",
            frame.columns[6]: "age",
        }
    )
    for column in ["recipient_sample_id", "post_sample_id", "donor_sample_id"]:
        frame[column] = frame[column].map(normalize_sample_id)
    frame["response_code"] = frame["response_code"].astype(float).astype(int)
    frame["disease_code"] = frame["disease_code"].astype(float).astype(int)
    frame["response_label"] = frame["response_code"].map(RESPONSE_LABELS).fillna("missing")
    frame["fmt_id"] = [
        f"cellreports_{idx:04d}_{row.recipient_sample_id}_{row.post_sample_id}_{row.donor_sample_id}"
        for idx, row in frame.reset_index(drop=True).iterrows()
    ]
    frame["cohort"] = assign_consecutive_disease_cohorts(frame["disease_code"].tolist())
    frame["study"] = frame["cohort"]
    frame["indication"] = frame["disease_code"].map(lambda value: f"disease_{int(value)}")
    return frame


def read_summary_workbook(path: Path, kingdom: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Read a Cell Reports summary workbook as sample ids, feature ids, matrix."""
    iterator = xlsx_rows(path)
    header = next(iterator)
    raw_features = [clean_feature_name(value) for value in header[1:]]
    feature_ids = [make_feature_id(kingdom, feature) for feature in raw_features]
    sample_rows: Dict[str, List[np.ndarray]] = {}
    for row in iterator:
        if not row:
            continue
        sample_id = normalize_sample_id(row[0])
        values = np.asarray([float(value or 0.0) for value in row[1 : len(raw_features) + 1]], dtype=np.float32)
        if values.shape[0] < len(raw_features):
            values = np.pad(values, (0, len(raw_features) - values.shape[0]))
        sample_rows.setdefault(sample_id, []).append(values)
    sample_ids = sorted(sample_rows)
    matrix = np.stack([np.stack(sample_rows[sample], axis=0).mean(axis=0) for sample in sample_ids]).astype(np.float32)
    return sample_ids, feature_ids, matrix


def write_h5_community(path: Path, sample_ids: List[str], feature_ids: List[str], matrix: np.ndarray) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        group = handle.create_group("cellreports")
        group.create_dataset("axis0", data=np.asarray(sample_ids, dtype=object), dtype=string_dtype)
        group.create_dataset("axis1", data=np.asarray(feature_ids, dtype=object), dtype=string_dtype)
        group.create_dataset("block0_values", data=matrix.T.astype(np.float32), compression="gzip", chunks=True)
