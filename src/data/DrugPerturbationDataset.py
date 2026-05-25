"""Dataset utilities for drug-conditioned microbiome perturbation prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.Preprocessing import AbundancePreprocessor
from src.data.utils import load_preprocessed_data


def _decode_h5_strings(values) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def _matched_controls(meta: pd.DataFrame, drug_row: pd.Series) -> List[str]:
    controls = meta[meta["Type"] == "control"]
    if drug_row["Experiment"] == "Exp06":
        mask = (
            (controls["Experiment"] == drug_row["Experiment"])
            & (controls["Donor"] == drug_row["Donor"])
            & (controls["Plate"].astype(str) == str(drug_row["Plate"]))
            & (controls["Condition"] == "DMSO")
        )
    else:
        mask = (
            (controls["Experiment"] == drug_row["Experiment"])
            & (controls["Donor"] == drug_row["Donor"])
            & (controls["Plate"].astype(str) == str(drug_row["Plate"]))
            & (controls["Condition"].isin(["control", "Ethanol_control"]))
        )
    return controls.loc[mask, "Run"].astype(str).tolist()


def _load_ntv3_embeddings(path: str | Path):
    with h5py.File(path, "r") as handle:
        embeddings = handle["embeddings"][:].astype(np.float32)
        otu_ids = _decode_h5_strings(handle["otu_ids"][:])
    return embeddings, {otu_id: idx for idx, otu_id in enumerate(otu_ids)}


def _materialize_item(
    item: Dict[str, Any],
    ntv3_embeddings: np.ndarray | None,
    cls_embedding: np.ndarray | None,
    fixed_embeddings: torch.Tensor | None = None,
) -> Dict[str, Any]:
    if fixed_embeddings is not None:
        embeddings = fixed_embeddings
    else:
        if ntv3_embeddings is None or cls_embedding is None:
            raise ValueError("ntv3_embeddings and cls_embedding are required when fixed_embeddings is not provided")
        ntv3_indices = np.asarray(item["ntv3_indices"], dtype=np.int64)
        if ntv3_indices.size:
            otu_embeddings = np.zeros((ntv3_indices.size, ntv3_embeddings.shape[1]), dtype=np.float32)
            valid = ntv3_indices >= 0
            if valid.any():
                otu_embeddings[valid] = ntv3_embeddings[ntv3_indices[valid]]
        else:
            otu_embeddings = np.zeros((0, ntv3_embeddings.shape[1]), dtype=np.float32)
        embeddings = torch.from_numpy(np.vstack([cls_embedding[None, :], otu_embeddings]).astype(np.float32)).float()
    return {
        "sample_id": item["sample_id"],
        "donor": item["donor"],
        "drug_key": item["drug_key"],
        "otu_ids": item["otu_ids"],
        "input_ids": torch.as_tensor(item["input_ids"], dtype=torch.long),
        "embeddings": embeddings if torch.is_tensor(embeddings) else torch.from_numpy(embeddings).float(),
        "attention_mask": torch.as_tensor(item["attention_mask"], dtype=torch.long),
        "before_abundance_values": torch.as_tensor(item["before_abundance_values"], dtype=torch.float32),
        "after_abundance_values": torch.as_tensor(item["after_abundance_values"], dtype=torch.float32),
        "before_abundance_raw": torch.as_tensor(item["before_abundance_raw"], dtype=torch.float32),
        "after_abundance_raw": torch.as_tensor(item["after_abundance_raw"], dtype=torch.float32),
        "before_abundance_eval_raw": torch.as_tensor(item.get("before_abundance_eval_raw", item["before_abundance_raw"]), dtype=torch.float32),
        "after_abundance_eval_raw": torch.as_tensor(item.get("after_abundance_eval_raw", item["after_abundance_raw"]), dtype=torch.float32),
        "before_present_mask": torch.as_tensor(item["before_present_mask"], dtype=torch.bool),
        "after_present_mask": torch.as_tensor(item["after_present_mask"], dtype=torch.bool),
        "supervision_mask": torch.as_tensor(item["supervision_mask"], dtype=torch.bool),
        "drug_embedding": torch.as_tensor(item["drug_embedding"], dtype=torch.float32),
        "donor_id": torch.tensor(int(item["donor_id"]), dtype=torch.long),
    }


class DrugPerturbationDataset(Dataset):
    """Build aligned before/after perturbation examples from raw HDF5 inputs."""

    def __init__(
        self,
        data_dir: str | Path,
        ntv3_embedding_path: str | Path,
        max_otus: int = 768,
        min_abundance: float = 1e-4,
        output_min: float = -2.0,
        output_max: float = 2.0,
        normalize: bool = False,
        min_read_sum: int = 100,
        min_control_read_sum: int = 100,
        donor_to_id: Optional[Dict[str, int]] = None,
        cache_aligned_examples: bool = False,
        examples_csv: str | Path | None = None,
        selected_otus_tsv: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.otu_counts_path = self.data_dir / "otu_counts.h5"
        self.ntv3_embedding_path = Path(ntv3_embedding_path)
        self.max_otus = int(max_otus)
        self.min_abundance = float(min_abundance)
        self.min_read_sum = int(min_read_sum)
        self.min_control_read_sum = int(min_control_read_sum)
        self.cache_aligned_examples = bool(cache_aligned_examples)
        self.examples_csv = Path(examples_csv) if examples_csv is not None else None
        self.selected_otus_tsv = Path(selected_otus_tsv) if selected_otus_tsv is not None else None
        self.preprocessor = AbundancePreprocessor(
            min_abundance=min_abundance,
            output_min=output_min,
            output_max=output_max,
            normalize=normalize,
        )
        self.preprocessor.fit([])

        self.meta = pd.read_csv(self.data_dir / "sample_drug_mapping.tsv", sep="\t")
        self.drug_embeddings, self.drug_name_to_idx = self._load_drug_embeddings()
        self.sample_ids, self.sample_to_idx, self.otu_ids = self._load_otu_index()
        self.ntv3_embeddings, self.otu_id_to_ntv3_idx = _load_ntv3_embeddings(self.ntv3_embedding_path)
        self.selected_otus = self._load_selected_otus(self.selected_otus_tsv) if self.selected_otus_tsv is not None else None
        if self.selected_otus is not None and self.max_otus > 0:
            self.selected_otus = self.selected_otus.head(self.max_otus).copy()
        self.selected_otu_indices = self.selected_otus["otu_idx"].astype(int).to_numpy() if self.selected_otus is not None else None
        self.selected_otu_ids = self.selected_otus["otu_id"].astype(str).to_list() if self.selected_otus is not None else None
        self.cls_embedding = np.zeros(self.ntv3_embeddings.shape[1], dtype=np.float32)
        self.donor_to_id = donor_to_id or {
            donor: idx for idx, donor in enumerate(sorted(self.meta["Donor"].dropna().astype(str).unique()))
        }
        self.examples = self._build_examples()
        if not self.examples:
            raise ValueError("No usable drug perturbation examples were found")
        self.items = [self._build_item_arrays(example) for example in self.examples] if self.cache_aligned_examples else None

    def _load_drug_embeddings(self):
        with h5py.File(self.data_dir / "drug_embeddings.h5", "r") as handle:
            embeddings = handle["embeddings"][:].astype(np.float32)
            drug_names = _decode_h5_strings(handle["drug_name"][:])
        return embeddings, {name: idx for idx, name in enumerate(drug_names)}

    def _load_otu_index(self):
        with h5py.File(self.otu_counts_path, "r") as handle:
            sample_ids = _decode_h5_strings(handle["sample_ids"][:])
            otu_ids = _decode_h5_strings(handle["otu_ids"][:])
        return sample_ids, {sample_id: idx for idx, sample_id in enumerate(sample_ids)}, otu_ids

    @staticmethod
    def _load_selected_otus(path: str | Path) -> pd.DataFrame:
        selected = pd.read_csv(path, sep="\t")
        required = {"otu_idx", "otu_id"}
        missing = required.difference(selected.columns)
        if missing:
            raise ValueError(f"selected OTU table is missing columns: {sorted(missing)}")
        return selected.reset_index(drop=True)

    def _sample_read_sum(self, sample_idx: int) -> int:
        if not hasattr(self, "_read_sum_cache"):
            self._read_sum_cache = {}
        sample_idx = int(sample_idx)
        if sample_idx not in self._read_sum_cache:
            with h5py.File(self.otu_counts_path, "r") as handle:
                self._read_sum_cache[sample_idx] = int(handle["counts"][:, sample_idx].sum())
        return self._read_sum_cache[sample_idx]

    def _control_indices_from_runs(self, raw: Any) -> List[int]:
        indices = []
        for run in str(raw or "").split(";"):
            run = run.strip()
            if run and run in self.sample_to_idx:
                indices.append(self.sample_to_idx[run])
        return indices

    def _build_examples_from_csv(self, path: str | Path) -> List[Dict[str, Any]]:
        frame = pd.read_csv(path)
        examples = []
        for _, row in frame.iterrows():
            run = str(row.get("Run", row.get("sample_id", ""))).strip()
            if not run or run not in self.sample_to_idx:
                continue
            control_indices = self._control_indices_from_runs(row.get("control_runs", ""))
            if not control_indices:
                continue
            sample_idx = int(row.get("sample_idx", self.sample_to_idx[run]))
            drug_key = str(row.get("drug_key", row.get("Drug_Name_in_Library", ""))).strip()
            if not drug_key:
                drug_key = str(row.get("Condition", "")).strip()
            if "embedding_idx" in row and pd.notna(row["embedding_idx"]):
                drug_idx = int(row["embedding_idx"])
            elif drug_key in self.drug_name_to_idx:
                drug_idx = self.drug_name_to_idx[drug_key]
            else:
                continue
            donor = str(row.get("Donor", row.get("donor", "")))
            if donor not in self.donor_to_id:
                self.donor_to_id[donor] = len(self.donor_to_id)
            examples.append({
                "sample_id": run,
                "sample_idx": sample_idx,
                "control_indices": control_indices,
                "donor": donor,
                "donor_id": self.donor_to_id[donor],
                "drug_key": drug_key,
                "drug_idx": drug_idx,
            })
        return examples

    def _build_examples(self) -> List[Dict[str, Any]]:
        if self.examples_csv is not None:
            return self._build_examples_from_csv(self.examples_csv)
        examples = []
        for _, row in self.meta[self.meta["Type"] == "drug"].iterrows():
            run = str(row["Run"])
            drug_name = str(row.get("Drug_Name_in_Library", "")).strip()
            if drug_name not in self.drug_name_to_idx:
                condition_name = str(row.get("Condition", "")).strip()
                if condition_name in self.drug_name_to_idx:
                    drug_name = condition_name
            if run not in self.sample_to_idx or drug_name not in self.drug_name_to_idx:
                continue
            control_runs = _matched_controls(self.meta, row)
            control_indices = []
            for item in control_runs:
                if item not in self.sample_to_idx:
                    continue
                control_idx = self.sample_to_idx[item]
                if self.min_control_read_sum > 0 and self._sample_read_sum(control_idx) < self.min_control_read_sum:
                    continue
                control_indices.append(control_idx)
            if not control_indices:
                continue
            sample_idx = self.sample_to_idx[run]
            if self.min_read_sum > 0 and self._sample_read_sum(sample_idx) < self.min_read_sum:
                continue
            examples.append(
                {
                    "sample_id": run,
                    "sample_idx": sample_idx,
                    "control_indices": control_indices,
                    "donor": str(row["Donor"]),
                    "donor_id": self.donor_to_id[str(row["Donor"])],
                    "drug_key": drug_name,
                    "drug_idx": self.drug_name_to_idx[drug_name],
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _load_profiles(self, sample_idx: int, control_indices: Sequence[int]):
        with h5py.File(self.otu_counts_path, "r") as handle:
            rel = handle["rel_abundance"]
            after = rel[:, sample_idx].astype(np.float32)
            before = rel[:, sorted(control_indices)].mean(axis=1).astype(np.float32)
        return before, after

    def _load_selected_profiles(self, sample_idx: int, control_indices: Sequence[int]):
        union_idx = self.selected_otu_indices.astype(int).copy()
        sorted_idx = np.sort(union_idx)
        order = np.argsort(union_idx)
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        sample_indices = [int(sample_idx)] + sorted(int(idx) for idx in control_indices)
        with h5py.File(self.otu_counts_path, "r") as handle:
            block = handle["rel_abundance"][sorted_idx, :][:, sample_indices]
        block = block[inverse_order].astype(np.float32)
        after = block[:, 0]
        before = block[:, 1:].mean(axis=1).astype(np.float32)
        return union_idx, before, after

    def _build_item_arrays(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if self.selected_otu_indices is not None:
            union_idx, before_eval_raw, after_eval_raw = self._load_selected_profiles(
                example["sample_idx"],
                example["control_indices"],
            )
            otu_ids = list(self.selected_otu_ids)
            ntv3_indices = [self.otu_id_to_ntv3_idx.get(otu_id, -1) for otu_id in otu_ids]
        else:
            before, after = self._load_profiles(example["sample_idx"], example["control_indices"])
            before_present = before > 0
            after_present = after > 0
            union_idx = np.flatnonzero(before_present | after_present)
            if union_idx.size == 0:
                union_idx = np.array([], dtype=int)
            if self.max_otus > 0 and union_idx.size > self.max_otus:
                score = np.maximum(before[union_idx], after[union_idx])
                keep = np.argpartition(-score, self.max_otus - 1)[: self.max_otus]
                union_idx = union_idx[keep]
                union_idx = union_idx[np.argsort(-score[keep])]

            otu_ids = [self.otu_ids[int(i)] for i in union_idx]
            ntv3_indices = [self.otu_id_to_ntv3_idx[otu_id] for otu_id in otu_ids if otu_id in self.otu_id_to_ntv3_idx]
            kept_positions = [pos for pos, otu_id in enumerate(otu_ids) if otu_id in self.otu_id_to_ntv3_idx]
            union_idx = union_idx[np.asarray(kept_positions, dtype=int)] if kept_positions else np.array([], dtype=int)
            otu_ids = [otu_ids[pos] for pos in kept_positions]
            before_eval_raw = before[union_idx].astype(np.float32) if union_idx.size else np.array([], dtype=np.float32)
            after_eval_raw = after[union_idx].astype(np.float32) if union_idx.size else np.array([], dtype=np.float32)

        before_present_raw = before_eval_raw > 0
        after_present_raw = after_eval_raw > 0
        before_raw = np.where(before_eval_raw > 0, before_eval_raw, self.min_abundance).astype(np.float32)
        after_raw = np.where(after_eval_raw > 0, after_eval_raw, self.min_abundance).astype(np.float32)

        before_values = self.preprocessor.transform(before_raw)
        after_values = self.preprocessor.transform(after_raw)
        input_ids = np.concatenate([np.array([-1], dtype=np.int64), union_idx.astype(np.int64)])
        before_values = np.concatenate([np.array([0.0], dtype=np.float32), before_values.astype(np.float32)])
        after_values = np.concatenate([np.array([0.0], dtype=np.float32), after_values.astype(np.float32)])
        before_raw = np.concatenate([np.array([0.0], dtype=np.float32), before_raw])
        after_raw = np.concatenate([np.array([0.0], dtype=np.float32), after_raw])
        before_eval_raw = np.concatenate([np.array([0.0], dtype=np.float32), before_eval_raw])
        after_eval_raw = np.concatenate([np.array([0.0], dtype=np.float32), after_eval_raw])
        before_present_mask = np.concatenate([np.array([False]), before_present_raw])
        after_present_mask = np.concatenate([np.array([False]), after_present_raw])
        supervision_mask = np.ones_like(before_present_mask, dtype=bool)
        supervision_mask[0] = False

        return {
            "sample_id": example["sample_id"],
            "donor": example["donor"],
            "drug_key": example["drug_key"],
            "otu_ids": otu_ids,
            "ntv3_indices": np.asarray(ntv3_indices, dtype=np.int64),
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": np.ones(len(input_ids), dtype=np.int64),
            "before_abundance_values": before_values.astype(np.float32),
            "after_abundance_values": after_values.astype(np.float32),
            "before_abundance_raw": before_raw.astype(np.float32),
            "after_abundance_raw": after_raw.astype(np.float32),
            "before_abundance_eval_raw": before_eval_raw.astype(np.float32),
            "after_abundance_eval_raw": after_eval_raw.astype(np.float32),
            "before_present_mask": before_present_mask.astype(bool),
            "after_present_mask": after_present_mask.astype(bool),
            "supervision_mask": supervision_mask.astype(bool),
            "drug_embedding": self.drug_embeddings[example["drug_idx"]].astype(np.float32),
            "donor_id": int(example["donor_id"]),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx] if self.items is not None else self._build_item_arrays(self.examples[idx])
        return _materialize_item(item, self.ntv3_embeddings, self.cls_embedding)


class DrugPerturbationPreprocessedDataset(Dataset):
    """Drug perturbation dataset backed by a compact prebuilt pickle."""

    def __init__(self, preprocessed_data: str | Path | Dict[str, Any], ntv3_embedding_path: str | Path | None = None):
        data = load_preprocessed_data(str(preprocessed_data)) if isinstance(preprocessed_data, (str, Path)) else preprocessed_data
        self.data = data
        self.items = data["items"]
        self.examples = data["examples"]
        self.samples = data["samples"]
        self.donor_to_id = data["donor_to_id"]
        self.drug_embedding_dim = int(data["drug_embedding_dim"])
        self.selected_otus = pd.DataFrame(data["selected_otus"]) if "selected_otus" in data else None
        self.fixed_embeddings = (
            torch.as_tensor(data["fixed_embeddings"], dtype=torch.float32)
            if "fixed_embeddings" in data and data["fixed_embeddings"] is not None
            else None
        )
        embedding_path = ntv3_embedding_path or data.get("ntv3_embedding_path")
        if self.fixed_embeddings is None:
            if embedding_path is None:
                raise ValueError("ntv3_embedding_path is required for preprocessed drug perturbation data")
            self.ntv3_embedding_path = Path(embedding_path)
            self.ntv3_embeddings, _ = _load_ntv3_embeddings(self.ntv3_embedding_path)
            self.cls_embedding = np.zeros(self.ntv3_embeddings.shape[1], dtype=np.float32)
        else:
            self.ntv3_embedding_path = Path(embedding_path) if embedding_path is not None else None
            self.ntv3_embeddings = None
            self.cls_embedding = None
        self.drug_embeddings = np.zeros((1, self.drug_embedding_dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return _materialize_item(
            self.items[idx],
            self.ntv3_embeddings,
            self.cls_embedding,
            fixed_embeddings=self.fixed_embeddings,
        )


class DrugPerturbationCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        result: Dict[str, Any] = {
            "sample_id": [item["sample_id"] for item in batch],
            "donor": [item["donor"] for item in batch],
            "drug_key": [item["drug_key"] for item in batch],
            "otu_ids": [item["otu_ids"] for item in batch],
            "drug_embedding": torch.stack([item["drug_embedding"] for item in batch]),
            "donor_id": torch.stack([item["donor_id"] for item in batch]),
        }
        for key in [
            "input_ids",
            "attention_mask",
            "before_abundance_values",
            "after_abundance_values",
            "before_abundance_raw",
            "after_abundance_raw",
            "before_abundance_eval_raw",
            "after_abundance_eval_raw",
            "before_present_mask",
            "after_present_mask",
            "supervision_mask",
        ]:
            values = [item[key] for item in batch]
            pad_value = -1 if key == "input_ids" else 0
            if values[0].dtype == torch.bool:
                pad_value = False
            padded = []
            for value in values:
                pad_len = max_len - value.shape[0]
                if pad_len > 0:
                    padded.append(torch.cat([value, torch.full((pad_len,), pad_value, dtype=value.dtype)]))
                else:
                    padded.append(value)
            result[key] = torch.stack(padded)

        first_embeddings = batch[0]["embeddings"]
        if all(item["embeddings"] is first_embeddings for item in batch) and first_embeddings.shape[0] == max_len:
            result["embeddings"] = first_embeddings.unsqueeze(0).repeat(len(batch), 1, 1).contiguous()
        else:
            emb_dim = first_embeddings.shape[1]
            padded_embeddings = []
            for item in batch:
                pad_len = max_len - item["embeddings"].shape[0]
                if pad_len > 0:
                    padded_embeddings.append(torch.cat([item["embeddings"], torch.zeros(pad_len, emb_dim)], dim=0))
                else:
                    padded_embeddings.append(item["embeddings"])
            result["embeddings"] = torch.stack(padded_embeddings)
        return result
