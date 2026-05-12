"""Dataset utilities for drug-conditioned microbiome perturbation prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.Preprocessing import AbundancePreprocessor


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


class DrugPerturbationDataset(Dataset):
    """Aligned before/after perturbation examples for MGM2-style models.

    Each example aligns the union of OTUs observed in matched controls and the
    drug-treated sample. Missing before/after taxa are assigned `min_abundance`
    in the raw abundance fields before applying the MGM2 abundance transform.
    """

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
        donor_to_id: Optional[Dict[str, int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.otu_counts_path = self.data_dir / "otu_counts.h5"
        self.ntv3_embedding_path = Path(ntv3_embedding_path)
        self.max_otus = int(max_otus)
        self.min_abundance = float(min_abundance)
        self.min_read_sum = int(min_read_sum)
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
        self.ntv3_embeddings, self.otu_id_to_ntv3_idx = self._load_ntv3_embeddings()
        self.cls_embedding = np.zeros(self.ntv3_embeddings.shape[1], dtype=np.float32)
        self.donor_to_id = donor_to_id or {
            donor: idx for idx, donor in enumerate(sorted(self.meta["Donor"].dropna().astype(str).unique()))
        }
        self.examples = self._build_examples()
        if not self.examples:
            raise ValueError("No usable drug perturbation examples were found")

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

    def _load_ntv3_embeddings(self):
        with h5py.File(self.ntv3_embedding_path, "r") as handle:
            embeddings = handle["embeddings"][:].astype(np.float32)
            otu_ids = _decode_h5_strings(handle["otu_ids"][:])
        return embeddings, {otu_id: idx for idx, otu_id in enumerate(otu_ids)}

    def _sample_read_sum(self, sample_idx: int) -> int:
        with h5py.File(self.otu_counts_path, "r") as handle:
            return int(handle["counts"][:, sample_idx].sum())

    def _build_examples(self) -> List[Dict[str, Any]]:
        examples = []
        for _, row in self.meta[self.meta["Type"] == "drug"].iterrows():
            run = str(row["Run"])
            drug_name = str(row.get("Drug_Name_in_Library", "")).strip()
            if run not in self.sample_to_idx or drug_name not in self.drug_name_to_idx:
                continue
            control_runs = _matched_controls(self.meta, row)
            control_indices = [self.sample_to_idx[item] for item in control_runs if item in self.sample_to_idx]
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
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

        before_raw = before[union_idx].astype(np.float32) if union_idx.size else np.array([], dtype=np.float32)
        after_raw = after[union_idx].astype(np.float32) if union_idx.size else np.array([], dtype=np.float32)
        before_present_raw = before_raw > 0
        after_present_raw = after_raw > 0
        before_raw = np.where(before_raw > 0, before_raw, self.min_abundance).astype(np.float32)
        after_raw = np.where(after_raw > 0, after_raw, self.min_abundance).astype(np.float32)

        before_values = self.preprocessor.transform(before_raw)
        after_values = self.preprocessor.transform(after_raw)
        embeddings = self.ntv3_embeddings[np.asarray(ntv3_indices, dtype=int)] if ntv3_indices else np.zeros((0, self.ntv3_embeddings.shape[1]), dtype=np.float32)

        input_ids = np.concatenate([np.array([-1], dtype=np.int64), union_idx.astype(np.int64)])
        embeddings = np.vstack([self.cls_embedding[None, :], embeddings]).astype(np.float32)
        before_values = np.concatenate([np.array([0.0], dtype=np.float32), before_values.astype(np.float32)])
        after_values = np.concatenate([np.array([0.0], dtype=np.float32), after_values.astype(np.float32)])
        before_raw = np.concatenate([np.array([0.0], dtype=np.float32), before_raw])
        after_raw = np.concatenate([np.array([0.0], dtype=np.float32), after_raw])
        before_present_mask = np.concatenate([np.array([False]), before_present_raw])
        after_present_mask = np.concatenate([np.array([False]), after_present_raw])
        supervision_mask = np.ones_like(before_present_mask, dtype=bool)
        supervision_mask[0] = False

        return {
            "sample_id": example["sample_id"],
            "donor": example["donor"],
            "drug_key": example["drug_key"],
            "otu_ids": otu_ids,
            "input_ids": torch.from_numpy(input_ids).long(),
            "embeddings": torch.from_numpy(embeddings).float(),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "before_abundance_values": torch.from_numpy(before_values).float(),
            "after_abundance_values": torch.from_numpy(after_values).float(),
            "before_abundance_raw": torch.from_numpy(before_raw).float(),
            "after_abundance_raw": torch.from_numpy(after_raw).float(),
            "before_present_mask": torch.from_numpy(before_present_mask).bool(),
            "after_present_mask": torch.from_numpy(after_present_mask).bool(),
            "supervision_mask": torch.from_numpy(supervision_mask).bool(),
            "drug_embedding": torch.from_numpy(self.drug_embeddings[example["drug_idx"]]).float(),
            "donor_id": torch.tensor(example["donor_id"], dtype=torch.long),
        }


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

        emb_dim = batch[0]["embeddings"].shape[1]
        padded_embeddings = []
        for item in batch:
            pad_len = max_len - item["embeddings"].shape[0]
            if pad_len > 0:
                padded_embeddings.append(torch.cat([item["embeddings"], torch.zeros(pad_len, emb_dim)], dim=0))
            else:
                padded_embeddings.append(item["embeddings"])
        result["embeddings"] = torch.stack(padded_embeddings)
        return result
