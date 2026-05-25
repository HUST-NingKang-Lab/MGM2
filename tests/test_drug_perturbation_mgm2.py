import pickle
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from src.data.DrugPerturbationDataset import DrugPerturbationDataset, DrugPerturbationCollator, DrugPerturbationPreprocessedDataset
from src.models.DrugPerturbationModel import DrugPerturbationCrossAttentionModel


class DrugPerturbationMGM2Tests(unittest.TestCase):
    def _write_small_dataset(self, root: Path):
        with h5py.File(root / "otu_counts.h5", "w") as handle:
            rel = np.array(
                [
                    [0.2, 0.0, 0.4],
                    [0.0, 0.3, 0.1],
                    [0.0, 0.0, 0.5],
                ],
                dtype=np.float32,
            )
            handle.create_dataset("rel_abundance", data=rel)
            handle.create_dataset("counts", data=(rel * 1000).astype(np.int32))
            handle.create_dataset("otu_ids", data=np.array([b"99_1", b"99_2", b"99_3"]))
            handle.create_dataset("sample_ids", data=np.array([b"drug_run", b"ctrl_a", b"ctrl_b"]))

        pd.DataFrame(
            [
                {
                    "Run": "drug_run",
                    "Experiment": "Exp01",
                    "Donor": "SIC0",
                    "Condition": "Drug_A",
                    "Type": "drug",
                    "Plate": "1",
                    "Drug_Name_in_Library": "Drug A",
                    "SMILES": "CC",
                    "Drug_Target": "bacteria",
                },
                {
                    "Run": "ctrl_a",
                    "Experiment": "Exp01",
                    "Donor": "SIC0",
                    "Condition": "control",
                    "Type": "control",
                    "Plate": "1",
                    "Drug_Name_in_Library": "",
                    "SMILES": "",
                    "Drug_Target": "",
                },
                {
                    "Run": "ctrl_b",
                    "Experiment": "Exp01",
                    "Donor": "SIC0",
                    "Condition": "control",
                    "Type": "control",
                    "Plate": "1",
                    "Drug_Name_in_Library": "",
                    "SMILES": "",
                    "Drug_Target": "",
                },
            ]
        ).to_csv(root / "sample_drug_mapping.tsv", sep="\t", index=False)

        with h5py.File(root / "drug_embeddings.h5", "w") as handle:
            handle.create_dataset("embeddings", data=np.ones((1, 4), dtype=np.float32))
            handle.create_dataset("drug_name", data=np.array([b"Drug A"]))
            handle.create_dataset("condition", data=np.array([b"Drug_A"]))
            handle.create_dataset("smiles", data=np.array([b"CC"]))
            handle.create_dataset("drug_target", data=np.array([b"bacteria"]))
        pd.DataFrame(
            [{"Condition": "Drug_A", "Drug_Name_in_Library": "Drug A", "embedding_idx": 0}]
        ).to_csv(root / "drug_embedding_index.tsv", sep="\t", index=False)

        with h5py.File(root / "ntv3.h5", "w") as handle:
            handle.create_dataset("embeddings", data=np.arange(12, dtype=np.float32).reshape(3, 4))
            handle.create_dataset("otu_ids", data=np.array([b"99_1", b"99_2", b"99_3"]))

    def test_dataset_aligns_before_after_union_and_floors_missing_taxa(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_small_dataset(root)

            dataset = DrugPerturbationDataset(
                data_dir=root,
                ntv3_embedding_path=root / "ntv3.h5",
                max_otus=8,
            )
            sample = dataset[0]

            self.assertEqual(sample["sample_id"], "drug_run")
            self.assertEqual(sample["donor_id"].item(), 0)
            self.assertEqual(sample["drug_embedding"].shape, (4,))
            self.assertEqual(sample["input_ids"].tolist()[0], -1)
            self.assertEqual(len(sample["otu_ids"]), 3)
            self.assertTrue(torch.all(sample["supervision_mask"][1:]))
            self.assertTrue((sample["before_abundance_raw"][1:] >= 1e-4).all())
            self.assertTrue((sample["after_abundance_raw"][1:] >= 1e-4).all())

    def test_dataset_can_reuse_baseline_examples_and_fixed_selected_otus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_small_dataset(root)
            pd.DataFrame(
                [
                    {
                        "Run": "drug_run",
                        "sample_idx": 0,
                        "Donor": "SIC0",
                        "drug_key": "Drug A",
                        "embedding_idx": 0,
                        "control_runs": "ctrl_a",
                    }
                ]
            ).to_csv(root / "examples.csv", index=False)
            pd.DataFrame(
                {
                    "otu_idx": [2, 1, 0],
                    "otu_id": ["99_3", "99_2", "99_1"],
                    "abundance_sum": [1.0, 0.5, 0.2],
                }
            ).to_csv(root / "selected_otus.tsv", sep="\t", index=False)

            dataset = DrugPerturbationDataset(
                data_dir=root,
                ntv3_embedding_path=root / "ntv3.h5",
                examples_csv=root / "examples.csv",
                selected_otus_tsv=root / "selected_otus.tsv",
                max_otus=0,
            )
            sample = dataset[0]

            self.assertEqual(sample["otu_ids"], ["99_3", "99_2", "99_1"])
            self.assertEqual(sample["input_ids"].tolist(), [-1, 2, 1, 0])
            np.testing.assert_allclose(sample["after_abundance_raw"][1:].numpy(), [1e-4, 1e-4, 0.2])
            np.testing.assert_allclose(sample["before_abundance_raw"][1:].numpy(), [1e-4, 0.3, 1e-4])
            np.testing.assert_allclose(sample["after_abundance_eval_raw"][1:].numpy(), [0.0, 0.0, 0.2])
            np.testing.assert_allclose(sample["before_abundance_eval_raw"][1:].numpy(), [0.0, 0.3, 0.0])

    def test_dataset_filters_low_depth_controls_like_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_small_dataset(root)
            with h5py.File(root / "otu_counts.h5", "r+") as handle:
                handle["counts"][:, 1] = 0

            dataset = DrugPerturbationDataset(
                data_dir=root,
                ntv3_embedding_path=root / "ntv3.h5",
                max_otus=8,
                min_read_sum=100,
                min_control_read_sum=100,
            )

            self.assertEqual(dataset.examples[0]["control_indices"], [2])


    def test_preprocessed_dataset_reads_compact_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_small_dataset(root)
            raw_dataset = DrugPerturbationDataset(root, root / "ntv3.h5", max_otus=8, cache_aligned_examples=True)
            payload = {
                "format": "mgm2_drug_perturbation",
                "format_version": 1,
                "samples": [example["sample_id"] for example in raw_dataset.examples],
                "examples": [
                    {
                        "sample_id": example["sample_id"],
                        "donor": example["donor"],
                        "donor_id": example["donor_id"],
                        "drug_key": example["drug_key"],
                    }
                    for example in raw_dataset.examples
                ],
                "items": raw_dataset.items,
                "donor_to_id": raw_dataset.donor_to_id,
                "drug_embedding_dim": raw_dataset.drug_embeddings.shape[1],
                "ntv3_embedding_path": str(root / "ntv3.h5"),
            }
            pkl_path = root / "drug_perturbation.pkl"
            with pkl_path.open("wb") as handle:
                pickle.dump(payload, handle)

            prebuilt_dataset = DrugPerturbationPreprocessedDataset(pkl_path)
            sample = prebuilt_dataset[0]

            self.assertEqual(len(prebuilt_dataset), len(raw_dataset))
            self.assertEqual(sample["sample_id"], "drug_run")
            self.assertEqual(sample["drug_embedding"].shape, (4,))
            self.assertTrue(torch.equal(sample["input_ids"], raw_dataset[0]["input_ids"]))

    def test_preprocessed_dataset_uses_fixed_embeddings_without_ntv3_lookup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_small_dataset(root)
            raw_dataset = DrugPerturbationDataset(root, root / "ntv3.h5", max_otus=8, cache_aligned_examples=True)
            fixed_embeddings = np.full((raw_dataset[0]["input_ids"].shape[0], 4), 7.0, dtype=np.float32)
            payload = {
                "format": "mgm2_drug_perturbation",
                "format_version": 1,
                "samples": [example["sample_id"] for example in raw_dataset.examples],
                "examples": [
                    {
                        "sample_id": example["sample_id"],
                        "donor": example["donor"],
                        "donor_id": example["donor_id"],
                        "drug_key": example["drug_key"],
                    }
                    for example in raw_dataset.examples
                ],
                "items": raw_dataset.items,
                "donor_to_id": raw_dataset.donor_to_id,
                "drug_embedding_dim": raw_dataset.drug_embeddings.shape[1],
                "selected_otus": {"otu_idx": [0, 1, 2], "otu_id": ["99_1", "99_2", "99_3"], "abundance_sum": [1.0, 1.0, 1.0]},
                "fixed_embeddings": fixed_embeddings,
            }
            pkl_path = root / "drug_perturbation.pkl"
            with pkl_path.open("wb") as handle:
                pickle.dump(payload, handle)

            prebuilt_dataset = DrugPerturbationPreprocessedDataset(pkl_path)
            sample = prebuilt_dataset[0]
            batch = DrugPerturbationCollator()([sample, sample])

            self.assertIsNone(prebuilt_dataset.ntv3_embeddings)
            self.assertTrue(torch.equal(sample["embeddings"], torch.from_numpy(fixed_embeddings)))
            self.assertEqual(batch["embeddings"].shape, (2, fixed_embeddings.shape[0], fixed_embeddings.shape[1]))
            self.assertTrue(batch["embeddings"].is_contiguous())
            self.assertNotEqual(batch["embeddings"].untyped_storage().data_ptr(), sample["embeddings"].untyped_storage().data_ptr())

    def test_cross_attention_model_predicts_token_delta(self):
        model = DrugPerturbationCrossAttentionModel(
            model_config={
                "embedding_dim": 4,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "max_position_embeddings": 16,
            },
            drug_embedding_dim=4,
            num_donors=3,
            num_drug_tokens=2,
            cross_attention_heads=2,
        )
        outputs = model(
            input_ids=torch.tensor([[-1, 0, 1]]),
            embeddings=torch.zeros(1, 3, 4),
            before_abundance_values=torch.zeros(1, 3),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            drug_embedding=torch.ones(1, 4),
            donor_id=torch.tensor([1]),
        )

        self.assertEqual(outputs["pred_delta"].shape, (1, 3))
        self.assertEqual(outputs["pred_after_abundance"].shape, (1, 3))

    def test_collator_pads_perturbation_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_small_dataset(root)
            dataset = DrugPerturbationDataset(root, root / "ntv3.h5", max_otus=8)
            batch = DrugPerturbationCollator()([dataset[0], dataset[0]])

            self.assertEqual(batch["input_ids"].shape[0], 2)
            self.assertEqual(batch["drug_embedding"].shape, (2, 4))
            self.assertEqual(batch["donor_id"].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
