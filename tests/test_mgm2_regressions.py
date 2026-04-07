import importlib
import importlib.util
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch

from src.configs.Config import MGM2Config
from src.data.build_dataset import MGM2DatasetBuilder
from src.data.Dataset import BatchCollatorMGM2, MicrobiomeDatasetMGM2, create_dataloaders_mgm2
from src.data.Preprocessing import AbundancePreprocessor, BinManager
from src.data.utils import load_ntv3_embeddings, load_preprocessed_data
from src.losses.MultiTaskLoss import MGM2MultiTaskLoss
from src.models import MGM2Model
from src.models.BaseModel import BaseModel
from src.training.TrainerModule import MGM2LightningModule


class MGM2RegressionTests(unittest.TestCase):
    def _create_embedding_file(self, path: Path, otu_ids, embedding_dim: int = 4):
        embeddings = np.arange(len(otu_ids) * embedding_dim, dtype=np.float32).reshape(len(otu_ids), embedding_dim)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("embeddings", data=embeddings)
            handle.create_dataset("otu_ids", data=np.array(otu_ids, dtype="S32"))

    def _create_qwen3_embedding_file(self, path: Path, sample_ids, embedding_dim: int = 4):
        embeddings = np.arange(len(sample_ids) * embedding_dim, dtype=np.float32).reshape(len(sample_ids), embedding_dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        with h5py.File(path, "w") as handle:
            handle.create_dataset("embeddings", data=embeddings.astype(np.float32))
            dt = h5py.special_dtype(vlen=str)
            handle.create_dataset("sample_ids", data=np.array(sample_ids, dtype=object), dtype=dt)

    def _build_dataset_components(self, embedding_dim: int = 3):
        tmpdir = tempfile.TemporaryDirectory()
        embedding_path = Path(tmpdir.name) / "embeddings.h5"
        self._create_embedding_file(embedding_path, ["otu_a", "otu_b"], embedding_dim=embedding_dim)

        data = {
            "samples": ["s1"],
            "otu_data": {
                "s1": {
                    "input_ids": np.array([-1, 0, 1], dtype=np.int64),
                    "attention_mask": np.array([1, 1, 1], dtype=np.int64),
                    "abundance_values": np.array([0.0, 0.01, 0.2], dtype=np.float32),
                }
            },
            "otu_id_to_idx": {"otu_a": 0, "otu_b": 1},
            "ntv3_embedding_path": str(embedding_path),
        }

        raw_abundance = [np.array([0.0, 0.01, 0.2], dtype=np.float32)]
        preprocessor = AbundancePreprocessor()
        preprocessor.fit(raw_abundance)
        processed_train = preprocessor.fit_transform(raw_abundance)
        manager = BinManager(num_bins=5, method="quantile")
        manager.fit(processed_train)
        return tmpdir, data, preprocessor, manager

    def _create_preprocessed_dataset_file(self, path: Path, sample_count: int = 6):
        payload = {
            "samples": [f"sample_{i}" for i in range(sample_count)],
            "otu_data": {},
            "otu_id_to_idx": {"otu_a": 0, "otu_b": 1},
            "format_version": "mgm2_v1",
        }
        for i in range(sample_count):
            payload["otu_data"][f"sample_{i}"] = {
                "input_ids": np.array([-1, 0, 1], dtype=np.int64),
                "attention_mask": np.array([1, 1, 1], dtype=np.int64),
                "abundance_values": np.array([0.0, 0.01 * (i + 1), 0.1], dtype=np.float32),
            }

        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    def test_load_preprocessed_data_accepts_mgm2_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "dataset.pkl"
            payload = {
                "samples": ["s1"],
                "otu_data": {
                    "s1": {
                        "input_ids": np.array([-1, 1], dtype=np.int64),
                        "attention_mask": np.array([1, 1], dtype=np.int64),
                        "abundance_values": np.array([0.0, 0.5], dtype=np.float32),
                    }
                },
                "otu_id_to_idx": {"otu_1": 1},
                "format_version": "mgm2_v1",
            }
            with open(data_path, "wb") as handle:
                pickle.dump(payload, handle)

            loaded = load_preprocessed_data(str(data_path))

            self.assertEqual(["s1"], loaded["samples"])
            self.assertIn("otu_data", loaded)

    def test_fixed_threshold_labels_stay_in_range(self):
        manager = BinManager(
            num_bins=5,
            method="fixed_threshold",
            fixed_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        manager.fit([np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32)])

        labels, _ = manager.transform(np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32))

        self.assertLessEqual(int(labels.max()), 4)

    def test_quantile_binning_is_sample_wise_and_ignores_global_fit_thresholds(self):
        sample = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        manager_a = BinManager(num_bins=5, method="quantile")
        manager_a.fit([np.array([0.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)])
        labels_a, _ = manager_a.transform(sample)

        manager_b = BinManager(num_bins=5, method="quantile")
        manager_b.fit([np.array([0.0, -2.0, -1.5, -1.0, -0.5], dtype=np.float32)])
        labels_b, _ = manager_b.transform(sample)

        self.assertTrue(np.array_equal(labels_a, labels_b))

    def test_build_dataset_top_k_otus_keeps_highest_relative_abundance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_path = Path(tmpdir) / "embeddings.h5"
            self._create_embedding_file(embedding_path, ["otu_a", "otu_b", "otu_c"], embedding_dim=3)

            builder = MGM2DatasetBuilder(str(embedding_path), embedding_dim=3, top_k_otus=2)
            builder.load_ntv3_embeddings()
            sample = builder.build_sample(
                ["otu_a", "otu_b", "otu_c"],
                np.array([0.6, 0.1, 0.3], dtype=np.float32),
            )

            self.assertEqual(sample["input_ids"].tolist(), [-1, 0, 2])
            self.assertTrue(np.allclose(sample["abundance_values"], np.array([0.0, 0.6, 0.3], dtype=np.float32)))

    def test_build_dataset_top_k_otus_zero_keeps_all_nonzero_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_path = Path(tmpdir) / "embeddings.h5"
            self._create_embedding_file(embedding_path, ["otu_a", "otu_b", "otu_c"], embedding_dim=3)

            builder = MGM2DatasetBuilder(str(embedding_path), embedding_dim=3, top_k_otus=0)
            builder.load_ntv3_embeddings()
            sample = builder.build_sample(
                ["otu_a", "otu_b", "otu_c"],
                np.array([0.6, 0.1, 0.3], dtype=np.float32),
            )

            self.assertEqual(sample["input_ids"].tolist(), [-1, 0, 1, 2])

    def test_build_dataset_stores_top_k_otus_in_preprocessing_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_path = Path(tmpdir) / "embeddings.h5"
            otu_table_path = Path(tmpdir) / "otu_table.csv"
            output_path = Path(tmpdir) / "dataset.pkl"
            self._create_embedding_file(embedding_path, ["otu_a", "otu_b", "otu_c"], embedding_dim=3)

            otu_table = pd.DataFrame(
                {"sample_1": [6.0, 1.0, 3.0]},
                index=["otu_a", "otu_b", "otu_c"],
            )
            otu_table.to_csv(otu_table_path)

            builder = MGM2DatasetBuilder(str(embedding_path), embedding_dim=3, top_k_otus=2)
            builder.load_ntv3_embeddings()
            builder.process_otu_table(str(otu_table_path), str(output_path), input_format="csv")

            with open(output_path, "rb") as handle:
                payload = pickle.load(handle)

            self.assertEqual(payload["preprocessing"]["top_k_otus"], 2)
            self.assertEqual(payload["otu_data"]["sample_1"]["input_ids"].tolist(), [-1, 0, 2])

    def test_regression_top_k_bins_expands_regression_supervision(self):
        manager = BinManager(
            num_bins=5,
            method="fixed_threshold",
            fixed_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
            regression_top_k_bins=2,
        )
        manager.fit([np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32)])

        labels, regression_mask = manager.transform(np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32))

        self.assertTrue(np.array_equal(labels, np.array([1, 2, 3, 4], dtype=np.int64)))
        self.assertTrue(np.array_equal(regression_mask, np.array([False, False, True, True])))

    def test_collator_keeps_padded_regression_mask_false(self):
        collator = BatchCollatorMGM2()
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "bin_labels": torch.tensor([1, 2]),
                "regression_mask": torch.tensor([True, False]),
                "abundance_values": torch.tensor([0.1, 0.2]),
                "regression_targets": torch.tensor([0.1, 0.0]),
                "embeddings": torch.randn(2, 3),
            },
            {
                "input_ids": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
                "bin_labels": torch.tensor([1]),
                "regression_mask": torch.tensor([False]),
                "abundance_values": torch.tensor([0.3]),
                "regression_targets": torch.tensor([0.0]),
                "embeddings": torch.randn(1, 3),
            },
        ]

        output = collator(batch)

        self.assertFalse(bool(output["regression_mask"][1, 1].item()))
        self.assertNotIn("presence_labels", output)

    def test_dataset_sample_has_no_presence_labels(self):
        tmpdir, data, preprocessor, manager = self._build_dataset_components()
        try:
            dataset = MicrobiomeDatasetMGM2(
                preprocessed_data=data,
                abundance_preprocessor=preprocessor,
                bin_manager=manager,
                mask_prob=1.0,
                is_training=False,
            )
            sample = dataset[0]
            self.assertNotIn("presence_labels", sample)
        finally:
            tmpdir.cleanup()

    def test_dataset_returns_qwen3_embedding_when_sample_id_matches(self):
        tmpdir, data, preprocessor, manager = self._build_dataset_components()
        try:
            qwen3_path = Path(tmpdir.name) / "qwen3.h5"
            self._create_qwen3_embedding_file(qwen3_path, ["s1"], embedding_dim=5)

            dataset = MicrobiomeDatasetMGM2(
                preprocessed_data=data,
                abundance_preprocessor=preprocessor,
                bin_manager=manager,
                qwen3_embedding_path=str(qwen3_path),
                enable_qwen3_alignment=True,
                is_training=False,
            )

            sample = dataset[0]

            self.assertIn("qwen3_embedding", sample)
            self.assertIn("qwen3_alignment_mask", sample)
            self.assertTrue(bool(sample["qwen3_alignment_mask"].item()))
            self.assertEqual((5,), tuple(sample["qwen3_embedding"].shape))
        finally:
            tmpdir.cleanup()


    def test_dataset_matches_qwen3_sample_ids_stored_as_numpy_bytes(self):
        tmpdir, data, preprocessor, manager = self._build_dataset_components()
        try:
            qwen3_path = Path(tmpdir.name) / "qwen3_bytes.h5"
            embeddings = np.arange(5, dtype=np.float32).reshape(1, 5)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            with h5py.File(qwen3_path, "w") as handle:
                handle.create_dataset("embeddings", data=embeddings.astype(np.float32))
                handle.create_dataset("sample_ids", data=np.array([b"b\'s1\'"], dtype="S16"))

            dataset = MicrobiomeDatasetMGM2(
                preprocessed_data=data,
                abundance_preprocessor=preprocessor,
                bin_manager=manager,
                qwen3_embedding_path=str(qwen3_path),
                enable_qwen3_alignment=True,
                is_training=False,
            )

            sample = dataset[0]

            self.assertTrue(bool(sample["qwen3_alignment_mask"].item()))
            self.assertFalse(torch.equal(sample["qwen3_embedding"], torch.zeros(5)))
        finally:
            tmpdir.cleanup()

    def test_dataset_marks_missing_qwen3_alignment_without_dropping_sample(self):
        tmpdir, data, preprocessor, manager = self._build_dataset_components()
        try:
            qwen3_path = Path(tmpdir.name) / "qwen3.h5"
            self._create_qwen3_embedding_file(qwen3_path, ["other_sample"], embedding_dim=5)

            dataset = MicrobiomeDatasetMGM2(
                preprocessed_data=data,
                abundance_preprocessor=preprocessor,
                bin_manager=manager,
                qwen3_embedding_path=str(qwen3_path),
                enable_qwen3_alignment=True,
                is_training=False,
            )

            sample = dataset[0]

            self.assertEqual(1, len(dataset))
            self.assertFalse(bool(sample["qwen3_alignment_mask"].item()))
            self.assertTrue(torch.equal(sample["qwen3_embedding"], torch.zeros(5)))
        finally:
            tmpdir.cleanup()

    def test_dataset_fails_fast_on_qwen3_projection_dim_mismatch(self):
        tmpdir, data, preprocessor, manager = self._build_dataset_components()
        try:
            qwen3_path = Path(tmpdir.name) / "qwen3.h5"
            self._create_qwen3_embedding_file(qwen3_path, ["s1"], embedding_dim=5)

            with self.assertRaises(ValueError):
                MicrobiomeDatasetMGM2(
                    preprocessed_data=data,
                    abundance_preprocessor=preprocessor,
                    bin_manager=manager,
                    qwen3_embedding_path=str(qwen3_path),
                    qwen3_projection_dim=6,
                    enable_qwen3_alignment=True,
                    is_training=False,
                )
        finally:
            tmpdir.cleanup()

    def test_collator_stacks_qwen3_alignment_fields(self):
        collator = BatchCollatorMGM2()
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "bin_labels": torch.tensor([1, 2]),
                "regression_mask": torch.tensor([True, False]),
                "abundance_values": torch.tensor([0.1, 0.2]),
                "regression_targets": torch.tensor([0.1, 0.0]),
                "embeddings": torch.randn(2, 3),
                "qwen3_embedding": torch.tensor([1.0, 0.0, 0.0]),
                "qwen3_alignment_mask": torch.tensor(True),
            },
            {
                "input_ids": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
                "bin_labels": torch.tensor([1]),
                "regression_mask": torch.tensor([False]),
                "abundance_values": torch.tensor([0.3]),
                "regression_targets": torch.tensor([0.0]),
                "embeddings": torch.randn(1, 3),
                "qwen3_embedding": torch.tensor([0.0, 1.0, 0.0]),
                "qwen3_alignment_mask": torch.tensor(False),
            },
        ]

        output = collator(batch)

        self.assertEqual((2, 3), tuple(output["qwen3_embedding"].shape))
        self.assertEqual((2,), tuple(output["qwen3_alignment_mask"].shape))
        self.assertTrue(bool(output["qwen3_alignment_mask"][0].item()))
        self.assertFalse(bool(output["qwen3_alignment_mask"][1].item()))

    def test_training_masking_changes_inputs_not_targets(self):
        tmpdir, data, preprocessor, manager = self._build_dataset_components()
        try:
            random.seed(0)
            torch.manual_seed(0)
            train_dataset = MicrobiomeDatasetMGM2(
                preprocessed_data=data,
                abundance_preprocessor=preprocessor,
                bin_manager=manager,
                mask_prob=1.0,
                is_training=True,
            )
            eval_dataset = MicrobiomeDatasetMGM2(
                preprocessed_data=data,
                abundance_preprocessor=preprocessor,
                bin_manager=manager,
                mask_prob=1.0,
                is_training=False,
            )

            train_sample = train_dataset[0]
            eval_sample = eval_dataset[0]

            self.assertFalse(torch.equal(train_sample["abundance_values"], eval_sample["abundance_values"]))
            supervised_positions = train_sample["bin_labels"] != -100
            self.assertTrue(torch.equal(train_sample["bin_labels"][supervised_positions], eval_sample["bin_labels"][supervised_positions]))
            self.assertNotIn("presence_labels", train_sample)
        finally:
            tmpdir.cleanup()


    def test_training_masking_preserves_masked_otu_embeddings(self):
        tmpdir, data, preprocessor, manager = self._build_dataset_components()
        try:
            random.seed(0)
            torch.manual_seed(0)
            train_dataset = MicrobiomeDatasetMGM2(
                preprocessed_data=data,
                abundance_preprocessor=preprocessor,
                bin_manager=manager,
                mask_prob=1.0,
                is_training=True,
            )
            eval_dataset = MicrobiomeDatasetMGM2(
                preprocessed_data=data,
                abundance_preprocessor=preprocessor,
                bin_manager=manager,
                mask_prob=1.0,
                is_training=False,
            )

            train_sample = train_dataset[0]
            eval_sample = eval_dataset[0]
            supervised_positions = train_sample["bin_labels"] != -100

            self.assertTrue(supervised_positions[1:].all())
            self.assertTrue(torch.equal(train_sample["embeddings"][supervised_positions], eval_sample["embeddings"][supervised_positions]))
        finally:
            tmpdir.cleanup()


    def test_legacy_presence_config_keys_fail_fast(self):
        with self.assertRaises(TypeError):
            MGM2Config.from_dict({"loss": {"lambda_presence": 0.5}})

    def test_dataset_reuses_embedding_cache_for_same_path(self):
        tmpdir, data, preprocessor, manager = self._build_dataset_components()
        try:
            MicrobiomeDatasetMGM2.clear_runtime_caches()
            with patch("h5py.File", wraps=h5py.File) as mocked_h5_file:
                first = MicrobiomeDatasetMGM2(
                    preprocessed_data=data,
                    abundance_preprocessor=preprocessor,
                    bin_manager=manager,
                    is_training=False,
                )
                second = MicrobiomeDatasetMGM2(
                    preprocessed_data=data,
                    abundance_preprocessor=preprocessor,
                    bin_manager=manager,
                    is_training=False,
                )

            self.assertEqual(1, mocked_h5_file.call_count)
            self.assertIs(first.ntv3_embeddings, second.ntv3_embeddings)
        finally:
            MicrobiomeDatasetMGM2.clear_runtime_caches()
            tmpdir.cleanup()

    def test_create_dataloaders_uses_deterministic_split_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "dataset.pkl"
            embedding_path = tmpdir / "embeddings.h5"
            output_dir = tmpdir / "outputs"
            self._create_preprocessed_dataset_file(data_path, sample_count=10)
            self._create_embedding_file(embedding_path, ["otu_a", "otu_b"], embedding_dim=4)

            abundance_cfg = MGM2Config().abundance.__dict__.copy()
            training_cfg = {
                "batch_size": 2,
                "seed": 123,
                "output_dir": str(output_dir),
            }

            loaders_a = create_dataloaders_mgm2(
                data_path=str(data_path),
                ntv3_embedding_path=str(embedding_path),
                abundance_bin_config=abundance_cfg,
                training_config=training_cfg,
                num_workers=0,
            )
            loaders_b = create_dataloaders_mgm2(
                data_path=str(data_path),
                ntv3_embedding_path=str(embedding_path),
                abundance_bin_config=abundance_cfg,
                training_config=training_cfg,
                num_workers=0,
            )

            self.assertEqual(loaders_a[0].dataset.data["samples"], loaders_b[0].dataset.data["samples"])
            self.assertEqual(loaders_a[1].dataset.data["samples"], loaders_b[1].dataset.data["samples"])
            self.assertEqual(loaders_a[2].dataset.data["samples"], loaders_b[2].dataset.data["samples"])

    def test_create_dataloaders_can_skip_test_loader_for_pretraining(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "dataset.pkl"
            embedding_path = tmpdir / "embeddings.h5"
            output_dir = tmpdir / "outputs"

            self._create_preprocessed_dataset_file(data_path, sample_count=6)
            self._create_embedding_file(embedding_path, ["otu_a", "otu_b"])

            loaders = create_dataloaders_mgm2(
                data_path=str(data_path),
                ntv3_embedding_path=str(embedding_path),
                abundance_bin_config={
                    "min_abundance": 1e-4,
                    "output_min": -2.0,
                    "output_max": 2.0,
                    "normalize": False,
                    "num_bins": 5,
                    "binning_method": "quantile",
                    "fixed_thresholds": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "enable_regression": True,
                    "regression_top_k_bins": 2,
                },
                training_config={
                    "batch_size": 2,
                    "output_dir": str(output_dir),
                    "seed": 7,
                },
                mask_prob=0.15,
                num_workers=0,
                train_ratio=0.8,
                val_ratio=0.2,
                include_test_loader=False,
            )

            self.assertEqual(2, len(loaders))
            train_loader, val_loader = loaders
            self.assertEqual(4, len(train_loader.dataset))
            self.assertEqual(2, len(val_loader.dataset))
            self.assertEqual(6, len(train_loader.dataset) + len(val_loader.dataset))

    def test_create_dataloaders_skips_startup_abundance_transform_for_quantile_binning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "dataset.pkl"
            embedding_path = tmpdir / "embeddings.h5"
            output_dir = tmpdir / "outputs"
            self._create_preprocessed_dataset_file(data_path, sample_count=6)
            self._create_embedding_file(embedding_path, ["otu_a", "otu_b"], embedding_dim=4)

            abundance_cfg = MGM2Config().abundance.__dict__.copy()
            abundance_cfg["binning_method"] = "quantile"
            abundance_cfg["normalize"] = False

            with patch.object(AbundancePreprocessor, "transform", side_effect=AssertionError("startup should not transform the full training set")):
                create_dataloaders_mgm2(
                    data_path=str(data_path),
                    ntv3_embedding_path=str(embedding_path),
                    abundance_bin_config=abundance_cfg,
                    training_config={"batch_size": 2, "seed": 42, "output_dir": str(output_dir)},
                    num_workers=0,
                )

    def test_default_config_uses_only_current_abundance_fields(self):
        config = MGM2Config.load("src/configs/mgm2_default.json")

        self.assertTrue(hasattr(config.abundance, "min_abundance"))
        self.assertTrue(hasattr(config.abundance, "output_min"))
        self.assertTrue(hasattr(config.abundance, "output_max"))
        self.assertTrue(hasattr(config.abundance, "normalize"))
        self.assertTrue(hasattr(config.abundance, "regression_top_k_bins"))
        self.assertFalse(hasattr(config.abundance, "high_abundance_threshold"))
        self.assertFalse(hasattr(config.abundance, "use_log1p"))
        self.assertFalse(hasattr(config.abundance, "apply_label_smoothing"))

    def test_default_config_disables_qwen3_alignment(self):
        config = MGM2Config.load("src/configs/mgm2_default.json")

        self.assertTrue(hasattr(config, "qwen3_alignment"))
        self.assertFalse(config.qwen3_alignment.enabled)
        self.assertEqual("", config.qwen3_alignment.embedding_path)

    def test_default_config_uses_full_train_val_split_for_pretraining(self):
        config = MGM2Config.load("src/configs/mgm2_default.json")

        self.assertAlmostEqual(0.9, config.training.train_ratio)
        self.assertAlmostEqual(0.1, config.training.val_ratio)
        self.assertAlmostEqual(1.0, config.training.train_ratio + config.training.val_ratio)

    def test_default_config_uses_small_model_preset(self):
        config = MGM2Config.load("src/configs/mgm2_default.json")

        self.assertEqual(224, config.model.hidden_size)
        self.assertEqual(6, config.model.num_hidden_layers)
        self.assertEqual(4, config.model.num_attention_heads)
        self.assertEqual(896, config.model.intermediate_size)


    def test_builder_uses_relative_abundance_for_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            embedding_path = tmpdir / "embeddings.h5"
            csv_path = tmpdir / "otu_table.csv"
            output_path = tmpdir / "dataset.pkl"
            self._create_embedding_file(embedding_path, ["otu_a", "otu_b"])

            csv_path.write_text("otu_id,sample_1\notu_a,1\notu_b,1\n", encoding="utf-8")

            builder = MGM2DatasetBuilder(str(embedding_path), embedding_dim=4)
            builder.load_ntv3_embeddings()
            builder.process_otu_table(str(csv_path), str(output_path), input_format="csv")

            with open(output_path, "rb") as handle:
                dataset = pickle.load(handle)

            abundances = dataset["otu_data"]["sample_1"]["abundance_values"]
            np.testing.assert_allclose(abundances[1:], np.array([0.5, 0.5], dtype=np.float32))

    def test_builder_raises_when_no_valid_samples_are_produced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            embedding_path = tmpdir / "embeddings.h5"
            csv_path = tmpdir / "otu_table.csv"
            output_path = tmpdir / "dataset.pkl"
            self._create_embedding_file(embedding_path, ["otu_in_vocab"])

            csv_path.write_text("otu_id,sample_1\notu_not_in_vocab,1\n", encoding="utf-8")

            builder = MGM2DatasetBuilder(str(embedding_path), embedding_dim=4)
            builder.load_ntv3_embeddings()

            with self.assertRaises(ValueError):
                builder.process_otu_table(str(csv_path), str(output_path), input_format="csv")

    def test_embedding_loader_returns_accessible_large_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_path = Path(tmpdir) / "embeddings.h5"
            otu_count = 50001
            with h5py.File(embedding_path, "w") as handle:
                handle.create_dataset("embeddings", data=np.arange(otu_count, dtype=np.float32).reshape(otu_count, 1))
                handle.create_dataset("otu_ids", data=np.array([f"otu_{idx}".encode("utf-8") for idx in range(otu_count)]))

            embeddings = load_ntv3_embeddings(str(embedding_path), use_memory_mapping=True)

            self.assertEqual((otu_count, 1), embeddings.shape)
            self.assertAlmostEqual(123.0, float(embeddings[123][0]))

    def test_base_model_uses_film_modules_for_input_conditioning(self):
        model = BaseModel(embedding_dim=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2)
        model.eval()

        self.assertTrue(hasattr(model, "embedding_projection"))
        self.assertTrue(hasattr(model, "abundance_conditioner"))
        self.assertFalse(hasattr(model, "input_projection"))

    def test_base_model_non_cls_hidden_states_change_with_abundance(self):
        torch.manual_seed(0)
        model = BaseModel(embedding_dim=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2)
        model.eval()
        embeddings = torch.ones(1, 2, 4)
        input_ids = torch.tensor([[-1, 0]])
        attention_mask = torch.tensor([[1, 1]])

        low = model(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=torch.tensor([[0.0, -2.0]]),
            attention_mask=attention_mask,
        )["last_hidden_state"]
        high = model(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=torch.tensor([[0.0, 2.0]]),
            attention_mask=attention_mask,
        )["last_hidden_state"]

        self.assertFalse(torch.allclose(low[:, 1, :], high[:, 1, :]))

    def test_base_model_cls_token_is_not_conditioned_on_abundance(self):
        torch.manual_seed(0)
        model = BaseModel(embedding_dim=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2)
        model.eval()
        embeddings = torch.randn(1, 1, 4)
        input_ids = torch.tensor([[-1]])
        attention_mask = torch.tensor([[1]])

        low = model(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=torch.tensor([[0.0]]),
            attention_mask=attention_mask,
        )["last_hidden_state"]
        high = model(
            input_ids=input_ids,
            embeddings=embeddings,
            abundance_values=torch.tensor([[1.0]]),
            attention_mask=attention_mask,
        )["last_hidden_state"]

        self.assertTrue(torch.allclose(low[:, 0, :], high[:, 0, :]))

    def test_model_and_loss_use_only_bin_and_regression(self):
        config = MGM2Config()
        config.model.embedding_dim = 4
        config.model.hidden_size = 8
        config.model.intermediate_size = 16
        config.model.num_hidden_layers = 1
        config.model.num_attention_heads = 2
        config.model.max_position_embeddings = 16

        module = MGM2LightningModule(config)
        batch = {
            "input_ids": torch.tensor([[-1, 0, 1]]),
            "embeddings": torch.randn(1, 3, 4),
            "abundance_values": torch.randn(1, 3),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "bin_labels": torch.tensor([[-100, 1, 4]]),
            "regression_targets": torch.tensor([[0.0, 0.0, 1.5]]),
            "regression_mask": torch.tensor([[False, False, True]]),
        }

        outputs = module.model(**batch)
        loss_dict = module.criterion(
            {
                "bin_logits": outputs["bin_logits"],
                "regression_preds": outputs["regression_preds"],
            },
            {
                "bin_labels": batch["bin_labels"],
                "regression_targets": batch["regression_targets"],
                "regression_mask": batch["regression_mask"],
            },
        )

        self.assertNotIn("presence_logits", outputs)
        self.assertNotIn("presence_loss", loss_dict)
        self.assertFalse(hasattr(config.loss, "lambda_presence"))
        self.assertIsInstance(module.criterion, MGM2MultiTaskLoss)

    def test_model_exposes_cls_hidden_and_optional_qwen3_projection(self):
        model_config = {
            "embedding_dim": 4,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 16,
            "qwen3_projection_dim": 6,
        }
        batch = {
            "input_ids": torch.tensor([[-1, 0, 1]]),
            "embeddings": torch.randn(1, 3, 4),
            "abundance_values": torch.randn(1, 3),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        base_model = MGM2Model(config=model_config, num_bins=5, enable_regression=True)
        base_outputs = base_model(**batch)
        self.assertIn("cls_hidden", base_outputs)
        self.assertNotIn("qwen3_proj", base_outputs)
        self.assertEqual((1, 8), tuple(base_outputs["cls_hidden"].shape))

        aligned_model = MGM2Model(
            config=model_config,
            num_bins=5,
            enable_regression=True,
            enable_qwen3_alignment=True,
            qwen3_projection_dim=6,
        )
        aligned_outputs = aligned_model(**batch)
        self.assertIn("cls_hidden", aligned_outputs)
        self.assertIn("qwen3_proj", aligned_outputs)
        self.assertEqual((1, 6), tuple(aligned_outputs["qwen3_proj"].shape))

    def test_multitask_loss_returns_zero_alignment_loss_with_fewer_than_two_valid_pairs(self):
        criterion = MGM2MultiTaskLoss(
            num_bins=5,
            enable_regression=False,
            enable_qwen3_alignment=True,
            lambda_qwen3_alignment=0.2,
            qwen3_alignment_temperature=0.07,
        )

        loss_dict = criterion(
            {
                "qwen3_proj": torch.tensor([[1.0, 0.0, 0.0]]),
            },
            {
                "qwen3_embedding": torch.tensor([[1.0, 0.0, 0.0]]),
                "qwen3_alignment_mask": torch.tensor([True]),
            },
        )

        self.assertIn("qwen3_alignment_loss", loss_dict)
        self.assertEqual(0.0, float(loss_dict["qwen3_alignment_loss"]))

    def test_multitask_loss_adds_alignment_loss_when_enabled(self):
        criterion = MGM2MultiTaskLoss(
            num_bins=5,
            enable_regression=False,
            enable_qwen3_alignment=True,
            lambda_qwen3_alignment=0.2,
            qwen3_alignment_temperature=0.07,
        )

        predictions = {
            "qwen3_proj": torch.nn.functional.normalize(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                p=2,
                dim=-1,
            ),
        }
        labels = {
            "qwen3_embedding": torch.nn.functional.normalize(
                torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32),
                p=2,
                dim=-1,
            ),
            "qwen3_alignment_mask": torch.tensor([True, True]),
        }

        loss_dict = criterion(predictions, labels)

        self.assertIn("qwen3_alignment_loss", loss_dict)
        self.assertGreater(float(loss_dict["qwen3_alignment_loss"]), 0.0)
        self.assertGreater(float(loss_dict["total_loss"]), 0.0)

    def test_save_and_load_pretrained_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MGM2Config()
            config.model.embedding_dim = 4
            config.model.hidden_size = 8
            config.model.intermediate_size = 16
            config.model.num_hidden_layers = 1
            config.model.num_attention_heads = 2
            config.model.max_position_embeddings = 16
            module = MGM2LightningModule(config)

            output_dir = Path(tmpdir) / "saved_model"
            module.save_pretrained(str(output_dir))
            loaded = MGM2LightningModule.load_pretrained(str(output_dir))

            self.assertIsInstance(loaded, MGM2LightningModule)
            self.assertTrue((output_dir / "mgm2_config.json").exists())

    def test_lightning_module_wires_qwen3_alignment_predictions_and_labels(self):
        config = MGM2Config.from_dict({
            "model": {
                "embedding_dim": 4,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 16,
            },
            "qwen3_alignment": {
                "enabled": True,
                "embedding_path": "/tmp/qwen3_dummy.h5",
                "projection_dim": 4,
                "temperature": 0.07,
                "loss_weight": 0.2,
            },
        })
        module = MGM2LightningModule(config)
        batch = {
            "input_ids": torch.tensor([[-1, 0, 1], [-1, 1, 0]]),
            "embeddings": torch.randn(2, 3, 4),
            "abundance_values": torch.randn(2, 3),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "bin_labels": torch.tensor([[-100, 1, 4], [-100, 2, 3]]),
            "regression_targets": torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.5, 1.0]]),
            "regression_mask": torch.tensor([[False, False, True], [False, True, True]]),
            "qwen3_embedding": torch.nn.functional.normalize(torch.randn(2, 4), p=2, dim=-1),
            "qwen3_alignment_mask": torch.tensor([True, True]),
        }

        loss = module.training_step(batch, 0)
        outputs = module.model(
            input_ids=batch["input_ids"],
            embeddings=batch["embeddings"],
            abundance_values=batch["abundance_values"],
            attention_mask=batch["attention_mask"],
        )
        predictions, labels = module._split_predictions_and_labels(outputs, batch)

        self.assertGreater(float(loss), 0.0)
        self.assertIn("qwen3_proj", predictions)
        self.assertIn("qwen3_embedding", labels)
        self.assertIn("qwen3_alignment_mask", labels)


    def test_src_package_exports_config(self):
        root_pkg = importlib.import_module("src")

        self.assertIs(root_pkg.MGM2Config, MGM2Config)

    def test_pretrain_entrypoint_replaces_train_script(self):
        repo_root = Path(__file__).resolve().parents[1]

        self.assertTrue((repo_root / "src" / "training" / "Pretrain.py").exists())
        self.assertFalse((repo_root / "src" / "training" / "Train.py").exists())

    def test_pretrain_entrypoint_does_not_run_test_evaluation(self):
        pretrain_path = Path(__file__).resolve().parents[1] / "src" / "training" / "Pretrain.py"
        pretrain_source = pretrain_path.read_text()

        self.assertNotIn("trainer.test(", pretrain_source)

    def test_pretrain_script_summary_mentions_binning_and_regression_top_k(self):
        train_module = importlib.import_module("src.training.Pretrain")
        config = MGM2Config()
        config.abundance.binning_method = "quantile"
        config.abundance.regression_top_k_bins = 2
        config.qwen3_alignment.enabled = True

        summary = train_module.summarize_experiment_config(config)

        self.assertIn("binning_method: quantile", summary)
        self.assertIn("regression_top_k_bins: 2", summary)
        self.assertIn("qwen3_alignment: enabled", summary)

    def test_pretrain_script_help_uses_qwen3_embedding_path_flag(self):
        script_path = Path(__file__).resolve().parents[1] / "src" / "training" / "Pretrain.py"
        env = os.environ.copy()
        env["MKL_SERVICE_FORCE_INTEL"] = "1"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            env=env,
        )

        help_text = result.stdout + result.stderr
        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIn("--qwen3_embedding_path", help_text)
        self.assertNotIn("--qwen3_embedding ", help_text)
        self.assertIn("src/training/Pretrain.py", help_text)

    def test_create_config_uses_repo_default_json_when_config_is_omitted(self):
        train_module = importlib.import_module("src.training.Pretrain")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "dataset.pkl"
            ntv3_path = tmpdir / "ntv3.h5"
            data_path.write_bytes(b"placeholder")
            ntv3_path.write_bytes(b"placeholder")

            args = SimpleNamespace(
                config=None,
                max_epochs=1,
                batch_size=2,
                devices=[0],
                ntv3_embedding_path=str(ntv3_path),
                data_file=str(data_path),
                output_dir=str(tmpdir / "outputs"),
                experiment_name="test_run",
                learning_rate=None,
                qwen3_embedding_path=None,
                enable_qwen3_alignment=False,
                resume=None,
            )

            config = train_module.create_config(args)

            self.assertEqual(4096, config.qwen3_alignment.projection_dim)

    def test_create_config_fails_when_qwen3_enabled_without_existing_path(self):
        train_module = importlib.import_module("src.training.Pretrain")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "dataset.pkl"
            ntv3_path = tmpdir / "ntv3.h5"
            data_path.write_bytes(b"placeholder")
            ntv3_path.write_bytes(b"placeholder")

            args = SimpleNamespace(
                config=None,
                max_epochs=1,
                batch_size=2,
                devices=[0],
                ntv3_embedding=str(ntv3_path),
                data_file=str(data_path),
                output_dir=str(tmpdir / "outputs"),
                experiment_name="test_run",
                learning_rate=None,
                qwen3_embedding_path=str(tmpdir / "missing_qwen3.h5"),
                enable_qwen3_alignment=True,
            )

            with self.assertRaises(FileNotFoundError):
                train_module.create_config(args)

    def test_pretrain_registers_safe_globals_for_resume(self):
        train_module = importlib.import_module("src.training.Pretrain")

        with patch.object(torch.serialization, "add_safe_globals") as mocked_add:
            train_module.register_trusted_checkpoint_safe_globals()

        mocked_add.assert_called_once()
        registered = mocked_add.call_args.args[0]
        self.assertIn(MGM2Config, registered)
        self.assertIn(train_module.AbundanceConfig, registered)
        self.assertIn(train_module.ModelConfig, registered)
        self.assertIn(train_module.LossConfig, registered)
        self.assertIn(train_module.TrainingConfig, registered)
        self.assertIn(train_module.Qwen3AlignmentConfig, registered)

    def test_pretrain_script_runs_from_outside_repo(self):
        script_path = Path(__file__).resolve().parents[1] / "src" / "training" / "Pretrain.py"
        env = os.environ.copy()
        env["MKL_SERVICE_FORCE_INTEL"] = "1"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            env=env,
        )

        help_text = (result.stdout + result.stderr).lower()
        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIn("dual-task", help_text)
        self.assertIn("qwen3", help_text)
        self.assertIn("contrastive", help_text)
        self.assertNotIn("hierarchical multi-task", help_text)


    def test_readme_examples_use_pretrain_entrypoint(self):
        readme_text = Path(__file__).resolve().parents[1].joinpath("README.md").read_text()
        src_readme_text = Path(__file__).resolve().parents[1].joinpath("src", "README.md").read_text()

        self.assertIn("src/training/Pretrain.py", readme_text)
        self.assertNotIn("src/training/Train.py", readme_text)
        self.assertIn("src/training/Pretrain.py", src_readme_text)
        self.assertNotIn("src/training/Train.py", src_readme_text)

    def test_repository_guide_uses_pretrain_entrypoint(self):
        guide_text = Path(__file__).resolve().parents[1].joinpath("docs", "repository_guide_zh.md").read_text()

        self.assertIn("src/training/Pretrain.py", guide_text)
        self.assertNotIn("src/training/Train.py", guide_text)
        self.assertIn("不跑测试集", guide_text)

    def test_stage1_grid_script_exists_and_uses_pretrain_entrypoint(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "grid_search_pretrain_stage1.sh"
        script_text = script_path.read_text()

        self.assertTrue(script_path.exists())
        self.assertIn("src/training/Pretrain.py", script_text)
        self.assertIn("outputs_grid/stage1", script_text)
        self.assertIn("val/bin_accuracy", script_text)
        self.assertIn("val/regression_mse", script_text)
        self.assertIn('DATA_FILE="', script_text)
        self.assertIn('if (( ${#QWEN3_CONFIG_OVERRIDES[@]} )); then', script_text)
        self.assertIn('if (( ${#QWEN3_PRETRAIN_ARGS[@]} )); then', script_text)
        self.assertIn('python -u src/training/Pretrain.py', script_text)
        self.assertIn('| tee "${run_dir}/train.log"', script_text)
        self.assertNotIn('Usage:', script_text)
        self.assertNotIn('if [[ $#', script_text)

    def test_stage2_grid_script_uses_curated_scaling_presets(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "grid_search_pretrain_stage2.sh"
        script_text = script_path.read_text()

        self.assertTrue(script_path.exists())
        self.assertIn("scale_small", script_text)
        self.assertIn("scale_medium", script_text)
        self.assertIn("scale_large", script_text)
        self.assertIn("scale_xlarge", script_text)
        self.assertIn('"scale_small|224|6|4|896"', script_text)
        self.assertIn('"scale_medium|320|6|5|1280"', script_text)
        self.assertIn('"scale_large|384|8|6|1536"', script_text)
        self.assertIn('"scale_xlarge|512|10|8|2048"', script_text)
        self.assertNotIn("scale_medium_large", script_text)
        self.assertIn("src/training/Pretrain.py", script_text)
        self.assertIn('BASE_CONFIG="', script_text)
        self.assertIn('if (( ${#QWEN3_PRETRAIN_ARGS[@]} )); then', script_text)
        self.assertIn('python -u src/training/Pretrain.py', script_text)
        self.assertIn('| tee "${run_dir}/train.log"', script_text)
        self.assertNotIn('Usage:', script_text)
        self.assertNotIn('if [[ $#', script_text)
        self.assertNotIn("for hidden_size in", script_text)
        self.assertNotIn("for num_hidden_layers in", script_text)
        self.assertNotIn("for num_attention_heads in", script_text)

    def test_grid_config_builder_applies_nested_overrides(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_grid_config.py"
        spec = importlib.util.spec_from_file_location("build_grid_config", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config = {
            "training": {"learning_rate": 1e-4, "weight_decay": 0.01},
            "loss": {"lambda_regression": 0.3},
            "qwen3_alignment": {"loss_weight": 0.1},
        }
        updated = module.apply_overrides(
            config,
            {
                "training.learning_rate": 5e-5,
                "loss.lambda_regression": 0.5,
                "qwen3_alignment.loss_weight": 0.2,
            },
        )

        self.assertEqual(5e-5, updated["training"]["learning_rate"])
        self.assertEqual(0.5, updated["loss"]["lambda_regression"])
        self.assertEqual(0.2, updated["qwen3_alignment"]["loss_weight"])
        self.assertEqual(1e-4, config["training"]["learning_rate"])

    def test_embedding_eval_selects_requested_sample_embedding_strategy(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_embeddings.py"
        spec = importlib.util.spec_from_file_location("evaluate_embeddings", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        outputs = {
            "cls_hidden": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
            "hidden_states": torch.tensor(
                [[[10.0, 20.0], [1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]],
                dtype=torch.float32,
            ),
            "qwen3_proj": torch.tensor([[0.5, 0.25]], dtype=torch.float32),
        }
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

        cls_embedding = module.select_sample_embeddings(outputs, attention_mask, strategy="cls")
        mean_embedding = module.select_sample_embeddings(outputs, attention_mask, strategy="mean_pool")
        qwen3_embedding = module.select_sample_embeddings(outputs, attention_mask, strategy="qwen3_proj")

        self.assertTrue(np.allclose(cls_embedding, np.array([[10.0, 20.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(mean_embedding, np.array([[3.0, 5.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(qwen3_embedding, np.array([[0.5, 0.25]], dtype=np.float32)))

    def test_embedding_eval_balanced_sampling_limits_each_label(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_embeddings.py"
        spec = importlib.util.spec_from_file_location("evaluate_embeddings", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        metadata = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(8)],
                "label": ["soil", "soil", "soil", "plant", "plant", "plant", "water", "water"],
            }
        )

        sampled = module.sample_metadata_balanced(
            metadata,
            label_column="label",
            max_samples_per_label=2,
            seed=42,
        )
        counts = sampled["label"].value_counts().to_dict()

        self.assertEqual(2, counts["soil"])
        self.assertEqual(2, counts["plant"])
        self.assertEqual(2, counts["water"])

    def test_embedding_eval_filters_labels_by_requested_keywords(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_embeddings.py"
        spec = importlib.util.spec_from_file_location("evaluate_embeddings", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        metadata = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(6)],
                "label": [
                    "Aquatic biome",
                    "animal gut",
                    "SOIL microbiome",
                    "plant rhizosphere",
                    "marine sediment",
                    "human oral",
                ],
            }
        )

        filtered = module.filter_metadata_by_keywords(metadata, "label", module.LABEL_KEYWORDS)

        self.assertEqual(["s0", "s1", "s2", "s3"], filtered["sample_id"].tolist())

    def test_embedding_eval_save_plot_writes_pdf_with_editable_font_setting(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_embeddings.py"
        spec = importlib.util.spec_from_file_location("evaluate_embeddings", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "demo.png"
            module._save_plot(
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                ["soil", "plant"],
                output_path,
                "demo",
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.with_suffix('.pdf').exists())
            import matplotlib
            self.assertEqual(42, matplotlib.rcParams['pdf.fonttype'])

    def test_embedding_eval_infers_config_by_searching_checkpoint_ancestors(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_embeddings.py"
        spec = importlib.util.spec_from_file_location("evaluate_embeddings", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            run_dir = tmpdir / "run_001"
            checkpoint_dir = run_dir / "artifacts" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            config_path = run_dir / "config.json"
            config_path.write_text("{}")
            checkpoint_path = checkpoint_dir / "last.ckpt"
            checkpoint_path.write_text("ckpt")

            resolved = module._resolve_config_path(checkpoint_path, None)

            self.assertEqual(config_path.resolve(), resolved)

    def test_embedding_eval_load_model_disables_weights_only_for_trusted_lightning_checkpoints(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_embeddings.py"
        spec = importlib.util.spec_from_file_location("evaluate_embeddings", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fake_loaded = SimpleNamespace()
        fake_loaded.eval = lambda: None
        fake_loaded.freeze = lambda: None
        fake_loaded.to = lambda device: None
        device = torch.device("cpu")
        config = object()

        with patch.object(MGM2LightningModule, "load_from_checkpoint", return_value=fake_loaded) as mocked_load:
            loaded = module._load_model(Path("demo.ckpt"), config=config, device=device)

        mocked_load.assert_called_once_with(
            str(Path("demo.ckpt")),
            config=config,
            map_location=device,
            weights_only=False,
        )
        self.assertIs(loaded, fake_loaded)

    def test_embedding_eval_extract_embeddings_uses_progress_wrapper(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_embeddings.py"
        spec = importlib.util.spec_from_file_location("evaluate_embeddings", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class DummyModel:
            def __call__(self, **kwargs):
                return {
                    "cls_hidden": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                    "hidden_states": torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
                }

        dummy_module = SimpleNamespace(
            model=DummyModel(),
            _extract_model_inputs=lambda batch: batch,
        )
        loader = [
            {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            }
        ]

        with patch.object(module, "_wrap_progress", side_effect=lambda iterable, total, desc, disable: iterable) as mocked_progress:
            embeddings, sample_ids = module._extract_embeddings(
                dummy_module,
                loader,
                ["sample_1"],
                strategy="cls",
                device=torch.device("cpu"),
                show_progress=True,
            )

        mocked_progress.assert_called_once_with(
            loader,
            total=1,
            desc="Extracting embeddings",
            disable=False,
        )
        self.assertEqual(["sample_1"], sample_ids)
        self.assertTrue(np.allclose(embeddings, np.array([[1.0, 2.0]], dtype=np.float32)))

    def test_stage1_grid_summary_helper_falls_back_to_latest_metrics_version(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_grid_results.py"
        spec = importlib.util.spec_from_file_location("summarize_grid_results", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            version_1 = tmpdir / "logs" / "demo" / "version_1"
            version_2 = tmpdir / "logs" / "demo" / "version_2"
            version_1.mkdir(parents=True)
            version_2.mkdir(parents=True)
            (version_1 / "metrics.csv").write_text(
                "epoch,val/total_loss,val/bin_accuracy,val/regression_mse\n"
                "0,0.80,0.60,0.20\n"
            )
            (version_2 / "metrics.csv").write_text(
                "epoch,val/total_loss,val/bin_accuracy,val/regression_mse\n"
                "0,0.70,0.65,0.18\n"
            )

            requested = tmpdir / "logs" / "demo" / "version_0" / "metrics.csv"
            summary = module.extract_best_metrics(requested)

        self.assertEqual(0.70, summary["best_val_total_loss"])
        self.assertEqual(0.65, summary["best_val_bin_accuracy"])
        self.assertEqual(0.18, summary["best_val_regression_mse"])

    def test_stage1_grid_summary_helper_records_primary_and_secondary_metrics(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_grid_results.py"
        spec = importlib.util.spec_from_file_location("summarize_grid_results", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            metrics_dir = tmpdir / "logs" / "demo" / "version_0"
            metrics_dir.mkdir(parents=True)
            metrics_path = metrics_dir / "metrics.csv"
            metrics_path.write_text(
                "epoch,val/total_loss,val/bin_accuracy,val/regression_mse\n"
                "0,1.20,0.40,0.30\n"
                "1,0.90,0.55,0.25\n"
                "2,1.10,0.60,0.35\n"
            )

            summary = module.extract_best_metrics(metrics_path)

        self.assertEqual(0.90, summary["best_val_total_loss"])
        self.assertEqual(0.55, summary["best_val_bin_accuracy"])
        self.assertEqual(0.25, summary["best_val_regression_mse"])


    def test_nonzero_otu_analysis_counts_exclude_cls_token(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_nonzero_otu_counts.py"
        spec = importlib.util.spec_from_file_location("analyze_nonzero_otu_counts", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        data = {
            "samples": ["s1", "s2"],
            "otu_data": {
                "s1": {"input_ids": np.array([-1, 1, 2, 3], dtype=np.int64)},
                "s2": {"input_ids": np.array([-1, 7], dtype=np.int64)},
            },
        }

        count_table = module.compute_nonzero_otu_counts(data)

        self.assertEqual([3, 1], count_table["nonzero_otu_count"].tolist())

    def test_nonzero_otu_analysis_summarizes_distribution(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_nonzero_otu_counts.py"
        spec = importlib.util.spec_from_file_location("analyze_nonzero_otu_counts", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        count_table = pd.DataFrame({"nonzero_otu_count": [1, 2, 3, 4, 5]})
        summary = module.summarize_counts(count_table)

        self.assertEqual(5, summary["num_samples"])
        self.assertEqual(1, summary["min"])
        self.assertEqual(5, summary["max"])
        self.assertAlmostEqual(3.0, summary["median"])
        self.assertAlmostEqual(4.6, summary["p90"])


    def test_nonzero_otu_analysis_summarizes_top_k_tradeoffs(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_nonzero_otu_counts.py"
        spec = importlib.util.spec_from_file_location("analyze_nonzero_otu_counts", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        data = {
            "samples": ["s1", "s2", "s3"],
            "otu_data": {
                "s1": {"abundance_values": np.array([0.0, 0.5, 0.2, 1e-4], dtype=np.float32)},
                "s2": {"abundance_values": np.array([0.0, 0.9, 0.8, 0.7], dtype=np.float32)},
                "s3": {"abundance_values": np.array([0.0, 0.6, 0.4], dtype=np.float32)},
            },
        }

        summary = module.summarize_top_k_tradeoffs(data, [2, 3], floor_abundance=1e-4)

        top2 = summary.loc[summary["top_k_otus"] == 2].iloc[0]
        top3 = summary.loc[summary["top_k_otus"] == 3].iloc[0]
        self.assertEqual(2, int(top2["truncated_samples"]))
        self.assertAlmostEqual(2 / 3, float(top2["truncated_fraction"]))
        self.assertEqual(2, int(top3["eligible_samples"]))
        self.assertAlmostEqual(0.5, float(top3["floor_or_below_fraction"]))

    def test_nonzero_otu_analysis_builds_top_k_recommendations(self):
        module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_nonzero_otu_counts.py"
        spec = importlib.util.spec_from_file_location("analyze_nonzero_otu_counts", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        tradeoff_table = pd.DataFrame(
            {
                "top_k_otus": [512, 768, 1024],
                "truncated_fraction": [0.11, 0.04, 0.008],
                "floor_or_below_fraction": [0.08, 0.20, 0.30],
            }
        )

        recommendations = module.build_top_k_recommendations(tradeoff_table)

        self.assertEqual(768, recommendations["balanced"])
        self.assertEqual(1024, recommendations["conservative"])
        self.assertEqual(512, recommendations["aggressive"])


if __name__ == "__main__":
    unittest.main()
