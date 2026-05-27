import pickle
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from src.configs.Config import MGM2Config
from src.data.FMTDataset import (
    FMTCommunityClassificationCollator,
    FMTCommunityClassificationDataset,
    FMTCommunityPredictionCollator,
    FMTCommunityPredictionDataset,
)
from src.models.FMTModel import FMTCommunityClassificationModel, FMTCommunityPredictionModel


def _load_fmt_mlp_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "downstreams" / "FMT" / "train_mlp_baseline.py"
    spec = importlib.util.spec_from_file_location("fmt_train_mlp_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fmt_mlp_classification_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "downstreams" / "FMT" / "train_mlp_classification_baseline.py"
    spec = importlib.util.spec_from_file_location("fmt_train_mlp_classification_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fmt_community_classification_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "src" / "training" / "FMTCommunityClassification.py"
    spec = importlib.util.spec_from_file_location("fmt_community_classification", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fmt_community_prediction_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "src" / "training" / "FMTCommunityPrediction.py"
    spec = importlib.util.spec_from_file_location("fmt_community_prediction", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fmt_5fold_utils_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "downstreams" / "FMT" / "fmt_5fold_utils.py"
    spec = importlib.util.spec_from_file_location("fmt_5fold_utils", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fmt_5fold_plot_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "downstreams" / "FMT" / "plot_fmt_5fold_results.py"
    spec = importlib.util.spec_from_file_location("plot_fmt_5fold_results", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FMTMGM2Tests(unittest.TestCase):
    def _write_fmt_inputs(self, root: Path):
        with h5py.File(root / "resolved.h5", "w") as handle:
            handle.create_dataset("embeddings", data=np.arange(12, dtype=np.float32).reshape(3, 4))
            handle.create_dataset("otu_ids", data=np.array([b"otu_a", b"otu_b", b"otu_c"]))

        payload = {
            "samples": ["donor_1", "recipient_1", "post_1"],
            "otu_data": {
                "donor_1": {
                    "input_ids": np.array([-1, 0, 2], dtype=np.int64),
                    "attention_mask": np.array([1, 1, 1], dtype=np.int64),
                    "abundance_values": np.array([0.0, 0.2, 0.8], dtype=np.float32),
                },
                "recipient_1": {
                    "input_ids": np.array([-1, 1], dtype=np.int64),
                    "attention_mask": np.array([1, 1], dtype=np.int64),
                    "abundance_values": np.array([0.0, 1.0], dtype=np.float32),
                },
                "post_1": {
                    "input_ids": np.array([-1, 0, 1], dtype=np.int64),
                    "attention_mask": np.array([1, 1, 1], dtype=np.int64),
                    "abundance_values": np.array([0.0, 0.4, 0.6], dtype=np.float32),
                },
            },
            "otu_id_to_idx": {"otu_a": 0, "otu_b": 1, "otu_c": 2},
            "ntv3_embedding_path": str(root / "resolved.h5"),
            "format_version": "mgm2_v1",
        }
        with (root / "community.pkl").open("wb") as handle:
            pickle.dump(payload, handle)

        pd.DataFrame(
            [
                {
                    "TargetID": "fmt_1::post_1",
                    "fmt_id": "fmt_1",
                    "donor_sample_id": "donor_1",
                    "recipient_sample_id": "recipient_1",
                    "post_sample_id": "post_1",
                    "timepoint_fmt": 14,
                    "response_label": "responder",
                }
            ]
        ).to_csv(root / "meta.csv", index=False)

        pd.DataFrame(
            [
                {
                    "TargetID": "fmt_1",
                    "fmt_id": "fmt_1",
                    "study": "test_study",
                    "donor_sample_id": "donor_1",
                    "recipient_sample_id": "recipient_1",
                    "response_label": "responder",
                },
                {
                    "TargetID": "fmt_missing",
                    "fmt_id": "fmt_missing",
                    "study": "test_study",
                    "donor_sample_id": "donor_1",
                    "recipient_sample_id": "recipient_1",
                    "response_label": "missing",
                },
            ]
        ).to_csv(root / "response_meta.csv", index=False)

    def test_dataset_materializes_donor_recipient_and_post_on_shared_feature_axis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fmt_inputs(root)

            dataset = FMTCommunityPredictionDataset(
                community_data_file=root / "community.pkl",
                metadata_file=root / "meta.csv",
                ntv3_embedding_path=root / "resolved.h5",
            )
            sample = dataset[0]

            self.assertEqual(sample["target_id"], "fmt_1::post_1")
            self.assertEqual(sample["input_ids"].tolist(), [-1, 0, 1, 2])
            self.assertEqual(sample["embeddings"].shape, (4, 4))
            self.assertTrue(torch.all(sample["supervision_mask"][1:]))
            np.testing.assert_allclose(sample["donor_abundance_raw"][1:].numpy(), [0.2, 0.0, 0.8])
            np.testing.assert_allclose(sample["recipient_abundance_raw"][1:].numpy(), [0.0, 1.0, 0.0])
            np.testing.assert_allclose(sample["post_abundance_raw"][1:].numpy(), [0.4, 0.6, 0.0])

    def test_collator_and_cross_attention_model_accept_two_communities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fmt_inputs(root)
            dataset = FMTCommunityPredictionDataset(root / "community.pkl", root / "meta.csv", root / "resolved.h5")
            batch = FMTCommunityPredictionCollator()([dataset[0], dataset[0]])

            config = MGM2Config()
            config.model.embedding_dim = 4
            config.model.hidden_size = 8
            config.model.num_hidden_layers = 1
            config.model.num_attention_heads = 2
            config.model.intermediate_size = 16
            model = FMTCommunityPredictionModel(config.model.__dict__, cross_attention_heads=2, dropout=0.0)

            outputs = model(
                input_ids=batch["input_ids"],
                embeddings=batch["embeddings"],
                recipient_abundance_values=batch["recipient_abundance_values"],
                donor_abundance_values=batch["donor_abundance_values"],
                attention_mask=batch["attention_mask"],
                timepoint=batch["timepoint_fmt"],
            )

            self.assertEqual(outputs["pred_post_abundance"].shape, batch["post_abundance_values"].shape)
            self.assertEqual(outputs["cross_attention_weights"].shape[0], 2)

    def test_mlp_baseline_builds_donor_recipient_time_inputs_and_normalized_predictions(self):
        module = _load_fmt_mlp_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fmt_inputs(root)
            dataset = FMTCommunityPredictionDataset(root / "community.pkl", root / "meta.csv", root / "resolved.h5")

            arrays = module.build_arrays(dataset)

            self.assertEqual(arrays.inputs.shape, (1, 7))
            self.assertEqual(arrays.targets.shape, (1, 3))
            np.testing.assert_allclose(arrays.recipient_raw, [[0.0, 1.0, 0.0]], atol=1e-6)
            np.testing.assert_allclose(arrays.donor_raw, [[0.2, 0.0, 0.8]], atol=1e-6)
            self.assertEqual(arrays.metadata.loc[0, "target_id"], "fmt_1::post_1")

            finalized = module.finalize_raw_predictions(np.array([[0.2, -0.1, 0.3]], dtype=np.float32))
            np.testing.assert_allclose(finalized, [[0.4, 0.0, 0.6]], atol=1e-6)

    def test_classification_dataset_filters_missing_response_and_encodes_two_communities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fmt_inputs(root)

            dataset = FMTCommunityClassificationDataset(
                community_data_file=root / "community.pkl",
                metadata_file=root / "response_meta.csv",
                ntv3_embedding_path=root / "resolved.h5",
            )
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.class_names, ["non-responder", "responder"])
            self.assertEqual(sample["target_id"], "fmt_1")
            self.assertEqual(sample["label"].item(), 1)
            np.testing.assert_allclose(sample["donor_abundance_raw"][1:].numpy(), [0.2, 0.0, 0.8])
            np.testing.assert_allclose(sample["recipient_abundance_raw"][1:].numpy(), [0.0, 1.0, 0.0])

    def test_classification_collator_and_model_emit_response_logits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fmt_inputs(root)
            dataset = FMTCommunityClassificationDataset(root / "community.pkl", root / "response_meta.csv", root / "resolved.h5")
            batch = FMTCommunityClassificationCollator()([dataset[0], dataset[0]])

            config = MGM2Config()
            config.model.embedding_dim = 4
            config.model.hidden_size = 8
            config.model.num_hidden_layers = 1
            config.model.num_attention_heads = 2
            config.model.intermediate_size = 16
            model = FMTCommunityClassificationModel(
                config.model.__dict__,
                num_classes=2,
                cross_attention_heads=2,
                dropout=0.0,
            )

            outputs = model(
                input_ids=batch["input_ids"],
                embeddings=batch["embeddings"],
                recipient_abundance_values=batch["recipient_abundance_values"],
                donor_abundance_values=batch["donor_abundance_values"],
                attention_mask=batch["attention_mask"],
            )

            self.assertEqual(outputs["logits"].shape, (2, 2))
            self.assertEqual(outputs["cross_attention_weights"].shape[0], 2)

    def test_mlp_classification_baseline_builds_response_arrays(self):
        module = _load_fmt_mlp_classification_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fmt_inputs(root)
            dataset = FMTCommunityClassificationDataset(root / "community.pkl", root / "response_meta.csv", root / "resolved.h5")

            arrays = module.build_arrays(dataset)

            self.assertEqual(arrays.inputs.shape, (1, 9))
            self.assertEqual(arrays.labels.tolist(), [1])
            self.assertEqual(arrays.class_names, ["non-responder", "responder"])
            self.assertEqual(arrays.metadata.loc[0, "target_id"], "fmt_1")

    def test_fmt_classification_predictions_writer_accepts_bfloat16_tensors(self):
        module = _load_fmt_community_classification_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.csv"
            predictions = [
                {
                    "target_id": ["fmt_1"],
                    "fmt_id": ["fmt_1"],
                    "study": ["test_study"],
                    "indication": ["test_indication"],
                    "response_label": ["responder"],
                    "donor_sample_id": ["donor_1"],
                    "recipient_sample_id": ["recipient_1"],
                    "label_id": torch.tensor([1], dtype=torch.long),
                    "pred_id": torch.tensor([1], dtype=torch.long),
                    "probabilities": torch.tensor([[0.25, 0.75]], dtype=torch.bfloat16),
                    "logits": torch.tensor([[-1.0, 1.0]], dtype=torch.bfloat16),
                }
            ]

            module.write_predictions(output_path, ["non-responder", "responder"], predictions, {"fold_index": 0})

            table = pd.read_csv(output_path)
            self.assertEqual(table.loc[0, "pred_label"], "responder")
            self.assertAlmostEqual(table.loc[0, "prob_responder"], 0.75, places=3)

    def test_fmt_prediction_resolve_eval_ckpt_returns_real_checkpoint_path(self):
        module = _load_fmt_community_prediction_module()

        class Checkpoint:
            best_model_path = "/tmp/best.ckpt"
            last_model_path = "/tmp/last.ckpt"

        class Trainer:
            checkpoint_callback = Checkpoint()

        self.assertEqual(module.resolve_eval_ckpt(Trainer(), "best"), "/tmp/best.ckpt")
        self.assertEqual(module.resolve_eval_ckpt(Trainer(), "last"), "/tmp/last.ckpt")

    def test_fmt_prediction_writer_accepts_bfloat16_tensors(self):
        module = _load_fmt_community_prediction_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fmt_inputs(root)
            dataset = FMTCommunityPredictionDataset(root / "community.pkl", root / "meta.csv", root / "resolved.h5")
            output_path = root / "predicted_abundance.h5"
            predictions = [
                {
                    "target_id": ["fmt_1::post_1"],
                    "fmt_id": ["fmt_1"],
                    "study": ["test_study"],
                    "response_label": ["responder"],
                    "donor_sample_id": ["donor_1"],
                    "recipient_sample_id": ["recipient_1"],
                    "post_sample_id": ["post_1"],
                    "timepoint_fmt": torch.tensor([14.0], dtype=torch.bfloat16),
                    "pred_post_raw": torch.tensor([[0.4, 0.6, 0.0]], dtype=torch.bfloat16),
                    "true_post_raw": torch.tensor([[0.4, 0.6, 0.0]], dtype=torch.bfloat16),
                    "recipient_raw": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.bfloat16),
                    "donor_raw": torch.tensor([[0.2, 0.0, 0.8]], dtype=torch.bfloat16),
                }
            ]

            module.write_predictions(output_path, dataset, predictions)

            metrics = pd.read_csv(output_path.with_suffix(".metrics.csv"))
            self.assertAlmostEqual(metrics.loc[0, "mgm2_bray_curtis"], 0.0, places=5)

    def test_nested_grouped_cv_indices_separates_outer_test_from_inner_val(self):
        module = _load_fmt_5fold_utils_module()
        metadata = pd.DataFrame(
            {
                "fmt_id": [f"fmt_{idx}" for idx in range(10)],
                "response_label": ["responder" if idx % 2 else "non-responder" for idx in range(10)],
            }
        )

        train_idx, val_idx, test_idx = module.nested_grouped_cv_indices(
            metadata,
            fold_index=0,
            num_folds=5,
            val_ratio=0.25,
            seed=11,
            group_column="fmt_id",
            label_column="response_label",
        )

        train_groups = set(metadata.iloc[train_idx]["fmt_id"])
        val_groups = set(metadata.iloc[val_idx]["fmt_id"])
        test_groups = set(metadata.iloc[test_idx]["fmt_id"])
        self.assertTrue(train_groups)
        self.assertTrue(val_groups)
        self.assertEqual(len(test_groups), 2)
        self.assertFalse(train_groups & val_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(val_groups & test_groups)

    def test_fmt_5fold_plot_loader_reads_mgm2_and_mlp_community_metrics(self):
        module = _load_fmt_5fold_plot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mgm2_dir = root / "community_prediction"
            mlp_dir = root / "mlp_community_prediction"
            mgm2_fold = mgm2_dir / "scale_small" / "seed_11" / "fold_0"
            mlp_fold = mlp_dir / "seed_11" / "fold_0"
            mgm2_fold.mkdir(parents=True)
            mlp_fold.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "target_id": "fmt_1::post_1",
                        "fmt_id": "fmt_1",
                        "study": "study_a",
                        "response_label": "responder",
                        "donor_sample_id": "donor_1",
                        "recipient_sample_id": "recipient_1",
                        "post_sample_id": "post_1",
                        "timepoint_fmt": 14,
                        "mgm2_bray_curtis": 0.3,
                        "recipient_baseline_bray_curtis": 0.5,
                        "mgm2_cosine": 0.8,
                        "recipient_baseline_cosine": 0.6,
                    }
                ]
            ).to_csv(mgm2_fold / "fold_validation_predictions.metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "target_id": "fmt_1::post_1",
                        "fmt_id": "fmt_1",
                        "study": "study_a",
                        "response_label": "responder",
                        "donor_sample_id": "donor_1",
                        "recipient_sample_id": "recipient_1",
                        "post_sample_id": "post_1",
                        "timepoint_fmt": 14,
                        "mlp_bray_curtis": 0.4,
                        "recipient_baseline_bray_curtis": 0.5,
                        "mlp_cosine": 0.7,
                        "recipient_baseline_cosine": 0.6,
                    }
                ]
            ).to_csv(mlp_fold / "predicted_abundance.metrics.csv", index=False)

            metrics = module.load_community_prediction_metrics(mgm2_dir, mlp_dir)

            self.assertEqual(set(metrics["method"]), {"MGM2-small", "MLP"})
            self.assertEqual(metrics.loc[metrics["method"] == "MGM2-small", "fold"].tolist(), [0])
            self.assertAlmostEqual(metrics.loc[metrics["method"] == "MLP", "bray_curtis"].iloc[0], 0.4)

    def test_fmt_5fold_plot_loader_reads_classification_metrics_and_predictions(self):
        module = _load_fmt_5fold_plot_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mgm2_dir = root / "community_classification"
            mlp_dir = root / "mlp_classification"
            mgm2_fold = mgm2_dir / "scale_small" / "seed_11" / "fold_0"
            mlp_fold = mlp_dir / "seed_11" / "fold_0"
            mgm2_fold.mkdir(parents=True)
            mlp_fold.mkdir(parents=True)
            for fold_dir, pred_name, accuracy in [
                (mgm2_fold, "fold_validation_predictions.csv", 0.75),
                (mlp_fold, "test_predictions.csv", 0.5),
            ]:
                with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "accuracy": accuracy,
                            "balanced_accuracy": accuracy,
                            "macro_f1": accuracy,
                            "roc_auc": accuracy,
                            "average_precision": accuracy,
                        },
                        handle,
                    )
                pd.DataFrame(
                    [
                        {
                            "target_id": "fmt_1",
                            "fmt_id": "fmt_1",
                            "study": "study_a",
                            "response_label": "responder",
                            "label_id": 1,
                            "pred_id": 1,
                            "prob_responder": accuracy,
                        }
                    ]
                ).to_csv(fold_dir / pred_name, index=False)

            metrics, predictions = module.load_classification_results(mgm2_dir, mlp_dir)

            self.assertEqual(set(metrics["method"]), {"MGM2-small", "MLP"})
            self.assertEqual(set(predictions["method"]), {"MGM2-small", "MLP"})
            self.assertAlmostEqual(metrics.loc[metrics["method"] == "MGM2-small", "accuracy"].iloc[0], 0.75)


if __name__ == "__main__":
    unittest.main()
