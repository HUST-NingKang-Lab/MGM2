import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "downstreams" / "drug_perturbation" / "train_mlp_baseline.py"


def load_mlp_module():
    spec = importlib.util.spec_from_file_location("drug_perturbation_mlp", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DrugPerturbationMLPTests(unittest.TestCase):
    def test_standardizer_uses_training_statistics_for_transform(self):
        mlp = load_mlp_module()
        train = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
        test = np.array([[5.0, 10.0]], dtype=np.float32)

        scaler = mlp.Standardizer.fit(train)
        transformed_train = scaler.transform(train)
        transformed_test = scaler.transform(test)

        self.assertTrue(np.allclose(transformed_train.mean(axis=0), [0.0, 0.0], atol=1e-6))
        self.assertTrue(np.allclose(transformed_train.std(axis=0), [1.0, 1.0], atol=1e-6))
        self.assertTrue(np.isfinite(transformed_test).all())

    def test_grouped_validation_split_keeps_groups_disjoint(self):
        mlp = load_mlp_module()
        examples = pd.DataFrame(
            {
                "Donor": ["SIC0", "SIC0", "SIC1", "SIC2"],
                "drug_key": ["A", "A", "B", "C"],
            }
        )

        train_idx, val_idx = mlp.grouped_train_val_split(examples, mode="pair", val_fraction=0.5, seed=7)

        train_groups = set(zip(examples.iloc[train_idx]["Donor"], examples.iloc[train_idx]["drug_key"]))
        val_groups = set(zip(examples.iloc[val_idx]["Donor"], examples.iloc[val_idx]["drug_key"]))
        self.assertTrue(val_groups)
        self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_mlp_forward_predicts_delta_dimension(self):
        mlp = load_mlp_module()
        model = mlp.DrugPerturbationMLP(
            community_dim=4,
            drug_dim=3,
            donor_dim=2,
            hidden_size=8,
            dropout=0.0,
        )

        output = model(
            torch.zeros(5, 4),
            torch.zeros(5, 3),
            torch.zeros(5, 2),
        )

        self.assertEqual(tuple(output.shape), (5, 4))

    def test_prediction_export_metadata_tracks_test_rows_and_methods(self):
        mlp = load_mlp_module()
        test_examples = pd.DataFrame(
            {
                "Run": ["r1", "r2"],
                "Donor": ["SIC0", "SIC1"],
                "drug_key": ["DrugA", "DrugB"],
            }
        )
        predictions = {
            "mlp_delta": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            "no_effect": np.array([[0.0, 0.2], [0.3, 0.0]], dtype=np.float32),
        }

        metadata, matrices = mlp.build_prediction_export(
            split_mode="pair",
            fold_idx=3,
            group_label="SIC0::DrugA",
            test_examples=test_examples,
            predictions=predictions,
        )

        self.assertEqual(metadata["Run"].tolist(), ["r1", "r2"])
        self.assertEqual(metadata["split_mode"].tolist(), ["pair", "pair"])
        self.assertEqual(metadata["fold"].tolist(), [3, 3])
        self.assertEqual(metadata["group"].tolist(), ["SIC0::DrugA", "SIC0::DrugA"])
        self.assertTrue(np.allclose(matrices["mlp_delta"], predictions["mlp_delta"]))
        self.assertTrue(np.allclose(matrices["no_effect"], predictions["no_effect"]))

    def test_finalize_raw_predictions_clips_and_optionally_normalizes(self):
        mlp = load_mlp_module()
        predictions = {
            "mlp_delta": np.array([[-1.0, 2.0, 3.0], [0.0, -0.5, 0.5]], dtype=np.float32)
        }

        clipped = mlp.finalize_raw_predictions(predictions, clip_nonnegative=True, normalize_prediction="none")
        normalized = mlp.finalize_raw_predictions(predictions, clip_nonnegative=True, normalize_prediction="topk")

        self.assertTrue(np.all(clipped["mlp_delta"] >= 0))
        self.assertTrue(np.allclose(clipped["mlp_delta"], [[0.0, 2.0, 3.0], [0.0, 0.0, 0.5]]))
        self.assertTrue(np.allclose(normalized["mlp_delta"].sum(axis=1), [1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
