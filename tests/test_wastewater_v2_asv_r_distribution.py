import unittest

import pandas as pd

from scripts.downstreams.wastewater_v2.plot_asv_trend_r_distribution import (
    compute_asv_trend_correlations,
    compute_box_statistics,
    compute_pairwise_tests,
    compute_plant_mean_correlations,
)


class WastewaterV2ASVRDistributionTests(unittest.TestCase):
    def test_computes_per_plant_asv_correlations_and_skips_constant_series(self):
        frame = pd.DataFrame(
            [
                *[
                    {
                        "run_label": "model_a",
                        "label": "Model A",
                        "plant": "P1",
                        "asv_id": "ASV1",
                        "horizon": idx + 1,
                        "y_true": value,
                        "y_pred": value,
                    }
                    for idx, value in enumerate([0.1, 0.2, 0.3, 0.4])
                ],
                *[
                    {
                        "run_label": "model_a",
                        "label": "Model A",
                        "plant": "P1",
                        "asv_id": "ASV2",
                        "horizon": idx + 1,
                        "y_true": value,
                        "y_pred": 0.5 - value,
                    }
                    for idx, value in enumerate([0.1, 0.2, 0.3, 0.4])
                ],
                *[
                    {
                        "run_label": "model_a",
                        "label": "Model A",
                        "plant": "P1",
                        "asv_id": "ASV3",
                        "horizon": idx + 1,
                        "y_true": value,
                        "y_pred": 0.2,
                    }
                    for idx, value in enumerate([0.1, 0.2, 0.3, 0.4])
                ],
            ]
        )

        result = compute_asv_trend_correlations(frame)

        self.assertEqual(set(result["asv_id"]), {"ASV1", "ASV2"})
        values = dict(zip(result["asv_id"], result["trend_pearson"]))
        self.assertAlmostEqual(values["ASV1"], 1.0)
        self.assertAlmostEqual(values["ASV2"], -1.0)

    def test_pairwise_tests_use_plant_level_mean_deltas(self):
        correlations = pd.DataFrame(
            [
                {"run_label": "token_xlarge", "label": "MGM2 xlarge", "plant": "P1", "asv_id": "A1", "trend_pearson": 0.8},
                {"run_label": "token_xlarge", "label": "MGM2 xlarge", "plant": "P1", "asv_id": "A2", "trend_pearson": 0.6},
                {"run_label": "gru", "label": "GRU", "plant": "P1", "asv_id": "A1", "trend_pearson": 0.4},
                {"run_label": "gru", "label": "GRU", "plant": "P1", "asv_id": "A2", "trend_pearson": 0.2},
                {"run_label": "token_xlarge", "label": "MGM2 xlarge", "plant": "P2", "asv_id": "A1", "trend_pearson": 0.5},
                {"run_label": "token_xlarge", "label": "MGM2 xlarge", "plant": "P2", "asv_id": "A2", "trend_pearson": 0.3},
                {"run_label": "gru", "label": "GRU", "plant": "P2", "asv_id": "A1", "trend_pearson": 0.1},
                {"run_label": "gru", "label": "GRU", "plant": "P2", "asv_id": "A2", "trend_pearson": -0.1},
            ]
        )

        plant_tests, _ = compute_pairwise_tests(correlations, candidate="token_xlarge", seed=1)
        row = plant_tests[plant_tests["reference"].eq("gru")].iloc[0]

        self.assertEqual(row["n_pairs"], 2)
        self.assertAlmostEqual(row["mean_delta"], 0.4)
        self.assertAlmostEqual(row["fraction_positive_delta"], 1.0)

    def test_computes_plant_mean_correlations_for_each_model(self):
        correlations = pd.DataFrame(
            [
                {"run_label": "token_xlarge", "label": "MGM2 xlarge", "plant": "P1", "asv_id": "A1", "trend_pearson": 0.8},
                {"run_label": "token_xlarge", "label": "MGM2 xlarge", "plant": "P1", "asv_id": "A2", "trend_pearson": 0.6},
                {"run_label": "gru", "label": "GRU", "plant": "P1", "asv_id": "A1", "trend_pearson": 0.4},
                {"run_label": "gru", "label": "GRU", "plant": "P1", "asv_id": "A2", "trend_pearson": 0.2},
            ]
        )

        plant_mean = compute_plant_mean_correlations(correlations)

        values = dict(zip(plant_mean["run_label"], plant_mean["plant_mean_r"]))
        self.assertAlmostEqual(values["token_xlarge"], 0.7)
        self.assertAlmostEqual(values["gru"], 0.3)

    def test_computes_box_statistics_for_asv_level_distribution(self):
        correlations = pd.DataFrame(
            [
                {"run_label": "gru", "label": "GRU", "plant": "P1", "asv_id": f"A{i}", "trend_pearson": value}
                for i, value in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0])
            ]
        )

        stats = compute_box_statistics(correlations)
        row = stats.iloc[0]

        self.assertEqual(row["run_label"], "gru")
        self.assertEqual(row["n"], 5)
        self.assertAlmostEqual(row["median"], 0.0)
        self.assertAlmostEqual(row["q25"], -0.5)
        self.assertAlmostEqual(row["q75"], 0.5)


if __name__ == "__main__":
    unittest.main()
