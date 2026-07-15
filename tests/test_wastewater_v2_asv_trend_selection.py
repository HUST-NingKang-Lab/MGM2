import unittest

import numpy as np
import pandas as pd

from scripts.downstreams.wastewater_v2.plot_token_asv_trends import select_best_token_asvs


class WastewaterV2ASVTrendSelectionTests(unittest.TestCase):
    def test_selects_high_variation_examples_from_low_medium_high_abundance_bins(self):
        rows = []
        abundances = [0.001, 0.002, 0.004, 0.007, 0.012, 0.02, 0.03, 0.05, 0.08]
        for idx, mean_abundance in enumerate(abundances):
            asv_id = f"ASV{idx}"
            wave = np.sin(np.linspace(0, np.pi, 10))
            variation_factor = 0.2 if idx % 3 == 0 else 0.8
            true = mean_abundance + mean_abundance * variation_factor * wave
            for run_label in ["token_large", "token_xlarge"]:
                pred = true[::-1] if run_label == "token_xlarge" else true
                for horizon, (y_true, y_pred) in enumerate(zip(true, pred), start=1):
                    rows.append(
                        {
                            "run_label": run_label,
                            "plant": "PlantA",
                            "window_idx": 0,
                            "horizon": horizon,
                            "sample_id": f"s{horizon}",
                            "date": horizon,
                            "asv_id": asv_id,
                            "y_true": float(y_true),
                            "y_pred": float(y_pred),
                        }
                    )
        predictions = pd.DataFrame(rows)

        selected = select_best_token_asvs(
            predictions,
            top_n=6,
            min_mean_abundance=0.0,
            min_range=0.0,
            min_std=1e-9,
            examples_per_bin=2,
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(selected["abundance_bin"].value_counts().to_dict(), {"Low": 2, "Medium": 2, "High": 2})
        self.assertNotIn("ASV0", set(selected["asv_id"]))
        self.assertNotIn("ASV3", set(selected["asv_id"]))
        self.assertNotIn("ASV6", set(selected["asv_id"]))


if __name__ == "__main__":
    unittest.main()
