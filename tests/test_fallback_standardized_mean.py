import importlib.util
from pathlib import Path
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "regular_analysis" / "fallback_umap_standardized_mean.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fallback_umap_standardized_mean", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FallbackStandardizedMeanTests(unittest.TestCase):
    def test_standardized_mean_matches_mean_of_standardized_otus(self):
        module = load_module()
        raw_matched = np.asarray(
            [
                [1.0, 2.0, 1.0],
                [3.0, 6.0, 5.0],
                [10.0, 4.0, 2.0],
                [12.0, 8.0, 6.0],
            ],
            dtype=np.float32,
        )
        top_genera = [("g1", [0, 1]), ("g2", [2, 3])]
        scaler = StandardScaler().fit(raw_matched)

        prototypes = module.build_standardized_mean_prototypes(raw_matched, top_genera, scaler)
        standardized = scaler.transform(raw_matched)

        self.assertTrue(np.allclose(prototypes[0], standardized[[0, 1]].mean(axis=0)))
        self.assertTrue(np.allclose(prototypes[1], standardized[[2, 3]].mean(axis=0)))


if __name__ == "__main__":
    unittest.main()
