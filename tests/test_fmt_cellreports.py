import importlib.util
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _load_cellreports_utils():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "downstreams" / "FMT_cellreports" / "cellreports_utils.py"
    spec = importlib.util.spec_from_file_location("fmt_cellreports_utils", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FMTCellReportsTests(unittest.TestCase):
    def test_normalize_sample_id_strips_only_trailing_replicate_suffix(self):
        module = _load_cellreports_utils()

        self.assertEqual(module.normalize_sample_id("B003_1"), "B003")
        self.assertEqual(module.normalize_sample_id("SAMN18106008_1"), "SAMN18106008")
        self.assertEqual(module.normalize_sample_id("FDM002W0"), "FDM002W0")
        self.assertEqual(module.normalize_sample_id("BL_ABC"), "BL_ABC")

    def test_cellreports_feature_id_preserves_kingdom_and_species_name(self):
        module = _load_cellreports_utils()

        feature_id = module.make_feature_id("bacteria", "Bacteroides fragilis")

        self.assertEqual(feature_id, "cellreports|bacteria|Bacteroides fragilis")
        self.assertEqual(module.parse_feature_id(feature_id), ("bacteria", "Bacteroides fragilis"))
        self.assertEqual(module.feature_genus("uncultured Methanoregula sp."), "Methanoregula")
        self.assertEqual(module.feature_genus("Bacteroides fragilis"), "Bacteroides")

    def test_loco_cohort_uses_consecutive_disease_runs(self):
        module = _load_cellreports_utils()

        cohorts = module.assign_consecutive_disease_cohorts([1, 1, 1, 2, 2, 2, 2, 1])

        self.assertEqual(
            cohorts,
            [
                "cohort_001__disease_1",
                "cohort_001__disease_1",
                "cohort_001__disease_1",
                "cohort_002__disease_2",
                "cohort_002__disease_2",
                "cohort_002__disease_2",
                "cohort_002__disease_2",
                "cohort_003__disease_1",
            ],
        )

    def test_cellreports_embedding_map_building_handles_canonical_otu_column(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "downstreams" / "FMT_cellreports" / "build_cellreports_embeddings.py"
        spec = importlib.util.spec_from_file_location("build_cellreports_embeddings", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        original_read_csv = module.pd.read_csv
        try:
            module.pd.read_csv = lambda *args, **kwargs: iter(
                [
                    pd.DataFrame(
                        {
                            "OTU": ["99_1"],
                            "Tax": ["Bacteria;Firmicutes;Bacilli"],
                            "RepSpecies": ["Bacillus subtilis"],
                        }
                    )
                ]
            )
            species_map, genus_map, kingdom_map, _ = module.build_maps(
                SimpleNamespace(otu_info=Path("unused.tsv"), chunk_size=10),
                np.ones((1, 4), dtype=np.float32),
                ["99_1"],
            )
        finally:
            module.pd.read_csv = original_read_csv

        self.assertIn(("bacteria", "bacillus subtilis"), species_map)
        self.assertIn(("bacteria", "bacillus"), genus_map)
        self.assertIn("bacteria", kingdom_map)


if __name__ == "__main__":
    unittest.main()
