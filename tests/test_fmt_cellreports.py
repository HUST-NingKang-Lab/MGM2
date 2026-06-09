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
        self.assertEqual(module.normalize_sample_id("BL62_combined"), "BL62")
        self.assertEqual(module.normalize_sample_id("BL62_R1"), "BL62")
        self.assertEqual(module.normalize_sample_id("ST0023FR001_S23"), "ST0023FR001")
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

    def test_similarity_label_follows_donor_similarity_increase_definition(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "downstreams" / "FMT_cellreports" / "prepare_similarity_random_splits.py"
        spec = importlib.util.spec_from_file_location("prepare_similarity_random_splits", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        donor = np.asarray([1.0, 0.0], dtype=np.float32)
        recipient_pre = np.asarray([0.0, 1.0], dtype=np.float32)
        recipient_post = np.asarray([0.8, 0.2], dtype=np.float32)

        _, _, delta, label_id, label_name = module.donor_similarity_delta(recipient_pre, recipient_post, donor)

        self.assertGreater(delta, 0)
        self.assertEqual(label_id, 0)
        self.assertEqual(label_name, "similarity_increased")

    def test_similarity_random_split_uses_72_8_20_ratios(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "downstreams" / "FMT_cellreports" / "prepare_similarity_random_splits.py"
        spec = importlib.util.spec_from_file_location("prepare_similarity_random_splits", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        train_idx, val_idx, test_idx = module.random_split_indices(466, 11, 0.72, 0.08, 0.20)

        self.assertEqual((len(train_idx), len(val_idx), len(test_idx)), (336, 37, 93))
        self.assertEqual(len(set(train_idx).intersection(val_idx)), 0)
        self.assertEqual(len(set(train_idx).intersection(test_idx)), 0)
        self.assertEqual(len(set(val_idx).intersection(test_idx)), 0)

    def test_similarity_split_is_stratified_by_disease_code(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "downstreams" / "FMT_cellreports" / "prepare_similarity_random_splits.py"
        spec = importlib.util.spec_from_file_location("prepare_similarity_random_splits", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        metadata = pd.DataFrame({"disease_code": [1] * 10 + [2] * 10 + [3] * 10})

        train_idx, val_idx, test_idx = module.stratified_split_indices(metadata, "disease_code", 11, 0.6, 0.2, 0.2)

        self.assertEqual((len(train_idx), len(val_idx), len(test_idx)), (18, 6, 6))
        for indices, expected_per_disease in [(train_idx, 6), (val_idx, 2), (test_idx, 2)]:
            counts = metadata.iloc[indices]["disease_code"].value_counts().to_dict()
            self.assertEqual(counts, {1: expected_per_disease, 2: expected_per_disease, 3: expected_per_disease})
        self.assertEqual(len(set(train_idx).intersection(val_idx)), 0)
        self.assertEqual(len(set(train_idx).intersection(test_idx)), 0)
        self.assertEqual(len(set(val_idx).intersection(test_idx)), 0)


if __name__ == "__main__":
    unittest.main()
