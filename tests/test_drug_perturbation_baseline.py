import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "downstreams" / "drug_perturbation" / "baseline.py"


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("drug_perturbation_baseline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_drug_key_uses_library_name_for_embedding_join():
    baseline = load_baseline_module()
    meta = pd.DataFrame(
        {
            "Run": ["r1", "r2"],
            "Type": ["drug", "drug"],
            "Condition": ["Pefloxacin-mesylate", "Pefloxacin_mesylate"],
            "Drug_Name_in_Library": ["Pefloxacin mesylate", "Pefloxacin mesylate"],
            "SMILES": ["S", "S"],
        }
    )
    index = pd.DataFrame(
        {
            "Condition": ["Pefloxacin_mesylate"],
            "Drug_Name_in_Library": ["Pefloxacin mesylate"],
            "embedding_idx": [17],
        }
    )

    annotated = baseline.annotate_canonical_drugs(meta, index)

    assert annotated["drug_key"].tolist() == ["Pefloxacin mesylate", "Pefloxacin mesylate"]
    assert annotated["embedding_idx"].tolist() == [17, 17]


def test_matched_controls_are_plate_matched_and_exclude_exp06_empty_controls():
    baseline = load_baseline_module()
    meta = pd.DataFrame(
        {
            "Run": ["d1", "c1", "c2", "d2", "dmso", "empty"],
            "Experiment": ["Exp01", "Exp01", "Exp01", "Exp06", "Exp06", "Exp06"],
            "Donor": ["SIC0", "SIC0", "SIC0", "SIC1", "SIC1", "SIC1"],
            "Plate": ["1", "1", "2", "SIC1", "SIC1", "SIC1"],
            "Condition": ["DrugA", "control", "control", "DrugB", "DMSO", "empty3"],
            "Type": ["drug", "control", "control", "drug", "control", "control"],
        }
    )

    matches = baseline.build_matched_control_map(meta)

    assert matches["d1"] == ["c1"]
    assert matches["d2"] == ["dmso"]


def test_leave_one_pair_folds_keep_replicates_together():
    baseline = load_baseline_module()
    examples = pd.DataFrame(
        {
            "Run": ["r1", "r2", "r3"],
            "Donor": ["SIC0", "SIC0", "SIC1"],
            "drug_key": ["DrugA", "DrugA", "DrugA"],
        }
    )

    folds = list(baseline.iter_leave_one_groups(examples, mode="pair"))

    assert len(folds) == 2
    held_out_runs = [set(examples.loc[test_idx, "Run"]) for _, test_idx, _ in folds]
    assert {"r1", "r2"} in held_out_runs
    assert {"r3"} in held_out_runs


def test_mean_delta_baselines_use_specific_effects_and_fallback_to_global():
    baseline = load_baseline_module()
    train = pd.DataFrame(
        {
            "Donor": ["SIC0", "SIC1", "SIC1"],
            "drug_key": ["DrugA", "DrugA", "DrugB"],
        }
    )
    test = pd.DataFrame(
        {
            "Donor": ["SIC0", "SIC2"],
            "drug_key": ["DrugA", "DrugC"],
        }
    )
    train_delta = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    baseline_profiles = np.zeros((2, 2), dtype=np.float32)

    predictions = baseline.predict_mean_delta_baselines(train, test, train_delta, baseline_profiles)

    assert np.allclose(predictions["global_mean_delta"][0], [4.0 / 3.0, 2.0 / 3.0])
    assert np.allclose(predictions["drug_mean_delta"][0], [2.0, 0.0])
    assert np.allclose(predictions["donor_mean_delta"][0], [1.0, 0.0])
    assert np.allclose(predictions["drug_mean_delta"][1], [4.0 / 3.0, 2.0 / 3.0])
    assert np.allclose(predictions["donor_mean_delta"][1], [4.0 / 3.0, 2.0 / 3.0])
