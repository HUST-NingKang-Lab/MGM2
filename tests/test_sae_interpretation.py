import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.sae_interpretation.activation_alignment import infer_activation_token_type
from src.sae_interpretation.controls import build_negative_controls
from src.sae_interpretation.data_sources import load_allinfo_taxonomy_map
from src.sae_interpretation.evidence import build_evidence_package
from src.sae_interpretation.normalization import add_normalized_activation_columns
from src.sae_interpretation.prompt_rendering import render_codex_task_files
from src.sae_interpretation.schemas import validate_evidence_package, validate_interpretation_output_dir
from src.sae_interpretation.sae_features import (
    dense_feature_activations,
    select_feature_ids,
    sparse_topk_feature_activations,
    sparse_topk_feature_events,
)
from src.sae_interpretation.taxonomy import canonical_otu_id, parse_taxonomy
from src.models.BatchTopKSAE import BatchTopKSAE


def _events():
    return pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "otu_id": "99_a",
                "activation": 10.0,
                "abundance": 0.10,
                "abundance_rank": 1,
                "split": "train",
                "primary_env": "animal",
                "project": "p1",
                "taxonomy": "Bacteria;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus",
            },
            {
                "sample_id": "s2",
                "otu_id": "99_a",
                "activation": 5.0,
                "abundance": 0.01,
                "abundance_rank": 10,
                "split": "val",
                "primary_env": "soil",
                "project": "p2",
                "taxonomy": "Bacteria;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus",
            },
            {
                "sample_id": "s3",
                "otu_id": "99_b",
                "activation": 1.0,
                "abundance": 0.02,
                "abundance_rank": 5,
                "split": "train",
                "primary_env": "animal",
                "project": "p1",
                "taxonomy": "Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia",
            },
            {
                "sample_id": "s4",
                "otu_id": "99_c",
                "activation": 0.1,
                "abundance": 0.03,
                "abundance_rank": 6,
                "split": "train",
                "primary_env": "animal",
                "project": "p3",
                "taxonomy": "Bacteria;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus",
            },
        ]
    )


def test_taxonomy_helpers_parse_canonical_ids_and_ranks():
    assert canonical_otu_id("90_x;96_y;99_abc") == "99_abc"
    assert canonical_otu_id("99_abc") == "99_abc"
    parsed = parse_taxonomy("Bacteria;Firmicutes;Bacilli;Order;Family;Genus")
    assert parsed["kingdom"] == "Bacteria"
    assert parsed["phylum"] == "Firmicutes"
    assert parsed["genus"] == "Genus"
    assert parse_taxonomy(float("nan"))["kingdom"] == "unknown"


def test_activation_token_type_infers_from_sidecar(tmp_path):
    path = tmp_path / "acts.h5"
    path.write_bytes(b"")
    assert infer_activation_token_type(path) == "all"
    Path(str(path) + ".json").write_text(json.dumps({"token_type": "otu"}))
    assert infer_activation_token_type(path) == "otu"
    assert infer_activation_token_type(path, "all") == "all"


def test_allinfo_taxonomy_loader_prefers_gtdb_taxonomy(tmp_path):
    path = tmp_path / "otus.99.allinfo"
    fields = ["MAPv3;90_x;96_y;99_abc", "1", "", "", "", "", "SEQ", "hit", "d__Bacteria;p__Firmicutes;c__Bacilli;o__Order;f__Family;g__Genus", "", "", "", "", "", "Bacteria;Legacy", "", "", "", "", ""]
    path.write_text("\t".join(fields) + "\n")
    taxonomy = load_allinfo_taxonomy_map(path, {"99_abc"})
    assert taxonomy["99_abc"]["taxonomy"] == "d__Bacteria;p__Firmicutes;c__Bacilli;o__Order;f__Family;g__Genus"
    assert taxonomy["99_abc"]["taxonomy_source"] == "otu99_allinfo_gtdb"


def test_normalization_adds_feature_specific_scores():
    events = add_normalized_activation_columns(_events())
    assert events.loc[0, "fmax"] == 1.0
    assert events["inverse_token_frequency"].notna().all()
    assert events["inverse_sample_frequency"].notna().all()
    assert events.loc[0, "normalized_activation"] > events.loc[2, "normalized_activation"]


def test_negative_controls_include_taxonomy_biome_abundance_and_study():
    controls = build_negative_controls(_events(), positive_otu_ids={"99_a"}, positive_sample_ids={"s1", "s2"})
    assert {"same_taxonomy", "same_biome", "same_abundance_rank", "same_study"} <= set(controls)
    assert controls["same_taxonomy"][0]["otu_id"] == "99_c"
    assert controls["same_biome"][0]["sample_id"] in {"s3", "s4"}


def test_evidence_package_validates_and_renders_codex_tasks(tmp_path):
    events = add_normalized_activation_columns(_events())
    evidence = build_evidence_package(feature_id=7, events=events, max_examples=3)
    validate_evidence_package(evidence)
    assert evidence["feature_id"] == 7
    assert evidence["global_stats"]["num_events"] == 4
    assert evidence["top_examples"][0]["otu_id"] == "99_a"
    assert "study_batch_artifact" in evidence["diagnostics"]

    out_dir = tmp_path / "tasks"
    render_codex_task_files(evidence, out_dir)
    initial = out_dir / "feature_000007" / "00_initial_observation.md"
    schema = out_dir / "feature_000007" / "schemas" / "unified_description.schema.json"
    assert initial.exists()
    assert "evidence_anchors" in initial.read_text()
    assert "Do not create template JSON" in initial.read_text()
    classification_prompt = out_dir / "feature_000007" / "06_classification.md"
    classification_text = classification_prompt.read_text()
    assert '"visualization_priority"' not in classification_text
    assert "Do not output `visualization_priority`" in classification_text
    assert schema.exists()
    json.loads(schema.read_text())


def test_taxonomy_signal_prefers_specific_rank_over_kingdom():
    events = pd.DataFrame(
        [
            {
                "sample_id": f"s{i}",
                "otu_id": f"99_{i}",
                "activation": 10.0 - i,
                "abundance": 0.1,
                "abundance_rank": i + 1,
                "split": "train",
                "primary_env": "soil",
                "project": "p1",
                "taxonomy": "Bacteria;Firmicutes_A;Clostridia;Lachnospirales;Lachnospiraceae;unknown",
            }
            for i in range(60)
        ]
        + [
            {
                "sample_id": f"x{i}",
                "otu_id": f"99_x{i}",
                "activation": 1.0,
                "abundance": 0.1,
                "abundance_rank": i + 100,
                "split": "train",
                "primary_env": "soil",
                "project": "p1",
                "taxonomy": "Bacteria;Proteobacteria;Gammaproteobacteria",
            }
            for i in range(40)
        ]
    )

    evidence = build_evidence_package(feature_id=8, events=events, max_examples=100)
    signal = evidence["diagnostics"]["taxonomy_signal"]

    assert signal["best_rank"] == "family"
    assert signal["best_value"] == "Lachnospiraceae"


def test_top_n_feature_selection_uses_scores_not_recent_firing_ties(tmp_path):
    checkpoint = tmp_path / "sae.pt"
    torch.save(
        {
            "model_state_dict": {
                "W_enc": torch.zeros(2, 6),
                "num_tokens_since_fired": torch.zeros(6),
            }
        },
        checkpoint,
    )
    scores = tmp_path / "feature_scores.csv"
    pd.DataFrame(
        {
            "feature_id": [0, 1, 2, 3, 4, 5],
            "activation_sum": [1.0, 50.0, 3.0, 40.0, 2.0, 30.0],
        }
    ).to_csv(scores, index=False)

    selected = select_feature_ids(
        checkpoint,
        top_n_features=3,
        feature_scores_path=scores,
        feature_selection_metric="activation_sum",
    )

    assert selected == [1, 3, 5]


def test_top_n_feature_selection_requires_scores(tmp_path):
    checkpoint = tmp_path / "sae.pt"
    torch.save({"model_state_dict": {"W_enc": torch.zeros(2, 6), "num_tokens_since_fired": torch.zeros(6)}}, checkpoint)

    try:
        select_feature_ids(checkpoint, top_n_features=3)
    except ValueError as exc:
        assert "feature scores" in str(exc)
    else:
        raise AssertionError("top-n feature selection should require feature scores")


def test_all_feature_selection_returns_full_dictionary(tmp_path):
    checkpoint = tmp_path / "sae.pt"
    torch.save({"model_state_dict": {"W_enc": torch.zeros(2, 6), "num_tokens_since_fired": torch.zeros(6)}}, checkpoint)

    selected = select_feature_ids(checkpoint, all_features=True)

    assert selected == [0, 1, 2, 3, 4, 5]


def test_sparse_topk_feature_activations_apply_batchtopk_gate():
    model = BatchTopKSAE(input_dim=3, dict_size=4, top_k=1, input_unit_norm=False)
    with torch.no_grad():
        model.b_dec.zero_()
        model.W_enc.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.5],
                ]
            )
        )
    x = torch.tensor([[3.0, 2.0, 1.0], [0.1, 4.0, 2.0]], dtype=torch.float32).numpy()

    dense = dense_feature_activations(model, x, [0, 1, 2, 3])
    sparse = sparse_topk_feature_activations(model, x, [0, 1, 2, 3])
    rows, cols, values = sparse_topk_feature_events(model, x, device="cpu")

    assert (dense > 0).sum() > (sparse > 0).sum()
    assert int((sparse > 0).sum()) == model.top_k * len(x)
    assert len(values) == model.top_k * len(x)
    assert sorted(zip(rows.tolist(), cols.tolist())) == sorted(zip(*np.nonzero(sparse)))


def test_interpretation_output_validation_requires_all_agent_stages(tmp_path):
    feature_dir = tmp_path / "feature_000001"
    feature_dir.mkdir()
    (feature_dir / "unified_description.json").write_text(
        json.dumps(
            {
                "short_name": "example",
                "summary": "example",
                "activation_pattern": "example",
                "taxonomic_scope": "example",
                "ecological_context": "example",
                "abundance_context": "example",
                "community_context": "example",
                "exemplars": [],
                "negative_cases": [],
                "confidence": "low",
                "limitations": [],
                "recommended_followup": [],
            }
        )
    )

    result = validate_interpretation_output_dir(feature_dir)

    assert result["valid"] is False
    assert "classification.json" in result["missing_files"]


def test_strict_quality_validation_flags_shallow_complete_outputs(tmp_path):
    feature_dir = tmp_path / "feature_000001"
    feature_dir.mkdir()
    shallow_unified = {
        "short_name": "example",
        "summary": "too short",
        "activation_pattern": "example",
        "taxonomic_scope": "example",
        "ecological_context": "example",
        "abundance_context": "example",
        "community_context": "example",
        "exemplars": [],
        "negative_cases": [],
        "confidence": "low",
        "limitations": [],
        "recommended_followup": [],
    }
    files = {
        "initial_observation.json": {"feature_id": 1, "observation": "summary"},
        "grouping.json": {"feature_id": 1, "groups": [{"name": "A"}, {"name": "B"}]},
        "group_descriptions.json": {"feature_id": 1, "group_descriptions": [{"group_id": "A"}, {"group_id": "B"}]},
        "group_validation.json": {"feature_id": 1, "validation_results": [{"group_id": "A"}, {"group_id": "B"}]},
        "unified_description.json": shallow_unified,
        "final_validation.json": {"feature_id": 1, "accepted": True},
        "classification.json": {"feature_id": 1, "primary_class": "Unclear", "visualization_priority": "high"},
    }
    for name, payload in files.items():
        (feature_dir / name).write_text(json.dumps(payload))

    basic = validate_interpretation_output_dir(feature_dir)
    strict = validate_interpretation_output_dir(feature_dir, strict_quality=True)

    assert basic["valid"] is True
    assert basic["quality_pass"] is False
    assert strict["valid"] is False
    assert strict["quality_warnings"]
    assert any("visualization_priority" in item["warning"] for item in strict["quality_warnings"])
    assert any("needs confidence" in item["warning"] for item in strict["quality_warnings"])
