"""JSON schemas and lightweight validation for SAE interpretation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


UNIFIED_DESCRIPTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "short_name",
        "summary",
        "activation_pattern",
        "taxonomic_scope",
        "ecological_context",
        "abundance_context",
        "community_context",
        "exemplars",
        "negative_cases",
        "confidence",
        "limitations",
        "recommended_followup",
    ],
    "properties": {
        "short_name": {"type": "string"},
        "summary": {"type": "string"},
        "activation_pattern": {"type": "string"},
        "taxonomic_scope": {"type": "string"},
        "ecological_context": {"type": "string"},
        "abundance_context": {"type": "string"},
        "community_context": {"type": "string"},
        "exemplars": {"type": "array"},
        "negative_cases": {"type": "array"},
        "confidence": {"type": "string"},
        "limitations": {"type": "array"},
        "recommended_followup": {"type": "array"},
    },
}


EVIDENCE_REQUIRED = {
    "feature_id",
    "global_stats",
    "top_examples",
    "top_otus",
    "top_samples",
    "enrichment",
    "negative_controls",
    "diagnostics",
}


REQUIRED_AGENT_OUTPUT_FILES = [
    "initial_observation.json",
    "grouping.json",
    "group_descriptions.json",
    "group_validation.json",
    "unified_description.json",
    "final_validation.json",
    "classification.json",
]


NEGATIVE_CONTROL_KEYS = {"same_taxonomy", "same_biome", "same_abundance_rank", "same_study"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return [value]


def _list_len(value: Any) -> int:
    return len(_as_list(value))


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def _has_negative_controls(value: Any) -> bool:
    text = _json_text(value)
    return all(key in text for key in NEGATIVE_CONTROL_KEYS)


def _quality_warnings_for_outputs(feature_dir: Path) -> list[Dict[str, str]]:
    warnings: list[Dict[str, str]] = []

    def warn(file_name: str, message: str) -> None:
        warnings.append({"file": file_name, "warning": message})

    loaded: Dict[str, Dict[str, Any]] = {}
    for name in REQUIRED_AGENT_OUTPUT_FILES:
        path = feature_dir / name
        if not path.exists():
            continue
        try:
            data = _load_json(path)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            warn(name, "top-level JSON value should be an object")
            continue
        loaded[name] = data

    initial = loaded.get("initial_observation.json", {})
    if initial:
        anchor_count = _list_len(initial.get("evidence_anchors"))
        observation_count = _list_len(initial.get("deterministic_observations"))
        key_evidence_count = len(initial.get("key_evidence", {})) if isinstance(initial.get("key_evidence"), dict) else 0
        if max(anchor_count, observation_count, key_evidence_count) < 5:
            warn("initial_observation.json", "needs at least 5 feature-specific evidence anchors or deterministic observations")
        if _list_len(initial.get("candidate_concepts", initial.get("likely_concepts"))) < 2:
            warn("initial_observation.json", "needs multiple candidate concepts, including ambiguity when present")
        if not _has_negative_controls(initial):
            warn("initial_observation.json", "should explicitly discuss same_taxonomy, same_biome, same_abundance_rank, and same_study controls")

    grouping = loaded.get("grouping.json", {})
    if grouping:
        groups = grouping.get("groups", [])
        if not isinstance(groups, list) or not 2 <= len(groups) <= 10:
            warn("grouping.json", "groups must be a list with 2-10 groups")
        else:
            for idx, group in enumerate(groups):
                if not isinstance(group, dict):
                    warn("grouping.json", f"group {idx} should be an object")
                    continue
                group_text = _json_text(group)
                if "holdout" not in group_text and "heldout" not in group_text:
                    warn("grouping.json", f"group {idx} should reserve holdout events/samples/OTUs")
                if "evidence" not in group_text and "support" not in group_text:
                    warn("grouping.json", f"group {idx} should include feature-specific supporting evidence")
        if _list_len(grouping.get("rejected_groupings")) < 1:
            warn("grouping.json", "should include at least one rejected grouping hypothesis")

    group_desc = loaded.get("group_descriptions.json", {})
    if group_desc:
        descriptions = group_desc.get("group_descriptions", group_desc.get("descriptions", group_desc.get("groups", [])))
        if _list_len(descriptions) < 2:
            warn("group_descriptions.json", "needs group-specific descriptions for at least two groups")
        for idx, desc in enumerate(_as_list(descriptions)):
            if not isinstance(desc, dict):
                continue
            text = _json_text(desc)
            if "exemplar" not in text:
                warn("group_descriptions.json", f"group description {idx} should include exemplar samples")
            if "community" not in text and "otu" not in text:
                warn("group_descriptions.json", f"group description {idx} should include community/OTU activation evidence")

    group_validation = loaded.get("group_validation.json", {})
    if group_validation:
        if _list_len(group_validation.get("validation_results", group_validation.get("group_validations"))) < 2:
            warn("group_validation.json", "needs validation results for grouped descriptions")
        if not _has_negative_controls(group_validation):
            warn("group_validation.json", "needs explicit negative control validation")

    unified = loaded.get("unified_description.json", {})
    if unified:
        if len(str(unified.get("summary", ""))) < 250:
            warn("unified_description.json", "summary is too short for a feature-specific interpretation")
        if _list_len(unified.get("evidence_anchors")) < 5:
            warn("unified_description.json", "needs at least 5 explicit evidence anchors")
        if _list_len(unified.get("alternative_hypotheses")) < 3:
            warn("unified_description.json", "needs at least 3 accepted/rejected/uncertain alternative hypotheses")
        if not _has_negative_controls(unified):
            warn("unified_description.json", "needs explicit same_taxonomy, same_biome, same_abundance_rank, and same_study assessment")
        if _list_len(unified.get("limitations")) < 2:
            warn("unified_description.json", "needs at least 2 limitations")
        if _list_len(unified.get("recommended_followup")) < 2:
            warn("unified_description.json", "needs at least 2 recommended follow-up checks")

    final_validation = loaded.get("final_validation.json", {})
    if final_validation:
        if isinstance(final_validation.get("accepted"), bool):
            warn("final_validation.json", "use an acceptance string, not a bare accepted boolean")
        if "acceptance" not in final_validation:
            warn("final_validation.json", "needs acceptance string: accepted, accepted_with_caveats, accepted_as_artifact, accepted_as_unclear, or rejected")
        for key in ["specificity", "overgeneralization_risk", "artifact_judgment", "test_split_use"]:
            if key not in final_validation:
                warn("final_validation.json", f"missing {key}")

    classification = loaded.get("classification.json", {})
    if classification:
        if "visualization_priority" in classification:
            warn("classification.json", "remove visualization_priority; choose figures later from deterministic evidence catalog metrics")
        rejected = classification.get("rejected_classes")
        if isinstance(rejected, dict):
            rejected_count = len(rejected)
        else:
            rejected_count = _list_len(rejected)
        if rejected_count < 4:
            warn("classification.json", "needs rejected_classes explaining non-selected interpretations")
        if not _has_negative_controls(classification.get("negative_controls_used", classification)):
            warn("classification.json", "needs all four negative controls in negative_controls_used")
        if "primary_class" not in classification:
            warn("classification.json", "missing primary_class")

    return warnings


def validate_evidence_package(data: Dict[str, Any]) -> None:
    missing = EVIDENCE_REQUIRED - set(data)
    if missing:
        raise ValueError(f"evidence package missing required fields: {sorted(missing)}")
    if not isinstance(data["top_examples"], list):
        raise TypeError("top_examples must be a list")
    if "study_batch_artifact" not in data["diagnostics"]:
        raise ValueError("diagnostics must include study_batch_artifact")
    for key in [
        "project_batch_artifact",
        "technology_bias",
        "sample_concentration",
        "environment_enrichment",
        "taxonomy_signal",
        "biological_signal_strength",
    ]:
        if key not in data["diagnostics"]:
            raise ValueError(f"diagnostics must include {key}")


def validate_required_schema_fields(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    missing = set(schema.get("required", [])) - set(data)
    if missing:
        raise ValueError(f"JSON object missing required schema fields: {sorted(missing)}")


def validate_interpretation_output_dir(feature_dir: Path, strict_quality: bool = False) -> Dict[str, Any]:
    feature_dir = Path(feature_dir)
    missing_files = [name for name in REQUIRED_AGENT_OUTPUT_FILES if not (feature_dir / name).exists()]
    invalid_json = []
    for name in REQUIRED_AGENT_OUTPUT_FILES:
        path = feature_dir / name
        if not path.exists():
            continue
        try:
            json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            invalid_json.append({"file": name, "error": str(exc)})

    schema_errors = []
    unified_path = feature_dir / "unified_description.json"
    if unified_path.exists():
        try:
            validate_required_schema_fields(json.loads(unified_path.read_text()), UNIFIED_DESCRIPTION_SCHEMA)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            schema_errors.append({"file": "unified_description.json", "error": str(exc)})

    quality_warnings = _quality_warnings_for_outputs(feature_dir)

    return {
        "feature_dir": str(feature_dir),
        "valid": not missing_files and not invalid_json and not schema_errors and (not strict_quality or not quality_warnings),
        "quality_pass": not quality_warnings,
        "missing_files": missing_files,
        "invalid_json": invalid_json,
        "schema_errors": schema_errors,
        "quality_warnings": quality_warnings,
    }
