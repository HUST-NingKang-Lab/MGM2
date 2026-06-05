"""Render Codex task packets for feature interpretation."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import UNIFIED_DESCRIPTION_SCHEMA


AGENTS = [
    (
        "00_initial_observation.md",
        "Initial observation agent",
        "Summarize deterministic evidence, likely microbial concepts, ambiguity, and what must not be overclaimed.",
        """Required JSON shape:
{
  "feature_id": int,
  "stage": "initial_observation",
  "description_basis": {"used_splits": ["train", "val"], "excluded_splits": ["test"], "note": "..."},
  "evidence_anchors": [{"field": "path.in.evidence", "value": "...", "interpretation": "..."}],
  "deterministic_observations": ["at least 5 feature-specific observations with numbers"],
  "candidate_concepts": [{"concept": "...", "supporting_evidence": ["..."], "conflicting_evidence": ["..."], "verdict": "keep|reject|uncertain"}],
  "artifact_assessment": {"project": "...", "technology": "...", "sample_concentration": "...", "verdict": "..."},
  "negative_control_assessment": {"same_taxonomy": "...", "same_biome": "...", "same_abundance_rank": "...", "same_study": "..."},
  "confidence": "high|medium|low",
  "limitations": ["..."]
}
""",
    ),
    (
        "01_grouping.md",
        "Grouping agent",
        "Group top activated events into 2-10 biological groups using taxonomy, ecology, abundance, and community context.",
        """Required JSON shape:
{
  "feature_id": int,
  "stage": "grouping",
  "grouping_strategy": "why these groups are biologically/statistically justified; do not always choose 3 groups",
  "evidence_anchors": [{"field": "path.in.evidence", "value": "...", "interpretation": "..."}],
  "groups": [
    {
      "group_id": "G1",
      "name": "...",
      "definition": "taxonomy/ecology/metadata/abundance rule",
      "supporting_events": ["specific top or heldout examples from evidence"],
      "heldout_events_or_samples": ["events/samples/OTUs reserved for validation"],
      "negative_controls_to_check": ["same_taxonomy", "same_biome", "same_abundance_rank", "same_study"],
      "caveats": ["..."]
    }
  ],
  "rejected_groupings": [{"hypothesis": "...", "reason_rejected": "..."}],
  "confidence": "high|medium|low"
}
""",
    ),
    (
        "02_group_descriptions.md",
        "Group description agent",
        "For each group, choose exemplar samples and explain group-level activation patterns.",
        """Required JSON shape:
{
  "feature_id": int,
  "stage": "group_descriptions",
  "evidence_anchors": [{"field": "path.in.evidence", "value": "...", "interpretation": "..."}],
  "group_descriptions": [
    {
      "group_id": "G1",
      "exemplar_samples": [{"sample_id": "...", "project": "...", "primary_env": "...", "why_exemplar": "..."}],
      "community_activation_table": [{"otu_id": "...", "taxonomy": "...", "activation": "...", "abundance_or_rank": "..."}],
      "activation_explanation": "...",
      "what_this_group_does_not_show": "..."
    }
  ],
  "cross_group_comparison": "...",
  "limitations": ["..."]
}
""",
    ),
    (
        "03_group_validation.md",
        "Group validation agent",
        "Use held-out events and negative controls to validate or revise each group description.",
        """Required JSON shape:
{
  "feature_id": int,
  "stage": "group_validation",
  "evidence_anchors": [{"field": "path.in.evidence", "value": "...", "interpretation": "..."}],
  "validation_results": [
    {
      "group_id": "G1",
      "heldout_result": "supports|partially_supports|fails",
      "negative_control_result": {"same_taxonomy": "...", "same_biome": "...", "same_abundance_rank": "...", "same_study": "..."},
      "revision_needed": true,
      "revision": "..."
    }
  ],
  "failed_or_ambiguous_groups": [{"group_id": "...", "reason": "..."}],
  "overall_group_validity": "high|medium|low"
}
""",
    ),
    (
        "04_unification.md",
        "Unification agent",
        "Integrate validated group descriptions into one feature description JSON.",
        """Required JSON shape:
Use the unified_description.schema.json fields, and also include:
{
  "evidence_anchors": [{"field": "path.in.evidence", "value": "...", "interpretation": "..."}],
  "alternative_hypotheses": [{"hypothesis": "...", "supporting_evidence": ["..."], "conflicting_evidence": ["..."], "verdict": "accepted|rejected|uncertain"}],
  "negative_control_assessment": {"same_taxonomy": "...", "same_biome": "...", "same_abundance_rank": "...", "same_study": "..."},
  "artifact_assessment": {"project_batch": "...", "technology": "...", "sample_concentration": "...", "final_judgment": "..."}
}
""",
    ),
    (
        "05_final_validation.md",
        "Final validation agent",
        "Evaluate applicability, specificity, overgeneralization risk, and acceptance.",
        """Required JSON shape:
{
  "feature_id": int,
  "stage": "final_validation",
  "evidence_anchors": [{"field": "path.in.evidence", "value": "...", "interpretation": "..."}],
  "applicability": "...",
  "specificity": "...",
  "overgeneralization_risk": "...",
  "artifact_judgment": "...",
  "test_split_use": "No test split was used. Test should be reserved for independent validation.",
  "acceptance": "accepted|accepted_with_caveats|accepted_as_artifact|accepted_as_unclear|rejected",
  "confidence": "high|medium|low",
  "limitations": ["..."],
  "recommended_followup": ["..."]
}
Do not use a bare boolean for acceptance.
""",
    ),
    (
        "06_classification.md",
        "Feature classification agent",
        "Classify as Taxonomic identity, Abundance/rank, Ecological niche, Host-associated, Disease-associated context, Community interaction/co-occurrence, Perturbation/transition, Study/batch/artifact, or Unclear.",
        """Required JSON shape:
{
  "feature_id": int,
  "stage": "classification",
  "primary_class": "...",
  "secondary_classes": ["..."],
  "rejected_classes": {
    "Taxonomic identity": "reason if rejected or limited",
    "Abundance/rank": "reason if rejected or limited",
    "Ecological niche": "reason if rejected or limited",
    "Host-associated": "reason if rejected or limited",
    "Disease-associated context": "reason if rejected or limited",
    "Community interaction/co-occurrence": "reason if rejected or limited",
    "Perturbation/transition": "reason if rejected or limited",
    "Study/batch/artifact": "reason if rejected or limited",
    "Unclear": "reason if rejected or limited"
  },
  "negative_controls_used": ["same_taxonomy", "same_biome", "same_abundance_rank", "same_study"],
  "confidence": "high|medium|low",
  "visualization_priority": "high|medium|low",
  "limitations": ["..."]
}
""",
    ),
]


GLOBAL_INTERPRETATION_GUARDRAILS = """Process requirements:
- Interpret exactly one feature per output directory. Do not write a batch summary for several features.
- Do not create template JSON by swapping feature_id. Every sentence should be tied to this feature's evidence.
- Read the evidence package before writing. Use at least 6 explicit evidence anchors with field names and numeric values when available.
- Evaluate at least 3 competing hypotheses before unifying the description. Include rejected or uncertain alternatives.
- Use negative controls explicitly: same_taxonomy, same_biome, same_abundance_rank, and same_study.
- Separate biological signal from artifact/bias. A high AMPLICON fraction alone is only a technology caveat unless the background-corrected technology diagnostic says suspected.
- Prefer conservative labels. If the evidence is mixed, write Unclear or accepted_with_caveats instead of inventing a clean biological concept.
- Never use the test split to generate the description.
"""


def _task_text(agent_name: str, instruction: str, output_contract: str, evidence_filename: str) -> str:
    return f"""# {agent_name}

You are Codex acting as a scientific feature-interpretation agent for an MGM2 OTU-token sparse autoencoder.

Use only the deterministic evidence package in `{evidence_filename}`. Do not calculate enrichment yourself. Do not use test data to generate the description.

{GLOBAL_INTERPRETATION_GUARDRAILS}

Interpretation rules:
- Do not treat high AMPLICON fraction alone as Study/batch/artifact. AMPLICON is common in the background; use `diagnostics.technology_bias.enrichment_ratio` and `diagnostics.technology_bias.suspected`.
- Treat `diagnostics.project_batch_artifact` as the primary study/batch artifact signal. Project fraction >=0.3 is high risk; >=0.5 is strong.
- Treat `diagnostics.sample_concentration` separately from project artifact. A feature concentrated in few samples may be sample-specific even without one dominant project.
- Use `diagnostics.environment_enrichment` as ecological/contextual evidence, not as artifact evidence.
- Use `diagnostics.taxonomy_signal` and `diagnostics.biological_signal_strength` to avoid overcalling artifact when taxonomy or ecology is strong and project concentration is low.
- If `biological_signal_strength.interpret_as_artifact_first` is true, explain the artifact/bias first. Otherwise, report technology as a caveat rather than the primary interpretation unless evidence supports it.

Task:
{instruction}

{output_contract}

Output must be JSON and must explicitly discuss confidence and limitations. If evidence is insufficient, say so. Shallow summaries without evidence anchors, negative controls, and rejected alternatives are not acceptable.
"""


def render_codex_task_files(evidence: dict, output_dir: Path) -> Path:
    feature_id = int(evidence["feature_id"])
    feature_dir = Path(output_dir) / f"feature_{feature_id:06d}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = feature_dir / f"feature_{feature_id:06d}.evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    for filename, agent_name, instruction, output_contract in AGENTS:
        (feature_dir / filename).write_text(_task_text(agent_name, instruction, output_contract, evidence_path.name))
    schema_dir = feature_dir / "schemas"
    schema_dir.mkdir(exist_ok=True)
    (schema_dir / "unified_description.schema.json").write_text(json.dumps(UNIFIED_DESCRIPTION_SCHEMA, indent=2))
    return feature_dir
