#!/usr/bin/env python3
"""Summarize the interpreted top-500 OTU SAE feature catalog."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config


BIOLOGICAL_CLASSES = {
    "Taxonomic identity",
    "Ecological niche",
    "Host-associated",
    "Abundance/rank",
    "Community interaction/co-occurrence",
    "Disease-associated context",
    "Perturbation/transition",
}
CORE_CLASSES = [
    "Taxonomic identity",
    "Ecological niche",
    "Host-associated",
    "Study/batch/artifact",
    "Abundance/rank",
    "Community interaction/co-occurrence",
    "Disease-associated context",
    "Perturbation/transition",
    "Unclear",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sae_interpretation/otu_xlarge_sample500k_k64_aux0p03125_sparse_topk_global.json"),
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--representatives-per-class", type=int, default=8)
    return parser.parse_args()


def _load_macro_summary(root: Path, summary_csv: Path) -> pd.DataFrame:
    if summary_csv.exists():
        return pd.read_csv(summary_csv)
    module_path = project_root / "scripts" / "sae_interpretation" / "plot_feature_macro_statistics.py"
    spec = importlib.util.spec_from_file_location("plot_feature_macro_statistics", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import summary collector from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._collect_summary(root)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).map(lambda x: str(x).lower() == "true")


def _clean_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value)
    return text if text else "unknown"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "taxonomy_strong",
        "taxonomy_moderate",
        "environment_strong",
        "project_artifact_suspected",
        "project_artifact_strong",
        "technology_artifact_suspected",
        "technology_caveat",
        "sample_concentration_suspected",
        "interpret_as_artifact_first",
    ]:
        if col in df:
            df[col] = _bool_series(df[col])
    for col in ["primary_class", "final_confidence", "classification_confidence", "acceptance", "taxonomy_best_rank", "taxonomy_best_value", "environment_top", "technology_top"]:
        if col in df:
            df[col] = df[col].map(_clean_value)
    numeric_cols = [
        "taxonomy_best_fraction",
        "environment_max_fraction",
        "environment_enrichment_ratio",
        "project_max_fraction",
        "project_enrichment_ratio",
        "technology_max_fraction",
        "technology_enrichment_ratio",
        "events_per_sample",
        "lookup_unique_otus",
        "lookup_unique_samples",
        "positive_fraction",
        "num_positive_events",
        "activation_max",
        "mean_positive_activation",
        "activation_sum",
        "activation_vs_abundance",
        "activation_vs_abundance_rank",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _split_labels(value: object) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _add_catalog_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["taxonomy_axis"] = df["taxonomy_strong"] | (df["taxonomy_best_fraction"] >= 0.70)
    df["environment_axis"] = df["environment_strong"] | (df["environment_max_fraction"] >= 0.60)
    df["artifact_axis"] = (
        df["interpret_as_artifact_first"]
        | df["project_artifact_suspected"]
        | df["technology_artifact_suspected"]
        | (df["project_max_fraction"] >= 0.50)
    )
    df["rank_axis"] = (df["primary_class"].eq("Abundance/rank")) | (df["activation_vs_abundance_rank"].abs() >= 0.10)
    df["broad_lookup_axis"] = (df["lookup_unique_otus"] >= df["lookup_unique_otus"].quantile(0.75)) | (
        df["lookup_unique_samples"] >= df["lookup_unique_samples"].quantile(0.75)
    )
    df["n_axis_flags"] = df[["taxonomy_axis", "environment_axis", "artifact_axis", "rank_axis", "broad_lookup_axis"]].sum(axis=1)

    def tier(row: pd.Series) -> str:
        if row["primary_class"] == "Unclear":
            return "unclear"
        if row["artifact_axis"] or row["primary_class"] == "Study/batch/artifact":
            return "artifact_or_batch_control"
        if row["final_confidence"] == "high" and row["primary_class"] in BIOLOGICAL_CLASSES:
            return "high_confidence_biological"
        if row["final_confidence"] == "medium" and row["primary_class"] in BIOLOGICAL_CLASSES:
            return "medium_confidence_biological_candidate"
        if row["rank_axis"]:
            return "rank_or_abundance_candidate"
        return "low_confidence_or_needs_review"

    df["catalog_tier"] = df.apply(tier, axis=1)
    return df


def _label_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in df.iterrows():
        primary = row["primary_class"]
        labels = [primary] + _split_labels(row.get("secondary_classes", ""))
        for label in labels:
            rows.append({"feature_id": row["feature_id"], "label": label, "role": "primary" if label == primary else "secondary"})
    label_df = pd.DataFrame(rows)
    label_counts = (
        label_df.groupby("label")
        .agg(total_features=("feature_id", "nunique"), primary_count=("role", lambda s: int((s == "primary").sum())), secondary_count=("role", lambda s: int((s == "secondary").sum())))
        .reset_index()
        .sort_values(["total_features", "primary_count"], ascending=False)
    )
    labels = [label for label in CORE_CLASSES if label in set(label_df["label"])] + sorted(set(label_df["label"]) - set(CORE_CLASSES))
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for _, sub in label_df.groupby("feature_id"):
        feature_labels = sorted(set(sub["label"]))
        for a in feature_labels:
            for b in feature_labels:
                matrix.loc[a, b] += 1
    matrix = matrix.reset_index(names="label")
    return label_counts, matrix


def _mean_median_table(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    parts = []
    grouped = df.groupby(group_col, dropna=False)
    for group, sub in grouped:
        row = {group_col: group, "n_features": len(sub)}
        for col in value_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_median"] = float(sub[col].median())
            row[f"{col}_p25"] = float(sub[col].quantile(0.25))
            row[f"{col}_p75"] = float(sub[col].quantile(0.75))
        parts.append(row)
    return pd.DataFrame(parts).sort_values("n_features", ascending=False)


def _top_by_group(df: pd.DataFrame, group_col: str, value_col: str, count_name: str = "count", top_n: int = 10) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        values = sub[value_col].astype("string").replace({"unknown": pd.NA, "<missing>": pd.NA, "": pd.NA}).dropna()
        counts = values.value_counts().head(top_n)
        total = len(sub)
        for value, count in counts.items():
            rows.append({group_col: group, value_col: value, count_name: int(count), "fraction_within_group": float(count / max(total, 1))})
    return pd.DataFrame(rows)


def _representative_score(row: pd.Series) -> float:
    cls = row["primary_class"]
    if cls == "Taxonomic identity":
        return row["taxonomy_best_fraction"]
    if cls in {"Ecological niche", "Host-associated"}:
        return row["environment_max_fraction"] * np.log1p(row["environment_enrichment_ratio"])
    if cls == "Study/batch/artifact":
        return row["project_max_fraction"] + 0.25 * row["technology_max_fraction"]
    if cls == "Abundance/rank":
        return abs(row["activation_vs_abundance_rank"])
    if cls == "Unclear":
        return -(max(row["taxonomy_best_fraction"], row["environment_max_fraction"], row["project_max_fraction"]))
    return max(row["taxonomy_best_fraction"], row["environment_max_fraction"])


def _write_tables(df: pd.DataFrame, output_dir: Path, representatives_per_class: int) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    def save_csv(name: str, frame: pd.DataFrame) -> None:
        path = output_dir / name
        frame.to_csv(path, index=False)
        paths[name] = path

    save_csv("top500_feature_catalog_enriched.csv", df)
    save_csv("class_confidence_crosstab.csv", pd.crosstab(df["primary_class"], df["final_confidence"]).reset_index())
    save_csv("class_acceptance_crosstab.csv", pd.crosstab(df["primary_class"], df["acceptance"]).reset_index())
    label_counts, label_overlap = _label_tables(df)
    save_csv("label_counts_primary_secondary.csv", label_counts)
    save_csv("label_overlap_matrix.csv", label_overlap)
    value_cols = [
        "taxonomy_best_fraction",
        "environment_max_fraction",
        "environment_enrichment_ratio",
        "project_max_fraction",
        "technology_max_fraction",
        "positive_fraction",
        "mean_positive_activation",
        "lookup_unique_otus",
        "lookup_unique_samples",
        "activation_vs_abundance_rank",
    ]
    save_csv("class_axis_distribution_summary.csv", _mean_median_table(df, "primary_class", value_cols))
    save_csv("tier_axis_distribution_summary.csv", _mean_median_table(df, "catalog_tier", value_cols))
    save_csv("catalog_tier_counts.csv", df["catalog_tier"].value_counts().rename_axis("catalog_tier").reset_index(name="n_features"))
    axis_cols = ["taxonomy_axis", "environment_axis", "artifact_axis", "rank_axis", "broad_lookup_axis"]
    save_csv("axis_flag_counts.csv", df[axis_cols].sum().rename_axis("axis").reset_index(name="n_features"))
    save_csv("axis_flags_by_class.csv", df.groupby("primary_class")[axis_cols].mean().reset_index())
    save_csv("taxonomy_rank_by_class.csv", pd.crosstab(df["primary_class"], df["taxonomy_best_rank"]).reset_index())
    save_csv("environment_by_class.csv", pd.crosstab(df["primary_class"], df["environment_top"]).reset_index())
    save_csv("technology_by_class.csv", pd.crosstab(df["primary_class"], df["technology_top"]).reset_index())
    save_csv("top_taxa_overall.csv", df["taxonomy_best_value"].replace(["unknown", "<missing>", ""], np.nan).dropna().value_counts().reset_index(name="n_features").rename(columns={"index": "taxonomy_best_value"}))
    save_csv("top_taxa_by_class.csv", _top_by_group(df, "primary_class", "taxonomy_best_value", top_n=12))
    save_csv("top_environments_by_class.csv", _top_by_group(df, "primary_class", "environment_top", top_n=8))
    artifact_cols = [
        "primary_class",
        "project_artifact_suspected",
        "technology_artifact_suspected",
        "technology_caveat",
        "sample_concentration_suspected",
        "interpret_as_artifact_first",
    ]
    save_csv("artifact_diagnostics_by_class.csv", df.groupby("primary_class")[artifact_cols[1:]].mean().reset_index())
    rank_candidates = df[df["rank_axis"]].sort_values("activation_vs_abundance_rank", key=lambda s: s.abs(), ascending=False)
    save_csv("rank_abundance_candidates.csv", rank_candidates)
    artifact_candidates = df[df["artifact_axis"] | df["primary_class"].eq("Study/batch/artifact")].sort_values(["interpret_as_artifact_first", "project_max_fraction"], ascending=False)
    save_csv("artifact_candidate_features.csv", artifact_candidates)
    high_bio = df[df["catalog_tier"].eq("high_confidence_biological")].sort_values(["primary_class", "taxonomy_best_fraction", "environment_max_fraction"], ascending=[True, False, False])
    save_csv("high_confidence_biological_features.csv", high_bio)
    reps = df.copy()
    reps["representative_score"] = reps.apply(_representative_score, axis=1)
    reps = (
        reps.sort_values(["primary_class", "representative_score"], ascending=[True, False])
        .groupby("primary_class", group_keys=False)
        .head(representatives_per_class)
    )
    save_csv("representative_features_by_class.csv", reps)
    return paths


def _overall_metrics(df: pd.DataFrame) -> dict:
    def count_true(col: str) -> int:
        return int(df[col].sum())

    return {
        "n_features": int(len(df)),
        "primary_class_counts": {str(k): int(v) for k, v in df["primary_class"].value_counts().items()},
        "confidence_counts": {str(k): int(v) for k, v in df["final_confidence"].value_counts().items()},
        "catalog_tier_counts": {str(k): int(v) for k, v in df["catalog_tier"].value_counts().items()},
        "axis_flag_counts": {
            "taxonomy_axis": count_true("taxonomy_axis"),
            "environment_axis": count_true("environment_axis"),
            "artifact_axis": count_true("artifact_axis"),
            "rank_axis": count_true("rank_axis"),
            "broad_lookup_axis": count_true("broad_lookup_axis"),
        },
        "median_positive_fraction": float(df["positive_fraction"].median()),
        "median_mean_positive_activation": float(df["mean_positive_activation"].median()),
        "median_lookup_unique_otus": float(df["lookup_unique_otus"].median()),
        "median_lookup_unique_samples": float(df["lookup_unique_samples"].median()),
        "top_environments": {str(k): int(v) for k, v in df["environment_top"].value_counts().head(10).items()},
        "top_taxa": {str(k): int(v) for k, v in df["taxonomy_best_value"].value_counts().head(10).items()},
        "top_technologies": {str(k): int(v) for k, v in df["technology_top"].value_counts().head(10).items()},
    }


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _format_counts(counts: dict[str, int]) -> str:
    return "；".join([f"{key}: {value}" for key, value in counts.items()])


def _write_readme(output_dir: Path, metrics: dict, df: pd.DataFrame, paths: dict[str, Path]) -> Path:
    high_bio = int(df["catalog_tier"].eq("high_confidence_biological").sum())
    medium_bio = int(df["catalog_tier"].eq("medium_confidence_biological_candidate").sum())
    artifact = int(df["catalog_tier"].eq("artifact_or_batch_control").sum())
    unclear = int(df["catalog_tier"].eq("unclear").sum())
    low_review = int(df["catalog_tier"].eq("low_confidence_or_needs_review").sum())
    lines = [
        "# SAE top500 feature 整体描述统计",
        "",
        f"本目录汇总 {metrics['n_features']} 个已解释的 OTU SAE top features。统计输入是 `feature_macro_summary.csv`，也就是已经完成 Codex interpretation 和 deterministic evidence 后的 feature-level 表。",
        "",
        "## 一句话概括",
        "",
        f"Top500 中，生态/环境轴非常常见，environment_axis={metrics['axis_flag_counts']['environment_axis']}；taxonomy_axis={metrics['axis_flag_counts']['taxonomy_axis']}；artifact_axis={metrics['axis_flag_counts']['artifact_axis']}；rank_axis={metrics['axis_flag_counts']['rank_axis']}。整体上 high-confidence biological features 只有 {high_bio} 个，medium-confidence biological candidates 有 {medium_bio} 个，artifact/control features 有 {artifact} 个，unclear features 有 {unclear} 个，另有 {low_review} 个低可信或需要复核的 feature。",
        "",
        "## Primary class 构成",
        "",
        _format_counts(metrics["primary_class_counts"]),
        "",
        "## Confidence 构成",
        "",
        _format_counts(metrics["confidence_counts"]),
        "",
        "## Catalog tier 构成",
        "",
        _format_counts(metrics["catalog_tier_counts"]),
        "",
        "## Top environments",
        "",
        _format_counts(metrics["top_environments"]),
        "",
        "## Top best taxa",
        "",
        _format_counts(metrics["top_taxa"]),
        "",
        "## Activation / lookup breadth 中位数",
        "",
        f"- positive_fraction median: {metrics['median_positive_fraction']:.4f}",
        f"- mean_positive_activation median: {metrics['median_mean_positive_activation']:.3f}",
        f"- lookup_unique_otus median: {metrics['median_lookup_unique_otus']:.0f}",
        f"- lookup_unique_samples median: {metrics['median_lookup_unique_samples']:.0f}",
        "",
        "## 输出表说明",
        "",
        "- `top500_feature_catalog_enriched.csv`: 每个 feature 的主表，包含新增的 catalog tier 和 axis flags。",
        "- `class_confidence_crosstab.csv`: primary class x final confidence。",
        "- `label_counts_primary_secondary.csv`: primary + secondary 多标签总体计数。",
        "- `label_overlap_matrix.csv`: 多标签共现矩阵，可用于看 taxonomy/ecology/host/artifact 的重叠。",
        "- `class_axis_distribution_summary.csv`: 每个 primary class 的 taxonomy、environment、artifact、activation 等数值分布。",
        "- `catalog_tier_counts.csv`: 高可信、medium 候选、artifact、unclear、低可信复核等 tier 的数量。",
        "- `axis_flag_counts.csv` 和 `axis_flags_by_class.csv`: taxonomy/environment/artifact/rank/broad lookup 五个解释轴的总体和分类别比例。",
        "- `taxonomy_rank_by_class.csv`、`top_taxa_overall.csv`、`top_taxa_by_class.csv`: taxonomy 层面的整体描述。",
        "- `environment_by_class.csv`、`top_environments_by_class.csv`: 环境/宿主 context 的整体描述。",
        "- `artifact_diagnostics_by_class.csv` 和 `artifact_candidate_features.csv`: artifact 风险描述和候选列表。",
        "- `rank_abundance_candidates.csv`: rank/abundance 候选 feature 列表。",
        "- `high_confidence_biological_features.csv`: 最适合作为生物学解释示例的高可信 feature。",
        "- `representative_features_by_class.csv`: 每个 primary class 的代表 feature 候选。",
        "",
        "## 使用建议",
        "",
        "- 写论文方法/结果时，可以用 `primary_class_counts` 和 `catalog_tier_counts` 描述 SAE feature catalog 的整体可解释性。",
        "- 选择展示 feature 时，优先从 `high_confidence_biological_features.csv` 和 `representative_features_by_class.csv` 中挑；artifact 对照从 `artifact_candidate_features.csv` 中挑。",
        "- 不建议把 `Unclear` 或 `low_confidence_or_needs_review` 当作负结果直接丢弃，它们可以用于说明当前自动解释流程的边界。",
    ]
    readme = output_dir / "README.zh.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    macro_dir = root / "reports" / "feature_macro_statistics"
    summary_csv = args.summary_csv or macro_dir / "feature_macro_summary.csv"
    output_dir = args.output_dir or root / "reports" / "top500_feature_catalog_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_macro_summary(root, summary_csv)
    df = _add_catalog_fields(_normalize_df(df))
    paths = _write_tables(df, output_dir, args.representatives_per_class)
    metrics = _overall_metrics(df)
    metrics_path = output_dir / "overall_metrics.json"
    _write_json(metrics_path, metrics)
    readme = _write_readme(output_dir, metrics, df, paths)
    print(
        json.dumps(
            {
                "features": int(len(df)),
                "output_dir": str(output_dir),
                "tables": len(paths),
                "overall_metrics": str(metrics_path),
                "readme": str(readme),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
