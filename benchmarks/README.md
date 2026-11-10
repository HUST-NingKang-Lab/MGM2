# Article benchmark data

These aggregate tables contain the source values used in the unified article
figures dated 9 November 2026.

| Directory | Task | Contents |
|---|---|---|
| mgnify | Hierarchical biome prediction | Model/level AUROC and parameter counts |
| fmt | Fecal microbiota transplantation | Community, clinical response and recovery summaries |
| drug | Drug perturbation | Nested-fold community and perturbation metric summaries |
| sae | Sparse-feature interpretation | Formal training and interpretation-agent summaries |

## Evaluation units

MGnify reports five hierarchy levels. Filtered subsets follow the historical
cohort definitions; the five-bin model uses the complete pretraining corpus.
FMT community estimates use patient-grouped cross-validation. Clinical response
and community reconstruction each retain their task-specific evaluation units
and denominators. Drug experiments use the observed SIC–drug pairs. Statistical
comparisons use the biological grouping appropriate to each task, with seeds
and folds representing repeated model fits and data partitions.

## Working with the results

The tables support model-level comparison, metric inspection and summary plotting.
The Python package provides community embedding, five-bin objective reference
functions and frozen formal SAE encoding. Model configurations identify the
architecture and selected checkpoint for each scale.
