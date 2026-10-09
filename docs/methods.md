# Methods

The question is whether view disagreement and model variation improve selection of cancer-positive breasts missed by a fixed classifier, compared with cancer score alone at the same review capacity.

## Data and model

- Patient-separated development: 2,383 patients, 4,766 breasts, 99 positive breasts.
- Holdout: 9,530 patients, 19,060 breasts, 393 positive breasts.
- Five author-released GMIC members; no RSNA weight fitting in the final study.
- Full-field, aspect-preserving resize/pad to 2944 × 1920. Repeated images are averaged within view, then CC and MLO receive equal weight, as do the five members.

The five inference shards are computational divisions of one holdout. Earlier CNN training attempts belong to development history, not the final model.

## Classification and review

Calibration and a cutoff targeting at least 90% empirical sensitivity were fitted on development patients. The fixed raw cutoff was `0.02242065602913494`. Only breasts classified negative were eligible for review ranking.

The combined score is the equal mean of development-reference midrank percentiles of calibrated cancer score, absolute CC–MLO disagreement and between-member population variance. References include all AI-negative development breasts. Ranking uses no holdout labels; ties use a fixed label-independent hash.

The primary comparison selects `floor(0.05 × N_AI-negative)` breasts globally with each method. The endpoint is the proportion of all classifier misses selected for potential review. Capture uses the missed-cancer denominator; review yield uses the selected-breast denominator.

## Uncertainty and scope

Two schemes each use 2,000 paired, site-stratified patient-cluster bootstrap draws, keeping both breasts together and recomputing global capacity. The primary scheme holds calibration fixed; the secondary scheme resamples development patients and refits calibration, cutoff and percentile references.

The analysis was frozen before final model outcomes after adaptive development; it was not publicly preregistered. The opened holdout cannot be reused as an untouched test for a revised score. No human-reader recovery or prospective clinical outcome was measured.

Exact definitions: [frozen protocol](../FIXED_GMIC_FINAL_PROTOCOL.md), [configuration](../fixed_gmic_final_config.json), [evaluator](../fixed_gmic_evaluation.py). See [results](results.md) and [reproducibility](reproducibility.md).
