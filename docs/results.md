# Results

The frozen experiment completed on 2026-10-01. At equal 5% global review capacity, adding view disagreement and model variation did not establish an improvement over cancer score alone.

## Primary comparison

The AI-negative holdout pool contained 8,315 breasts, including 39 cancer-positive classifier misses. Both methods selected 415 breasts (4.99% after rounding down).

| Method | Selected breasts | Misses flagged | Capture | Review yield |
|---|---:|---:|---:|---:|
| Cancer score alone | 415 | 3 / 39 | 7.69% | 0.72% |
| Combined score | 415 | 2 / 39 | 5.13% | 0.48% |

The capture difference (combined minus cancer score) was **−2.56 percentage points**.

| Patient-bootstrap scheme | 95% CI for the difference | Draws |
|---|---|---:|
| Fixed calibration, primary | −13.79 to +8.11 percentage points | 2,000 |
| Calibration refitted, secondary | −14.29 to +8.00 percentage points | 2,000 |

The intervals cross zero. Incremental benefit is inconclusive; the study does not establish equivalence or definitive inferiority. Selection against dataset labels is not measured recovery by a radiologist.

## Classifier and secondary results

On 19,060 holdout breasts: TP 354, FN 39, TN 8,276, FP 10,391; sensitivity 90.08%, specificity 44.33%, ROC-AUC 0.77245 and average precision 0.12907.

All six prespecified review scores and four budgets remain in the [tables](../reports/TABLES.md) and [aggregate JSON](../validation/fixed_gmic_final_results_20261001.json). At 20% capacity, combined scoring flagged 13/39 misses and cancer score alone flagged 18/39. These descriptive comparisons do not replace the primary comparison. Site and density summaries inherit global selections; density is unavailable at site 2.

## Limitations

One dataset and one external ensemble family, only 39 holdout misses, low operating-point specificity, adaptive development, finite calibration data and possible external-pretraining overlap limit generalization. Technical image inspection does not establish preservation of every clinically relevant detail. No independent external replication, reader study, interval-cancer endpoint or prospective benefit was evaluated.

## Evidence

- [Figures](../reports/) and [captions](../reports/CAPTIONS.md).
- [All 4,000 bootstrap draws](../validation/fixed_gmic_bootstrap_draws_20261001.csv) and [extraction verification](../validation/fixed_gmic_final_result_verification_20261001.json).
- [Source provenance](../validation/release_provenance.json); freeze SHA-256: `6cdc4c4f976c54b751eb15b85c9a1d1fa1db6877a554c6cb4ac468963c35b0ef`.
- [Methods](methods.md), [reproduction commands](reproducibility.md) and [development archive](README.md#development-archive).

Original patient-level outputs remain in private Kaggle runs. This repository contains code and aggregate results.
