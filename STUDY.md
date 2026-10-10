# Study summary

## Research question

I examine whether a score combining cancer probability, CC–MLO disagreement and between-model variation improves prioritization of cancer-positive breasts missed by a fixed mammography classifier, compared with cancer score alone at equal review capacity.

The contribution is a controlled retrospective comparison with explicit denominators and uncertainty estimates. The study does not introduce a new image-model architecture or establish a novel clinical intervention.

## Design and methods

The audited RSNA cohort contained 11,913 patients and 23,826 breasts. The patient-separated development group contained 2,383 patients, 4,766 breasts and 99 cancer-positive breasts. The holdout contained 9,530 patients, 19,060 breasts and 393 cancer-positive breasts. Both breasts of each patient remained together.

The final classifier used all five author-released GMIC members at upstream revision `594029efe6184bcca3a151fa1dd46c5fd620cb25`. No model weights were fitted on RSNA data in this final experiment. Images underwent aspect-preserving full-field resizing/padding to 2944 × 1920. Repeated images were averaged within view; CC and MLO then received equal weight, as did the five members. Five GPU shards were computational partitions of one holdout, not cross-validation folds.

Calibration and a cutoff targeting at least 90% empirical sensitivity were fitted on development patients. The raw cutoff was `0.02242065602913494`; breasts at or above it were classified positive. Only classifier-negative breasts were eligible for review.

The combined score averaged development-reference midrank percentiles of calibrated cancer score, absolute CC–MLO disagreement and between-member population variance. References used all classifier-negative development breasts. Rankings used no holdout labels, and ties followed a fixed label-independent hash.

The primary endpoint was the fraction of all classifier misses selected at a global budget of `floor(0.05 × number of classifier-negative breasts)`. Capture and review yield therefore use different denominators: all missed cancers and all selected breasts, respectively.

Two schemes each used 2,000 paired, site-stratified patient-cluster bootstrap draws, keeping both breasts together and recomputing global capacity. The primary scheme held calibration fixed. The secondary scheme resampled development patients and refitted calibration, cutoff and percentile references.

## Results

The frozen evaluation completed on October 1, 2026. The classifier-negative pool contained 8,315 breasts, including 39 cancer-positive misses. Each method selected 415 breasts, or 4.99% after rounding down.

| Method | Selected breasts | Misses flagged | Capture | Review yield |
|---|---:|---:|---:|---:|
| Cancer score alone | 415 | 3 / 39 | 7.69% | 0.72% |
| Combined score | 415 | 2 / 39 | 5.13% | 0.48% |

Combined minus cancer-score capture was **−2.56 percentage points**. The fixed-calibration 95% CI was **−13.79 to +8.11** points; the development-refit CI was **−14.29 to +8.00** points. Both intervals include zero.

The classifier yielded TP 354, FN 39, TN 8,276 and FP 10,391: sensitivity 90.08%, specificity 44.33%, ROC-AUC 0.77245 and average precision 0.12907. At the secondary 20% capacity, the combined score flagged 13/39 misses and cancer score alone flagged 18/39. All six prespecified scores and four tabulated budgets remain in [the results tables](reports/TABLES.md); additional curve points are shown in Figure 3. Site and density summaries inherit global selections. Density is unavailable at site 2.

## Interpretation and limitations

My interpretation is that this experiment does not establish added benefit from these disagreement and variation descriptors at the primary review capacity. The wide intervals also prevent a conclusion of equivalence or definitive inferiority. Retrospective selection against dataset labels is not measured recovery by a radiologist.

The final freeze followed adaptive development, including unsuccessful CNN trials and unfavorable development comparisons; it was not a public preregistration. The opened holdout cannot serve as untouched evaluation data for a revised method. One dataset, one ensemble family, 39 misses, low specificity, finite calibration data and possible external-pretraining overlap limit generalization. Technical image inspection does not establish diagnostic validity or preservation of every fine detail. No independent external replication, reader study, interval-cancer endpoint or prospective benefit was evaluated.

Exact implementation and settings are in [src](src/). Aggregate results, all 4,000 bootstrap draws and original hashes are in [validation](validation/). [Reproducibility](REPRODUCIBILITY.md) identifies the untouched original workflow and dated development records.
