# Frozen fixed-GMIC holdout findings — 2026-10-01

**The primary interval includes zero, so incremental benefit from combined scoring remains inconclusive.** At the prespecified 5% review budget, combined scoring flagged 2 of 39 AI-missed cancer-positive breasts; cancer score alone flagged 3. Each selected 415 breasts. The capture difference was -2.56 percentage points.

This endpoint means selection for possible review against existing breast-level labels. No radiologist recovery, diagnosis, patient benefit or clinical safety was measured. No score, weight, cutoff or budget was changed after final outcomes.

## Provenance and validation

Private Kaggle notebook `19_rsna_fixed_gmic_final_analysis`, version 1 (354389561), executed the frozen evaluator after all five inference shards completed. The evaluator verified every file digest, image membership, independently re-pooled breast features, consistent inference environments/model identities, complete coverage and absence of overlaps before joining cancer labels.

- Holdout: 9,530 patients, 19,060 breasts, 43,787 standard-view images and 393 cancer-positive breasts.
- Development calibration: 2,383 separate patients, 4,766 breasts and 99 cancer-positive breasts.
- Raw operating cutoff: `0.02242065602913494`; calibrated cutoff: `0.006898700768467533`.
- Five fixed external GMIC members, equal CC/MLO weighting after averaging repeated images within view; no RSNA weight training.
- Freeze SHA-256: `6cdc4c4f976c54b751eb15b85c9a1d1fa1db6877a554c6cb4ac468963c35b0ef`.

The design was frozen before final model outcomes, following adaptive development; it was not publicly preregistered. Final-cohort metadata and labels had previously been used for auditing and stratified assignment. The five shards are computational divisions of one holdout, not five independent experiments.

## Primary comparison and uncertainty

| Method | Review slots | Misses flagged / all misses | Capture | Review yield | Misses left unflagged |
|---|---|---|---|---|---|
| Cancer score alone | 415 | 3 / 39 | 7.69% | 0.72% | 36 |
| Combined three-score method | 415 | 2 / 39 | 5.13% | 0.48% | 37 |

| Patient bootstrap | 95% interval for capture difference | Draws | Defined | Zero-miss draws | Numerical failures |
|---|---|---|---|---|---|
| Fixed calibration (primary) | -13.79 to +8.11 pp | 2000 | 2000 | 0 | 0 |
| Calibration refitted (secondary) | -14.29 to +8.00 pp | 2000 | 2000 | 0 | 0 |

Positive differences favor combined scoring. Resampling retained both breasts per patient, preserved site patient counts, paired methods on the same holdout draw and recomputed the global review capacity. The secondary interval also resampled development patients and refitted calibration, cutoff and percentile references. It does not capture all adaptive model-selection uncertainty or possible external-pretraining overlap.

The realized review fraction was 4.99% of 8,315 AI-negative breasts. Random selection has an expected capture fraction of 4.99% at that capacity; this is an expectation, not another fitted method.

## Classifier performance at the frozen cutoff

| Group | Breasts | Cancer-positive | TP | FN | TN | FP | Sensitivity | Specificity | ROC-AUC | AP |
|---|---|---|---|---|---|---|---|---|---|---|
| All holdout breasts | 19060 | 393 | 354 | 39 | 8276 | 10391 | 90.08% | 44.33% | 0.77245 | 0.12907 |
| Site 1 | 9308 | 203 | 173 | 30 | 4732 | 4373 | 85.22% | 51.97% | 0.78662 | 0.16452 |
| Site 2 | 9752 | 190 | 181 | 9 | 3544 | 6018 | 95.26% | 37.06% | 0.78327 | 0.13857 |
| Site 1 density A/B | 4906 | 104 | 83 | 21 | 2830 | 1972 | 79.81% | 58.93% | 0.78315 | 0.14828 |
| Site 1 density C/D | 4386 | 99 | 90 | 9 | 1896 | 2391 | 90.91% | 44.23% | 0.79134 | 0.18828 |
| Site 1 density missing | 16 | 0 | 0 | 0 | 6 | 10 | undefined | 37.50% | undefined | undefined |

Pooled study-sample PPV = 3.29%; NPV = 99.53%. These are predictive values in this dataset, not estimates for a different screening population. The development target of 90% sensitivity was an empirical fitting target, not a holdout guarantee.

## All prespecified methods and budgets

Entries are flagged missed cancers / all missed cancers (capture fraction). Review capacity is global across sites and shards; all six methods have identical breast capacity at each budget. Secondary comparisons are descriptive and do not replace the primary result.

| Method | 1% budget | 5% budget | 10% budget | 20% budget |
|---|---|---|---|---|
| Cancer score alone | 0 / 39 (0.00%) | 3 / 39 (7.69%) | 8 / 39 (20.51%) | 18 / 39 (46.15%) |
| CC–MLO disagreement | 0 / 39 (0.00%) | 1 / 39 (2.56%) | 3 / 39 (7.69%) | 6 / 39 (15.38%) |
| Between-model variance | 0 / 39 (0.00%) | 1 / 39 (2.56%) | 4 / 39 (10.26%) | 13 / 39 (33.33%) |
| Cancer score + disagreement | 0 / 39 (0.00%) | 2 / 39 (5.13%) | 5 / 39 (12.82%) | 16 / 39 (41.03%) |
| Cancer score + variance | 0 / 39 (0.00%) | 2 / 39 (5.13%) | 7 / 39 (17.95%) | 14 / 39 (35.90%) |
| Combined three-score method | 0 / 39 (0.00%) | 2 / 39 (5.13%) | 5 / 39 (12.82%) | 13 / 39 (33.33%) |

| Budget | Breast selections |
|---|---|
| 1.00% | 83 |
| 5.00% | 415 |
| 10.00% | 831 |
| 20.00% | 1663 |

## Primary-budget workload and overlap

| Method | Selected patients | Selected patients with other breast already AI-positive |
|---|---|---|
| Cancer score alone | 404 | 200 |
| Combined three-score method | 398 | 140 |

The two primary selections overlapped on 163 breasts and 163 patients. Breast-selection counts do not measure radiologist time or additional whole-examination workload.

## Descriptive site and density comparisons at 5% global capacity

| Group | Method | AI-negative breasts | All misses | Selected breasts | Misses flagged | Capture |
|---|---|---|---|---|---|---|
| Site 1 | Cancer score alone | 4762 | 30 | 252 | 2 | 6.67% |
| Site 2 | Cancer score alone | 3553 | 9 | 163 | 1 | 11.11% |
| Site 1 density A/B | Cancer score alone | 2851 | 21 | 117 | 0 | 0.00% |
| Site 1 density C/D | Cancer score alone | 1905 | 9 | 135 | 2 | 22.22% |
| Site 1 density missing | Cancer score alone | 6 | 0 | 0 | 0 | undefined |
| Site 1 | Combined three-score method | 4762 | 30 | 184 | 2 | 6.67% |
| Site 2 | Combined three-score method | 3553 | 9 | 231 | 0 | 0.00% |
| Site 1 density A/B | Combined three-score method | 2851 | 21 | 92 | 2 | 9.52% |
| Site 1 density C/D | Combined three-score method | 1905 | 9 | 92 | 0 | 0.00% |
| Site 1 density missing | Combined three-score method | 6 | 0 | 0 | 0 | undefined |

Subgroups inherit the global selections; no subgroup receives a fresh 5% allocation. Density comparisons are limited to recorded site-1 density, with missing values separate. All site-2 density values are missing. Small missed-cancer denominators limit subgroup interpretation; these tables do not establish a density effect.

## Probability-quality diagnostics

| Pooled predictor | Brier score | Log loss |
|---|---|---|
| raw | 0.03024 | 0.12614 |
| calibrated | 0.01925 | 0.08982 |
| constant_development_prevalence | 0.02019 | 0.10044 |

Lower is better for these two metrics. The constant comparator uses development prevalence. Exact subgroup diagnostics, all ten fixed calibration bins and the full 0–20% review curve are preserved in the aggregate results JSON.

## Limits and next work

This evaluates one fixed external ensemble with an adapted full-field preprocessing pipeline in one retrospective dataset. Fine-detail loss, possible pretraining overlap, development adaptation, finite calibration samples and limited missed-cancer counts constrain interpretation. Between-member variance is a descriptive score, not a calibrated error probability. No human-reader study, interval-cancer endpoint or prospective workflow was evaluated. Novelty is not established.

The frozen experiment is complete. Next work is student review of methods and findings, completion of remaining related-work access leads, and a manuscript draft that retains unfavorable and inconclusive findings. Any changed score or threshold would require a separate future experiment and independent evaluation data; the opened holdout must not be reused as an untouched test set. No paper submission or public release is authorized by completion of this analysis.

## Records

Complete outputs remain in private Kaggle version 354389561. The aggregate JSON and all 4,000 bootstrap rows were also saved locally. Their contents match the original log; saved draws reproduce both intervals. A bulk archive was not obtained, so patient-level outputs remain in the private Kaggle originals.

- Bootstrap records: `validation/fixed_gmic_bootstrap_draws_20261001.csv`.
- Extraction and numerical verification: `validation/fixed_gmic_final_result_verification_20261001.json`.
- Aggregate results: `validation/fixed_gmic_final_results_20261001.json`.
- Versioned run and final log: `validation/kaggle_fixed_gmic_analysis_completed_20261001.json`.
- Exact run status, inference hashes and restart record: `validation/kaggle_fixed_gmic_final_status.json`.
- Frozen design: `FIXED_GMIC_FINAL_PROTOCOL.md`.
- Assistant contribution disclosure: `AI_ASSISTANCE.md`.
