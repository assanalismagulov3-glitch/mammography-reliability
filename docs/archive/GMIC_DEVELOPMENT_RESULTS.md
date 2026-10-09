# GMIC development results — 28 September 2026

The fixed mammography ensemble produces useful development ranking, but the proposed failure-prioritization benefit is not established. Its fit-chosen sensitivity target did not transfer to the separate tuning group. These are adaptive development findings, not final test results, clinical validation, or a novelty claim.

## What ran

Private [notebook 11, version 2](https://www.kaggle.com/code/aseqwss/11-rsna-gmic-reserved-pilot/output?scriptVersionId=352364754) completed in 18,611.9 seconds, approximately 5 hours 10 minutes. It scored 10,881 images from 2,383 reserved development patients, producing predictions for 4,766 breasts. All five external GMIC members stayed fixed. Repeated images were averaged within view, then CC and MLO received equal weight, followed by equal averaging of the five members.

The saved predictions, plan and summary passed their recorded SHA256 checks. The original patient-role checksum matched. Independent image-to-breast aggregation and all 16 original AP/AUC summaries reproduced. No final-evaluation predictions were opened.

The operating-point plan was written after inspecting the pilot AP/AUC values and before calculating its thresholds and review endpoints. It therefore documents adaptive development and is not a preregistration. Two artificial software checks passed: tuning labels cannot change fitted scores or selection, and site summaries inherit the single global review allocation.

## Base model and operating point

Calibration and the threshold used only the fit group. The threshold was the largest score attaining at least 90% empirical fit sensitivity; it was applied unchanged to the tune group. Positive-slope calibration preserves score ordering and cannot fix missed cancers merely by changing the probability scale.

| Group | Breasts | Cancer-positive | ROC-AUC | Average precision | Detected / positive | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit, pooled | 3,574 | 73 | 0.824 | 0.221 | 66 / 73 | 90.4% | 52.0% |
| Tune, pooled | 1,192 | 26 | 0.726 | 0.147 | 18 / 26 | 69.2% | 56.7% |
| Tune, site 1 | 582 | 13 | 0.761 | 0.442 | 9 / 13 | 69.2% | 66.6% |
| Tune, site 2 | 610 | 13 | 0.653 | 0.043 | 9 / 13 | 69.2% | 47.2% |

Raw threshold: 0.0293174902. Locally calibrated threshold: approximately 0.007237284. Calibration slope/intercept: 0.885357 / −1.822660. Small optimizer differences across software environments are recorded; confusion and review counts reproduced unchanged on Kaggle.

The tune cohort contains 523 AI-positive and 669 AI-negative breasts. There are 8 false negatives, four at each site. A nominal 90% fitting target does not guarantee 90% sensitivity on another sample. With only 26 tune positives, these observations cannot establish the cause of the decline or a reliable site effect.

## Simulated review of AI-negative breasts

All methods review the same number of breasts at each budget, selected without using tune cancer labels. Entries below are **false negatives flagged out of 8**. A flag is a candidate for review; no human review or clinical recovery was measured.

| Method | 1%: 6 reviews | 5%: 33 reviews | 10%: 66 reviews | 20%: 133 reviews |
|---|---:|---:|---:|---:|
| Cancer score alone | 0 | 0 | 0 | 5 |
| CC–MLO disagreement | 0 | 1 | 3 | 3 |
| Ensemble variance | 0 | 0 | 2 | 2 |
| Score + disagreement | 0 | 1 | 3 | 4 |
| Score + variance | 0 | 0 | 2 | 3 |
| Score + disagreement + variance | 0 | 1 | 2 | 3 |

The focal combined-versus-score comparison at 5% is 1/8 versus 0/8: a 12.5 percentage-point capture difference. The conditional paired patient-bootstrap 95% percentile interval is **0 to 50 percentage points**; 1,997 of 2,000 replicates were defined. This sparse, adaptive result does not demonstrate an improvement. The three methods flagging a miss at 5% flag the same single miss, not three different cancers.

The actual focal review fraction is 33/669 = 4.93% after rounding down, or 2.77% of all tune breasts. Combined-score yield is 1/33 = 3.03%. At 20%, cancer score alone flags more misses than the combined method. This unfavorable comparison must remain in the research record.

## Calibration and technical limits

On pooled tune data, Brier score changed from 0.02824 raw to 0.02032 calibrated; a constant fit-prevalence predictor scores 0.02134. Log loss changed from 0.12457 to 0.09678, versus 0.10506 for that constant predictor. Lower is better for both metrics. At site 2, however, the calibrated model is slightly worse than the constant predictor on both measures. Pooled improvement does not establish subgroup calibration.

Among AI-negative tune breasts, Spearman correlation with cancer score is 0.589 for view disagreement and 0.730 for ensemble variance. These signals overlap with cancer score; they are not automatically independent evidence of failure.

Three fit-group, site-1 CC images met the predeclared low-orientation-contrast flag. They are retained, with no label-dependent exclusion or prediction change. All three original/processed pairs passed the subsequent technical visual check: no obvious orientation, inversion or boundary-exclusion error at contact-sheet scale. Notebook 12 version 2 reproduced the analysis and prepared this audit in 46.7 seconds. This is AI engineering inspection, not radiologist review. Downsampling and heuristic orientation remain limitations even if a small visual check passes.

## Decision

Continue as an exploratory reliability study using a fixed external mammography classifier, while preserving its limitations. Do not describe the method as clinically reliable or switch the primary score to the best-looking development variant. No best-member selection, site exclusion, density imputation, or hormone-receptor rule was introduced.

Operating-point stability has now been assessed using development patients; see `THRESHOLD_STABILITY_RESULTS.md`. The cutoff is sample-sensitive, and combined scoring does not show a stable advantage. Before final inference, finish the substantive prior-art review and explicitly revise the final protocol for fixed external models and probability pooling. The original trained-CNN final pipeline remains locked. Density is a later site-1 exploratory subgroup analysis, and concept bottlenecks require annotations not supplied by this release.

Reproduction: from the source directory run `python3 gmic_operating_point.py --input /path/to/private/notebook11/gmic_reserved_pilot`. Inputs must be the verified private notebook outputs. Full aggregate tables, software versions, provenance and the conditional bootstrap are in `validation/gmic_operating_point/`. Patient-level predictions remain private.
