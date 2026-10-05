# Development threshold stability — 28 September 2026

**The cutoff is sample-sensitive, and the combined warning score has no stable demonstrated advantage over cancer score alone.** These findings support revising the evaluation design; they do not validate a clinical warning system or explain the original sensitivity gap conclusively.

## What was completed

The declared `THRESHOLD_STABILITY_PLAN.md` was executed on the unchanged predictions from 2,383 reserved development patients (4,766 breasts). The original split has 1,787 fit patients with 73 positive breasts and 596 tune patients with 26 positive breasts. All five external GMIC models, image inclusion, view pooling and original patient assignments remained fixed. The 9,530 final-evaluation patients were not used.

We generated 2,000 independent pairs of fit/tune bootstrap samples. A bootstrap samples observed patients with replacement: both breasts travel together, and each site's patient count is preserved. Each pair produces three analyses: change the fit sample only, change the tune sample only, or change both. An additional 100 patient-separated 75%/25% partitions use only the pooled development cohort, stratified by site and whether a patient has a positive breast. All repetitions are retained.

The cutoff targets at least 90% sensitivity **in its own calibration sample**. Sensitivity means the fraction of cancer-positive breasts classified positive; specificity is the fraction of cancer-negative breasts classified negative. We also compare the existing combined score with cancer score alone when reviewing floor(5% × AI-negative breasts). Flagging a known false negative retrospectively is not evidence that a doctor would recover it.

## How variable is the operating point?

The original raw cutoff, 0.02931749, detects 18/26 tune cancers: 69.2% sensitivity, with 56.7% specificity. The original results reproduced exactly before resampling.

| Analysis | What changes? | Median sensitivity | 2.5th–97.5th percentiles | Draws reaching 90% |
|---|---|---:|---:|---:|
| Fit bootstrap | Patients setting the cutoff; original tune patients fixed | 69.2% | 57.7%–84.6% | 0/2,000 |
| Tune bootstrap | Patients measuring performance; original cutoff fixed | 69.6% | 50.0%–88.2% | 36/2,000 |
| Joint bootstrap | Both patient samples | 69.2% | 42.9%–92.9% | 112/2,000 |
| Alternative development splits | Membership of disjoint calibration/evaluation groups | 92.0% | 74.9%–100.0% | 53/100 |

The fit-bootstrap cutoff's central 95% range is **0.02207–0.05078**. On the same original tune patients, the central range corresponds to detecting 15–22 of the 26 cancers. Across all 2,000 draws, detection ranges from 14 to 23; at least 24 would be needed to reach 90%. Specificity's central range is 47.0%–71.3%, and the 5% review allocation varies from 27 to 42 breasts over its central range. This demonstrates a meaningful sensitivity/workload tradeoff, not a reliable 90% guarantee.

All 100 alternative splits have sensitivity above the original split's 69.2%; their full range is 73.1%–100%. Their lower cutoffs also produce more positive calls: median specificity is **44.4%**, versus the original 56.7%. They reuse the same 99 positive development breasts. Their more favorable sensitivity is not an independent validation, does not erase the original result, and must not be used to select a preferred split. The count of splits below the original is not a confirmatory p-value.

The fit-bootstrap and alternative-split ranges are empirical stability summaries, not confidence intervals for generalization. The tune and joint percentile intervals are conditional bootstrap summaries. None includes uncertainty from earlier model selection, retraining, an external population, or technical image-processing validity.

## Does combined scoring add value?

The combined score is the equal mean of three percentile ranks: cancer score, CC–MLO disagreement, and ensemble variance. The percentile reference is recomputed using only AI-negative calibration observations. Evaluation labels do not set the cutoff, reference distributions, or review order.

The original 5% comparison flagged **1 of 8 misses versus 0 of 8** for cancer score alone. Changing only tune patients retains that sparse pattern, but changing the calibration sample can reverse it:

| Analysis | Combined flags more misses | Equal | Combined flags fewer | Undefined: no misses |
|---|---:|---:|---:|---:|
| Fit bootstrap | 720 | 338 | 942 | 0 |
| Tune bootstrap | 1,371 | 626 | 0 | 3 |
| Joint bootstrap | 613 | 707 | 671 | 9 |
| Alternative development splits | 6 | 47 | 42 | 5 |

With both samples varying, the median difference in the fraction of misses flagged is **0 percentage points**, and its central 95% interval is **−50 to +37.5 percentage points** among the 1,991 defined draws. These draw counts are not independent patients, clinical success probabilities, or a formal superiority test. A changing cutoff also changes which cancers count as false negatives.

Site summaries use the same global review selection. They do not assign each site a separate 5% budget. Full site distributions and undefined-denominator counts are preserved in the aggregate results; the small denominators preclude strong subgroup claims.

## Decision and next step

Keep the original cutoff and all unfavorable results in the record. Do not choose a more favorable split, increase the target after inspecting tune outcomes, or change the primary score to the best development variant. Probability calibration alone cannot fix these threshold/ranking results: a strictly increasing transformation preserves the ordering used here.

The next step is a substantive prior-art comparison followed by a prospective revision of the final evaluation for fixed external models. That revision must specify calibration patients, cutoff selection, the mandatory cancer-score comparator, equal review workload, patient-level uncertainty analysis, and how the limited number of missed cancers constrains the endpoint. A useful outcome may be a careful negative reliability study; innovation and clinical benefit remain unestablished. The original trained-CNN final pipeline stays locked until an explicit compatible protocol is recorded.

## Reproduction and checks

Seven artificial-data checks passed, including weighted-versus-literal duplicate-row equivalence, preservation of patient clusters, label isolation, original-evaluator agreement, and disjoint reproducible partitions. Verified source inputs reproduce all original pooling and ranking summaries. The run produced 18,303 aggregate rows, with no skipped calibration draws. Analysis took approximately 14 seconds on CPU using already computed predictions; no GPU inference or retraining occurred. The four-panel figure was visually inspected.

From the project directory:

```bash
python3 -m unittest discover -s tests -p 'test_threshold_stability.py' -v
python3 threshold_stability.py --input /path/to/private/gmic_reserved_pilot
```

`threshold_stability_config.json` records the seeds and settings. `validation/threshold_stability/threshold_stability_results.json` records input/source hashes, software versions, and summaries. `resampling_draws.csv` retains every aggregate draw, and `threshold_stability.png` illustrates the findings. Patient split membership stays in the private output archive. This is an AI-assisted, adaptive development analysis declared after the original operating-point results, not a preregistration or final paper.
