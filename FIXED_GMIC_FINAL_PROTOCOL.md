# Fixed-GMIC final evaluation protocol

**Version:** 0.3, design specified 2026-09-28. **Execution status:** implemented with a separate local freeze record; each GPU job also requires successful Kaggle CPU verification, exact mounted cohort identities and development-fixture reproduction before inference. The authoritative freeze is `private_runs/fixed_gmic_final/frozen_fixed_gmic.json`. This is a prospective record relative to final model outcomes, not a public preregistration or a claim that choices preceded development results.

The machine-readable specification is `fixed_gmic_final_config.json`. Prior decisions and unsuccessful experiments remain in `DEVELOPMENT_DECISIONS.md`. The rationale and closest evidence are in `RELATED_WORK_DECISION_2026_09_28.md`.

## 1. Objective and estimand

Test whether an equal-weight combination of cancer score, CC–MLO disagreement and between-member variation selects more cancer-positive breasts among a fixed classifier's negative predictions than cancer score alone, when both methods can select the same number of breasts.

The primary estimand is:

`Delta = (false negatives flagged by combined score - false negatives flagged by cancer score) / all false negatives`

at a global review budget of 5% of AI-negative breasts. Positive Delta favors combined scoring. This is a retrospective selection endpoint against released breast-level labels. It does not establish diagnosis, recovery by a reader, clinical safety or reduced mortality.

## 2. Data roles and adaptation record

| Role | Patients | Breasts | Positive breasts | Permitted use |
|---|---:|---:|---:|---|
| All reserved development | 2,383 | 4,766 | 99 | Final calibration, cutoff and percentile reference distributions; software verification |
| Locked final holdout | 9,530 | 19,060 | 393 | One final evaluation after execution freeze |

Use the existing patient membership without repartitioning. Both breasts remain together. Original fit/tune labels and outer-fold IDs remain as provenance; they no longer define separate final calibration groups or five independently trained models. Computation shards may use those IDs, but do not receive separate cutoffs or budgets.

The change from the old trained-CNN cross-validation design follows adoption of externally trained, fixed GMIC models. All-development calibration increases the available positive calibration sample; it is not evidence of improved held-out performance. Original fit/tune results remain unchanged and must be disclosed as adaptive development.

Final-cohort metadata and labels have previously been used for audit and stratified assignment, and technical image preprocessing has occurred. No final-cohort model-performance assessment has been used in these decisions. Do not describe the entire dataset as previously unseen.

Include every audited breast with its standard CC/MLO images; all audited breasts have both views. Retain repeated standard views and average them as below. Exclude only the already specified nonstandard view labels from model input, not their breast groups. Failed decoding, missing predictions or unexpected identifiers stop the run; no silent or outcome-dependent case removal is allowed.

## 3. Fixed image model and preprocessing

- Five author-released GMIC members, indices 1–5, at upstream commit `594029efe6184bcca3a151fa1dd46c5fd620cb25`; verify files against `gmic_upstream_manifest.json` and record runtime SHA-256 digests.
- Same reviewed full-field preprocessing as completed notebook 11 version 2: deterministic orientation heuristic, aspect-preserving downscaling/padding to 2944 × 1920, FP32 inference, no test-time augmentation, no RSNA weight fitting.
- Preserve all orientation flags as technical metadata; flags alone exclude no case. Stop for actual software/data-integrity failures and log repairs without inspecting cancer outcomes to select a repair.
- Use malignant output channel 1. For each model, average probabilities within repeated CC images and within repeated MLO images, then average those two view values equally. Average the five breast probabilities equally for `p_raw`.
- `view_disagreement = abs(mean_model(p_CC) - mean_model(p_MLO))`.
- `ensemble_variance = population variance (ddof=0) of the five pooled breast probabilities`.

The released members differ through external hyperparameter selection. Variance is a descriptive warning feature, not a calibrated error probability. Full-field adaptation and possible small-feature loss remain study limitations.

## 4. Calibration and operating point

Fit the existing monotone Platt procedure on all development breasts: logistic regression of cancer on clipped logit(`p_raw`), constrained to a positive slope. Preserve clipping at 1e-7, L-BFGS-B fitting, log-slope bounds [-4, 4], intercept bounds [-20, 20], and the existing 1e-6 parameter penalty. Fail if fitting does not converge; do not silently substitute another calibrator.

Choose the raw cutoff as the largest observed positive-breast score attaining at least 90% empirical development sensitivity. With 99 positive breasts, this is the 90th highest positive score; tied values are all positive. Predict positive for `p_raw >= cutoff`. Derive the corresponding calibrated cutoff and require identical classification decisions; detect saturation/clipping discrepancies before freeze.

Calibrated probabilities are used for probability-quality reporting and the established review-score definitions. A monotone calibration cannot improve ranking by itself. The empirical 90% target is not a lower confidence bound and is not a sensitivity guarantee. Do not move the cutoff after final outcomes are known.

## 5. Review scores and capacity

Compute midrank empirical percentiles from **all AI-negative development breasts**, including any missed cancers; do not restrict the reference to true negatives. For reference values R and value x, the percentile is `(count(R < x) + count(R <= x)) / (2 * len(R))`.

| Stored method | Definition | Role |
|---|---|---|
| `score_confidence` | Calibrated cancer score; higher ranks first | Primary comparator |
| `score_disagreement` | Absolute ensemble CC–MLO difference | Secondary |
| `score_uncertainty` | Between-member population variance | Secondary; legacy column name |
| `score_confidence_disagreement` | Mean of the two development-reference percentiles | Secondary |
| `score_confidence_uncertainty` | Mean of the two development-reference percentiles | Secondary |
| `score_combined` | Mean of cancer, disagreement and variation percentiles | Primary candidate |

Select only among AI-negative holdout breasts. For fraction q, choose exactly `floor(q * N_negative)` breasts, descending by score. Resolve ties by ascending SHA-256 of `20260909:patient_id:laterality`, never by label or input row order. One global allocation spans sites and computational shards.

The primary q is 0.05. Report all six methods at q = 0.01, 0.05, 0.10 and 0.20; a descriptive curve uses q = 0.00 through 0.20 in 0.01 steps. No method, weight, cutoff or budget is selected using final results.

Capacity is measured in breast selections. Also report unique selected patients, overlap between methods, and selected patients whose other breast is already AI-positive. These are not measurements of radiologist time or additional whole-examination workload.

## 6. Outcomes and interpretation

Report exact TP, FN, TN and FP; sensitivity, specificity, study-sample PPV/NPV, ROC-AUC and average precision. Report raw and calibrated Brier score/log loss against a constant predictor set to development prevalence. If shown, calibration bins are ten fixed equal-width bins, with counts and no adaptive bin selection; rare positive labels limit calibration interpretation.

For each review method/budget, report eligible negatives, selection count, false-negative denominator, misses flagged, capture fraction, review yield and misses left unflagged. The random-selection expected capture fraction is the actual selected fraction, not a trained comparator. A descriptive residual-risk curve uses `unflagged misses / unreviewed AI-negative breasts`; label it as conditional on the AI-negative cohort, not overall screening risk.

For the primary comparison report Delta and its interval. If its 95% interval includes zero, report inconclusive evidence of incremental benefit. If it is wholly positive, describe evidence only within this fixed-model retrospective sample; do not infer clinical benefit. If wholly negative, report the unfavorable result. Secondary analyses do not replace a failed primary comparison. No confirmatory significance claims will be made from unadjusted secondary comparisons.

## 7. Patient-cluster uncertainty

Primary interval: 2,000 paired percentile bootstrap draws of holdout patients, sampled with replacement separately within each site, retaining both breasts and the observed per-site patient counts. All methods use the same draw. Keep the final calibration/rule fixed, recompute the global budget in each draw, and use the 2.5th and 97.5th percentiles. Do not resample breasts independently or stratify the bootstrap by cancer outcome.

Secondary interval: independently resample development and holdout patients within site for 2,000 paired draws. Refit calibration, cutoff and percentile references on each development draw, then evaluate all methods on the paired holdout draw. This propagates calibration-sample variability but not retraining, unknown pretraining overlap or all adaptive-selection uncertainty.

Use multiplicity weights equivalent to literal duplicated patient clusters, including partial selection at the last duplicated score when capacity is exhausted. Preserve fixed identifier tie ordering. Seeds are declared in the JSON configuration. Report undefined zero-FN draws and all numerical failures; no replacement of unfavorable or undefined draws. An observed zero-FN cohort has no defined capture comparison and is not evidence that the warning method succeeds. Bootstrap fractions of positive differences are not p-values.

## 8. Secondary groups and limits

Report site 1 and site 2 descriptively. Within site 1, compare available density A/B and C/D and report missing density separately. Every group inherits the global selections; do not reallocate review slots within a group. Report exact group and FN denominators; sparse groups may be uninterpretable. All site-2 density values are missing, so density is not a full-cohort primary endpoint.

Cancer labels are joined only for evaluation. Density/site may be used for the stated reporting or bootstrap strata, not as cancer-model or warning-score features. BI-RADS, biopsy, invasive status and difficult-negative-case labels are not predictors. Hormone-receptor information and lesion-concept supervision are unavailable; there is no clinical subgroup protocol or Concept Bottleneck Model in this experiment.

There are 393 labeled positive holdout breasts, but the number of false negatives is unknown before inference. For illustration only, sensitivities of 95%, 90%, 80% and 70% would leave roughly 20, 39, 79 and 118 misses. A single additional selected miss would then change capture by about 5.0, 2.6, 1.3 or 0.85 percentage points. These are sample-size scenarios, not observed outcomes, power estimates or forecasts.

## 9. Execution freeze and release record

Before any final inference, require all of the following:

1. Exact original cohort/metadata identities, expected counts, patient disjointness and complete standard-view coverage verified.
2. Model/source identities and preprocessing reviewed; an existing development inference fixture reproduces within the previously declared 1e-6 tolerance.
3. Separate fixed-model aggregation/scoring/evaluation code verified against the completed development results. The scorer accepts no cancer labels; labels enter only the evaluation stage. Perturbing evaluation labels must leave scores/selections unchanged.
4. Synthetic checks cover repeated-view pooling, percentile ties, monotone decisions, global capacity across shards/sites, duplicate-weight bootstrap equivalence and zero-FN cases.
5. All-development rule fitted and saved; protocol, configuration, evaluator/launcher, calibration artifact and cohort-manifest hashes recorded together. No fabricated or manually bypassed freeze token.
6. Private job configuration set to fail closed on missing/duplicate/nonfinite outputs; resumable image predictions retained. Final outcomes are produced only after full coverage and the frozen-rule checks pass.

The old `PROTOCOL.md`, original configuration and trained-CNN freeze gate remain historical and unchanged. This document specifies the implemented separate fixed-model path. The five original fold IDs define computation shards only; a single global analysis follows complete coverage. A verified recovery can resume the exact expected image prefix with the same environment, model files and freeze. A runtime repair requires a recorded deviation and rerun of relevant technical checks, not outcome-driven changes to the scientific rule.

Preserve aggregate results, code, environments, hashes and AI-assistance disclosure. Keep patient/image-level outputs and medical images private. Any later public release needs the existing data/license and artifact review, and the student must be able to explain the final methods. No paper or public release is produced by this protocol.
