# Working experiment protocol — version 0.2

This is an implementation protocol prepared before model results. It is not a preregistration, literature review, research paper, or claim of originality. Changes made after reviewing only development results must receive a new configuration/source record before the held-out predictions are opened.

## Question and scope

Working question: does CC–MLO disagreement and variation across independently trained model members improve the prioritization of false-negative breast predictions beyond the model's cancer score alone?

Primary scope: retrospective failure detection for the experimental model on the labeled RSNA training release. Density is a secondary subgroup variable. No radiologist review, clinical intervention, mortality benefit, or prospective screening performance is measured.

The RSNA paper describes breast-level cancer labels, density availability at site 1, exclusion of interval cancers, and absence of precise lesion-localization annotations. The release used here must be audited directly: earlier user outputs contained extra image records even though the publication describes a standardized view selection. [Dataset paper](https://pubs.rsna.org/doi/10.1148/ryai.250375)

Concept bottleneck modeling is deferred: the supplied table does not contain the lesion-concept annotations needed to validate the proposed clinical concepts. BI-RADS, biopsy, invasive status, and difficult-negative-case flags will not be inputs to the baseline or failure score. Density is not imputed for site 2.

## Cohort and data audit

- Start with every breast having at least one CC and one MLO image; the user's current audit reports complete coverage. Missing standard views trigger an explicit cohort decision.
- Retain additional CC/MLO images. Record other view types and do not feed them to the paired-view baseline. Preserve the original CSV.
- Stop on invalid identifiers, inconsistent cancer labels, repeated image identifiers, or exact decoded-pixel duplicates across patient identifiers. Investigate duplicated patients or define connected patient groups before splitting if needed.
- Mark conflicting density labels as unavailable for subgroup analysis. Distinguish partial from complete missingness.
- Record source CSV hash, software versions, image dimensions, display transforms, cache integrity, patient counts and positive counts by partition.

## Patient partitions

Before any outcome-guided model development, reserve 20% of patients as a pilot cohort, stratified by site and whether the patient has cancer in either breast. Use 75% of that cohort for pilot fitting and 25% for pilot checkpoint selection and global method decisions. These patients never enter a final test endpoint. The selection seed and assignments are recorded; do not search for a favorable split.

The remaining 80% form the evaluation cohort. Five outer folds within this cohort are stratified by site and patient cancer status. Within each outer fold, 60% of evaluation-cohort patients fit the model, 10% select checkpoints, 10% fit calibration/choose the operating point, and 20% form the held-out test fold. After global settings are frozen, all pilot patients additionally contribute to final model fitting. Approximate fractions of the whole release per final fold are therefore 68% fit, 8% tune, 8% calibrate, and 16% test. Both breasts and all images of a patient remain together.

The fractions are approximate because patients, strata, and integer counts are indivisible. Five-fold results cover the evaluation cohort only. Pilot reuse for fitting does not make pilot patients eligible for test endpoints. The calibration partition is never used for gradient updates or checkpoint selection. Test labels are used only for the prespecified final calculations, apart from their initial use in stratification. Final-fold tuning results select checkpoints automatically; they cannot trigger new global method choices, since those patients may be test patients in another fold.

**Resolved before model outcomes on 2026-09-09:** version 0.1 allowed a fold-0 pilot whose development patients overlapped other folds' later test patients. Version 0.2 implements the independent pilot cohort above in `study.py`. The previous 20-image check and GPU benchmark were engineering checks and did not evaluate cancer-model performance. The revised experiment uses a new output directory; old audit records remain preserved.

The cost is a smaller final evaluation sample, particularly for false negatives and density subgroups. Report exact denominators and wide or inconclusive intervals honestly. An isolated validation cohort keeps model selection separate from evaluation; this design does not run a full nested hyperparameter search. [scikit-learn model-selection guidance](https://scikit-learn.org/stable/modules/cross_validation.html)

Final-fold training, evaluation, and summary require `frozen_protocol.json`. This records the code fingerprint, configuration, partition-file hashes, and a concrete review of completed pilot trials, resolution, related research, and limitations. A freeze cannot be created retrospectively after final-fold training or test predictions exist.

## Image and model specification

- Process grayscale DICOM pixels using modality and VOI transforms, handle MONOCHROME1 inversion and padding, and preserve the complete field of view.
- The initial configuration pads/resizes to 1024 × 768. It uses lossless 16-bit caches when storage permits. This resolution is a feasibility starting point; it is not established as sufficient for small lesions. Inspect real images and assess a higher-resolution development run before freezing final experiments.
- A separate pilot-only cache at 1536 × 1152 is prepared directly from original DICOM pixels for comparison with 1024 × 768 using the same pilot patients and initial model seed. PNG compression is lossless for the resized arrays; downsampling itself removes spatial detail. Higher resolution is a candidate to test, not a proven improvement. Both trials require the complete original-pixel duplicate audit across the full study cohort before training.
- The real engineering sample estimated a 27.70 GiB cache against a 19.50 GiB notebook output limit. Two private storage parts retain this resolution and 16-bit intensity precision. Storage membership uses patient ID modulo two and does not assign evaluation roles. Validate complete image coverage and cross-patient pixel duplicates again after joining parts.
- Shared ImageNet-initialized ResNet-18 produces an image logit. Average repeated-image logits within CC and within MLO; average the two view logits to obtain one breast logit. The label and loss are at breast level.
- Freeze batch-normalization running statistics for small batches; retain trainable affine parameters. Use horizontal flips only during fitting, with the full image retained.
- Three independent seeds use identical patient partitions. AdamW, class-weighted binary cross entropy, and validation average precision are specified in `config.json`. Class weight uses fitting labels only. Each member is selected by tuning average precision.

## Calibration and failure scores

The base cancer score is the mean of member probabilities. Fit a positive-slope logistic calibration map on the separate calibration partition. Its intercept can correct the probability shift introduced by class-weighted fitting; successful calibration must still be demonstrated empirically.

Choose the largest threshold achieving at least 90% sensitivity on calibration positives, using `p >= threshold` as a positive prediction. Ninety percent is a research operating target, not a clinical recommendation. Report the actual test sensitivity without changing the threshold to force that target on the test set.

For breasts predicted negative, compare:

1. Cancer score alone: the essential confidence comparison.
2. Absolute difference between mean CC and mean MLO probabilities.
3. Variance of breast probabilities across the three members.
4. Equal-weight percentile combination of cancer score and view disagreement.
5. Equal-weight percentile combination of cancer score and ensemble variance.
6. Equal-weight percentile combination of all three signals: the primary candidate.

Percentile reference distributions use predicted-negative calibration breasts only. The combination has fixed weights and does not fit a separate error classifier. This limits extra fitting demands when false negatives are few. Entropy of one cancer probability is a deterministic function of that probability, so it is not treated as independent uncertainty evidence.

The view-specific sigmoid outputs are uncalibrated model scores. Their disagreement is a candidate signal whose usefulness must be established by the held-out comparison.

## Endpoints and uncertainty

Primary comparison: difference in the fraction of false negatives flagged when reviewing the highest-scoring 5% of predicted-negative breasts in each held-out fold. Use floor rounding, and deterministic label-independent tie ordering. Aggregate numerators and false-negative denominators across folds. Also report actual review counts, review yield, and the expected capture fraction under random review.

Prespecified secondary budgets: 1%, 10%, and 20%. Other score comparisons are secondary. Report base-model ROC-AUC, average precision, confusion counts, Brier score, negative log likelihood, and 10-bin calibration error. Predictive values describe the study sample.

Use a paired bootstrap resampling patients within each held-out fold, retaining both breasts together. The 95% percentile interval is conditional on fitted models, calibration, and thresholds; it does not include retraining uncertainty or establish external generalization. Report undefined resamples. A zero false-negative denominator produces an undefined capture fraction rather than a claimed success.

Density and site summaries inherit the global review selections. Report subgroup denominators and captured counts. Do not silently reallocate the budget separately to each density group. Density comparisons are restricted to site 1 and are exploratory when counts are small. Cross-site held-out validation is a potential extension requiring separate source-only calibration and a frozen protocol; it is not implemented as evidence by ordinary site-stratified cross-validation.

## Required real-data decisions before final evaluation

1. Confirm the raw metadata audit, remaining density questions, file integrity, and image decoding.
2. Visually inspect full-frame processed examples across sites, views, and display encodings. Investigate decoding failures and exact pixel duplicates.
3. Measure GPU memory, disk use, and preprocessing/training time. Run a development-only pilot; inspect learning curves and validation discrimination relative to the positive fraction and random ranking.
4. If the model cannot learn useful mammographic information, improve the baseline using development data before failure-detector claims. Check higher resolution on development data. Log every configuration tried.
5. Freeze the final configuration and analysis plan before evaluating the held-out folds. Preserve negative and inconclusive results.
6. Conduct the substantive literature review and have the methodology critically reviewed before a novelty claim or paper submission.
