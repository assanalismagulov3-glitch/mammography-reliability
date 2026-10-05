# Development threshold-stability plan — 28 September 2026

Declared after the original development operating-point findings, before computing the resampling results below. This is an adaptive diagnostic, not preregistration or final validation.

## Question

How sensitive are the 90%-target decision threshold, achieved sensitivity, review workload and combined-versus-cancer-score flagging comparison to the development sample?

## Fixed inputs and rules

- Use only the verified notebook 11 version-2 predictions from the 2,383 reserved development patients. Reuse the existing complete-input/hash/role checks. All 9,530 final-evaluation patients remain excluded.
- Keep the five external GMIC models, probabilities, pooling, image inclusion, original fit/tune roles and 90% empirical sensitivity target unchanged. Do not fit a new cancer classifier or select an ensemble member.
- Set positive if raw score >= threshold. Choose the largest observed positive-case score reaching the target on calibration observations, counting bootstrap multiplicities. The existing positive-slope Platt map preserves ordering, so recalibrating its numerical probability scale is unnecessary for this threshold/rank-only diagnostic. Do not claim this assesses probability calibration.
- Define the combined review score as the equal mean of three empirical midrank percentiles: raw cancer score, absolute CC–MLO disagreement, and five-member probability variance. Learn reference distributions from AI-negative calibration observations only. Include cancer cases among those predicted negatives; true-negative-only normalization would use outcome information incorrectly.
- Compare only the existing focal combined score and cancer score alone, at floor(5% × AI-negative breasts). Use the original label-independent SHA256 tie ordering with seed 20260909. Site summaries inherit global decisions and review allocation.
- Use seed 20260928 for calibration resampling, seed 20260929 for evaluation resampling, and 20261028 for the sequence of alternative splits. Record all seeds and results; no favorable-draw selection.

## Analyses

1. **Calibration-sample variation:** 2,000 bootstrap draws of original fit patients, sampled with replacement separately within each site while preserving that site's patient count. Do not condition on cancer counts. Both breasts receive the same multiplicity. Recompute the raw cutoff and normalization references; evaluate each unchanged rule on all original tune patients. Report cutoff, sensitivity, specificity, false negatives, review workload and capture difference. These are distributions conditional on the observed tune sample, not confidence intervals for generalization.
2. **Evaluation-sample variation:** 2,000 bootstrap draws of original tune patients, also clustered by patient and stratified by site. First evaluate the original fixed fit-derived rule. Then pair each draw with the corresponding independently generated fit draw and evaluate the refitted rule, capturing both sample sources. The paired joint-bootstrap percentile interval is conditional on fixed GMIC predictions and the chosen development procedure; it omits retraining, earlier model selection and external-population uncertainty. Keep zero-FN capture fractions undefined and count them explicitly.
3. **Alternative development partitions:** 100 independent random 75%/25% patient splits within the same pooled reserved development cohort, stratified by site and whether either breast is cancer-positive. Fit threshold/reference distributions on the 75% group only; evaluate on its disjoint 25%. Never change the saved original roles. Report all split results, median and central 95% empirical ranges. These overlapping repetitions are not independent studies, do not add patients, and do not provide a confidence interval for final performance. Do not interpret the rank of the original split as a confirmatory p-value.

For all three analyses, preserve fixed case inclusion and report pooled plus site-specific sensitivity/specificity and global review counts. Use integer multiplicities rather than physically copying private rows. Verify the weighted implementation against literal duplicate-row resampling and the existing unweighted evaluator on artificial fixtures.

## Interpretation and next decision

The analysis can show instability and workload tradeoffs; it cannot prove why the original fit/tune sensitivity differed. Do not lower the threshold to force a desired tune result or use bootstrap repetitions as extra cancer cases. Leave the current operating point unchanged. Use the findings to design any prospective conservative-threshold comparison and final protocol revision, with a substantive prior-art review before a novelty claim.

Methodological context: Rutter (2000), *Bootstrap estimation of diagnostic accuracy with patient-clustered data*, DOI 10.1016/S1076-6332(00)80381-5, supports accounting for within-patient correlation (author abstract consulted: https://pubmed.ncbi.nlm.nih.gov/10845400/). Cawley and Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, describes selection bias from optimizing finite-sample evaluation criteria (https://www.jmlr.org/beta/papers/v11/cawley10a.html). These references support methodological caution, not novelty or clinical validity of our particular analysis.
