# Reserved-development operating-point analysis — 2026-09-28

Declared after reviewing notebook 11 ensemble/member AP and ROC-AUC, before calculating its operating thresholds or false-negative review results. This is adaptive development, not preregistration or final validation.

## Scope and fixed choices

- Use only notebook 11 version 2 (352364754), completed successfully on the 2,383 reserved development patients. Verify saved file hashes and exact fit/tune counts. No final-evaluation patient may enter this analysis.
- Retain all five external GMIC members, their original equal probability pooling, all scored breasts and both sites. Do not select a member, site, view or patient based on outcomes.
- Fit one pooled positive-slope Platt calibrator using the 3,574 fit breasts only. Estimate a global threshold on those same fit breasts at the previously proposed 90% sensitivity target, with a positive decision defined by score >= threshold. The target is a research choice, not a clinical recommendation or guarantee.
- Apply the fitted calibrator and threshold unchanged to all 1,192 tune breasts. Report confusion counts, sensitivity, specificity, the number of AI-negative breasts, AP/AUC, Brier score and log loss. Report raw and calibrated Brier/log loss and compare with a constant probability equal to fit prevalence. A monotone calibrator cannot improve ranking.
- Normalize review features using the empirical distributions among AI-negative fit breasts, irrespective of their cancer labels. Reuse the previously specified six methods: cancer score alone; absolute CC–MLO disagreement; ensemble probability variance; equal percentile averages of score + disagreement, score + variance, and all three.
- Apply review budgets of 1%, 5%, 10% and 20% of AI-negative tune breasts, rounded down. Five percent remains the focal comparison. Use existing deterministic identifier-based tie ordering, never outcomes, to fill an exact budget.
- Report the number of false negatives flagged, total false negatives, review yield and global review workload. A flag is a simulated review candidate; no radiologist review or clinical cancer recovery is measured.
- Report site subgroups using the same global threshold and the same globally selected candidates. Do not separately allocate budgets or tune site-specific rules. Density remains a later exploratory analysis; this analysis will not infer missing density.
- Calculate a paired patient-cluster bootstrap with 2,000 replicates for the combined-minus-score FN capture difference at 5%, retaining both breasts together. Fitted calibration and threshold stay fixed. Report undefined replicates and acknowledge that very few misses make the interval unstable. This interval is not a confirmatory hypothesis test after adaptive development.
- Describe score distributions and correlations by role/site to detect practical concerns. Do not attribute site differences to breast density, device, or biology without supporting evidence.

## Decision after analysis

Use these results to assess whether a failure-prioritization study is feasible and whether the proposed uncertainty signals show any development promise beyond cancer score alone. Preserve negative results. Do not alter the final protocol or run final inference automatically. Adopting the fixed external ensemble requires an explicit prospective protocol revision and a substantive literature-gap review.
