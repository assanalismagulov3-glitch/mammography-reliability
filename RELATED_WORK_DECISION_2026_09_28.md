# Related work and research decision — 2026-09-28

**Decision:** continue as an empirical evaluation of false-negative prioritization. Do not describe uncertainty-based mammography review, CC–MLO disagreement, or the equal-weight combined score as our invention. The additional scores may fail to improve on the cancer score; that outcome must remain reportable.

This is a targeted primary-source review for an experimental decision, not a systematic review or a completed novelty assessment. No final-cohort model outcomes were examined for this decision.

## Closest evidence

| Study and access | What was examined | Consequence for this project |
|---|---|---|
| [Verboom et al., Radiology 2025](https://pubs.rsna.org/doi/10.1148/radiol.242594), main methods/results/limitations; supplement unavailable | Uncertainty-based referral to double human reading, with a cancer-score-only comparator and density analysis. Exam-level retrospective screening outcomes include follow-up. | The broad warning-and-review idea and score-only comparison already exist. Our narrower endpoint ranks released cancer labels among AI-negative breasts; it does not measure reader recovery. Exact supplemental uncertainty definitions remain unverified. |
| [Ahmed and Lasserre, CEI-Net, MICCAI 2026](https://papers.miccai.org/miccai-2026/1158-Paper2986.html), accepted main paper read; tables/figure visually checked | Trained cross-view evidential fusion with disagreement gating. CBIS-DDSM mass malignancy and VinDr BI-RADS proxy targets; classification, calibration and disagreement/error analyses. | Direct overlap with cross-view conflict as an uncertainty signal. The examined main paper does not evaluate fixed-budget false-negative ranking restricted to an unchanged model's negative predictions. That difference is an experimental distinction, not proof of novelty. |
| [Vu et al., M&M, 2023](https://arxiv.org/abs/2308.06420), full main methods/results/discussion now read | Multi-view sparse detection and multi-instance classification across five datasets, including negative cases; localization and classification endpoints. | Multi-view fusion is established. This study emphasizes false-positive control and detection/classification performance, rather than the proposed post hoc miss-ranking endpoint. Its retrained GMIC comparator is not our released-weight model. |
| [Woo et al., Radiology 2025](https://pubs.rsna.org/doi/10.1148/radiol.242408), main methods/results read | Retrospective review of invasive cancers missed by a localization-based AI definition; some misses were judged actionable. Cancer-only cohort. | Motivates error analysis, but neither validates our selection rule nor estimates population false-positive rates. Molecular subtype is not an available screening-time input in our data. |
| [Lu et al., 2021](https://arxiv.org/html/2107.02716v2), main methods/results now read | MC-dropout uncertainty, rejection and subgroup disparities for density classification. | Supports examining subgroup behavior and uncertainty-score choice. Density is the prediction target there, not a cancer false-negative endpoint. |
| [Jaeger et al., ICLR 2023](https://arxiv.org/html/2211.15259v2), failure-detection formulation and evaluation recommendations read | Evaluates classifiers and confidence measures together, emphasizes strong simple baselines and risk–coverage behavior; permits application-specific working points. | Keep the base classifier identical across review methods, compare at equal capacity, and retain the cancer-score baseline. Calibration quality and error-ranking quality require separate reporting. |

The CEI-Net PDF was obtained from the official conference link; M&M was obtained from arXiv. Their hashes and access scope are in `literature_search_log_2026_09_28.json`. Full copyrighted papers are reference material and are not included in the project archive.

## Important constraints from the data and baseline

The [RSNA dataset publication](https://pubs.rsna.org/doi/10.1148/ryai.250375) describes an enriched dataset, excludes interval cancers and does not provide precise lesion annotations. Therefore, study-sample predictive values cannot be presented as population screening values, and our endpoint cannot establish interval-cancer prevention. Our audit, rather than another paper's cohort size, determines the available sample.

The [GMIC paper](https://arxiv.org/html/2002.07613v1) selected five models using external validation performance across a hyperparameter search. These are not five otherwise identical seed-only models. Their dispersion will be called **between-member variation**, a heuristic whose usefulness we test. It is not a validated probability of error or a clean measurement of epistemic uncertainty.

Our full-field preprocessing is a documented adaptation of the author pipeline. Technical image inspection did not establish diagnostic equivalence or preservation of all small lesions. Distinct reported source institutions also do not prove that individual patients never overlap external pretraining data.

## Question and hypotheses

> Among breasts classified negative by a fixed external mammography ensemble, do CC–MLO disagreement and between-member variation improve false-negative prioritization over cancer score alone at equal review capacity, and how sensitive is the comparison to cutoff estimation?

The primary comparison remains the equal-weight three-component score versus cancer score alone when reviewing 5% of AI-negative breasts. The scientific alternative is a positive difference in the fraction of false negatives selected. A confidence interval containing zero is inconclusive; it does not prove equivalence. Component comparisons and other budgets are secondary.

This is a narrower, testable question. A defensible contribution would be the transparent comparison, threshold-estimation analysis, reproducible implementation and retained negative findings. A claim of a previously unstudied gap still needs broader searching and resolution of the access limits below.

## Decisions made after reviewing development evidence

1. Adopt all five fixed GMIC models as the baseline under evaluation. Do not select a favorable member or introduce another architecture in response to the held-out result.
2. Use all 2,383 reserved development patients for the final calibration/refit of the review rule. Their 99 positive breasts provide more calibration information than the original 73. Their historical fit/tune outcomes remain adaptive development evidence.
3. Evaluate once on the remaining 9,530 patients. Because no RSNA classifier training is planned, the prior trained-CNN five-fold experiment is superseded for this fixed-model branch by one patient-separated holdout. Original assignments remain preserved.
4. Preserve the 90% empirical calibration-sensitivity target and the original primary score/budget. Do not choose a favorable alternative split, recalibrate on the holdout, or switch the primary method based on development rankings.
5. Report intervals both conditional on the fixed rule and, secondarily, with calibration-sample uncertainty propagated. Neither accounts for every effect of adaptive development or external model selection.
6. Keep density exploratory within site 1. No hormone-receptor rule or Concept Bottleneck Model can be supported by the current annotations.

These are project design judgments, not conclusions demonstrated by the cited studies. The original tune sensitivity of 18/26 and the unfavorable larger-budget combined-score comparison are retained. A larger calibration set cannot guarantee that the sensitivity target transfers.

## Remaining evidence limits

- [SafeRecall, SSRN abstract 7324281](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7324281) is a relevant 2026 preprint lead: the indexed author abstract describes BI-RADS-guided uncertainty and recall reduction. The full page/PDF could not be accessed; detailed methods and peer-review status were not established. No performance claims are adopted.
- The CEI-Net reference chain identifies additional evidential multi-view work, including [Gudhe et al., IEEE Access 2024](https://doi.org/10.1109/ACCESS.2024.3399204). Bibliographic identification does not constitute a completed full-text review.
- The Verboom supplement remains unavailable. Its exact metric definitions cannot be claimed as reproduced.
- Searches used both web search engines and primary-source/citation follow-up. Some broad queries returned unrelated or general-information results. This is a coverage limitation, not evidence of absent research. No exhaustive PubMed, Scopus or Web of Science search/export or PRISMA count is claimed.

## Immediate next execution step

Implement a separate fixed-model evaluator and private inference launcher matching `FIXED_GMIC_FINAL_PROTOCOL.md`; verify label isolation, pooling, calibration, global budgets and patient-cluster resampling. Then fit the all-development rule, record immutable hashes and pass the new final-run gate before scoring the held-out cohort. This document does not unlock the old trained-CNN gate or represent completed final inference.
