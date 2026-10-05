# Working evidence review

Primary-source checks began on 2026-09-10 and were expanded through 2026-10-01. This is a targeted methodological review, not a systematic review or a novelty determination. The current access updates and post-outcome interpretation are in `RELATED_WORK_ADDENDUM_2026_10_01.md` and `literature_search_log_2026_10_01.json`. The frozen pre-outcome decision remains in `RELATED_WORK_DECISION_2026_09_28.md`; dated earlier access limits below are historical and should be read with the addendum.

## Assessment

The broad idea of using AI uncertainty to decide which mammograms need human review is established. Our candidate question is narrower: **among breasts classified as negative, do CC–MLO disagreement and between-member variation improve false-negative prioritization beyond the cancer score, at the same review budget?** Whether this precise comparison fills a defensible gap remains unresolved.

The 2026-09-28 update identified [CEI-Net, MICCAI 2026](https://papers.miccai.org/miccai-2026/1158-Paper2986.html), whose main paper explicitly models disagreement with evidential fusion. This strengthens the overlap and rules out treating cross-view conflict itself as our new idea. The new decision document compares endpoints and records the remaining access limitations.

Our proposed endpoint measures which released cancer labels a review rule selects. It does not measure cancers actually recovered by radiologists, interval-cancer prevention, or safety of replacing standard screening.

## Evidence and access limits

| Primary source | Evidence examined | Implication |
|---|---|---|
| [Verboom et al., 2025, AI Should Read Mammograms Only When Confident](https://pubs.rsna.org/doi/10.1148/radiol.242594) | Main methods, results, limitations. Retrospective hybrid screening with region-classifier uncertainty, including structured Monte Carlo dropout; patient-separated development and assessment. High uncertainty triggered double reading. A cancer-score-only strategy was a comparator. Density was analyzed. | Strong overlap with both the technical warning and double-check ideas. Uncertainty-based triage and comparison against cancer scores are already studied. Our proposed breast-level, predicted-negative, fixed-budget endpoint differs, but that distinction alone cannot establish novelty. The supplemental PDF returned 403; exact supplemental metric definitions remain unverified. |
| [Woo et al., 2025, Invasive Breast Cancers Missed by AI Screening of Mammograms](https://pubs.rsna.org/doi/10.1148/radiol.242408) | Main methods and results. Cancer-only, two-center cohort: 1,097 invasive cancers and 154 AI misses; 95 misses judged actionable by retrospective readers. Misses depended on lesion localization and an abnormality-score threshold. | Supports investigating AI misses, but does not validate our review rule. Its localization endpoint differs from breast-label classification; a cancer-only cohort cannot establish population false-positive rates. Molecular subtype is postdiagnostic and absent from our RSNA metadata. |
| [RSNA dataset paper](https://pubs.rsna.org/doi/10.1148/ryai.250375) | Relevant methods and limitations. Breast-level labels; interval cancers excluded; precise lesion locations not supplied. Small findings can be lost by aggressive downsampling. | Our endpoint concerns errors against the released labels. A lesion-concept bottleneck and interval-cancer analysis require additional data. Our own audit governs counts and density availability. |
| [Chen et al., MICCAI 2022, Multi-view Local Co-occurrence and Global Consistency Learning](https://arxiv.org/pdf/2209.10478) | Main text through methods, results, and conclusion. BRAIxMVCCL learns global consistency and local relationships between CC and MLO; breast-level supervision, ImageNet initialization, large private training data, and public cross-dataset evaluation. | Cross-view consistency and fusion are established. The examined experiments assess classification/generalization rather than our proposed post hoc, fixed-budget miss ranking. Penalizing disagreement during training differs from using it afterward as a warning. |
| [Vu et al., 2023, M&M](https://arxiv.org/abs/2308.06420) | Updated 2026-09-28: main methods/results/discussion read; table page 6 visually checked. Multi-view sparse detection and classification, including negative cases. | Established cross-view modeling with localization/classification endpoints; no claim that it directly tests our fixed-budget miss ranking. The earlier abstract-only access limitation is resolved for the main paper. |
| [Shen et al., GMIC, 2020 preprint](https://arxiv.org/abs/2002.07613) | Updated 2026-09-24: methods and classification results (sections 2, 3.1–3.3) and pinned author implementation examined. Global/local model with fixed released weights. | Transfer candidate now undergoing engineering preflight. Not a concept bottleneck or a new classifier of ours; clinical-concept and transfer-validity claims remain unsupported. |
| [Lu et al., 2021, Evaluating subgroup disparity using epistemic uncertainty in mammography](https://arxiv.org/abs/2107.02716) | Updated 2026-09-28: main methods/results read. MC-dropout uncertainty, rejection and subgroup analysis for density classification. | Relevant to reliability evaluation, but the target is density rather than cancer. It cannot establish cancer false-negative detection. |
| [Wu et al., 2019, Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening](https://arxiv.org/abs/1903.08297) | Abstract. Large-scale mammography modeling with breast-level and pixel-level information and a reader study. | Established modeling and reader-study literature. Different data, training resources, and endpoints prevent direct comparison of reported performance with our small pilot. |

## Consequences for the design

1. Retain the cancer score as the primary comparator at identical review workload. Report component scores and combinations; a combined improvement cannot show that each component helped.
2. Treat view disagreement as a candidate signal, not a probability of error. Two views can agree on a wrong answer; a finding can also be clearer in one view. These are reasons to test the signal empirically.
3. Keep adaptive model development isolated from final evaluation. The failed pilots and all subsequent choices remain part of the research record.
4. Keep density secondary and restricted to the site where it is available. Our audit found all site-2 density values missing. An apparent density association would require careful consideration of site and sample size.
5. A retrospective review-policy simulation can follow successful technical evaluation. We have no reader experiment demonstrating that doctors recover the selected cancers, and no screening-time hormone-receptor variable.
6. Do not describe familiar scores combined by a simple rule as a new uncertainty algorithm. A reproducible empirical comparison or negative result may still be useful, provided the gap and limitations are justified.

These are project judgments informed by the sources and data audit, not claims that the papers tested our method.

## Search limitations and next review work

The targeted searches covered mammography uncertainty, false-negative screening errors, CC–MLO disagreement, multi-view consistency, ensembles, and high-resolution baseline training. Follow-up query strings included `mammography "view disagreement" "uncertainty"`, `mammography "CC" "MLO" "false negative" uncertainty ensemble`, and `"242594" "suppl" uncertainty mammograms`.

Unrelated and nonprimary hits were excluded. Sparse search results do not show that prior work is absent. The review has not completed systematic database searching, citation chaining, or a full text assessment of every candidate. Before a novelty claim, obtain the missing methods, follow references and citing papers, search medical and computer-science indexes for selective prediction/abstention and view inconsistency, and record inclusion/exclusion decisions. No PRISMA screening count is claimed.

Follow-up on 2026-09-23: requested searches for `"M&M: Tackling False Positives" mammography full paper` and `mammography "view disagreement" uncertainty false negative`. The search service reported rewriting the first query; unrelated confectionery and nonmammography results were excluded. The [official MICCAI 2023 entry](https://conferences.miccai.org/2023/papers/394-Paper3448.html) was identified, but its full-page fetch returned an error; the arXiv PDF fetch also failed. M&M therefore remains abstract-level evidence in this record. No absence-of-prior-art conclusion follows from this unsuccessful access attempt.

## Diagnostic implementation references

The [official PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) describes fixed pretrained feature extraction. Our diagnostic uses that approach with the existing breast/view pooling and a simple fitted classifier. [Scikit-learn's LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) documents regularization and class weights. Scaling and fitting use fitting patients only. These sources do not establish optimal mammography hyperparameters.

The [authors' GMIC repository](https://github.com/nyukat/GMIC) provides pretrained models and describes a 2944 × 1920 input pipeline and AGPLv3 license. Its documented dependencies are old. Since the initial candidate inspection, an adapted full-field pipeline has completed development inference. That adaptation is not an exact reproduction of the original input geometry; execution evidence and limitations are in `GMIC_DEVELOPMENT_RESULTS.md`.

## GMIC feasibility follow-up — 2026-09-24

The versioned [HTML paper](https://arxiv.org/html/2002.07613v1) was accessible, enabling methods review beyond the abstract. The PDF was too large for the web reader. Author code was retrieved and verified against Git commit `594029efe6184bcca3a151fa1dd46c5fd620cb25`. Preflight and paper differences, source-data provenance limits, and pending reference/crop checks are recorded in `DEVELOPMENT_DECISIONS.md`. These checks concern baseline feasibility, not novelty of the proposed failure-detection comparison.
