# Literature comparison and manuscript framing — 2026-10-01

**Decision:** the targeted literature check is complete enough to frame this project as a transparent empirical evaluation. It does not establish methodological novelty. The completed experiment did not establish an incremental benefit from the combined review score.

This addendum was written **after final outcomes were available**. It updates source access and manuscript interpretation only. The September 28 decision, frozen protocol, score definitions, primary endpoint and results remain unchanged. This is neither a systematic review nor a preregistration.

## Closest comparisons

The distinctions below are our interpretation of the examined sources, not proof that no other study has tested a similar question.

| Study | Relevant overlap | Distinction from this experiment |
|---|---|---|
| [Verboom et al., Radiology 2025](https://pubs.rsna.org/doi/10.1148/radiol.242594) | Uncertainty selects examinations for double reading; a malignancy-score strategy is a comparator. | Retrospective examination-level reading outcomes differ from ranking cancer-positive breasts within a fixed classifier's negative predictions at equal breast capacity. We did not measure reader recovery. |
| [Ahmed and Lasserre, CEI-Net, MICCAI 2026](https://papers.miccai.org/miccai-2026/1158-Paper2986.html) | Trained evidential fusion explicitly uses cross-view disagreement. | Classification/calibration on CBIS-DDSM and VinDr differs from post hoc review ranking of an unchanged external ensemble. Cross-view conflict is established prior work. |
| [Gudhe et al., IEEE Access 2024](https://doi.org/10.1109/ACCESS.2024.3399204) | MV-DEFEAT fuses view evidence using Dempster–Shafer reasoning. Density is its main target; additional experiments transfer to VinDr BI-RADS and DDSM benign/malignant/normal classification. | It must not be dismissed as density-only. The examined sections evaluate trained classification and transfer, rather than our negative-only, equal-capacity miss-selection endpoint. |
| [Vu et al., M&M, 2023](https://arxiv.org/abs/2308.06420) | Multi-view and multi-instance mammography modeling, with emphasis on false positives. | Detection/classification differs from our review ranking. Its retrained GMIC comparison is not a direct benchmark for our released-weight pipeline. |
| [Lu et al., 2021](https://arxiv.org/html/2107.02716v2) | Uncertainty, rejection and subgroup behavior in density classification. | Relevant evaluation context; the target differs from cancer-label false negatives. |
| [Jaeger et al., ICLR 2023](https://arxiv.org/html/2211.15259v2) | Failure-detection evaluation and strong simple confidence baselines. | Methodological support for retaining the score baseline; not validation of our particular heuristic. |

The September 28 review retains the full access record for CEI-Net, M&M, Lu, Jaeger, Woo, the dataset and GMIC. This update does not imply those papers were all newly reread in full.

## Two access questions resolved

### Verboom supplemental methods

The [publisher supplement](https://pubs.rsna.org/doi/suppl/10.1148/radiol.242594/suppl_file/ry242594suppa1.pdf) was visually inspected on PDF pages 1–4, including Appendix S1 equations and the beginning of Appendix S2. Intermittent retrieval failures preceded successful page rendering; the remaining pages were not fully reviewed.

Appendix S1 defines four families, each evaluated using raw classifier output and transformed malignancy scores, giving eight candidates:

| Family | Quantity summarized |
|---|---|
| All-region variation | Sum of MC-dropout variances across detected regions |
| Single-region variation | MC-dropout variance for the most suspicious region |
| Entropy of MC mean | Binary entropy of the mean dropout score for that region |
| Single-score entropy | Binary entropy of the region's ordinary malignancy score |

The dropout calculations use 200 samples. These definitions establish prior use of variance and entropy in mammography referral. They are not a reproduction of our between-GMIC-member score. In particular, entropy of a dropout mean should not silently be equated with entropy of a single deterministic prediction. We did not reimplement or numerically verify their metrics.

### Gudhe published paper

The IEEE landing page and PDF did not provide readable full text through the available retrieval paths. The published article's [full-text transcription on ResearchGate](https://www.researchgate.net/publication/380470401_A_Multi-view_deep_evidential_learning_approach_for_mammogram_density_classification) was available, with matching DOI, title, authors and journal pagination. The [authors' research-group publication list](https://www.behravans.com/publications/) corroborated the bibliographic identity.

We examined the abstract, evidential fusion and learning sections, and results sections IV.C–F. We did not visually verify every numerical table or independently reproduce performance. No cross-paper numerical superiority claim is made.

## Unresolved leads and search limits

- [SafeRecall, SSRN 7324281](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7324281): the indexed author abstract and a delivery-PDF hit identify a relevant uncertainty/recall preprint. Full page and PDF remained inaccessible. Detailed methods, denominators, validation quality and publication status remain unverified; its numerical claims are not adopted.
- [MMDC-CLIP-F, 2026 publisher-indexed lead](https://www.sciencedirect.com/science/article/abs/pii/S0925231226021934): identified during the updated search. The publisher page returned 403; full methods were not examined. It remains a follow-up lead, not evidence for a precise method comparison.
- The earlier medRxiv failure-benchmark lead remains discovery-only, as recorded on September 28.
- Both available web search engines were used. Some heavily constrained searches returned unrelated or generic pages. These retrieval failures cannot establish absence of prior work. There was no exhaustive database search, deduplicated screening count or PRISMA process.

The exact retained queries and access scope are in `literature_search_log_2026_10_01.json`. These limitations permit a cautious empirical paper; they prevent a claim that the method or application is the first of its kind. A future novelty claim would require further evidence.

## Defensible contribution

**Research question:** for breasts classified negative by a fixed mammography ensemble, do view disagreement and between-member variation improve the fraction of missed cancer labels selected for possible review, compared with cancer score alone at the same capacity?

The contribution is the completed, auditable comparison in this specific setting:

1. A patient-separated RSNA holdout evaluation of five fixed external GMIC members, with documented preprocessing adaptation and development-only calibration.
2. An unchanged cancer-score comparator, component comparisons, identical global breast-selection capacity, and retention of all prespecified methods and budgets.
3. Patient-cluster uncertainty estimates, including a secondary analysis that refits calibration and the cutoff on resampled development patients.
4. Transparent reporting of weak development trials, implementation failures, the inconclusive primary result and limits of transfer to clinical workflows.

These are empirical and reproducibility contributions. They are not new architecture, first-use, clinical-safety or demonstrated-benefit claims.

## Result wording for the manuscript

The 9,530-patient holdout contained 19,060 breasts and 393 cancer-positive breasts. At the frozen cutoff, 39 positive breasts were classified negative. Reviewing 5% of the 8,315 AI-negative breasts allocated 415 selections to each method. The combined score flagged 2/39 missed positive breasts (5.13%); cancer score alone flagged 3/39 (7.69%). The difference was −2.56 percentage points, with a primary 95% paired patient-bootstrap interval of −13.79 to +8.11 points. Incremental benefit was not established in this setting. The interval does not demonstrate equivalence or rule out benefit in other settings.

The secondary calibration-refit interval was −14.29 to +8.00 points. At 20% capacity, combined scoring flagged 13/39 misses versus 18/39 for cancer score alone. That descriptive comparison must remain visible; it cannot replace the primary endpoint. Numerical provenance is `FIXED_GMIC_FINAL_RESULTS.md` and its linked aggregate records.

| Proposed statement | Decision |
|---|---|
| “We evaluated whether the added signals improve miss prioritization beyond cancer score.” | Supported by the design. |
| “The combined score improved false-negative detection.” | Unsupported by the primary result. |
| “Uncertainty was proven useless / the methods are equivalent.” | Unsupported; the interval is wide. |
| “Two missed cancers were recovered by radiologists.” | Unsupported; two labeled positive breasts were flagged, with no reader study. |
| “Five independent test cohorts confirmed the result.” | Incorrect; five inference shards form one holdout. |
| “The protocol was publicly preregistered.” | Incorrect; it was internally frozen after adaptive development. |
| “This is the first uncertainty-based mammography review system.” | Unsupported by prior work and the incomplete search. |

Suggested neutral working title: **False-negative prioritization using view disagreement and between-model variation: a fixed-ensemble evaluation on RSNA mammography.**

## Preparation progress

One of the five previously listed preparation blocks is complete at the bounded scope above. **Four remain; zero main experiment steps remain.**

| Remaining block | Completion criterion |
|---|---|
| Student methods/results review | The student can explain the split, pooling, cutoff, denominators, bootstrap and conclusion in their own words. Assistant-written explanations alone do not demonstrate this. |
| Paper figures and tables | Create and verify figures/tables from the frozen aggregate outputs, with denominators and uncertainty clearly labeled. |
| Reproducible repository packaging | Make the final fixed-GMIC path the README entry point; document exact environments, seeds, commands and official input access. Preserve historical failed branches. |
| Public-content and attribution review | Inspect the concrete release contents, data restrictions, borrowed code/weights, licenses and AI-assistance disclosure before publication. |

Methods and Results drafting can proceed alongside these preparations. No additional GPU experiment is required to report the completed result. Revised scoring would be a separate future study requiring independent evaluation data; this opened holdout is not a fresh test set.
