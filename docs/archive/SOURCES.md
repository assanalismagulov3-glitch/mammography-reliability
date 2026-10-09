# Implementation sources checked on 2026-09-09

- [RSNA dataset paper](https://pubs.rsna.org/doi/10.1148/ryai.250375): dataset context and limitations. Actual release contents must be established by the audit.
- [Official RSNA access and license](https://registry.opendata.aws/rsna-screening-mammography-breast-cancer-detection/): access conditions and data citation.
- [pydicom pixel data guide](https://pydicom.github.io/pydicom/stable/guides/user/working_with_pixel_data.html), [modality transform](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_modality_lut.html), [VOI transform](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_voi_lut.html): DICOM implementation references.
- [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): model and ImageNet weights. This project preserves the full mammogram and does not apply the documented natural-image center crop.
- [Scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html): probability calibration background.
- [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html): used on a unique-patient table, not on image rows.
- [scikit-learn cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) and [nested versus non-nested selection](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): why global model choices must be separated from final evaluation. Version 0.2 uses an independent pilot cohort followed by frozen cross-validation; it does not implement a full nested hyperparameter search.
- [Lakshminarayanan et al., Deep Ensembles](https://arxiv.org/abs/1612.01474): established foundation for using independently trained members to assess predictive variation. This project is not a replication of every method in that paper.

This source list supports implementation choices. It is not a systematic search for a research gap.

## Targeted prior-art check, 2026-09-09

- [Verboom et al., 2025, AI Should Read Mammograms Only When Confident](https://pubs.rsna.org/doi/10.1148/radiol.242594): reviewed the abstract and main methods describing eight uncertainty metrics, a region-based model, and retrospective hybrid AI/radiologist reading. Uncertainty-driven mammography triage is already studied. Our potential contribution must be narrower: incremental false-negative prioritization from breast-view disagreement and ensemble variation, tested against the cancer score at the same review budget. That possible distinction still requires a wider search.
- [Woo et al., 2025, Invasive Breast Cancers Missed by AI Screening of Mammograms](https://pubs.rsna.org/doi/10.1148/radiol.242408): identified as directly relevant prior work. Full methods and endpoint extraction remain pending; do not infer its results from the title.
- [Vu et al., M&M, 2023](https://arxiv.org/abs/2308.06420): the authors describe a multi-view, multi-instance sparse detection method targeting false positives, with five-dataset validation. Multi-view mammography modeling is established. Its relationship to post hoc false-negative ranking needs detailed comparison before any gap claim.

This is a targeted design check, not a completed systematic or comprehensive literature review. No novelty conclusion has been reached.

## Expanded working review

See `LITERATURE_WORKING_REVIEW.md` for the retained primary-source evidence table, exact access limits, implications for the proposed comparison, and remaining gap-review work. It is not a completed systematic review or a novelty conclusion.

## Sampling diagnostic implementation, 2026-09-23

- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html): checked the positive-weight and mean-reduction definition. The balanced sampler's objective scaling is derived in `DEVELOPMENT_DECISIONS.md` and checked numerically, not claimed as a new method.
- [PyTorch mixed precision examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html): checked gradient accumulation, unscaling before clipping, and skipped updates when gradients are nonfinite. The new trial records these behaviors. Kaggle package versions are saved separately; the previous environment used PyTorch 2.10.0.

## Mammography transfer feasibility, 2026-09-24

- [GMIC paper, methods and classification results](https://arxiv.org/html/2002.07613v1): native-resolution global/local modeling, NYU training data and evaluation design. Relevant sections 2 and 3.1–3.3 were examined.
- [Author repository at the inspected commit](https://github.com/nyukat/GMIC/tree/594029efe6184bcca3a151fa1dd46c5fd620cb25): README, inference, preprocessing, architecture, Git tree, LICENSE and NOTICE inspected. Every file used by the preflight is pinned in `gmic_upstream_manifest.json`; runtime downloads require matching Git blob identities. The published sample predictions provide an engineering reference.
- [RSNA dataset methods](https://pubs.rsna.org/doi/10.1148/ryai.250375): source institutions are Emory and BreastScreen Victoria. This provenance comparison cannot prove the absence of individual patient overlap with an external model's training data.

Preflight settings and deviations are recorded in `DEVELOPMENT_DECISIONS.md`. This source review does not establish that GMIC transfers successfully to our cohort.

The GMIC reproducibility investigation additionally inspected [initial author revision](https://github.com/nyukat/GMIC/tree/febe2431dbe6efd75bcb37ddea85dc48c0df0b07) and repository history. Historical fixture identities are in `gmic_historical_manifest.json`. Current original-CLI parity is an observed software result; the historical-reference check is recorded separately and must not be inferred from source inspection alone.


## Development resampling context — 2026-09-28

- [Rutter, 2000, Bootstrap estimation of diagnostic accuracy with patient-clustered data](https://pubmed.ncbi.nlm.nih.gov/10845400/), DOI 10.1016/S1076-6332(00)80381-5: author abstract consulted; full article was not accessed. Supports accounting for correlations within a patient when estimating uncertainty in diagnostic accuracy. It does not validate this project's particular warning score or chosen bootstrap percentile intervals.
- [Cawley and Talbot, 2010, On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/beta/papers/v11/cawley10a.html): author abstract consulted. Supports keeping adaptive selection separate from final performance evaluation. No claim of a full-text methods review is made here.

The implemented plan and conditional limitations are recorded separately in `THRESHOLD_STABILITY_PLAN.md`. These methodological references do not complete the substantive mammography gap review.


## Fixed-model design evidence update — 2026-09-28

The expanded primary-source comparison, examined sections and access limits are recorded in `RELATED_WORK_DECISION_2026_09_28.md` and `literature_search_log_2026_09_28.json`. Earlier dated entries above remain historical.

- [CEI-Net, official MICCAI 2026 accepted paper](https://papers.miccai.org/miccai-2026/1158-Paper2986.html) and its linked PDF: main paper read, including figures/tables.
- [M&M](https://arxiv.org/abs/2308.06420): main full text now accessed; supersedes the earlier abstract-only record.
- [Lu et al.](https://arxiv.org/html/2107.02716v2): main methods/results now accessed.
- [Jaeger et al., ICLR 2023](https://arxiv.org/html/2211.15259v2): failure-detection evaluation recommendations examined.
- Verboom, Woo, the RSNA dataset publication and GMIC model-selection methods were rechecked at their previously recorded primary URLs.
- [SafeRecall](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7324281) and [Gudhe et al.](https://doi.org/10.1109/ACCESS.2024.3399204) remain limited-access follow-up leads; they are not treated as fully reviewed methods.

No complete systematic search, novelty determination or clinical validation is claimed.


## Post-outcome literature access update — 2026-10-01

See `RELATED_WORK_ADDENDUM_2026_10_01.md` for the updated comparison and `literature_search_log_2026_10_01.json` for the retained query/access record. The September 28 frozen decision remains unchanged.

- Verboom supplemental PDF pages 1–4 were visually examined after intermittent retrieval failures; the earlier total-access failure is superseded at that scope.
- Gudhe's published-paper transcription was examined in selected methods/results sections, with bibliographic corroboration from the authors' research-group page. The earlier bibliographic-only record is superseded.
- SafeRecall and an additional 2026 publisher-indexed lead remain access-limited. Their numerical claims are not adopted.

The targeted check supports cautious empirical framing. It does not establish novelty or constitute an exhaustive literature review. No frozen scientific rule changed.
