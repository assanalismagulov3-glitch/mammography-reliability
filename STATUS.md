# Execution status — 2026-10-05

## Current state

- **The frozen fixed-GMIC experiment is complete and documented. Zero main experiment steps remain.** Private notebook 19 version 1 (354389561) completed successfully in 752.9 seconds. No further inference or analysis run is needed.
- **Primary finding:** combined scoring flagged 2/39 AI-missed cancer-positive breasts, versus 3/39 for cancer score alone, at equal 5% global capacity (415 breasts each). Capture difference: −2.56 percentage points; primary 95% patient-bootstrap interval: −13.79 to +8.11 points. Added benefit is not established. See `FIXED_GMIC_FINAL_RESULTS.md`.
- **Both uncertainty calculations completed:** 2,000 fixed-calibration draws and 2,000 calibration-refit draws, with no numerical failures or undefined zero-miss draws. The calibration-refit interval is −14.29 to +8.00 percentage points. All 4,000 saved draw rows reproduce the two reported intervals.
- **Full coverage and integrity passed before the label join:** 43,787 images, 9,530 holdout patients, 19,060 breasts and 393 cancer-positive breasts; all five shard digests, image identities, independently pooled features, nonoverlap and model/environment consistency were verified by the unchanged evaluator.
- At the unchanged raw cutoff 0.02242065602913494, the classifier detected 354/393 positive breasts (90.08% sensitivity), missed 39, and had 44.33% specificity. Study-sample ROC-AUC was 0.77245 and average precision 0.12907. These are retrospective breast-level results; no human-reader recovery or clinical benefit was measured.
- The report retains all six methods at 1%, 5%, 10% and 20% budgets, exact subgroup denominators, both intervals, probability-quality diagnostics and limitations. The complete curve, calibration bins and subgroup records remain in the aggregate JSON.
- The original 32 frozen scientific/configuration/protocol files and five calibration/cohort artifacts remain unchanged. The holdout is now opened and must not be reused as untouched evaluation data for a revised score.
- Original complete outputs remain in their private Kaggle versions. Full aggregate JSON and all 4,000 bootstrap rows were additionally saved and checked. `validation/fixed_gmic_final_result_verification_20261001.json` records extraction completeness, identities and interval reproduction.
- The October 1 private Kaggle upload/import succeeded after the user's explicit continuation; the September 30 approval block is historical and resolved. The audited source-and-aggregate package was uploaded to the owner's private GitHub repository on October 5. No manuscript or public release has occurred.

## Practical release completed — 2026-10-05

The user requested completion of the practical package and GitHub upload, with detailed explanations deferred until before paper writing. Teaching or quizzes are not a release prerequisite. No manuscript is being drafted in this step.

| Practical step | State | Evidence |
|---|---|---|
| Figures and tables | Complete | Three figures in PDF/SVG/PNG; six table sections; both intervals reproduced from all 4,000 saved draws. |
| Reproducible source and release audit | Complete | Audited source-and-aggregate archive; all 32 frozen source hashes and five private artifact hashes unchanged; 18 tests pass; reproduction succeeds from the staged release. |
| GitHub upload and remote verification | Complete | All 126 original release files downloaded from GitHub match the audited stage byte-for-byte; all 125 manifest digests and 32 frozen scientific digests pass. |

The main experiment and practical release have **zero steps remaining**. Detailed teaching comes next, before paper writing. The literature review is complete for cautious empirical framing, with its documented access limits and no established novelty claim.

Repository: https://github.com/assanalismagulov3-glitch/mammography-reliability (private). The original package was independently checked at commit `1bc39f08896bdb6fbbb8dc001247be3b2475f2a0`; `validation/github_upload_verification_20261005.json` records the downloaded archive digest and exact verification scope. Subsequent hosting-status documentation updates preserve all frozen files and original aggregate results. Hosted CI and a clean-room patient-level rerun are not claimed.

## Historical implementation and execution record

The entries below describe earlier checkpoints. Their pending-work statements are historical; the current state and practical-release table above supersede them.

- Continued from the student's existing RSNA audit; preserved the reported aggregates with their provenance.
- Implemented metadata validation, density auditing, patient partitions, pixel decoding/auditing, a breast-level CNN, ensemble inference, calibration, false-negative review comparisons, subgroup accounting, and paired patient bootstrap.
- Prepared an executable Kaggle notebook and a working experiment protocol.
- Ran software checks on the local CPU, including CNN training on artificial DICOMs, checkpoint recovery after a simulated interruption, and cache-integrity checks. Exact test results are in `validation/`.
- Connected to the student's Kaggle account and independently reran the original metadata audit successfully.
- Confirmed 23,826 breasts, 492 positive breasts, 486 positive patients, and no breast-level density conflicts.
- Confirmed usable density for 11,618 breasts at site 1; 18 site-1 breasts and all 12,190 site-2 breasts lack density. Dense categories C/D contain 120 positive breasts at site 1.
- Decoded and visually inspected 20 fit-partition images covering both sites and all machines represented in that partition. No obvious blank images, grayscale inversion failure, or unintended crop was observed. This was technical inspection, not diagnostic review.
- Measured an approximate 27.70 GiB 16-bit PNG cache against a 19.50 GiB notebook output limit. Implemented two private cache parts without reducing resolution or intensity precision. The storage-part assembly test passed, including remounting and global duplicate detection.
- Saved the audit privately as version 1 of `02_rsna_reliability_workflow` and submitted the two full preprocessing jobs to Kaggle.
- Verified two Tesla T4 GPUs with approximately 14.56 GiB each and 30 hours/week of available account quota at the initial check. Artificial-tensor forward/backward benchmarks passed at 1024 × 768 with 2, 4, and 8 images per breast; the largest case used approximately 1.65 GiB allocated GPU memory.
- Implemented version 0.2 before any cancer-model outcomes: reserve 20% of patients for development, keep them out of all five final test folds, and reuse them only for fitting final models after global settings are frozen.
- Added a prospective evaluation gate checking the configuration, source, patient assignments, and recorded review of completed pilot trials. Final-fold training and evaluation are blocked before this review.
- Passed four new isolation/cache/gate tests. The three existing model and training-regression tests also passed after the training instrumentation changes. These are artificial software fixtures.
- Submitted private notebook `05_rsna_pilot_preparation`, version 1, on CPU. Its real-data code reached higher-resolution preparation for 10,881 images belonging to 2,383 reserved pilot patients. It prepares 1536 × 1152 arrays directly from original DICOMs.
- Confirmed the new cohort totals from the versioned real-data log: pilot = 2,383 patients / 4,766 breasts / 99 positive breasts; final evaluation cohort = 9,530 patients / 19,060 breasts / 393 positive breasts. Pilot fitting has 73 positive breasts; pilot tuning has 26. Exact site counts are saved in `validation/kaggle_study_design.json`.
- Recorded a targeted prior-art check. Closely related uncertainty-driven mammography triage already exists; no research-gap or novelty conclusion has been reached.

- Verified all three preprocessing commits completed successfully: 27,240 base images in part 0 (20,071.8 seconds), 27,428 in part 1 (18,388.3 seconds), and 10,881 higher-resolution pilot images (7,699.6 seconds).
- Created a standalone two-resolution pilot launcher without changing the audited model package. It pins the three verified cache manifests, checks combined image coverage and duplicates, verifies the high-resolution pilot against the original decoded-pixel hashes and patient roles, then runs seed 17 at both resolutions. The notebook schema and combined Python code were validated locally.
- Verified private notebook `06_rsna_development_pilots`, version 1 (348655595), completed successfully in 4,394.7 seconds. Combined validation confirmed full coverage, zero exact cross-patient pixel duplicates, and high-resolution pilot identity. No final test predictions were opened.
- Recorded all 11 development epochs and both completed model summaries from the versioned visible log. The 1024 × 768 pilot selected epoch 3 by average precision (AP 0.02316; ROC-AUC 0.4890). The 1536 × 1152 pilot selected epoch 2 (AP 0.02528; ROC-AUC 0.5525). The tuning set has 26 positive breasts among 1,192. These weak, selected development results are not final research estimates.
- Prepared a separate diagnostic using only reserved pilot patients. It checks inputs and initial floating-point behavior, attempts to memorize 16 fitting breasts, fits a frozen-feature linear probe, and conditionally runs one stability trial. Local notebook schema, Python syntax, embedded source identity, and deterministic fitting-subset isolation checks passed. GPU behavior is verified on Kaggle, not locally.

## Work after practical completion

- Review the code, frozen design and final findings with the student so they can explain and defend the work.
- Use the October 1 literature addendum for empirical manuscript framing. Any stronger novelty claim would require further evidence and resolution of relevant access limits.
- After the methods/results discussion, prepare the manuscript while retaining the inconclusive primary result and unfavorable secondary comparisons. Any eventual change from private to public repository visibility remains a separate decision.

## Earlier development checkpoints (historical)

Earlier local software validation used CPU PyTorch. The current local runtime has no PyTorch or CUDA; the real-data GPU work runs in private Kaggle notebooks. Exact historical audit packages and engineering measurements are recorded in `validation/kaggle_technical_preflight.json`; completed pilot environments are in `validation/kaggle_development_status.json`.

The local validation images, labels, predictions, and training are artificial fixtures. Kaggle audits and development pilots use real data. Weak pilot performance is available; final false-negative detection performance, density effects, and novelty remain unestablished.

All three preprocessing commits and the initial GPU pilot commit have completed. Kaggle can continue submitted commits when the conversation is inactive; the assistant does not automatically monitor or launch subsequent work between messages.

The completed preprocessing results and exact manifest identities are recorded in `validation/kaggle_preprocessing_status.json`. The completed GPU commit is [06_rsna_development_pilots](https://www.kaggle.com/code/aseqwss/06-rsna-development-pilots/log?scriptVersionId=348655595). It uses `rsna_development_pilots.ipynb`, generated by `build_development_notebook.py`. The original working notebook `02_rsna_reliability_workflow` preserves the earlier audit and technical cells.

Private notebook `07_rsna_development_diagnostics`, version 1 (352083246), completed on GPU T4 x2. Its code includes the conditional stability trial and leaves final evaluation locked. Completed observations are recorded in `validation/kaggle_diagnostics_status.json`.

The diagnostic run completed successfully in 4,184.5 seconds. It revalidated all 54,668 base images. Inputs in the small diagnostic sample were finite and nonconstant; initial FP32 and mixed-precision logits differed by at most 0.000575. Tiny-set memorization passed (training AUC 1.0), but the stability trial still had weak tuning performance: selected epoch 12, AP 0.03094 and AUC 0.56040. The frozen-feature probe had tuning AP 0.04006 and AUC 0.64890, with fitting AP 0.9523. These are development observations, not final performance estimates.

Prepared `rsna_development_balanced.ipynb` to reproduce the selected reference checkpoint and test equal-class training updates. The new sampler passed three local checks covering update composition, determinism, invalid inputs, and equivalent expected loss/gradients. Notebook schema, Python syntax, and all ten embedded original modules were verified. Actual GPU execution is checked on Kaggle. The final pipeline remains unchanged and locked pending a useful baseline and protocol review.

Private notebook `08_rsna_balanced_training`, version 1 (352110149), was submitted on GPU T4 x2. Its draft session remains off. Official RSNA data and the verified outputs of notebooks 03, 04, and 07 are attached. Submission and full training completion are distinct; current observations are recorded in `validation/kaggle_balanced_status.json`.

Notebook 08 completed successfully in 2,965.3 seconds with 209.19 MB of output. The selected epoch was 5, with 9 epochs run. Fitting AP/AUC were 0.50733/0.98158; tuning AP/AUC were 0.03785/0.61294. The reference checkpoint reproduced its earlier tuning AP exactly. Two AMP updates were skipped across all epochs. Full aggregate records are preserved. Improved fitting did not establish useful generalization.

Prepared and locally checked `rsna_gmic_preflight.ipynb`: original source and five fixed mammography models from a pinned NYU commit; reference reproduction before fit-only technical imaging checks. Four synthetic adapter tests passed, notebook schema/syntax passed, and all ten original package modules remain unchanged. The new notebook does not evaluate RSNA tuning or final patients, does not claim a concept bottleneck, and requires real-image visual review before a full-cohort transfer experiment. Its actual execution status is recorded separately in `validation/kaggle_gmic_status.json`.

The first GMIC preflight attempt stopped during strict checkpoint loading because the historical checkpoint contains `shared_rep_filter.weight`, unused by the released network. All 48 downloaded files matched the pinned Git blobs. The adapter was narrowed to permit that specific legacy tensor while retaining strict checks on active parameters; the added regression test passed. No reference or RSNA predictions were produced by version 1.

Notebook 09 version 4 (352355871) completed in 176.4 seconds. Historical author-fixture predictions matched the original table to a maximum absolute error of 0.00004945; current-pipeline processed arrays matched exactly, with scores within four-decimal rounding. The mismatch against the unchanged README table for newer sample files remains recorded. This establishes engineering compatibility, not RSNA accuracy.

The subsequent real-image visual check found a substantial crop failure: sample 11 (site 1, CC) produced a mostly background fixed window. The original NYU crop/window adaptation was therefore rejected before any RSNA cohort accuracy was calculated. A full-image, aspect-preserving downscale-and-pad adaptation was implemented and tested on the same 20 fitting images. Notebook 10 version 1 (352360012) stopped during thumbnail generation because of Pillow uint16-mode handling; a display-only float conversion fixed that failure. Version 2 (352360882) completed numeric checks in 99.2 seconds. All 20 image pairs passed technical visual review; the previous sample-11 exclusion is corrected. Fine-detail preservation and cohort accuracy are still unestablished. Private notebook 11 version 1 stopped before inference at the reproduction probe; the corrected version 2 has been submitted on GPU T4 x2. Startup verification and its current progress are recorded in `validation/kaggle_gmic_pilot_status.json`; full-cohort results are pending. No final test result has been opened.

The current reserved-pilot job is notebook `11_rsna_gmic_reserved_pilot`, version 2 (352364754). Version 1 (352363590) is retained as a failed startup attempt. It attaches notebook 09 version 4 and notebook 10 version 2 plus the official dataset. It will score 10,881 images with all five fixed models, average repeated images within view, weight CC/MLO equally, and report descriptive pilot AP/AUC. No RSNA fitting, calibration, operating threshold, false-negative endpoint, or final patient inference is performed. The rough runtime extrapolation from the 20-image audit is approximately 4.5 hours; actual timing can differ.

Version 2's mounted-data, role, source and model checks passed. The five audited sample-11 predictions reproduced with zero absolute error before the cohort loop. The run is in progress; no cohort AP/AUC is yet available. Verified run page: https://www.kaggle.com/code/aseqwss/11-rsna-gmic-reserved-pilot/log?scriptVersionId=352364754

At the latest recorded check, version 2 had processed 100 / 10,881 images (178.24 seconds from script startup), with no errors shown. This is progress evidence, not completed cohort inference. Kaggle runs the submitted commit independently; the assistant does not monitor it automatically between messages.

## Completed CPU reproduction and orientation review — 2026-09-28

Private notebook 12 version 2 completed in 46.7 seconds. Six confusion-matrix groups and 24 review comparisons exactly matched the local analysis; small numerical calibration differences did not alter counts. The three flagged fit-image original/processed pairs showed no obvious gross inversion, orientation failure or boundary exclusion at contact-sheet scale. No images or predictions were changed. The first failed decoder attempt remains in `validation/kaggle_gmic_analysis_status.json`.

At that checkpoint the next step was operating-point stability. It has now completed; see the current state above and `THRESHOLD_STABILITY_RESULTS.md`. Next: complete the substantive related-work comparison and prospectively revise the fixed-model final evaluation. No GPU job is currently required or represented as running by this record. The assistant does not monitor or start new work between messages.
