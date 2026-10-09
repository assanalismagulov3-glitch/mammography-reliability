# Fixed-GMIC implementation, calibration and execution — 2026-10-01

**Completed:** implementation, calibration, all 18 local/Kaggle checks, historical parity, all five inference shards, full global integrity validation, the frozen outcome analysis and both bootstrap calculations. **Final result:** combined scoring flagged 2/39 misses versus 3/39 for cancer score alone at 415 breast selections each; capture difference −2.56 percentage points, primary 95% interval −13.79 to +8.11 points. Incremental benefit is not established. Zero main experiment steps remain. Read `FIXED_GMIC_FINAL_RESULTS.md`; dated sections below preserve earlier checkpoints.

## What the checks establish

All 18 local checks pass. Pooling, scoring and review selection accept label-free schemas; extra outcome/clinical columns are rejected. The checks cover repeated-view averaging, deterministic ties, equal global capacity, patient separation, site-preserving patient resampling, duplicate-weight equivalence, zero-false-negative cases, configuration/rule tampering and incompatible recovery attempts.

Using the exact completed development predictions, the new evaluator independently reproduced breast pooling, all six warning scores, the original tune classification statistics and all 24 method/budget comparisons. The historical raw cutoff remains 0.029317490197718, with eight tune false negatives. Those historical findings were not overwritten.

The first test pass exposed an unclear empty-input exception in pooling. The input validator was corrected, and the complete 18-check suite subsequently passed. This was a software repair with no change to the scientific rule.

These checks establish software behavior on fixtures and preserved development outputs. They do not establish generalization, diagnostic accuracy, or clinical benefit.

## All-development calibration

| Quantity | Result |
|---|---:|
| Development patients | 2,383 |
| Development breasts | 4,766 |
| Positive development breasts | 99 |
| Raw decision cutoff | 0.02242065602913494 |
| Corresponding calibrated cutoff | 0.006898700768467533 |
| Platt slope | 0.8264477443737436 |
| Platt intercept | −1.8495793534642264 |
| Positive breasts detected at the fitted cutoff | 90/99 = 90.9% |
| Calibration false negatives | 9 |
| AI-negative calibration breasts used for percentiles | 2,053 |
| Calibration specificity | 2,044/4,667 = 43.8% |

These are fitting diagnostics. The cutoff was chosen to attain at least 90% sensitivity in these same data. The 90.9% value is therefore not an independent performance estimate and does not predict held-out sensitivity. The combined warning score has not acquired evidence of benefit simply because calibration is complete.

## Preserved holdout and computational division

The original versioned patient and image identity tables were recovered through Kaggle's visible output interface. Complete row coverage was checked after avoiding browser-preview truncation. Canonical, sorted CSV digests make the identities independent of line-ending presentation. The original dataset checksum and development-role checksum matched the existing record.

The holdout is unchanged: 9,530 patients, 19,060 breasts and 43,787 standard-view images. Five computational shards follow the original fold IDs:

| Shard | Patients | Breasts | Images |
|---|---:|---:|---:|
| 0 | 1,906 | 3,812 | 8,793 |
| 1 | 1,906 | 3,812 | 8,765 |
| 2 | 1,906 | 3,812 | 8,735 |
| 3 | 1,906 | 3,812 | 8,735 |
| 4 | 1,906 | 3,812 | 8,759 |

This is one holdout evaluation. The shards are computational divisions, not five independent model experiments. Outcome evaluation requires all five to pass coverage and identity checks; review slots are allocated globally afterward.

## Freeze and runtime requirements

The local freeze was created at **2026-09-28 16:38:35 UTC**. It binds 32 source/configuration/protocol files and five calibration/cohort artifacts. The original trained-CNN gate and model package remain unchanged. Source or rule changes invalidate the new gate rather than silently changing the experiment.

Every GPU job additionally requires a successful Kaggle CPU verification record matching this freeze, exact mounted cohort identities, verified released model files and reproduction of the original development image within 1e-6. No held-out pixels are scored before those runtime checks pass. Prediction recovery accepts only a complete, matching image prefix with the same recorded environment and model identity.

The final evaluator computes the fixed primary comparison and both prescribed patient-bootstrap intervals. It has been exercised on synthetic cases and historical development data, not on the final cohort. All five Kaggle GPU launchers have passed the runtime checks; the first four have completed and the fifth is running. Startup and completion logs do not establish all-five-shard integrity or final performance; the global evaluator must independently verify every completed output.

## Kaggle verification and current restart point — 2026-09-30

The September 28 automatic approval rejection is retained as history in the original block record. The user then continued after the explicit request to transmit the prepared private notebooks and embedded development calibration data to the existing private Kaggle account. The upload succeeded. No frozen rule or source file changed.

Private notebook 13 version 1 (353817243) completed CPU verification in 31.7 seconds: all 18 tests passed; breast pooling, all six scores, historical classification and all 24 review comparisons reproduced; cohort identity matched; the frozen calibration remained unchanged. Its verification record is `fixed_gmic_cpu_verification_passed` with freeze digest `6cdc4c4f976c54b751eb15b85c9a1d1fa1db6877a554c6cb4ac468963c35b0ef`.

| Shard | Kaggle notebook | State at this checkpoint |
|---|---|---|
| 0 | 14_rsna_fixed_gmic_shard_0, v1 (353818900) | Completed: 8,793 images / 3,812 breasts; 16,319.7 seconds |
| 1 | 15_rsna_fixed_gmic_shard_1, v1 (353819520) | Completed: 8,765 images / 3,812 breasts; 16,837.8 seconds |
| 2 | 16_rsna_fixed_gmic_shard_2, v1 (354080590) | Completed: 8,735 images / 3,812 breasts; 14,637.7 seconds |
| 3 | 17_rsna_fixed_gmic_shard_3, v1 (354080654) | Completed: 8,735 images / 3,812 breasts; 15,066.4 seconds |
| 4 | 18_rsna_fixed_gmic_shard_4, v1 (354143213) | Running; gates passed; at least 200 / 8,759 images reported |

All four completed runs have matching freeze identities, expected image/breast counts and `outcomes_evaluated: false`. They cover 35,028 images (80.0% of the holdout), 7,624 patients and 15,248 breasts. These are completion records; full output hashes, pooling, coverage and cross-shard environment equality remain the final evaluator's responsibility. The final job, notebook 18 v1 (354143213), started at 12:25:03 UTC on September 30 and passed the original startup gates with reference error 0.0. No inference draft remains to launch. The previous output-download timeout is preserved; completed outputs remain in their private Kaggle versions. Submitted runs continue independently of the conversation.

Four main steps remain to final research results: finish inference; verify all-five-shard coverage and integrity; calculate the frozen global comparisons and prescribed intervals; interpret and document the findings. Paper writing, mentoring and public release follow afterward. No final cancer-outcome comparison, density effect, clinical benefit or novelty conclusion has been established.

The private checkpoint archive retains the rule, original identities and prepared notebooks. Embedded-data notebooks and patient-level artifacts remain private. The public-source project contains no new patient-level predictions. The assistant's implementation contribution remains documented for disclosure and student mentoring.

## Continuation while inference runs — 2026-09-29, requested at 14:51 UTC+05:00

Both submitted jobs remain running without a failure shown in their observed logs. Each has reported at least 2,700 processed images. No GPU slot is available, so notebooks 16–18 remain saved drafts and no duplicate or extra inference run was started. The full visible progress records are retained in `validation/kaggle_fixed_gmic_progress_20260929_1451.json`.

Prepared `rsna_fixed_gmic_final_analysis.ipynb` locally for CPU execution after all five shards complete. It embeds the exact setup from the successful notebook-13 verification, including all 32 frozen source/configuration/protocol files and all five rule/cohort artifacts plus the freeze. Syntax compilation, required notebook-structure checks and literal embedded-file identity checks pass. It uses the same notebook builder and metadata as the successfully imported CPU notebook. No scientific source, operating point, review rule or bootstrap setting was changed.

The launcher requires the successful CPU verification and all five completed shard outputs, then invokes the existing frozen global evaluator. That program independently verifies file hashes, image identities, re-pooled breast features, model/environment equality, complete coverage and no overlaps before joining outcome labels. Static launcher validation does not establish successful final execution. The notebook has not been uploaded or run, and all final comparison results remain pending.

## Continuation — 2026-09-30

Completion and startup records were saved in `validation/kaggle_fixed_gmic_progress_20260930.json`; exact version IDs, hashes, counts and next actions are in `validation/kaggle_fixed_gmic_final_status.json`. The active-events screenshot confirms two running jobs. The local freeze was reverified unchanged. The completed first pair took approximately 4.5–4.7 hours per job; this is a rough timing guide, not a guarantee for the new pair. No final cancer-outcome comparison was run.

## Afternoon continuation — 2026-09-30, requested at 17:20 UTC+05:00

Verified successful completion of notebooks 16 and 17, preserved their full visible logs and completion digests, and launched the prepared final shard without changing source or calibration. The original freeze was reverified intact. Four main research steps remain: finish the last inference run; verify all outputs; compute frozen comparisons and intervals; interpret and document findings.

Attempted to prepare the CPU analysis in private draft 19. Automatic approval review blocked the upload because of embedded patient identifiers, cancer labels and calibration data. Prior-conversation retrieval recovered the original specific approval request and the user's continuation, but review still rejected the Import action because the recovered permission was tool/assistant context. No further transmission route was attempted. The import dialog was cancelled and the copied CPU-check notebook renamed `19_rsna_final_analysis_pending_import`. It still contains copied notebook-13 source and inherited inputs, is not the final-analysis program, and has not executed. Explicit approval is requested before the prepared analysis file is imported. The exact local notebook remains unchanged and available for review.

See `validation/kaggle_fixed_gmic_second_pair_completed_20260930.json`, `validation/kaggle_fixed_gmic_final_launch_20260930.json`, and `validation/kaggle_fixed_gmic_analysis_import_blocked_20260930.json`. The final running job is unaffected by the import block. No final cancer-outcome analysis has been performed.


## Global analysis launched — 2026-10-01

The fifth inference job completed successfully in 15,096.9 seconds, with the expected 8,759 images and 3,812 breasts and no outcome analysis. All five jobs now cover the complete original holdout. The user's continuation followed the explicit private-upload and final-analysis authorization request; the prepared notebook imported successfully and replaced the pending copied source. The notebook SHA-256 remains `769017c9d0dc8a1c2eaa414cdaff5bb7bd720b4cbde06a407e92370e933fc9aa`.

Private notebook 19 version 1 (354389561) was submitted on CPU at 11:21:07 UTC, with the pinned original environment and official RSNA data plus completed notebook outputs 13–18. The log confirms all five shards are present. Full file/pooling/coverage/model checks precede the outcome boundary in the unchanged evaluator. Final results and both 2,000-draw bootstrap intervals remain pending at this launch checkpoint. The old September 30 block record remains preserved; it is no longer an active blocker.


## Final analysis completed and preserved — 2026-10-01

Private notebook 19 v1 (354389561) completed in 752.9 seconds. The original evaluator passed all five shard integrity, pooling, coverage, nonoverlap and environment/model checks before outcome labels entered. The complete holdout contained 393 positive breasts; the frozen classifier detected 354 and missed 39 (90.08% sensitivity; 44.33% specificity). All methods used the same 8,315 AI-negative breasts and global budget.

The primary combined-versus-cancer-score comparison was 2 versus 3 flagged misses among 415 selected breasts, with a capture difference of −2.56 percentage points. The fixed-calibration 95% interval was [−13.79, +8.11] points; calibration-inclusive interval [−14.29, +8.00] points. All 2,000 draws per mode were defined and numerically successful. The result does not establish incremental benefit. At the descriptive 20% budget, combined scoring flagged 13/39 misses versus 18/39 for cancer score alone; no secondary comparison replaces the primary endpoint.

The full 164,862-character result JSON and all 4,000 bootstrap rows were extracted from the complete raw browser documents, avoiding Kaggle's truncated preview and the browser tool's single-return limit. Saved draws independently reproduce both intervals. The bulk output download returned no archive; complete patient-level output files remain privately preserved in their original Kaggle versions. No source, threshold, weight, model or budget changed after outcomes. The report and exact extraction/provenance record are saved alongside the final status.
