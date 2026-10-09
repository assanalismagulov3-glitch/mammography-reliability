# Development decision record — 2026-09-24

## Evidence prompting the next experiment

Private Kaggle notebook 06, version 348655595, completed both initial pilots in 4,394.7 seconds. Both used seed 17 and identical reserved development roles: 3,574 fitting breasts with 73 positives; 1,192 tuning breasts with 26 positives. Checkpoints were selected by tuning average precision.

| Trial | Selected epoch | Tuning AP | Tuning ROC-AUC | Epochs run |
|---|---:|---:|---:|---:|
| 1024 × 768, original optimization | 3 | 0.0231565 | 0.489016 | 6 |
| 1536 × 1152, original optimization | 2 | 0.0252779 | 0.552547 | 5 |

The tuning positive fraction is 26/1192 = 0.0218121. AP is close to that reference, and ranking is weak. The apparent difference between resolutions does not establish a useful resolution improvement. Only one seed and 26 tuning positives were used; both checkpoint selection and subsequent development adapt to these patients.

Weighted fitting losses remained high. Code inspection found no obvious label-join, pooling, scaling, or accumulation error. Optimization instability under severe imbalance is a hypothesis, not a confirmed cause. Small lesions, full-image pooling, limited fitting positives, and preprocessing could also limit learning.

## Declared diagnostic sequence

1. Revalidate the two pinned image-cache manifests and the existing patient roles. Preserve the model package used in notebook 06.
2. Select eight positive and eight negative breasts from 16 distinct fitting patients, deterministically with seed 17. No tuning or final-evaluation patient enters this small set. Check finite, nonconstant inputs and initial FP32 versus mixed-precision logits.
3. Attempt memorization with augmentation off, unweighted binary cross-entropy, FP32 computation, AdamW at learning rate 0.00001, zero weight decay, and 100 optimizer steps. Each step accumulates all 16 breasts. Log every ten steps. The engineering gate requires final fitting loss at most half the initial loss and fitting ROC-AUC at least 0.95. This arbitrary, declared gate checks basic learning capacity; it is not a medical performance target.
4. Extract fixed ImageNet ResNet-18 features for the full reserved cohort, with equal within-view averaging and equal CC/MLO averaging. Fit feature scaling and a class-balanced logistic regression only on fitting patients. Use C=1, LBFGS, and at most 1,000 iterations. Report fitting and tuning results and convergence iterations. No hyperparameter search is planned for this probe.
5. If the memorization gate passes, run `stable_1024_lr1e5_acc32_v1`: learning rate 0.00001, accumulation over 32 breasts, a maximum of 12 epochs, and early-stopping patience four. Other package settings are unchanged. This is a joint adjustment, not an ablation that can isolate the effect of one setting.

All settings and identities are saved before diagnostic outcomes. Failing the small-set gate prevents the larger stability trial. Passing it does not show generalization. All private patient lists, features, and predictions stay in Kaggle.

## Decision after results

Inspect the numeric checks, fitting behavior, linear probe, and complete learning curves. If the baseline remains weak, investigate architecture or mammography-specific transfer learning before final experiments. If optimization improves, a fair resolution comparison requires the same improved settings at both resolutions. Do not choose a resolution from the failed original comparison alone.

No final method freeze is authorized by the existing weak results. The held-out evaluation cohort remains unavailable for model development. Keep negative outcomes and every adaptive trial in the research record. Avoid interpreting a successful software run as a successful research hypothesis.

## Completed diagnostic and next sampling trial — 2026-09-23

Notebook 07, version 352083246, completed in 4,184.5 seconds. The 16-breast memorization diagnostic passed: final fitting BCE 0.000316, AUC 1.0. This rules out a gross inability to learn that tiny set; it does not validate preprocessing clinically or show generalization. The frozen-feature probe had fitting AP 0.9523 and tuning AP 0.04006 / AUC 0.64890, a substantial fitting–tuning gap. The stability trial selected epoch 12 with tuning AP 0.03094 / AUC 0.56040. Its scaled fitting loss remained 1.94. All 12 epochs and all 11 memorization measurements are preserved in the diagnostic status file. No useful base model is established yet.

The next trial tests a specific optimization hypothesis: rare, heavily weighted positive examples may produce unstable updates when accumulated with naturally sampled negatives. This mechanism has not been measured in the old trials and is not established by the loss curves. Before changing the architecture, test balanced accumulation with one declared seed and an approximately matched update budget.

- Preserve the original ResNet-18, 1024 × 768 cache, seed 17, learning rate 0.00001, weight decay 0.0001, frozen BN statistics, horizontal flips, mixed precision, clipping norm 5, scheduler, 12-epoch maximum, and patience four.
- For each update, sample 16 positive and 16 negative fitting breasts uniformly within each class, with replacement. Each epoch has 112 updates and 3,584 breast draws, compared with 3,574 breast draws in the earlier natural-sampling trial. Repeated draws are not additional independent patients. Save their indices privately.
- Use unweighted BCE multiplied by `2*n_negative/n_fit`. If the within-class mean losses are L+ and L−, both this balanced objective in expectation and the previous naturally sampled weighted objective equal `(n_negative/n_fit)*(L+ + L−)`. A numerical test checks both the objective and its expected logit gradient. Equal expectation does not imply equal optimizer trajectories, gradient variance, clipping, or overfitting.
- Reevaluate the verified selected checkpoint from notebook 07 on the original fit and tune groups before training; require reproduction of its AP. This supplies previously missing fitting performance and class-conditional logit summaries.
- Record class-specific fitting losses, preclipping gradient norms, clipping counts, nonfinite-gradient counts, skipped AMP updates, scale values, unique breasts seen, and complete tuning curves. Evaluate the selected new checkpoint on all fit and tune breasts, including site summaries. Tuning prevalence is unchanged.
- Keep the implementation in standalone scripts, preserving the package fingerprint of completed experiments. It cannot be silently substituted into the final pipeline. If adopted, integrate and test it in a new package version before freezing the final protocol.

This is adaptive development using only 26 tuning positives; it cannot prove superiority or identify the sole cause of weak learning. A successful sampling trial still needs a fair resolution check and seed assessment. If it remains weak, use the fitting/tuning gap and numeric diagnostics to decide between stronger mammography-specific transfer or a revised architecture. Do not expand a blind hyperparameter search or open final test results to rescue development.

Implementation references: [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html) and [mixed precision examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html), checked 2026-09-23. The executed Kaggle environment is recorded separately (previously PyTorch 2.10.0); the reference pages describe the weighting and accumulation semantics used here.

## Precision of the eventual failure-detection comparison

The final evaluation cohort has 393 positive breasts. If eventual held-out sensitivity were 90%, there would be roughly 39 false-negative breasts; this is an illustrative calculation, not an observed or guaranteed outcome. With 40 false negatives, flagging one additional miss changes the capture fraction by 2.5 percentage points. Density subgroup denominators would be smaller still. The study may therefore yield wide intervals or an inconclusive comparison even if base-model learning improves. Report the paired patient-bootstrap interval and exact counts; do not promise statistical significance or strengthen the claim after seeing results.

## Balanced result and transfer preflight — 2026-09-24

Notebook 08, version 352110149, completed in 2,965.3 seconds. The reference checkpoint reproduced its tuning AP exactly. Balanced training selected epoch 5 and stopped after epoch 9. All nine epoch records and class/site summaries are retained in `validation/kaggle_balanced_status.json`.

| Model | Fitting AP | Fitting ROC-AUC | Tuning AP | Tuning ROC-AUC |
|---|---:|---:|---:|---:|
| Previous stability trial, reevaluated | 0.08277 | 0.75710 | 0.03094 | 0.56040 |
| Balanced updates, selected epoch 5 | 0.50733 | 0.98158 | 0.03785 | 0.61294 |

The fitting improvement is large, while tuning ranking is still weak. This indicates overfitting and limited transfer to separate development patients. It does not establish a statistically reliable improvement or prove that sampling explains all earlier failure. Only two optimizer updates were skipped by AMP across nine epochs; heavy clipping remained common. The frozen-feature probe from notebook 07 remains a useful diagnostic comparator (tuning AP 0.04006, AUC 0.64890), not a validated superior method.

**Next decision:** check the feasibility of externally pretrained mammography-specific features before another full CNN tuning run. The candidate is the authors' GMIC implementation: [paper](https://arxiv.org/html/2002.07613v1), [repository](https://github.com/nyukat/GMIC), inspected commit `594029efe6184bcca3a151fa1dd46c5fd620cb25`. The paper describes NYU data; the RSNA dataset paper identifies Emory and BreastScreen Victoria. Different reported institutions support separation of source datasets, but do not constitute a patient-identity overlap audit.

The declared private notebook 09 is deliberately an engineering preflight:

1. Download the pinned source, five released weights, and 16 public author example images. Verify all 48 files against the inspected Git blob identities; record SHA-256 and retain the original LICENSE/NOTICE. No upstream code edits are planned.
2. Strictly load all five checkpoints. Use FP32 inference with no training. Reproduce the README predictions for the default model 1 using the original crop, center, flip, window and normalization functions. The declared maximum absolute tolerance is 0.005 across both outputs of every example; failure stops RSNA inference. This is an engineering gate, not clinical validation.
3. Use only the unchanged reserved pilot fitting patients for a deterministic image sample: two per site/view/laterality and one per machine, deduplicated. Decode original DICOMs at native dimensions rather than enlarge the 1024 cache. Record presentation, pixel spacing, orientation, crop retention, finite outputs, runtime and memory. No RSNA accuracy is calculated.
4. RSNA lacks NYU's horizontal-flip metadata. The provisional adapter uses relative intensity mass in the outer fifths to propose chest-wall orientation. Synthetic mirror/gain checks verify implementation invariants only; actual reliability requires visual inspection. Record ambiguous orientation and crop retention, and save private native/window contact sheets. No silent image rejection or automated full-cohort continuation is allowed.
5. Review the real-image contact sheets and any reference discrepancy before running the complete reserved cohort. Retain every adaptation and failed gate. Original CNN code, partitions and final-evaluation lock remain unchanged.

This is not an exact paper replication: the released deterministic inference code uses one center, whereas the paper reports ten test-time augmentations. Nor is GMIC a concept bottleneck model. Its five published members were selected after the authors' hyperparameter search; they are not merely our proposed three equal-configuration random seeds. If adopted, revise and test the ensemble, pooling, calibration, resampling and final-freeze protocol explicitly before opening final outcomes. Do not describe the borrowed classifier or its saliency maps as our novel contribution or validated clinical concepts.

### Reference-load repair recorded before a second attempt

Notebook 09 version 1 (352351181) failed after 61.5 seconds at strict checkpoint loading. All 48 external files passed their Git blob checks. The only reported extra tensor was `shared_rep_filter.weight`; the pinned model source does not define or use this module. The authors' loader uses `strict=False`, so it ignores such extras. Our adapter now permits only that named unused tensor, rejects any other extra or missing active weight, and loads the remaining state strictly. A synthetic regression test checks that this exception cannot hide missing or unexpected active weights. No source-network computation, learned tensor, sample tolerance or reference prediction was changed. Version 1 produced no reference or RSNA predictions. Both attempts must remain in the record.

### Published-reference discrepancy and independent entry-point check

Version 2 (352352316) loaded all five checkpoints and evaluated all 16 public author examples. Four images from exam 0 closely matched, but the largest discrepancy was 0.212959, exceeding the unchanged 0.005 gate. No RSNA image inference followed. Version 3 adds a diagnostic only: run the original author crop, center and inference CLI scripts unchanged; compare every processed-array hash, center and output against the adapter and README. This isolates adapter behavior from reference/environment differences. It does not relax the gate or authorize RSNA inference.

### Provenance-based fixture correction, declared before version 4

The original current CLI produced identical processed-array hashes for all 16 images and matched the adapter to within 0.00004850, consistent with its four-decimal CSV output. It also disagreed with the README by up to 0.213. Thus the observed discrepancy is reproducible using the authors' released current entry points; it is not caused by our direct-array adapter. This does not yet prove its historical cause.

Git history shows that commit `7ad1d03` replaced `sample_data/cropped_images` and its metadata with new images and a new preprocessing pipeline in December 2020. The initial commit `febe2431dbe6efd75bcb37ddea85dc48c0df0b07` contains the already-windowed reference fixtures; its README table is identical to the current table and all five checkpoint blobs are unchanged. The original inference script loads those images without flipping or further cropping and standardizes them.

Version 4 pins these historical fixtures and adds a separate check against the same unchanged 0.005 reference tolerance. It also reruns the current original-CLI parity check, requiring identical processed-array hashes and at most 0.00015 score difference (allowing four-decimal rounding). Only both checks together allow the fit-only engineering sample. The failed current-fixture/README comparison stays recorded as failed. This is an explicitly revised compatibility gate based on source history and independent entry points, not a relaxed numeric tolerance or a change selected using RSNA performance. Visual image review remains required before cohort-level transfer evaluation.

## Full-field geometry repair, declared for notebook 10

Notebook 09 version 4 completed in 176.4 seconds. Historical reference predictions matched to within 0.00004946; current-CLI parity remained within 0.00004850. All five models loaded and the numeric checks ran on 20 reserved fitting images. This establishes the checked inference compatibility, not transfer accuracy.

The technical image audit rejected the original fixed-window preprocessing for RSNA: sample 11's displayed crop largely contains background and excludes the breast. Its window retains only 21.1% of nonzero source pixels; other large images also lose substantial image extent. Nonzero-pixel retention is a technical proxy, not a measure of lesion retention.

The next check preserves the complete original image extent. After the same provisional chest-wall orientation, images are downscaled only if necessary to fit 2944 × 1920, keeping aspect ratio; padding is on the right and either vertically centered (CC) or below (MLO). There is no tissue bounding-box selection or fixed-window cropping. Bilinear downsampling can still remove fine details; this is an explicit RSNA adaptation, not the original NYU preprocessing. Pixel spacing is unavailable in the inspected headers, so physical-scale matching cannot be assumed.

Notebook 10 reuses the verified weights and the exact same 20 fit images, calculates no cancer accuracy, and saves correctly proportioned native/full-field contact sheets. Three synthetic geometry tests passed, checking corner-region retention after downscaling, mirrored-view equivalence, and padding without enlargement. Four adapter regression checks also passed. Review actual images before submitting full reserved-cohort inference.

Notebook 10 version 1 (352360012) stopped after the first model inference because Pillow could not reduce a uint16 display thumbnail. A display-only float conversion fixed this; it did not change the numeric model input. Version 2 (352360882) completed in 99.2 seconds. All 20 correctly proportioned original/input pairs across five private sheets were visually inspected. No gross orientation, blank-image, inversion, or boundary-exclusion failure was seen; sample 11 now retains the visible breast. Some images contain small breast areas and large backgrounds. These checks do not establish lesion preservation or diagnostic quality. The reviewed plan hash and limitations are pinned in `gmic_visual_review.json`.

## Fixed external ensemble pilot, declared before notebook 11 outcomes

Run all five verified fixed GMIC models on all 10,881 standard-view images from the unchanged 2,383 reserved pilot patients. Reproduce sample 11's five audited predictions within 1e-6 before full inference. Keep weights fixed; perform no RSNA fitting, calibration, threshold selection, final evaluation, or density-effect testing. Record image predictions incrementally and require exact coverage before breast aggregation. A failed image stops the run; no outcome-dependent exclusions are permitted.

Pool malignant probabilities (output channel 1) by arithmetic mean within repeated views, then equal CC/MLO means, then equal means across five models. Report AP and ROC-AUC for all fixed members and the ensemble in both pilot roles; also report ensemble performance by site. Do not select the highest-scoring member automatically. This probability pooling is distinct from the existing ResNet logit-pooling pipeline and is not silently substituted into the final protocol. The five externally selected hyperparameter models are not a three-seed ensemble. Population variance and absolute CC/MLO probability difference are saved as descriptive features, without interpreting them as calibrated uncertainty.

Before any final inference, adopting GMIC would require an explicit prospective protocol revision covering the fixed-model design, probability aggregation, calibration partitions, review-score definition, and statistical evaluation. The final cohort stays unopened throughout this pilot. The AP/AUC observations will be adaptive development evidence only.

Notebook 11 version 1 (352363590) passed mounted-artifact, patient-role, source and model-loading checks, then stopped before the reproduction forward pass because `Series.view` resolved to a pandas method instead of the metadata column. The probe now uses a named-tuple row, consistent with the main inference loop. Its scalar view/identifier selection was checked locally. No cohort predictions or outcomes were produced by the failed attempt. Version 2 retains the same images, transforms, weights, metrics and declared scientific design.

## Completed fixed-model pilot and operating-point analysis — 2026-09-28

Notebook 11 version 2 completed all 10,881 images, 4,766 breasts and 2,383 reserved patients. Pooled ensemble fit/tune AP was 0.22096/0.14683 and AUC was 0.82394/0.72612. Tune site AP was 0.44247 and 0.04334, with 13 positives per site. These adaptive point estimates prompted a declared operating-point feasibility analysis, not selection of a member or site.

The fixed 90%-fit-sensitivity threshold detected 66/73 fit cancers but 18/26 tune cancers. Among 669 AI-negative tune breasts, reviewing 33 flagged 1/8 misses with the combined score and 0/8 with cancer score alone. The conditional bootstrap capture-difference interval spans 0–50 percentage points. At 20% review, cancer score alone flags 5 misses versus 3 for combined. Retain this negative evidence. Do not interpret sparse development findings as established benefit or clinical performance.

All run checksums, independent pooling and original ranking metrics reproduced. Three low-orientation-contrast flags are fit/site-1/CC images and remain included. A private CPU follow-up reproduces the analysis and prepares a technical display for these images. Its first attempt reproduced quantitative results but stopped at a decoder dependency-discovery issue; fresh-process decoding repairs setup without changing the experiment. The final trained-CNN freeze gate remains unchanged and locked. A fixed-model final design must be prospectively revised rather than silently inserted into that pipeline.

See `GMIC_OPERATING_POINT_PLAN.md` and `GMIC_DEVELOPMENT_RESULTS.md` for the exact choices, outcomes and remaining limits.

Notebook 12 version 2 (353577896) subsequently completed in 46.7 seconds. All six confusion groups and 24 review comparisons reproduced. The assistant inspected all three original/processed image pairs and found no obvious gross orientation, inversion or boundary-exclusion error at contact-sheet scale. This is technical inspection, not diagnostic review; it does not establish fine-detail preservation. All cases and scores remain unchanged.


## Completed threshold-stability diagnostic — 2026-09-28

The plan and configuration were declared before the new resampling outcomes, following the observed 90.4% fit versus 69.2% tune sensitivity. All 2,000 paired patient bootstrap draws and 100 stratified patient-separated development splits completed. The original fixed rule and 1-versus-0 review comparison reproduced. No classifier, prediction, inclusion rule or original role was changed.

Fit resampling alone yields a central raw-cutoff range of 0.02207–0.05078 and tune sensitivity range of 57.7%–84.6%. The joint-bootstrap combined-minus-cancer-score capture difference has median 0 and central interval −50 to +37.5 percentage points. Among 100 alternative splits, combined flags more in 6, fewer in 42, equal in 47; 5 have no false negatives and undefined capture. Alternative-split median sensitivity is 92.0% but median specificity falls to 44.4%; these overlapping repetitions are not new validation data.

Decision: retain the original operating point and negative findings; do not select a favorable split or score variant. A nominal empirical 90% target is not a guaranteed transfer sensitivity. The next decision requires a substantive prior-art comparison and prospective final-protocol revision for fixed external models, with explicit calibration roles, uncertainty in threshold estimation, equal review workload and a feasible false-negative endpoint. No novelty, model superiority, causal explanation for the original gap, or clinical benefit is claimed. Full evidence is in `THRESHOLD_STABILITY_RESULTS.md`.


## Related work and fixed-model final design — 2026-09-28

The targeted review now includes the main CEI-Net MICCAI 2026 paper, M&M full main text, Lu et al. methods and general failure-detection evaluation recommendations. The direct prior work on uncertainty referral and cross-view disagreement prevents claiming either broad idea as new. Proceed as an empirical reliability comparison, with a possible negative result and unresolved novelty. Access limitations and the precise endpoint distinction are retained in the new related-work decision record.

After all observed development findings, specify a separate fixed-GMIC evaluation: unchanged five released members and preprocessing; all 2,383 reserved development patients for final calibration, cutoff and percentile references; the original 9,530 other patients for one holdout assessment. No RSNA classifier fitting is needed, so retraining five folds is not part of this branch. Original roles and historical results remain intact. More calibration positives motivate refitting; they do not guarantee improved sensitivity.

Keep the 90% empirical calibration target, combined-versus-cancer-score primary comparison at 5%, secondary components/budgets, global allocation and fixed tie ordering. Propagate patient sampling and, secondarily, calibration variability; keep density exploratory at site 1. Do not choose a favorable method or split from development and do not inspect final outcomes to alter the rule.

`FIXED_GMIC_FINAL_PROTOCOL.md` and `fixed_gmic_final_config.json` are design specifications, not an execution freeze. Implement the compatible evaluator/launcher, fit and hash the all-development rule, and pass the new gate before final inference. The original trained-CNN protocol/configuration/gate are unchanged. No final predictions were generated for this review.


## Fixed-model implementation and local freeze — 2026-09-28

Implemented the prospectively specified label-free pooling/scoring/selection, outcome evaluator, patient/site bootstrap, original-cohort identity checks, separate freeze gate and resumable inference shards. All six historical scores and 24 review comparisons reproduced. Eighteen local checks passed after repairing an explicit empty-input validation case. No scientific scoring choice was changed.

The all-development rule uses raw cutoff 0.02242065602913494, calibrated cutoff 0.006898700768467533 and 2,053 AI-negative development reference breasts. It detects 90/99 calibration cancers with specificity 2,044/4,667. These are in-sample fitting diagnostics; no final sensitivity claim follows.

At 16:38:35 UTC, the local fixed-model freeze bound 32 source/configuration/protocol files and five calibration/cohort artifacts. The original 9,530 holdout patients and 43,787 images are partitioned only computationally by preserved outer-fold IDs; all five shards must complete before one global evaluation. A matching Kaggle CPU verification record and fresh development-fixture probe are mandatory before GPU inference.

Six private notebooks were generated. Kaggle draft 13 was copied and renamed, but the file-chooser import was interrupted before completion. The draft still has inherited code; no CPU verification or final job was run. The precise continuation state is saved. Do not treat a generated notebook or local freeze as evidence of completed Kaggle execution.
