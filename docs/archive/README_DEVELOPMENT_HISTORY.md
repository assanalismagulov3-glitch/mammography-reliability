# Historical README snapshot

Archived 2026-10-01 before switching the entry point to the completed fixed-GMIC experiment. Status statements below describe earlier checkpoints and are superseded by README.md and STATUS.md.

# Mammography AI failure-detection experiments

This is an executable research foundation for the existing RSNA breast-cancer AI reliability project. It includes a data audit, patient-separated validation, DICOM processing, a breast-level CNN, a three-member ensemble, calibration, and comparisons of false-negative review scores.

**Status, 2026-10-01: the frozen fixed-model holdout experiment is complete.** All five inference jobs, full integrity checks and both 2,000-draw uncertainty analyses finished successfully. At equal 5% review capacity, combined scoring flagged 2/39 missed cancers versus 3/39 for cancer score alone; the 95% interval includes zero, so added benefit is not established. See `FIXED_GMIC_FINAL_RESULTS.md` for all prespecified findings and `STATUS.md` for later work. The trained-CNN instructions below are historical. No human-reader benefit, novelty or public release is established.

**Preparation update:** the targeted literature/contribution check is complete for cautious empirical framing. See `RELATED_WORK_ADDENDUM_2026_10_01.md` and `literature_search_log_2026_10_01.json`. Four preparation blocks remain: student understanding, paper figures/tables, repository packaging, and public-content/attribution review. This does not establish novelty; the frozen experiment is unchanged.

For a beginner-friendly explanation of the final study, read `STUDY_EXPLANATION.md`. It explains the classifier versus review ranking, the patient split, the three scores, result denominators and uncertainty intervals. The student's methods/results review is in progress.

## Start on Kaggle

Import `rsna_reliability_workflow.ipynb` into a **private** notebook and attach the official RSNA Screening Mammography Breast Cancer Detection competition dataset. The notebook contains the package source, so no GitHub upload is needed to run it. Its initial stage is `audit`. The established path is `/kaggle/input/competitions/rsna-breast-cancer-detection/train.csv`.

Stages are explicit so image processing, resource use, and validation performance can be checked before opening held-out results. Account access is established and the assistant has started the practical workflow.

1. `audit`: validate labels and identifiers, report density availability, reserve 20% of patients for independent development, and save folds for the remaining patients.
2. `prepare`: decode all standard-view images, check exact cross-patient pixel duplicates, and optionally create lossless 16-bit PNGs.
3. `pilot`: fit one model using only the reserved pilot patients. Compare recorded trial configurations without opening final-fold results.
4. `freeze`: record a concrete review of completed pilot trials, resolution, related research, limitations, and the selected settings. Final training cannot start without this record.
5. `train`: fit the three ensemble members for the selected outer fold. Pilot patients now contribute only to fitting; checkpoint selection uses that fold's tuning partition.
6. `evaluate`: fit calibration and choose the threshold on the calibration partition, then evaluate that outer fold's held-out patients.
7. `summarize`: combine all five completed held-out folds from the evaluation cohort and calculate the primary paired comparison. Pilot patients are excluded from every test endpoint.

Follow the feasibility decisions in `PROTOCOL.md`. A successful software test is not evidence of useful mammography performance. Version 0.2 uses `/kaggle/working/mammo_private_v0_2` so the original audit remains preserved.

The measured lossless cache exceeds a single Kaggle notebook's storage limit. `rsna_cache_part_0.ipynb` and `rsna_cache_part_1.ipynb` create separate private cache parts on CPU. They are already submitted as the student's notebooks `03_rsna_lossless_cache_part_0` and `04_rsna_lossless_cache_part_1`. Both have completed. Attach their versioned outputs to the training notebook and run `mammo_reliability.cache_parts assemble` with their actual mounted directories. Assembly verifies metadata, complete image coverage, preprocessing identity, and cross-patient duplicates. Training checks each cached image's content hash. No cache pixels are copied into this source package.

The current `rsna_development_pilots.ipynb` automates cache assembly, higher-resolution validation, and both pilot trials. Attach version 1 of notebooks 03, 04, and 05 plus the official competition, then select a GPU. The launcher verifies the recorded cache hashes before training. It uses the same seed and reserved development patients at both resolutions, preserves learning curves and checkpoints, and leaves final evaluation locked. It is submitted privately as notebook `06_rsna_development_pilots`.

`rsna_development_diagnostics.ipynb`, generated by `build_diagnostics_notebook.py`, preserves the completed diagnostic experiment. Attach the official competition and the same version 1 outputs from notebooks 03 and 04. Its standalone `development_diagnostics.py` leaves the audited model package unchanged. All diagnostic settings are written before outcomes are produced; the small-set memorization score must never be presented as validation or test performance.

`rsna_development_balanced.ipynb`, generated by `build_balanced_notebook.py`, preserves the completed sampling trial. Attach the official competition plus version 1 outputs from notebooks 03, 04, and 07. It verifies and reevaluates the selected reference checkpoint, then tests equal positive/negative accumulation groups using the same expected weighted objective and a comparable update budget. Standalone scripts preserve the original package. See `DEVELOPMENT_DECISIONS.md` for the rationale and limitations; this adaptive pilot does not unlock final evaluation.

`rsna_gmic_preflight.ipynb`, generated by `build_gmic_notebook.py`, implements the new compatibility check. Attach only the official RSNA competition, enable Internet and a GPU, and keep the notebook private. It downloads 48 blob-verified files from a pinned author repository, loads five fixed GMIC models, and checks model 1 against the historical author sample outputs, with a separate parity check against the current author pipeline. The repository changed its sample preprocessing after publication of the reference table; all earlier failures are retained. Only after both compatibility checks pass does it process a deterministic sample of reserved fitting images. Review the private orientation/crop contact sheets before any full-cohort transfer evaluation. The upstream source is unmodified and retains its AGPLv3 license and NOTICE. This is an engineering preflight, not a replication of the paper's full evaluation.

`rsna_gmic_fullfield_preflight.ipynb` preserves the completed geometry repair audit. Attach notebook 09 version 4 and the official dataset. Its versioned numeric results and technical visual review are in `validation/kaggle_fullfield_status.json` and `gmic_visual_review.json`.

`rsna_gmic_reserved_pilot.ipynb`, generated by `build_gmic_pilot_notebook.py`, evaluates all five fixed external models on the unchanged pilot patients. Attach notebook 09 version 4, notebook 10 version 2, and the official dataset. The launcher verifies their identities and the image review, reproduces the audited sample's predictions, and uses explicit probability pooling. It does not invoke final evaluation. Full image predictions are written every 100 images; there is no automatic cross-commit recovery, and partial outputs require a separate integrity review before reuse.

## Command-line use

From this directory, install the dependencies in `pyproject.toml`. The `train` extra includes PyTorch and DICOM decoders; the `dev` extra includes the test tools. A suitable GPU is required by the research configuration. Kaggle generally supplies PyTorch; install only missing compatible packages there.

```bash
python -m pip install -e '.[train,dev]'
python -m mammo_reliability.pipeline audit --data-root /path/to/official/data --run-root /path/to/private_runs --config config.json
python -m mammo_reliability.pipeline prepare --data-root /path/to/official/data --run-root /path/to/private_runs --config config.json
python -m mammo_reliability.pipeline pilot --data-root /path/to/official/data --run-root /path/to/private_runs --config config.json --trial baseline_1024_v1
python -m mammo_reliability.pipeline freeze --data-root /path/to/official/data --run-root /path/to/private_runs --config config.json --development-review /path/to/completed_development_review.json
python -m mammo_reliability.pipeline train --data-root /path/to/official/data --run-root /path/to/private_runs --config config.json --fold 0
python -m mammo_reliability.pipeline evaluate --data-root /path/to/official/data --run-root /path/to/private_runs --config config.json --fold 0
```

Repeat training/evaluation for folds 0–4 only after freezing the protocol. Then run the `summarize` stage. Every stage checks the metadata/configuration identity. Training additionally records the source fingerprint, software versions, checkpoint hashes, and learning curves. Interrupted training restarts from the last completed epoch; it can repeat an interrupted partial epoch.

`rsna_pilot_preparation.ipynb` records revised real-data cohort counts and prepares 1536 × 1152 images for reserved pilot patients. After completion, `pipeline pilot --trial higher_resolution_1536_v1 --pilot-model-config /path/to/overrides.json --pilot-cache-root /actual/private/cache/mount` uses that cache. The overrides are `{"height": 1536, "width": 1152}`. The baseline pilot uses `--trial baseline_1024_v1` and the assembled full 1024 × 768 cache. Both use seed 17 and identical pilot roles. Actual mounted paths must be supplied after attaching private versioned outputs.

## Files and results

- `mammo_reliability/audit.py`: metadata validation and patient partitions.
- `study.py`: independent pilot selection, final patient roles, and prospective protocol freeze.
- `pilot.py`: higher-resolution development cache and isolated training trials.
- `imaging.py`: presentation transforms, full-frame resizing, pixel audit and cache.
- `cache_parts.py`: parallel private cache generation and validated remounting across storage parts.
- `model.py`: ResNet-18 and breast/view pooling.
- `training.py`: fitting, early stopping, checkpoint recovery, inference.
- `evaluation.py`: calibration, operating threshold, review selection, cluster bootstrap.
- `pipeline.py`: stage orchestration and aggregate/private outputs.
- `tests/`: artificial fixtures and meaningful software checks.
- `observed_metadata.json`: previously reported aggregate counts, with explicit provenance.
- `development_preflight.py` and `gpu_preflight.py`: reproducible real-image technical checks and artificial-tensor resource measurements.
- `validation/kaggle_technical_preflight.json` and `validation/kaggle_gpu_preflight.json`: actual outputs of those checks, with their limits.

Output directories contain patient-level annotations, predictions, and possibly derived images. Keep them private. A later GitHub release can contain source, documentation, aggregate results, and access instructions after checking the release contents. The dataset's license prohibits redistributing images or annotations. [Official dataset license](https://registry.opendata.aws/rsna-screening-mammography-breast-cancer-detection/)

There is no automatic upload to GitHub, no clinical deployment, and no background training job implied by downloading this package.

## Latest development analysis — 28 September 2026

Notebook 11's fixed-GMIC pilot completed. `GMIC_DEVELOPMENT_RESULTS.md` reports the full development operating-point and review comparisons, including negative findings. `gmic_operating_point.py` verifies private input hashes, independently reproduces pooling/ranking, and applies fit-only calibration and threshold selection to tune patients. Run it with `--input /path/to/notebook11/gmic_reserved_pilot`. It requires CPU dependencies only.

`rsna_gmic_operating_point.ipynb`, built by `build_gmic_analysis_notebook.py`, reproduces those results on Kaggle and prepares a technical orientation display using `gmic_orientation_review.py`. Attach notebook 11 version 2 and the official competition; keep it private and use CPU. This follow-up neither retrains GMIC nor unlocks final evaluation. Private scored tables are stored under the ignored `private_runs/` directory and are excluded from the source archive.

`threshold_stability.py` implements the completed development-only patient resampling diagnostic. Run `python3 threshold_stability.py --input /path/to/private/gmic_reserved_pilot` from this directory. Its declared plan, fixed configuration, seven artificial software checks, all 18,303 aggregate result rows, figure and interpretation are included. It uses existing predictions and CPU dependencies; it does not require another Kaggle/GPU run. Patient split membership is private. These scripts must not be silently substituted into the older trained-CNN final pipeline.
