# Reproducing the completed fixed-GMIC study

The original experiment is complete. The release supports immediate reproduction of the aggregate figures, tables and bootstrap interval summaries, and includes the unchanged scientific implementation and its tests. Patient-level inference requires separately authorized RSNA data and the original private execution artifacts described below. A clean-room, end-to-end rerun from this archive alone has **not** been demonstrated.

## 1. Public aggregate reproduction

Use Python 3.12 in a virtual environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python scripts/verify_release.py
python -m unittest discover -s tests -p 'test_fixed_gmic_*.py' -v
python scripts/make_release_figures.py --output reproduced_reports
```

The figure script reads only the saved aggregate JSON, aggregate bootstrap draws, extraction-verification record and frozen configuration. It checks original SHA-256 digests, independently reconstructs both percentile intervals from 4,000 saved draws, and checks the global review capacities before writing PDF/SVG/PNG figures and Markdown/JSON tables. It does not rerun inference, refit calibration, resample patients again or inspect individual patient records. PDF/SVG file hashes may vary across plotting-library/platform versions without changing the numeric content.

The 18 artificial-data unit tests check patient clustering, site counts, label isolation, equal-view pooling, global selection capacity, deterministic ties, weighted resampling, freeze integrity and safe resume behavior. Passing these tests is not external model validation.

## 2. Completed experiment and fixed settings

| Setting | Value |
|---|---|
| GMIC upstream revision | `594029efe6184bcca3a151fa1dd46c5fd620cb25` |
| Models | All five externally trained released members; malignant channel 1 |
| RSNA weight fitting | None in the final experiment |
| Input | Full-field aspect-preserving resize/pad to 2944 × 1920 |
| Pooling | Repeats within view; equal CC/MLO; equal five members |
| Inference | Seed 17; FP32; no TF32; deterministic cuDNN; no benchmarking |
| Development | 2,383 patients; 4,766 breasts; 99 positive breasts |
| Holdout | 9,530 patients; 19,060 breasts; 393 positive breasts |
| Raw positive threshold | `0.02242065602913494`, positive at or above cutoff |
| Calibrated cutoff | `0.006898700768467533` |
| Main review budget | `floor(0.05 × number of AI-negative breasts)` globally |
| Ranking ties | Ascending SHA-256 of `20260909:patient_id:laterality` |
| Holdout bootstrap seed | 20260930 |
| Development-refit bootstrap seed | 20261001 |
| Bootstrap | 2,000 paired patient draws per scheme, stratified by site |

The exact rules are in `fixed_gmic_final_config.json` and [frozen protocol](../FIXED_GMIC_FINAL_PROTOCOL.md). These are historical frozen records: their pre-execution status fields are intentionally unchanged. Current completion status is in [results](results.md) and the aggregate result. The freeze followed adaptive development and was not a public preregistration. Five shards are computational partitions, not cross-validation folds or separately fitted models.

## 3. Original private workflow

Obtain the RSNA data through its official access process; do not obtain it from this repository. The original workflow also requires restricted outputs from the technical preflight and development jobs. Do not substitute placeholders or fabricate passing verification records.

| Stage | Included implementation | Required private inputs / output |
|---|---|---|
| Compatibility audit | `gmic_preflight.py`, `gmic_adapter.py` | Official DICOMs/metadata; pinned upstream files and fixtures; produces cohort identities and engineering sample |
| Full-field geometry audit | `gmic_fullfield_preflight.py`, `gmic_fullfield.py` | Successful preflight; original 20 fit images and fixed weights; produces reviewed geometry record |
| Development inference | `gmic_pilot.py`, `gmic_pilot_analysis.py` | Reviewed preflight and geometry outputs; produces development image/breast predictions |
| Development decisions | `gmic_operating_point.py`, `threshold_stability.py` | Development predictions and roles; historical failed comparisons remain documented |
| Final calibration | `fixed_gmic_prepare.py` | Original development outputs, patient/image membership and experiment declaration; five private rule/cohort files |
| Source and rule freeze | `fixed_gmic_gate.py` | Successful checks tied to exact source and rule hashes |
| CPU verification and five GPU shards | `build_fixed_gmic_notebooks.py`, `gmic_final_inference.py` | Original freeze, private rule files, audited fixtures and successful CPU verification |
| Global final analysis | `fixed_gmic_final_analysis.py` | All five verified shards, frozen bundle and official outcome metadata |

The historic launchers contain Kaggle mount paths and hashes for the author's exact private jobs. They intentionally fail when those artifacts or identities are absent. Merely changing a path is insufficient: source, rule, fixture, environment and membership verification must all pass. The generated final notebooks embed patient-level calibration tables, so **the generator source is included but those generated notebooks are excluded**.

With the original authorized artifacts restored at their expected locations, the final stage commands are:

```bash
# BUNDLE and all referenced paths must point to the original verified private files.
python build_fixed_gmic_notebooks.py --bundle "$BUNDLE" --output private_notebooks

# Run each shard separately on GPU after the original CPU verification succeeds.
python gmic_final_inference.py --bundle "$BUNDLE" --shard 0 \
  --output private_outputs/final_shard_0 --verification "$CPU_VERIFICATION"
# Repeat for shard indices 1, 2, 3 and 4, using distinct output directories.

# Run once on CPU; output must be a new directory.
python fixed_gmic_final_analysis.py --bundle "$BUNDLE" \
  --shards private_outputs/final_shard_0 private_outputs/final_shard_1 \
           private_outputs/final_shard_2 private_outputs/final_shard_3 \
           private_outputs/final_shard_4 \
  --metadata "$RSNA_ROOT/train.csv" --output private_outputs/reproduced_analysis
```

`BUNDLE` contains `calibration_rule_private.json`, `cohort_manifest.json`, `preparation_result.json`, `development_features_private.csv`, `development_outcomes_private.csv` and `frozen_fixed_gmic.json`. The original technical-preflight and development outputs are additional runtime prerequisites; they are not recreated by the last two commands. Original private outputs remain with the study owner in Kaggle. They are not promised as publicly downloadable artifacts.

For an independent rebuild from newly obtained data, rerun the technical and development preparation and record a separately identified reproduction. Recreating the historical private freeze byte-for-byte, or bypassing its guards, is not an acceptable substitute. The now-opened holdout cannot be called untouched evaluation data for a revised method.

## 4. Environments and verification scope

`validation/release_environment.json` records the actual CPU environment used for this release's figures and 18 tests. `requirements-report.txt` pins its direct analysis dependencies; it is not a GPU lockfile.

The original development GPU record reports Python 3.12.13, NumPy 2.0.2, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.6.1, Pillow 11.3.0, pydicom 3.0.2, torch 2.10.0+cu128, torchvision 0.25.0+cu128, pylibjpeg 2.1.0 and pylibjpeg-openjpeg 2.6.0. This is a **development** environment reference, not an independently exported final-shard lockfile. The frozen final evaluator verified that all five shard environment records and model identities agreed. The final Kaggle jobs retained the platform's original environment dated 2026-07-01. Their complete private run identities are retained with the original outputs; this release does not claim a portable, bit-for-bit GPU environment.

Original GMIC execution also needs its imageio, h5py, OpenCV, Pillow, DICOM decoders and compatible CUDA/PyTorch dependencies, as specified in the included launcher sources. Obtain upstream source, weights and fixtures separately using the inspected manifest; retain their upstream license and notice.

## 5. Data, attribution and limits

See [attribution](attribution.md) and [original release contents](archive/RELEASE_CONTENTS.md). This archive contains code, documentation and aggregate statistics, not medical images, patient identifiers/annotations, individual predictions, embedded private notebooks or model weights. Figures describe retrospective label-based flagging; no human-reader recovery or clinical benefit was measured.
