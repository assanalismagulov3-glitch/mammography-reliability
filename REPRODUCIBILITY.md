# Reproducibility

## Aggregate reproduction

Use Python 3.12 and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python scripts/verify_release.py
PYTHONPATH=src python -m unittest discover -s src/tests -p 'test_fixed_gmic_*.py' -v
python scripts/make_release_figures.py --output reproduced_reports
```

On Windows, set the `PYTHONPATH` environment variable to `src` before the unittest command. The requirements pin direct CPU reporting dependencies, not a historical GPU environment or every transitive dependency.

The verifier checks the package manifest, unchanged included scientific-file hashes and the original result/freeze identities. The reporting script checks the original aggregate inputs, reconstructs both percentile intervals from 4,000 saved bootstrap draws, checks global review capacities, and regenerates the figures and tables. It does not rerun model inference, patient resampling or calibration. PDF/SVG byte hashes may vary with plotting-library versions.

The 18 original synthetic tests cover pooling, label isolation, patient clustering, deterministic ties, global capacity, weighted resampling, freeze integrity and safe resume behavior. These tests verify software invariants, not external or clinical model validity.

## What this submission package contains

`src/` contains the unchanged implementation, supporting modules and settings used by the final study. The two final-study test files are in `src/tests/`. [The layout manifest](validation/submission_manifest.json) maps **29 included files** to their original frozen paths and SHA-256 values. Their bytes are unchanged despite the new layout.

Three frozen context records (the original protocol, literature decision and technical-review record), historical notebooks and development logs remain in the original Git revision. They are not silently rewritten or represented as new prospective records. The full original 32-file freeze is recorded in [release provenance](validation/release_provenance.json). Its historical status fields and release date describe that earlier record, not this packaging revision.

## Original patient-level workflow

The compact layout is for code inspection, synthetic checks and aggregate reproduction. Original inference and calibration must use the original layout because its integrity gate verifies exact paths and all 32 frozen files.

Restore the complete preserved workflow in a separate worktree:

```bash
git worktree add --detach ../mammography-original f068a9ac54e7bd3ab54885da1847fb3a45c15554
cd ../mammography-original
python scripts/verify_release.py
```

[Open the preserved version](https://github.com/assanalismagulov3-glitch/mammography-reliability/tree/f068a9ac54e7bd3ab54885da1847fb3a45c15554) and its [original workflow instructions](https://github.com/assanalismagulov3-glitch/mammography-reliability/blob/f068a9ac54e7bd3ab54885da1847fb3a45c15554/docs/reproducibility.md). This also retains failed experiments, dated decisions, upstream fixture audits and notebook generators.

Original patient-level execution additionally requires official RSNA access, the pinned GMIC source/weights, original technical-preflight and development outputs, the private calibration/cohort bundle, and original CPU verification and shard outputs. These materials are not distributed here. The historic launchers contain exact Kaggle mount paths and hashes and deliberately stop if an identity or integrity check fails. Do not bypass those gates or fabricate substitute verification records.

A clean-room end-to-end reconstruction from this repository alone has not been demonstrated. Any independent rebuild must be identified as a new reproduction. The already examined holdout is not a new test set.

## Provenance

The experiment freeze is `6cdc4c4f976c54b751eb15b85c9a1d1fa1db6877a554c6cb4ac468963c35b0ef`, created September 28, 2026 before final model outcomes. Final results were extracted October 1. The October 10 packaging changes do not alter model weights, analysis rules, original source bytes or numerical outputs.

`validation/release_environment.json` describes the original aggregate-reporting environment. `validation/submission_verification.json` records verification of this compact package. See [data and upstream credits](THIRD_PARTY_NOTICES.md) for access and licensing.
