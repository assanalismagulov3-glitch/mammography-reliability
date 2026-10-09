# Mammography reliability

Code and aggregate results from a retrospective RSNA study: does adding CC–MLO disagreement and between-model variation improve missed-cancer prioritization over cancer score alone at equal review capacity?

The study used five fixed, externally trained GMIC members, with 2,383 development patients and a separate holdout of 9,530 patients. No GMIC weights were trained on RSNA.

## Main result

Both methods selected **415 of 8,315 AI-negative breasts** at a 5% review budget. There were **39 missed cancer-positive breasts** in this pool.

| Review score | Misses flagged | Capture |
|---|---:|---:|
| Cancer score alone | 3 / 39 | 7.69% |
| Combined score | 2 / 39 | 5.13% |

Combined minus cancer-score capture: **−2.56 percentage points**, paired patient-bootstrap 95% CI **[−13.79, +8.11]**. Added benefit remains inconclusive; equivalence and clinical benefit were not established.

[Results and limitations](docs/results.md) · [Tables](reports/TABLES.md) · [Figures](reports/)

## Reproduce the aggregate results

Python 3.12; no medical data or GPU required:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python scripts/verify_release.py
python -m unittest discover -s tests -p 'test_fixed_gmic_*.py' -v
python scripts/make_release_figures.py --output reproduced_reports
```

These commands verify the release, run 18 synthetic-data checks and regenerate figures/tables from saved aggregate outputs. Patient-level inference needs separately authorized data and private artifacts; see [reproducibility](docs/reproducibility.md).

## Repository map

| Location | Contents |
|---|---|
| `fixed_gmic_*.py`, `gmic_*.py`, `mammo_reliability/` | Evaluation, inference, preprocessing and original development code |
| `scripts/`, `tests/` | Aggregate reporting, integrity checks and tests |
| `reports/`, `validation/` | Figures, tables, aggregate results, bootstrap draws and provenance |
| [docs/](docs/README.md) | Methods, results, reproduction, attribution and archived development notes |

Four protocol/plan documents remain at their original root paths because the frozen or historical code reads them directly. Scientific code and original numerical outputs are unchanged by the documentation reorganization.

[Data, upstream credits and assistance](docs/attribution.md). No patient-level data or model weights are distributed.
