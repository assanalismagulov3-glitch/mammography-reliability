# Mammography Reliability

**Review prioritization after a negative mammography classification**

I investigate whether CC–MLO disagreement and variation between fixed GMIC models help prioritize missed cancers beyond cancer score alone at the same review capacity.

## Main finding

On a patient-separated holdout of **9,530 patients / 19,060 breasts**, the classifier missed 39 cancer-positive breasts. Each method selected **415 of 8,315 classifier-negative breasts** for potential review.

| Review score | Misses flagged | Capture |
|---|---:|---:|
| Cancer score alone | 3 / 39 | 7.69% |
| Combined score | 2 / 39 | 5.13% |

The difference was **−2.56 percentage points** (primary 95% patient-bootstrap CI: **−13.79 to +8.11**). This experiment did not establish incremental benefit; it does not demonstrate equivalence or clinical recovery.

![Primary comparison](reports/figure_2_primary_comparison.png)

## Read and reproduce

| Material | Contents |
|---|---|
| [Study summary](STUDY.md) | Question, methods, findings and limitations |
| [Reproducibility](REPRODUCIBILITY.md) | Verified commands and original execution archive |
| [Source](src/) and [tests](src/tests/) | Unchanged final-study implementation and synthetic checks |
| [Figures and tables](reports/) | Three figures, captions and all secondary results |
| [Validation records](validation/) | Aggregate outputs, bootstrap draws and provenance |
| [Contributions](CONTRIBUTIONS.md) | Complete contribution and assistance disclosure |
| [Data and credits](THIRD_PARTY_NOTICES.md) | Dataset access, upstream model and licensing |

```bash
python -m pip install -r requirements-report.txt
python scripts/verify_release.py
PYTHONPATH=src python -m unittest discover -s src/tests -p 'test_fixed_gmic_*.py' -v
python scripts/make_release_figures.py --output reproduced_reports
```

This repository is the code-and-results companion to the research manuscript. It reproduces aggregate reporting; original patient-level execution requires authorized data and private artifacts. Historical notebooks and working logs remain in Git history. No patient records, images or model weights are distributed.
