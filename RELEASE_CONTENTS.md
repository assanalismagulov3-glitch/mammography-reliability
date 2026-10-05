# Release contents — 2026-10-01

This source-and-aggregate release was prepared after completion of the frozen experiment. It is separate from the owner's private working archive. It contains no new inference, threshold tuning or manuscript.

## Included

- All 32 frozen scientific/configuration/protocol files, byte-for-byte unchanged.
- Supporting development source, synthetic-data tests and earlier unexecuted launcher notebooks, preserving the project's unsuccessful development history.
- The complete aggregate final-result JSON and all 4,000 aggregate bootstrap draw records, with the original extraction-verification record.
- Three publication-resolution figures, each in PDF, SVG and PNG; six Markdown table sections; complete machine-readable aggregate table exports and captions.
- A final-experiment README, reproduction commands, fixed settings, environment scope, third-party attribution, assistance disclosure and known reproducibility limits.
- A SHA-256 manifest and scripts to verify original source digests and regenerate figures/tables.

## Excluded

- DICOMs, medical-image contact sheets, dataset metadata rows and annotations.
- Patient/image membership, individual predictions, scores, outcome tables and private calibration reference arrays.
- All `private/` and `private_runs/` contents, including generated final notebooks with embedded private calibration tables.
- Downloaded model weights, upstream source/fixture images, private archives, account screenshots, raw execution logs and credentials.

The small public-style launcher notebooks included at the repository root are historical source templates with no cell outputs or embedded patient tables. They are not the private final execution notebooks. Synthetic patient identifiers in software tests are invented fixtures, not cohort records.

## Verified for this release

The frozen source hashes match the original execution freeze. The 18 fixed-GMIC tests pass on the recorded CPU environment. Both reported confidence intervals reproduce from the saved 4,000 aggregate draws. All 126 review rows obey the frozen global floor-capacity rule. The three PDFs were rendered and visually inspected; figure/table values were derived from the unmodified aggregate result.

No GPU inference was rerun for packaging. No clean-room reproduction of private execution prerequisites is claimed. No independent external validation or open-source license grant is implied. See `REPRODUCIBILITY.md` and `THIRD_PARTY_NOTICES.md`.

## Publication state

The GitHub-ready archive is prepared locally. A remote upload is still pending GitHub installation/connection. No repository URL, commit or successful upload is claimed until remote contents can be checked. Detailed teaching is deferred until before paper writing at the user's request; it is not a prerequisite for this practical release.
