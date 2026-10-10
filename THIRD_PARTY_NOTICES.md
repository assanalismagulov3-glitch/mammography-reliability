# Data and third-party attribution

## RSNA screening mammography dataset

The study uses the RSNA Screening Mammography Breast Cancer Detection dataset. Obtain it through the official access process and comply with its terms. The official registry describes use for non-commercial research and educational purposes and prohibits redistributing any portion of the data or annotations. This repository does not redistribute these materials.

- Official record and access: https://registry.opendata.aws/rsna-screening-mammography-breast-cancer-detection/
- Dataset citation DOI: https://doi.org/10.1148/dataset.smbc.2024
- Original competition: https://www.kaggle.com/competitions/rsna-breast-cancer-detection

The reported aggregate findings are this study's results, not RSNA endorsement. Any future data use must follow the official terms applicable to that access.

## NYU GMIC

The fixed image models, upstream implementation and original preprocessing/reference fixtures belong to the GMIC authors. The released model architecture and weights were not developed or trained in this study. This study adds an RSNA adaptation, reliability descriptors and an evaluation of review prioritization.

- Upstream project: https://github.com/nyukat/GMIC
- Pinned revision: `594029efe6184bcca3a151fa1dd46c5fd620cb25`
- Upstream license recorded by the inspected manifest: AGPL-3.0
- Pinned LICENSE: https://github.com/nyukat/GMIC/blob/594029efe6184bcca3a151fa1dd46c5fd620cb25/LICENSE
- Pinned NOTICE: https://github.com/nyukat/GMIC/blob/594029efe6184bcca3a151fa1dd46c5fd620cb25/NOTICE

The manifest records LICENSE Git-blob SHA-1 `29ebfa545f5580919a4e884d7014d7a3eb2df762` and NOTICE `8306b2920b99173f2f0d0aead128cbe66701dc11`. `gmic_adapter.acquire_upstream` verifies these and every pinned file when obtaining upstream materials. Preserve those files and applicable obligations when obtaining, using, modifying or redistributing upstream code. This release contains the manifest and adapter source, not a redistributed upstream source tree, weights or fixture images. It does not relicense upstream materials.

## Project code

No new open-source license grant is made by this package. It supports inspection of the study source and aggregate outputs; absence of a project license is not permissive licensing. Upstream materials retain their own terms.
