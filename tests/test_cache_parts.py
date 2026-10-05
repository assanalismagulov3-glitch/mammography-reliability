"""Cache partitioning must preserve pixels, identifiers, and global leakage checks."""
import shutil

import pandas as pd
import pytest

from mammo_reliability.audit import audit_metadata
from mammo_reliability.cache_parts import assemble_parts, build_part
from mammo_reliability.model import BreastDataset
from test_integrity import artificial_metadata
from test_model_and_workflow import save_artificial_dicom


def test_parallel_cache_parts_can_be_remounted_without_changing_cohort(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    metadata = artificial_metadata(4)
    metadata.to_csv(source / 'train.csv', index=False)
    for row in metadata.itertuples(index=False):
        save_artificial_dicom(source / 'train_images' / str(row.patient_id) / f'{row.image_id}.dcm', row.image_id)
    roots = []
    for part in [0, 1]:
        root = tmp_path / f'part_{part}'
        build_part(source, root, part, workers=2, height=32, width=32)
        destination = tmp_path / 'mounted_inputs' / f'part_{part}'
        destination.parent.mkdir(exist_ok=True)
        shutil.move(root, destination)
        roots.append(destination)
    manifest = assemble_parts(roots, source, tmp_path / 'joined', height=32, width=32)
    assert len(manifest) == len(metadata)
    breasts, _, _ = audit_metadata(metadata)
    dataset = BreastDataset(manifest, breasts, height=32, width=32)
    assert dataset[0]['images'].shape == (2, 3, 32, 32)
    with pytest.raises(ValueError, match='Duplicate or missing'):
        assemble_parts([roots[0], roots[0]], source, tmp_path / 'bad', height=32, width=32)
    # Each part was valid individually. A cross-part duplicate must stop assembly.
    import json
    from mammo_reliability.audit import file_sha256
    first = pd.read_csv(roots[0] / 'manifest.csv')
    second = pd.read_csv(roots[1] / 'manifest.csv')
    second.loc[0, 'decoded_pixel_sha256'] = first.loc[0, 'decoded_pixel_sha256']
    second.to_csv(roots[1] / 'manifest.csv', index=False)
    complete_path = roots[1] / 'part_complete.json'
    complete = json.loads(complete_path.read_text())
    complete['manifest_sha256'] = file_sha256(roots[1] / 'manifest.csv')
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match='Identical decoded images'):
        assemble_parts(roots, source, tmp_path / 'bad_duplicates', height=32, width=32)
