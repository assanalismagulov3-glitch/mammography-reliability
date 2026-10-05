"""Isolation, real file identities, and prospective evaluation gates."""
import json

import pandas as pd
import pytest

from mammo_reliability.audit import audit_metadata, write_json
from mammo_reliability.pipeline import initialize
from mammo_reliability.study import (freeze_protocol, make_study_patients, pilot_roles,
                                     require_frozen_protocol, study_outer_roles)
from test_integrity import artificial_metadata


def study_config():
    return {'experiment_id': 'ARTIFICIAL_STUDY_FIXTURE', 'n_outer_folds': 5, 'split_seed': 20260909,
            'pilot_fraction': .20, 'pilot_tune_fraction': .25,
            'ensemble_seeds': [17, 41, 73], 'model': {'height': 16, 'width': 16}}


def initialized_study(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    artificial_metadata(160).to_csv(data / 'train.csv', index=False)
    root = tmp_path / 'study'
    config = study_config()
    initialize(data, root, config)
    return root, config


def test_pilot_is_excluded_from_every_test_fold_and_all_other_patients_test_once():
    breasts, _, _ = audit_metadata(artificial_metadata(160))
    patients = make_study_patients(breasts)
    pd.testing.assert_frame_equal(patients, make_study_patients(breasts.sample(frac=1, random_state=8)))
    pilot = pilot_roles(patients)
    assert len(pilot) == 32 and set(pilot.role) == {'fit', 'tune'}
    held_out = []
    for fold in range(5):
        roles = study_outer_roles(patients, fold)
        assert set(roles.patient_id) == set(patients.patient_id)
        assert (roles.loc[roles.patient_id.isin(pilot.patient_id), 'role'] == 'fit').all()
        assert set(roles.loc[roles.role == 'test', 'patient_id']).isdisjoint(pilot.patient_id)
        assert breasts.merge(roles[['patient_id', 'role']], on='patient_id').groupby('patient_id').role.nunique().eq(1).all()
        held_out.extend(roles.loc[roles.role == 'test', 'patient_id'])
    assert len(held_out) == len(set(held_out)) == 128
    assert set(held_out) == set(patients.loc[patients.cohort == 'evaluation', 'patient_id'])


def fake_review(root):
    trial = root / 'pilot' / 'trials' / 'ARTIFICIAL_GATE_FIXTURE'
    (trial / 'seed_17').mkdir(parents=True)
    roles = pd.read_csv(root / 'pilot' / 'roles_private.csv')
    roles.to_csv(trial / 'roles_private.csv', index=False)
    write_json(trial / 'pilot_trial.json', {'purpose': 'development_only',
               'metadata_sha256': json.loads((root / 'experiment.json').read_text())['metadata_sha256']})
    write_json(trial / 'seed_17' / 'completed.json', {'status': 'training_completed', 'fixture': 'ARTIFICIAL_GATE_TEST_ONLY'})
    review = root / 'review_input.json'
    write_json(review, {'decision': 'Artificial software fixture only', 'resolution_assessment': 'Not assessed',
                        'literature_assessment': 'Not assessed', 'limitations': 'No research evidence',
                        'completed_pilot_directories': [str(trial)]})
    return review


def test_final_evaluation_gate_rejects_missing_freeze_changed_code_and_changed_partitions(tmp_path, monkeypatch):
    root, config = initialized_study(tmp_path)
    with pytest.raises(ValueError, match='requires'):
        require_frozen_protocol(root, config)
    review = fake_review(root)
    freeze_protocol(root, config, review)
    assert require_frozen_protocol(root, config)['config'] == config
    with monkeypatch.context() as patch:
        patch.setattr('mammo_reliability.training.source_fingerprint', lambda: 'CHANGED')
        with pytest.raises(ValueError, match='Source changed'):
            require_frozen_protocol(root, config)
    path = root / 'fold_0' / 'roles_private.csv'
    roles = pd.read_csv(path)
    roles.loc[0, 'role'] = 'CHANGED'
    roles.to_csv(path, index=False)
    with pytest.raises(ValueError, match='assignments changed'):
        require_frozen_protocol(root, config)


def test_freeze_cannot_be_created_after_test_predictions_or_without_completed_pilot(tmp_path):
    root, config = initialized_study(tmp_path)
    review = fake_review(root)
    completed = next(root.glob('pilot/trials/*/seed_*/completed.json'))
    completed.unlink()
    with pytest.raises(ValueError, match='not completed'):
        freeze_protocol(root, config, review)
    (root / 'fold_0' / 'test_scored_private.csv').write_text('ARTIFICIAL_EXISTING_RESULT')
    with pytest.raises(ValueError, match='retrospective'):
        freeze_protocol(root, config, review)


def test_higher_resolution_pilot_cache_covers_only_reserved_patients_and_checks_raw_pixels(tmp_path):
    from mammo_reliability.imaging import prepare_images
    from mammo_reliability.pilot import prepare_pilot_cache, load_pilot_cache
    from test_model_and_workflow import save_artificial_dicom
    data = tmp_path / 'data'
    data.mkdir()
    metadata = artificial_metadata(160)
    metadata.to_csv(data / 'train.csv', index=False)
    for row in metadata.itertuples(index=False):
        save_artificial_dicom(data / 'train_images' / str(row.patient_id) / f'{row.image_id}.dcm', row.image_id)
    config = study_config()
    full = prepare_images(metadata, data, tmp_path / 'base_cache', 16, 16)
    cache = tmp_path / 'pilot_cache'
    complete = prepare_pilot_cache(data, cache, config, 32, 32, workers=2)
    roles = pd.read_csv(cache / 'roles_private.csv')
    mounted = load_pilot_cache(cache, full, roles, {'height': 32, 'width': 32}, complete['metadata_sha256'])
    assert set(mounted.patient_id) == set(roles.patient_id)
    assert len(mounted) == 128
    changed = full.copy()
    changed.loc[changed.image_id == mounted.iloc[0].image_id, 'decoded_pixel_sha256'] = 'CHANGED'
    with pytest.raises(ValueError, match='original pixels'):
        load_pilot_cache(cache, changed, roles, {'height': 32, 'width': 32}, complete['metadata_sha256'])
    wrong_roles = roles.copy()
    wrong_roles.loc[0, 'role'] = 'test'
    with pytest.raises(ValueError, match='assignments differ'):
        load_pilot_cache(cache, full, wrong_roles, {'height': 32, 'width': 32}, complete['metadata_sha256'])
