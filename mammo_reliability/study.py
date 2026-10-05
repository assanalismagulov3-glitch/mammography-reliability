"""Keep outcome-guided development patients outside every final test fold."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .audit import (BREAST_KEY, file_sha256, make_patient_folds,
                    partition_outer_fold, require, write_json)


def make_study_patients(breasts: pd.DataFrame, n_splits: int = 5,
                        seed: int = 20260909, pilot_fraction: float = 0.20,
                        pilot_tune_fraction: float = 0.25) -> pd.DataFrame:
    require(0 < pilot_fraction < 0.5 and 0 < pilot_tune_fraction < 0.5,
            "Invalid development cohort fractions.")
    require(not breasts[BREAST_KEY].duplicated().any(), "Expected unique breasts.")
    require(breasts['has_both_views'].all(), "Resolve missing standard views first.")
    patients = breasts.groupby('patient_id', as_index=False).agg(
        site_id=('site_id', 'first'), patient_cancer=('cancer', 'max'))
    strata = patients.site_id.astype(str) + ':' + patients.patient_cancer.astype(str)
    evaluation_idx, pilot_idx = train_test_split(
        patients.index, test_size=pilot_fraction, stratify=strata, random_state=seed + 7000)
    pilot = patients.loc[pilot_idx]
    pilot_strata = pilot.site_id.astype(str) + ':' + pilot.patient_cancer.astype(str)
    pilot_fit_idx, pilot_tune_idx = train_test_split(
        pilot.index, test_size=pilot_tune_fraction, stratify=pilot_strata, random_state=seed + 7001)
    evaluation_breasts = breasts.loc[breasts.patient_id.isin(patients.loc[evaluation_idx, 'patient_id'])]
    evaluation_folds = make_patient_folds(evaluation_breasts, n_splits, seed)
    fold_map = evaluation_folds.set_index('patient_id').outer_fold
    patients['cohort'] = 'evaluation'
    patients.loc[pilot_idx, 'cohort'] = 'pilot'
    patients['outer_fold'] = patients.patient_id.map(fold_map).fillna(-1).astype(int)
    patients['pilot_role'] = 'unused'
    patients.loc[pilot_fit_idx, 'pilot_role'] = 'fit'
    patients.loc[pilot_tune_idx, 'pilot_role'] = 'tune'
    require((patients.loc[patients.cohort == 'pilot', 'outer_fold'] == -1).all(),
            "Development patients received a test fold.")
    require((patients.loc[patients.cohort == 'evaluation', 'outer_fold'] >= 0).all(),
            "An evaluation patient lacks a fold.")
    return patients.sort_values('patient_id').reset_index(drop=True)


def pilot_roles(patients: pd.DataFrame) -> pd.DataFrame:
    result = patients.loc[patients.cohort == 'pilot'].copy()
    require(len(result) > 0 and result.patient_id.is_unique, "Invalid pilot patients.")
    require(set(result.pilot_role.unique()) == {'fit', 'tune'}, "Pilot role mismatch.")
    result['role'] = result.pilot_role
    return result.reset_index(drop=True)


def study_outer_roles(patients: pd.DataFrame, fold: int, seed: int = 20260909) -> pd.DataFrame:
    require(patients.patient_id.is_unique, "Patient identifiers repeat.")
    evaluation = patients.loc[patients.cohort == 'evaluation'].reset_index(drop=True)
    roles = partition_outer_fold(evaluation, fold, seed)
    # After global settings are fixed, pilot patients contribute to final model
    # fitting. They never contribute to tune, calibration, or test endpoints.
    pilot = pilot_roles(patients)
    pilot['role'] = 'fit'
    result = pd.concat([roles, pilot], ignore_index=True).sort_values('patient_id').reset_index(drop=True)
    require(set(result.patient_id) == set(patients.patient_id), "Partition omitted patients.")
    require((result.loc[result.cohort == 'pilot', 'role'] == 'fit').all(),
            "Pilot patients leaked into an evaluation role.")
    return result


def partition_identity(run_root: Path, config: dict) -> dict:
    names = ['experiment.json', 'patients_private.csv', 'standard_images_private.csv', 'pilot/roles_private.csv']
    names += [f'fold_{fold}/roles_private.csv' for fold in range(config['n_outer_folds'])]
    return {name: file_sha256(run_root / name) for name in names}


def require_frozen_protocol(run_root: Path, config: dict) -> dict:
    from .training import source_fingerprint
    path = run_root / 'frozen_protocol.json'
    require(path.is_file(), "Final training/evaluation requires a reviewed, frozen protocol; run development pilots first.")
    frozen = json.loads(path.read_text())
    require(frozen['config'] == config, "Configuration changed after protocol freeze.")
    require(frozen['source_sha256'] == source_fingerprint(), "Source changed after protocol freeze.")
    require(frozen['partition_files'] == partition_identity(run_root, config),
            "Patient assignments changed after protocol freeze.")
    require(frozen['development_review_sha256'] == file_sha256(run_root / 'development_review.json'),
            "Development review changed after protocol freeze.")
    return frozen


def freeze_protocol(run_root: Path, config: dict, review_path: Path) -> dict:
    """Record a concrete development review before final-fold training starts."""
    from .training import source_fingerprint
    require(not any(run_root.glob('fold_*/test*predictions*')) and
            not any(run_root.glob('fold_*/test_scored_private.csv')),
            "Held-out predictions already exist; a retrospective freeze is invalid.")
    require(not any(run_root.glob('fold_*/seed_*/signature.json')),
            "Freeze the protocol before starting final-fold training.")
    review = json.loads(review_path.read_text())
    for field in ['decision', 'resolution_assessment', 'literature_assessment', 'limitations']:
        require(isinstance(review.get(field), str) and bool(review[field].strip()),
                f"Development review needs {field}.")
    require(isinstance(review.get('completed_pilot_directories'), list) and
            len(review['completed_pilot_directories']) > 0, "Review must reference completed pilot runs.")
    evidence = []
    expected_patients = pd.read_csv(run_root / 'pilot' / 'roles_private.csv')
    metadata_sha256 = json.loads((run_root / 'experiment.json').read_text())['metadata_sha256']
    for directory in review['completed_pilot_directories']:
        directory = Path(directory)
        details = json.loads((directory / 'pilot_trial.json').read_text())
        require(details['purpose'] == 'development_only', "Expected an isolated development trial.")
        require(details['metadata_sha256'] == metadata_sha256, 'Reviewed pilot uses different source data.')
        roles = pd.read_csv(directory / 'roles_private.csv')
        require(roles.sort_values('patient_id').reset_index(drop=True).equals(
            expected_patients.sort_values('patient_id').reset_index(drop=True)),
            "Reviewed pilot uses different patients or roles.")
        completed = list(directory.glob('seed_*/completed.json'))
        require(len(completed) > 0, "A referenced pilot has not completed training.")
        require(all(json.loads(p.read_text()).get('status') == 'training_completed' for p in completed),
                'A referenced pilot did not complete successfully.')
        evidence.append({'trial': str(directory), 'trial_sha256': file_sha256(directory / 'pilot_trial.json'),
                         'completed_files': {str(p): file_sha256(p) for p in completed}})
    destination = run_root / 'development_review.json'
    if destination.exists():
        require(json.loads(destination.read_text()) == review, "A different development review already exists.")
    else:
        write_json(destination, review)
    record = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
              'source_sha256': source_fingerprint(), 'partition_files': partition_identity(run_root, config),
              'development_review_sha256': file_sha256(destination), 'pilot_evidence': evidence}
    target = run_root / 'frozen_protocol.json'
    require(not target.exists(), "A frozen protocol already exists; do not overwrite it.")
    write_json(target, record)
    return record
