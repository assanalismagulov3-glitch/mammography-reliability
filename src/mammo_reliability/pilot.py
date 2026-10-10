"""Development-only image preparation and model trials with explicit provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .audit import (STANDARD_VIEWS, audit_metadata, file_sha256, locate_dataset,
                    require, write_json)
from .cache_parts import _prepare_shard, preprocessing_fingerprint
from .imaging import check_cross_patient_duplicates
from .study import make_study_patients, pilot_roles


def prepare_pilot_cache(data_root: Path, output: Path, config: dict,
                        height: int = 1536, width: int = 1152, workers: int = 4) -> dict:
    images = pd.read_csv(data_root / 'train.csv')
    breasts, _, _ = audit_metadata(images)
    patients = make_study_patients(breasts, config['n_outer_folds'], config['split_seed'],
                                   config['pilot_fraction'], config['pilot_tune_fraction'])
    roles = pilot_roles(patients)
    selected = images.loc[images.patient_id.isin(roles.patient_id) & images.view.isin(STANDARD_VIEWS)].copy()
    require(workers > 0 and min(height, width) > 0, 'Invalid pilot cache settings.')
    identity = {'purpose': 'development_only', 'metadata_sha256': file_sha256(data_root / 'train.csv'),
                'height': height, 'width': width, 'workers': workers, 'n_images': len(selected),
                'image_processing_source_sha256': preprocessing_fingerprint(),
                'roles_sha256': hashlib.sha256(roles.to_csv(index=False).encode()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'pilot_cache_identity.json'
    if path.exists():
        require(json.loads(path.read_text()) == identity, 'Pilot cache settings changed.')
    else:
        write_json(path, identity)
    roles.to_csv(output / 'roles_private.csv', index=False)
    roles.groupby(['role', 'site_id', 'patient_cancer']).size().rename('patients').to_csv(
        output / 'partition_counts.csv')
    selected['worker'] = selected.patient_id % workers
    tasks = [(g.drop(columns='worker'), data_root, output / f'shard_{worker}', height, width)
             for worker, g in selected.groupby('worker')]
    print(f'Preparing development-only {height} x {width} cache: {len(selected)} images, {len(roles)} patients.', flush=True)
    if workers == 1:
        frames = [_prepare_shard(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_prepare_shard, tasks))
    manifest = pd.concat(frames, ignore_index=True).sort_values('image_id').reset_index(drop=True)
    require(manifest.image_id.is_unique and set(manifest.image_id) == set(selected.image_id), 'Pilot cache coverage mismatch.')
    check_cross_patient_duplicates(manifest)
    manifest['cache_relative_path'] = manifest.cache_path.map(lambda p: str(Path(p).relative_to(output)))
    manifest.to_csv(output / 'manifest.csv', index=False)
    result = {**identity, 'manifest_sha256': file_sha256(output / 'manifest.csv')}
    write_json(output / 'pilot_cache_complete.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def load_pilot_cache(root: Path, reference: pd.DataFrame, roles: pd.DataFrame,
                     model_config: dict, metadata_sha256: str) -> pd.DataFrame:
    complete = json.loads((root / 'pilot_cache_complete.json').read_text())
    require(complete['purpose'] == 'development_only' and complete['metadata_sha256'] == metadata_sha256,
            'Pilot cache uses another dataset or purpose.')
    require(complete['height'] == model_config['height'] and complete['width'] == model_config['width'],
            'Pilot cache resolution differs from the trial.')
    require(complete['image_processing_source_sha256'] == preprocessing_fingerprint(), 'Pilot preprocessing changed.')
    require(complete['manifest_sha256'] == file_sha256(root / 'manifest.csv'), 'Pilot cache manifest changed.')
    require(complete['roles_sha256'] == hashlib.sha256(roles.to_csv(index=False).encode()).hexdigest(),
            'Pilot cache patient assignments differ from this study.')
    manifest = pd.read_csv(root / 'manifest.csv')
    expected = reference.loc[reference.patient_id.isin(roles.patient_id)]
    columns = ['image_id', 'patient_id', 'laterality', 'view', 'decoded_pixel_sha256']
    require(manifest.image_id.is_unique and len(manifest) == complete['n_images'] and
            manifest[columns].sort_values('image_id').reset_index(drop=True).equals(
                expected[columns].sort_values('image_id').reset_index(drop=True)),
            'Pilot image coverage or original pixels differ from the full audit.')
    paths = []
    for relative in manifest.cache_relative_path:
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Invalid pilot cache path.')
        paths.append(str(root / path))
        require(Path(paths[-1]).is_file(), 'A pilot cached image is missing.')
    manifest['cache_path'] = paths
    return manifest


def run_pilot(run_root: Path, breasts: pd.DataFrame, config: dict, trial: str,
              model_overrides: dict | None = None, cache_root: Path | None = None) -> dict:
    from .pipeline import require_pixel_audit
    from .training import train_member
    require(re.fullmatch(r'[a-zA-Z0-9_-]+', trial) is not None, 'Use a simple trial name.')
    require(not (run_root / 'frozen_protocol.json').exists(), 'Global development is closed after the protocol freeze.')
    reference = require_pixel_audit(run_root, config)
    roles = pd.read_csv(run_root / 'pilot' / 'roles_private.csv')
    require((roles.cohort == 'pilot').all() and (roles.outer_fold == -1).all(), 'Nonpilot patients entered development.')
    model_config = {**config['model'], **(model_overrides or {})}
    require(set(model_config) == set(config['model']), 'Unknown pilot model option.')
    metadata_sha256 = json.loads((run_root / 'experiment.json').read_text())['metadata_sha256']
    if cache_root is not None:
        manifest = load_pilot_cache(cache_root, reference, roles, model_config, metadata_sha256)
    else:
        require((model_config['height'], model_config['width']) ==
                (config['model']['height'], config['model']['width']), 'Attach a matching higher-resolution pilot cache.')
        manifest = reference.loc[reference.patient_id.isin(roles.patient_id)].copy()
    subset = breasts.loc[breasts.patient_id.isin(roles.patient_id)]
    folder = run_root / 'pilot' / 'trials' / trial
    folder.mkdir(parents=True, exist_ok=True)
    trial_record = {'purpose': 'development_only', 'trial': trial, 'model': model_config,
                    'seed': config['ensemble_seeds'][0], 'metadata_sha256': metadata_sha256,
                    'role_sha256': hashlib.sha256(roles.to_csv(index=False).encode()).hexdigest(),
                    'final_test_results_opened': False}
    identity_path = folder / 'pilot_trial.json'
    if identity_path.exists():
        require(json.loads(identity_path.read_text()) == trial_record, 'Use a new name for changed pilot settings.')
    else:
        write_json(identity_path, trial_record)
    roles.to_csv(folder / 'roles_private.csv', index=False)
    seed = config['ensemble_seeds'][0]
    result = train_member(manifest, subset, roles, model_config, seed, folder / f'seed_{seed}', metadata_sha256)
    return {**result, 'purpose': 'development_only', 'trial': trial,
            'interpretation': 'Model development; these patients are excluded from final test endpoints.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--height', type=int, default=1536)
    parser.add_argument('--width', type=int, default=1152)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    prepare_pilot_cache(args.data_root or locate_dataset(), args.output,
                        json.loads(args.config.read_text()), args.height, args.width, args.workers)


if __name__ == '__main__':
    main()
