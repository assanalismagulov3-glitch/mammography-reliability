"""One label-free final inference shard. No cancer performance is calculated."""
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from fixed_gmic_gate import verify_freeze
from fixed_gmic_prepare import cohort_record, sha, write_json, table_digest
from fixed_gmic_evaluation import IMAGE_KEY, PREDICTIONS, pool_images
from mammo_reliability.audit import locate_dataset, require, environment


def mounted_final_images(base, bundle, config, shard, data_root):
    require(sha(data_root/'train.csv') == config['provenance']['metadata_sha256'], 'Dataset metadata changed.')
    patients = pd.read_csv(base/'cohort_identity/patients_private.csv')
    images = pd.read_csv(base/'cohort_identity/standard_images_private.csv')
    roles = pd.read_csv(base/'cohort_identity/pilot/roles_private.csv')
    require(sha(base/'cohort_identity/pilot/roles_private.csv') == config['provenance']['development_roles_sha256'],
            'Original development roles changed.')
    actual = cohort_record(patients, images, config, roles)
    expected = json.loads((bundle/'cohort_manifest.json').read_text())
    require(actual == expected, 'Mounted cohort differs from frozen cohort.')
    # Read only the four label-free columns for inference identity verification.
    official = pd.read_csv(data_root/'train.csv', usecols=IMAGE_KEY)
    official = official.loc[official.view.isin(['CC', 'MLO'])]
    require(table_digest(official, IMAGE_KEY, ['image_id']) == expected['standard_image_table_sha256'],
            'Original metadata/image correspondence changed.')
    patient_ids = patients.loc[patients.cohort.eq('evaluation') & patients.outer_fold.eq(shard), 'patient_id']
    selected = images.loc[images.patient_id.isin(patient_ids), IMAGE_KEY]
    selected = selected.sort_values(['patient_id', 'laterality', 'view', 'image_id']).reset_index(drop=True)
    require(len(selected) == expected['shards'][shard]['images'], 'Incomplete shard.')
    return selected


def checked_resume(path, expected_ids, fields, signature):
    path = Path(path)
    require(json.loads((path/'run_identity.json').read_text()) == signature, 'Incompatible resume source.')
    table = pd.read_csv(path/'image_predictions_private.csv')
    require(list(table) == fields, 'Incomplete resume schema.')
    require(table.image_id.is_unique and len(table) <= len(expected_ids), 'Duplicate/extra resume images.')
    require(np.array_equal(table.image_id.to_numpy(), np.asarray(expected_ids)[:len(table)]), 'Resume is not the expected prefix.')
    require(not table.isna().any().any(), 'Incomplete resume row; explicit repair required.')
    p = table[PREDICTIONS].to_numpy(float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid resumed predictions.')
    return table


def main(bundle, shard, output, verification, resume=None):
    # GPU-only imports are delayed so identity/recovery functions can be tested locally.
    import torch
    from gmic_adapter import decode_native, digest_file
    from gmic_fullfield import fullfield_array
    from gmic_preflight import infer
    from gmic_fullfield_preflight import verified_preflight, load_fixed_models
    from gmic_pilot import verified_geometry
    project = Path(__file__).resolve().parent
    bundle, output = Path(bundle), Path(output)
    require(shard in range(5), 'Unknown shard.')
    verify_freeze(project, bundle)
    verified = json.loads(Path(verification).read_text())
    require(verified['status'] == 'fixed_gmic_cpu_verification_passed' and verified['tests_run'] >= 18,
            'Complete the separate Kaggle CPU verification first.')
    require(verified['freeze_sha256'] == sha(bundle/'frozen_fixed_gmic.json') and verified['cohort_matched']
            and verified['calibration_rule_unchanged'], 'CPU verification does not match this freeze.')
    output.mkdir(parents=True, exist_ok=True)
    require(not (output/'image_predictions_private.csv').exists(), 'Use an explicit separate verified resume source.')
    require(torch.cuda.is_available(), 'GPU required.')
    torch.manual_seed(17); np.random.seed(17)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    config = json.loads((project/'fixed_gmic_final_config.json').read_text())
    base, upstream = verified_preflight()
    geometry, review = verified_geometry()
    data_root = locate_dataset()
    images = mounted_final_images(base, bundle, config, shard, data_root)
    images.to_csv(output/'image_identity_private.csv', index=False)
    models = load_fixed_models(upstream, output)
    # Repeat the original development probe before accessing held-out pixels.
    sample = next(pd.read_csv(base/'engineering_sample_private.csv').iloc[[10]].itertuples(index=False))
    roles = pd.read_csv(base/'cohort_identity/pilot/roles_private.csv')
    require(sample.patient_id in set(roles.loc[roles.role.eq('fit'), 'patient_id']), 'Probe must remain development-only.')
    raw, _ = decode_native(data_root/'train_images'/str(sample.patient_id)/f'{sample.image_id}.dcm')
    array, _, _ = fullfield_array(raw, f'{sample.laterality}-{sample.view}')
    error = float(np.max(abs(infer(models, array)-np.load(geometry/'sample_11_predictions_private.npy', allow_pickle=False))))
    require(error <= 1e-6, 'Original development probe did not reproduce.')
    write_json(output/'reproduction_probe.json', {'max_absolute_error': error, 'tolerance': 1e-6})
    signature = {'shard': shard, 'freeze_sha256': sha(bundle/'frozen_fixed_gmic.json'),
        'image_identity_sha256': table_digest(images, IMAGE_KEY, ['image_id']),
        'fixed_models_sha256': sha(output/'fixed_models.json'), 'environment': environment()}
    write_json(output/'run_identity.json', signature)
    fields = ['image_id'] + PREDICTIONS + ['scale', 'edge_mass_contrast', 'actual_flip_applied',
                                         'raw_height', 'raw_width', 'native_file_sha256']
    done = 0
    if resume is not None:
        previous = checked_resume(resume, images.image_id.to_numpy(), fields, signature)
        done = len(previous)
        shutil.copyfile(Path(resume)/'image_predictions_private.csv', output/'image_predictions_private.csv')
    started = time.monotonic()
    print('FINAL SHARD GATES PASSED', json.dumps({'shard': shard, 'images': len(images), 'resumed': done,
        'fixture_max_error': error, 'freeze_sha256': signature['freeze_sha256'], 'outcomes_evaluated': False}), flush=True)
    with (output/'image_predictions_private.csv').open('a' if resume else 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not resume: writer.writeheader()
        for index, row in enumerate(images.iloc[done:].itertuples(index=False), done+1):
            path = data_root/'train_images'/str(row.patient_id)/f'{row.image_id}.dcm'
            raw, _ = decode_native(path)
            array, _, shape = fullfield_array(raw, f'{row.laterality}-{row.view}')
            probabilities = infer(models, array)
            require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(),
                    'Nonfinite/out-of-range predictions; stopping.')
            writer.writerow({'image_id': row.image_id, **dict(zip(PREDICTIONS, probabilities.ravel().tolist())),
                'scale': shape['scale'], 'edge_mass_contrast': shape['edge_mass_contrast'],
                'actual_flip_applied': shape['actual_flip_applied'], 'raw_height': raw.shape[0],
                'raw_width': raw.shape[1], 'native_file_sha256': digest_file(path)})
            if index % 100 == 0 or index == len(images):
                stream.flush(); os.fsync(stream.fileno())
                progress = {'shard': shard, 'images_completed': index, 'images_total': len(images),
                    'elapsed_seconds': time.monotonic()-started, 'outcomes_evaluated': False}
                write_json(output/'progress.json', progress)
                print('FINAL SHARD PROGRESS', json.dumps(progress), flush=True)
    predictions = pd.read_csv(output/'image_predictions_private.csv')
    features = pool_images(images, predictions[['image_id']+PREDICTIONS])
    features.to_csv(output/'breast_features_private.csv', index=False)
    completed = {'status': 'label_free_final_shard_completed', 'shard': shard,
        'images': len(images), 'breasts': len(features), 'outcomes_evaluated': False,
        'freeze_sha256': signature['freeze_sha256'], 'run_identity_sha256': sha(output/'run_identity.json'),
        'images_sha256': sha(output/'image_identity_private.csv'),
        'predictions_sha256': sha(output/'image_predictions_private.csv'),
        'features_sha256': sha(output/'breast_features_private.csv')}
    write_json(output/'completed.json', completed)
    print('FINAL SHARD COMPLETED', json.dumps(completed), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    a = parser.parse_args(); main(a.bundle, a.shard, a.output, a.verification, a.resume)
