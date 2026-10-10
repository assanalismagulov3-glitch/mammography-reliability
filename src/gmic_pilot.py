"""Fixed external mammography ensemble; reserved development patients only."""
from pathlib import Path
import csv
import json
import os
import shutil
import time
import numpy as np
import pandas as pd
import torch

from gmic_adapter import decode_native, digest_file
from gmic_fullfield import fullfield_array
from gmic_preflight import infer
from gmic_fullfield_preflight import verified_preflight, load_fixed_models
from gmic_pilot_analysis import aggregate_predictions, summarize_development, PREDICTION_COLUMNS
from mammo_reliability.audit import locate_dataset, require, write_json, environment
from mammo_reliability.training import source_fingerprint

OUTPUT = Path('/kaggle/working/gmic_reserved_pilot')
EXPECTED_ROLES = 'fc0308c63300be6849c7ac9b830d09480c2b62d64df4f1d036bab12548a7e15f'
EXPECTED_METADATA = '35408731b67b87272f85874ee3c4c9d5cea7a2c4fbcaeac3fbe415ba1a5817c1'


def verified_geometry():
    candidates = [Path('/kaggle/input/notebooks/aseqwss/10-rsna-gmic-fullfield-check/gmic_fullfield_preflight'),
                  Path('/kaggle/input/10-rsna-gmic-fullfield-check/gmic_fullfield_preflight')]
    existing = [p for p in candidates if p.is_dir()]
    require(len(existing) == 1, 'Attach notebook 10 with the reviewed geometry audit.')
    root = existing[0]
    review = json.loads(Path('gmic_visual_review.json').read_text())
    require(review['engineering_visual_review_passed'] is True and review['images_reviewed'] == 20,
            'Complete technical visual review first.')
    require(digest_file(root/'declared_plan.json') == review['declared_plan_sha256'], 'Unreviewed audit version.')
    complete = json.loads((root/'completed.json').read_text())
    require(complete['images'] == 20 and not complete['final_test_results_opened'], 'Incomplete image audit.')
    declaration = json.loads((root/'declared_plan.json').read_text())
    for key, name in [('fullfield_sha256', 'gmic_fullfield.py'), ('adapter_sha256', 'gmic_adapter.py')]:
        require(declaration[key] == digest_file(name), f'Unreviewed processing changes: {name}')
    return root, review


def main():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # A new run must never accidentally combine partial outputs with new code.
    require(not (OUTPUT/'image_predictions_private.csv').exists(), 'Existing predictions require explicit verified recovery.')
    require(torch.cuda.is_available(), 'GPU required.')
    require(source_fingerprint() == 'f4178761b59233ffe1048218d234624c22764c8129d11c48fe30821cbe079a7c',
            'Original model package changed.')
    torch.manual_seed(17)
    np.random.seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, upstream = verified_preflight()
    geometry_root, review = verified_geometry()
    roles_path = base/'cohort_identity/pilot/roles_private.csv'
    require(digest_file(roles_path) == EXPECTED_ROLES, 'Patient roles changed.')
    roles = pd.read_csv(roles_path)
    require(roles.patient_id.is_unique and set(roles.cohort) == {'pilot'} and (roles.outer_fold == -1).all(),
            'A final evaluation patient entered development.')
    require(set(roles.role) == {'fit', 'tune'} and len(roles) == 2383, 'Wrong development roles.')
    data_root = locate_dataset()
    require(digest_file(data_root/'train.csv') == EXPECTED_METADATA, 'Metadata changed.')
    all_images = pd.read_csv(data_root/'train.csv')
    images = all_images.loc[all_images.patient_id.isin(roles.patient_id) & all_images.view.isin(['CC', 'MLO'])]
    images = images.merge(roles[['patient_id', 'role']], on='patient_id', validate='many_to_one')
    images = images.sort_values(['patient_id', 'laterality', 'view', 'image_id']).reset_index(drop=True)
    require(len(images) == 10881 and images.image_id.is_unique, 'Wrong pilot image coverage.')
    require(set(images.patient_id) == set(roles.patient_id), 'Missing development patients.')
    require(len(images[['patient_id', 'laterality']].drop_duplicates()) == 4766, 'Wrong breast count.')
    # The labels below are used only in the final descriptive analysis.
    images.to_csv(OUTPUT/'images_private.csv', index=False)
    shutil.copyfile(roles_path, OUTPUT/'roles_private.csv')
    shutil.copyfile('gmic_visual_review.json', OUTPUT/'visual_review.json')
    declaration = {
        'purpose': 'reserved_development_transfer_feasibility', 'rsna_model_training': False,
        'patients': 2383, 'breasts': 4766, 'images': 10881,
        'roles_sha256': EXPECTED_ROLES, 'metadata_sha256': EXPECTED_METADATA,
        'upstream_commit': '594029efe6184bcca3a151fa1dd46c5fd620cb25',
        'models': [1, 2, 3, 4, 5], 'class_order': ['benign', 'malignant'],
        'inference': 'FP32, one deterministic full-field image, fixed external weights, no TTA',
        'pooling': 'arithmetic malignant probability mean within repeated view; equal CC/MLO mean; equal five-model mean',
        'uncertainty_descriptors': 'absolute ensemble CC-MLO difference; population variance of five breast probabilities',
        'analysis': 'AP and ROC-AUC for all fixed members and ensemble by pilot role; ensemble also by site',
        'threshold_selection': False, 'calibration': False, 'density_effect_testing': False,
        'final_test_results_opened': False, 'model_or_member_selection_in_job': False,
        'orientation_low_contrast_flag': .2, 'orientation_flags_exclude_images': False,
        'environment': environment(), 'gpu': torch.cuda.get_device_name(0),
        'source_sha256': {p: digest_file(p) for p in [
            'gmic_pilot.py', 'gmic_pilot_analysis.py', 'gmic_adapter.py', 'gmic_preflight.py',
            'gmic_fullfield.py', 'gmic_fullfield_preflight.py', 'gmic_visual_review.json']},
        'geometry_review': review,
        'limitations': ['Adapted image scale can reduce small-feature visibility.',
                       'Heuristic orientation is not validated for every image.',
                       'Five published models were selected in external hyperparameter searches, not a seed-only ensemble.',
                       'Different reported source institutions do not prove patient-level independence.',
                       'Adaptive development estimates are not final research results.']}
    write_json(OUTPUT/'declared_plan.json', declaration)
    print('GMIC PILOT DECLARED PLAN', json.dumps(declaration), flush=True)
    models = load_fixed_models(upstream, OUTPUT)
    # Reproduce every model on the image that exposed the earlier crop failure.
    # A pandas Series has a .view method; use a named row for the view column.
    sample = next(pd.read_csv(base/'engineering_sample_private.csv').iloc[[10]].itertuples(index=False))
    require(sample.patient_id in set(roles.loc[roles.role.eq('fit'), 'patient_id']), 'Probe is not a fitting image.')
    raw, _ = decode_native(data_root/'train_images'/str(sample.patient_id)/f'{sample.image_id}.dcm')
    array, _, _ = fullfield_array(raw, f'{sample.laterality}-{sample.view}')
    probe = infer(models, array)
    expected = np.load(geometry_root/'sample_11_predictions_private.npy', allow_pickle=False)
    error = float(np.max(abs(probe-expected)))
    require(error <= 1e-6, 'The reviewed image predictions did not reproduce.')
    write_json(OUTPUT/'reproduction_probe.json', {'sample': 11, 'max_absolute_error': error, 'tolerance': 1e-6})
    print('GMIC PILOT REPRODUCTION', json.dumps({'sample': 11, 'max_absolute_error': error}), flush=True)
    fields = ['image_id'] + PREDICTION_COLUMNS + ['scale', 'edge_mass_contrast', 'actual_flip_applied',
                                                'raw_height', 'raw_width', 'native_file_sha256']
    with (OUTPUT/'image_predictions_private.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for count, row in enumerate(images.itertuples(index=False), 1):
            path = data_root/'train_images'/str(row.patient_id)/f'{row.image_id}.dcm'
            raw, _ = decode_native(path)
            array, _, geometry = fullfield_array(raw, f'{row.laterality}-{row.view}')
            predictions = infer(models, array)
            record = {'image_id': row.image_id, **dict(zip(PREDICTION_COLUMNS, predictions.ravel().tolist())),
                      'scale': geometry['scale'], 'edge_mass_contrast': geometry['edge_mass_contrast'],
                      'actual_flip_applied': geometry['actual_flip_applied'],
                      'raw_height': raw.shape[0], 'raw_width': raw.shape[1], 'native_file_sha256': digest_file(path)}
            writer.writerow(record)
            if count % 100 == 0 or count == len(images):
                stream.flush(); os.fsync(stream.fileno())
                progress = {'images_completed': count, 'images_total': len(images),
                            'elapsed_seconds': time.monotonic()-started, 'final_test_results_opened': False}
                write_json(OUTPUT/'progress.json', progress)
                print('GMIC PILOT PROGRESS', json.dumps(progress), flush=True)
    predictions = pd.read_csv(OUTPUT/'image_predictions_private.csv')
    breasts = aggregate_predictions(images, predictions)
    require(len(breasts) == 4766 and int(breasts.cancer.sum()) == 99, 'Wrong outcome coverage.')
    require(breasts.groupby('role').size().to_dict() == {'fit': 3574, 'tune': 1192}, 'Wrong role coverage.')
    breasts.to_csv(OUTPUT/'breast_predictions_private.csv', index=False)
    summary = summarize_development(breasts)
    summary['geometry'] = {'minimum_scale': float(predictions.scale.min()),
                           'maximum_scale': float(predictions.scale.max()),
                           'low_orientation_contrast_images': int((predictions.edge_mass_contrast < .2).sum()),
                           'minimum_edge_mass_contrast': float(predictions.edge_mass_contrast.min())}
    write_json(OUTPUT/'development_summary.json', summary)
    for record in summary['metrics']: print('GMIC PILOT METRICS', json.dumps(record), flush=True)
    result = {'status': 'reserved_development_inference_completed', 'images': len(images), 'breasts': len(breasts),
              'elapsed_seconds': time.monotonic()-started, 'final_test_results_opened': False,
              'declared_plan_sha256': digest_file(OUTPUT/'declared_plan.json'),
              'image_predictions_sha256': digest_file(OUTPUT/'image_predictions_private.csv'),
              'breast_predictions_sha256': digest_file(OUTPUT/'breast_predictions_private.csv'),
              'summary_sha256': digest_file(OUTPUT/'development_summary.json')}
    write_json(OUTPUT/'completed.json', result)
    print('GMIC PILOT COMPLETED', json.dumps(result), flush=True)


if __name__ == '__main__': main()
