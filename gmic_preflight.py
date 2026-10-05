"""Reference reproduction and fit-only engineering check, no RSNA accuracy claim."""
from pathlib import Path
import json
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

from gmic_adapter import (UPSTREAM_COMMIT, HISTORICAL_FIXTURE_COMMIT, acquire_upstream, digest_file, decode_native,
                          infer_horizontal_flip, preprocess_array, filter_legacy_state)
from mammo_reliability.audit import environment, locate_dataset, require, write_json
from mammo_reliability.pipeline import initialize
from mammo_reliability.training import source_fingerprint

ROOT = Path('/kaggle/working/gmic_transfer_preflight')
SETTINGS = {
    'purpose': 'reference_reproduction_and_fit_only_technical_preflight',
    'upstream_commit': UPSTREAM_COMMIT,
    'models': [1, 2, 3, 4, 5], 'weights_frozen': True, 'precision': 'FP32',
    'upstream_source_modifications': [],
    'checkpoint_compatibility': 'Discard only unused shared_rep_filter.weight if present; strictly load active tensors.',
    'reference': 'README table with default run.sh MODEL_INDEX=1',
    'reference_max_absolute_error_gate': 0.005,
    'reference_gate_limit': 'Engineering tolerance; not an accuracy test or exact old-environment replication.',
    'rsna_sample': 'Pilot fit only; 2 images per site/view/laterality plus 1 per machine, deduplicated.',
    'sample_seed': 20260924,
    'orientation': 'provisional label-free outer-fifth intensity-mass heuristic; visual review required',
    'rsna_preprocessing': 'native DICOM presentation, 16-bit conversion, unmodified NYU crop/center/normalize',
    'upstream_crop_erosion_iterations': 100, 'upstream_crop_buffer_pixels': 50,
    'image_size': [2944, 1920], 'patches': 6, 'patch_size': [256, 256],
    'test_time_augmentation': 'one deterministic center, matching released inference code',
    'paper_replication': False,
    'rsna_tuning_predictions': False, 'final_test_results_opened': False,
    'automatic_full_cohort_run': False,
    'on_reference_mismatch': 'Retain current-fixture failure; require historical reference agreement and current CLI parity.',
    'gate_revision': 'Historical fixture must match README; current adapter must match original current CLI inputs/outputs.',
    'historical_fixture_commit': HISTORICAL_FIXTURE_COMMIT,
    'current_cli_max_rounding_difference': 0.00015,
}


def load_models(upstream):
    from src.modeling.gmic import GMIC
    from src.constants import PERCENT_T_DICT
    models = []
    loading_records = []
    for index in SETTINGS['models']:
        params = {'device_type': 'gpu', 'gpu_number': 0, 'cam_size': (46, 30),
                  'K': 6, 'crop_shape': (256, 256), 'post_processing_dim': 256,
                  'num_classes': 2, 'use_v1_global': False,
                  'percent_t': PERCENT_T_DICT[str(index)]}
        model = GMIC(params)
        state = torch.load(upstream/'models'/f'sample_model_{index}.p', map_location='cpu', weights_only=True)
        active_state, ignored = filter_legacy_state(state, model.state_dict().keys())
        model.load_state_dict(active_state, strict=True)
        loading_records.append({'model': index, 'ignored_unused_keys': ignored,
                                'ignored_tensor_shapes': {k:list(state[k].shape) for k in ignored},
                                'active_state_loaded_strictly': True})
        model.eval().to('cuda')
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
        print('GMIC MODEL LOADED', index, flush=True)
    write_json(ROOT/'checkpoint_loading.json', loading_records)
    print('GMIC CHECKPOINT LOADING', json.dumps(loading_records), flush=True)
    return models


def infer(models, array):
    x = torch.from_numpy(np.ascontiguousarray(array))[None, None].to('cuda')
    predictions = []
    with torch.inference_mode():
        for model in models:
            out = model(x)
            require(out.shape == (1, 2) and torch.isfinite(out).all().item(), 'Invalid model output.')
            require(torch.isfinite(model.saliency_map).all().item(), 'Invalid saliency map.')
            require(np.isfinite(model.patch_locations).all(), 'Invalid crop coordinates.')
            require(((out >= 0) & (out <= 1)).all().item(), 'Output outside [0,1].')
            predictions.append(out.cpu().numpy()[0].tolist())
    return np.asarray(predictions)


def reproduce_reference(upstream, models):
    expected = {}
    for line in (upstream/'README.md').read_text().splitlines():
        if re.match(r'^[0-3]_[LR]-(CC|MLO)\s*\|', line):
            fields = [x.strip() for x in line.split('|')]
            expected[fields[0]] = [float(fields[1]), float(fields[2])]
    require(len(expected) == 16, 'Reference table changed.')
    with (upstream/'sample_data/exam_list_before_cropping.pkl').open('rb') as stream:
        exams = pickle.load(stream)  # Pinned, blob-verified public author fixture.
    records = []
    for exam in exams:
        for view in ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']:
            name = exam[view][0]
            raw = np.asarray(Image.open(upstream/'sample_data/images'/f'{name}.png'))
            processed, _, details = preprocess_array(raw, view, exam['horizontal_flip'])
            predictions = infer(models, processed)
            records.append({'image': name, 'expected_default_model': expected[name],
                            'predictions_by_model': predictions.tolist(),
                            'model_1_max_absolute_error': float(np.max(np.abs(predictions[0]-expected[name]))),
                            'preprocessing': details})
            print('GMIC REFERENCE IMAGE', name, records[-1]['model_1_max_absolute_error'], flush=True)
    require({r['image'] for r in records} == set(expected), 'Reference images missing.')
    max_error = max(r['model_1_max_absolute_error'] for r in records)
    result = {'images': len(records), 'max_absolute_error': max_error,
              'tolerance': SETTINGS['reference_max_absolute_error_gate'],
              'passed': max_error <= SETTINGS['reference_max_absolute_error_gate'],
              'records': records}
    write_json(ROOT/'reference_reproduction.json', result)
    print('GMIC REFERENCE RESULT', json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)
    if not result['passed']:
        parity = diagnose_original_pipeline(upstream, result)
        historical = reproduce_historical_reference(models, expected)
        require(parity['all_preprocessed_arrays_identical'] and
                parity['max_adapter_vs_cli_error'] <= SETTINGS['current_cli_max_rounding_difference'],
                'Adapter does not reproduce the original current pipeline.')
        require(historical['passed'], 'Historical fixtures do not reproduce the published reference.')
        result['compatibility_passed'] = True
        result['historical_reference_max_error'] = historical['max_absolute_error']
        result['current_cli_max_error'] = parity['max_adapter_vs_cli_error']
        result['interpretation'] = 'Current fixture/table mismatch retained; original fixtures and current adapter parity verified.'
    else:
        result['compatibility_passed'] = True
    write_json(ROOT/'reference_reproduction.json', result)
    return result


def diagnose_original_pipeline(upstream, reference):
    """Compare against the author's complete released entry points, unchanged."""
    destination = ROOT/'official_cli_diagnostic'
    cropped = destination/'cropped'
    metadata = destination/'data.pkl'
    predictions = destination/'inference'
    destination.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(upstream)
    commands = [
        ['src/cropping/crop_mammogram.py', '--input-data-folder', str(upstream/'sample_data/images'),
         '--output-data-folder', str(cropped), '--exam-list-path', str(upstream/'sample_data/exam_list_before_cropping.pkl'),
         '--cropped-exam-list-path', str(destination/'cropped.pkl'), '--num-processes', '2'],
        ['src/optimal_centers/get_optimal_centers.py', '--cropped-exam-list-path', str(destination/'cropped.pkl'),
         '--data-prefix', str(cropped), '--output-exam-list-path', str(metadata), '--num-processes', '2'],
        ['src/scripts/run_model.py', '--model-path', str(upstream/'models'), '--data-path', str(metadata),
         '--image-path', str(cropped), '--segmentation-path', str(destination/'no_segmentations'),
         '--output-path', str(predictions), '--device-type', 'gpu', '--gpu-number', '0', '--model-index', '1'],
    ]
    print('GMIC ORIGINAL CLI DIAGNOSTIC STARTING', flush=True)
    for command in commands:
        subprocess.run([sys.executable, '-u', *command], cwd=upstream, env=env, check=True)
    from src.data_loading import loading
    original = pd.read_csv(predictions/'predictions.csv').set_index('image_index')
    with metadata.open('rb') as stream: exams = pickle.load(stream)
    original_inputs = {}
    for exam in exams:
        for view in ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']:
            name = exam[view][0]
            image = loading.load_image(str(cropped/f'{name}.png'), view, exam['horizontal_flip'])
            array = loading.process_image(image, view, exam['best_center'][view][0])
            import hashlib
            original_inputs[name] = {'processed_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                                     'best_center': [int(x) for x in exam['best_center'][view][0]]}
    rows = []
    for record in reference['records']:
        name = record['image']
        scores = original.loc[name, ['benign_pred', 'malignant_pred']].to_numpy(float)
        row = {'image': name, 'original_cli_predictions': scores.tolist(),
               'adapter_predictions': record['predictions_by_model'][0],
               'published_predictions': record['expected_default_model'],
               'adapter_vs_cli_max_error': float(np.max(np.abs(scores-record['predictions_by_model'][0]))),
               'cli_vs_published_max_error': float(np.max(np.abs(scores-record['expected_default_model']))),
               'preprocessed_arrays_identical': original_inputs[name]['processed_sha256'] == record['preprocessing']['processed_sha256'],
               'original_center': original_inputs[name]['best_center'],
               'adapter_center': record['preprocessing']['best_center']}
        rows.append(row)
        print('GMIC ORIGINAL CLI COMPARISON', json.dumps(row), flush=True)
    result = {'status': 'diagnosis_only_original_reference_gate_unchanged', 'records': rows,
              'all_preprocessed_arrays_identical': all(x['preprocessed_arrays_identical'] for x in rows),
              'max_adapter_vs_cli_error': max(x['adapter_vs_cli_max_error'] for x in rows),
              'max_cli_vs_published_error': max(x['cli_vs_published_max_error'] for x in rows),
              'rsna_inference_allowed': False}
    write_json(ROOT/'original_cli_comparison.json', result)
    print('GMIC ORIGINAL CLI SUMMARY', json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)
    return result


def reproduce_historical_reference(models, current_expected):
    """Use exact already-windowed author fixtures for the original README table.

    Git history shows their replacement in December 2020. The initial author
    inference code only converted these arrays to float and standardized them;
    it explicitly disabled flipping. The checkpoint blobs are unchanged.
    """
    from src.data_loading.loading import standard_normalize_single_image
    destination = ROOT/'historical_fixtures'
    files = acquire_upstream('gmic_historical_manifest.json', destination)
    write_json(ROOT/'verified_historical_files.json', files)
    expected = {}
    for line in (destination/'README.md').read_text().splitlines():
        if re.match(r'^[0-3]_[LR]-(CC|MLO)\s*\|', line):
            parts = [x.strip() for x in line.split('|')]
            expected[parts[0]] = [float(parts[1]), float(parts[2])]
    require(expected == current_expected, 'The historical and current reference tables differ.')
    records = []
    for name in expected:
        array = np.array(Image.open(destination/'sample_data/cropped_images'/f'{name}.png'), dtype=np.float32)
        require(array.shape == (2944, 1920), 'Historical author fixture has unexpected dimensions.')
        standard_normalize_single_image(array)
        predicted = infer([models[0]], array)[0]
        record = {'image': name, 'expected': expected[name], 'prediction': predicted.tolist(),
                  'max_absolute_error': float(np.max(np.abs(predicted-expected[name])))}
        records.append(record)
        print('GMIC HISTORICAL REFERENCE IMAGE', json.dumps(record), flush=True)
    max_error = max(x['max_absolute_error'] for x in records)
    result = {'images': len(records), 'fixture_commit': HISTORICAL_FIXTURE_COMMIT,
              'max_absolute_error': max_error, 'tolerance': SETTINGS['reference_max_absolute_error_gate'],
              'passed': max_error <= SETTINGS['reference_max_absolute_error_gate'], 'records': records}
    write_json(ROOT/'historical_reference_reproduction.json', result)
    print('GMIC HISTORICAL REFERENCE SUMMARY', json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)
    return result


def check_rsna(models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data_root = locate_dataset()
    config = json.loads(Path('config.json').read_text())
    metadata_root = ROOT/'cohort_identity'
    initialize(data_root, metadata_root, config)
    roles_path = metadata_root/'pilot/roles_private.csv'
    require(digest_file(roles_path) == 'fc0308c63300be6849c7ac9b830d09480c2b62d64df4f1d036bab12548a7e15f',
            'Reserved development roles changed.')
    roles = pd.read_csv(roles_path)
    require(roles.patient_id.is_unique and set(roles.cohort) == {'pilot'} and (roles.outer_fold == -1).all(),
            'Invalid reserved pilot roles.')
    fit_ids = set(roles.loc[roles.role.eq('fit'), 'patient_id'])
    images = pd.read_csv(data_root/'train.csv')
    fit = images.loc[images.patient_id.isin(fit_ids) & images.view.isin(['CC', 'MLO'])]
    ordered = fit.sample(frac=1, random_state=SETTINGS['sample_seed'])
    sample = pd.concat([ordered.groupby(['site_id', 'view', 'laterality'], sort=True).head(2),
                        ordered.groupby('machine_id', sort=True).head(1)]).drop_duplicates('image_id')
    require(set(sample.patient_id).issubset(fit_ids), 'Engineering sample includes nonfit patients.')
    sample.to_csv(ROOT/'engineering_sample_private.csv', index=False)
    records, thumbnails = [], []
    for count, row in enumerate(sample.itertuples(index=False), 1):
        started = time.monotonic()
        raw, header = decode_native(data_root/'train_images'/str(row.patient_id)/f'{row.image_id}.dcm')
        view = f'{row.laterality}-{row.view}'
        flip, orientation = infer_horizontal_flip(raw, view)
        processed, window, details = preprocess_array(raw, view, flip)
        predictions = infer(models, processed)
        record = {'sample': count, 'site_id': int(row.site_id), 'machine_id': int(row.machine_id),
                  'view': view, **header, **orientation, **details,
                  'elapsed_seconds': time.monotonic()-started}
        records.append(record)
        # No RSNA scores or cancer labels in aggregate logs or contact-sheet titles.
        np.save(ROOT/f'sample_{count:02d}_predictions_private.npy', predictions)
        thumbnails.append((np.asarray(Image.fromarray(raw).resize((300, 460))),
                           np.asarray(Image.fromarray(window).resize((300, 460)))))
        print('GMIC RSNA TECHNICAL IMAGE', json.dumps(record), flush=True)
    for start in range(0, len(records), 4):
        fig, axes = plt.subplots(4, 2, figsize=(8, 16))
        for offset, pair in enumerate(axes):
            idx = start+offset
            for ax in pair: ax.axis('off')
            if idx >= len(records): continue
            item = records[idx]
            for col, image in enumerate(thumbnails[idx]):
                pair[col].imshow(image, cmap='gray')
                pair[col].set_title(f"Sample {idx+1} site {item['site_id']} {item['view']} | "
                                    + ('native presentation' if col == 0 else 'GMIC window'))
        fig.suptitle('Private technical orientation/crop review; no diagnostic interpretation')
        fig.tight_layout()
        fig.savefig(ROOT/f'contact_sheet_{start//4+1:02d}_private.png', dpi=100)
        plt.close(fig)
    result = {'images': len(records), 'sampling_partition': 'pilot_fit', 'records': records,
              'visual_review_status': 'pending', 'full_cohort_transfer_evaluation_authorized_by_gate': False}
    write_json(ROOT/'rsna_technical_summary.json', result)
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available(), 'Kaggle GPU required.')
    require(source_fingerprint() == 'f4178761b59233ffe1048218d234624c22764c8129d11c48fe30821cbe079a7c',
            'Original model package changed.')
    torch.manual_seed(17)
    np.random.seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    declared = {**SETTINGS, 'environment': environment(),
                'script_sha256': digest_file(__file__),
                'adapter_sha256': digest_file('gmic_adapter.py'),
                'manifest_sha256': digest_file('gmic_upstream_manifest.json')}
    write_json(ROOT/'declared_plan.json', declared)
    print('GMIC DECLARED PLAN', json.dumps(declared), flush=True)
    upstream = ROOT/'upstream'
    files = acquire_upstream('gmic_upstream_manifest.json', upstream)
    write_json(ROOT/'verified_upstream_files.json', files)
    sys.path.insert(0, str(upstream))
    models = load_models(upstream)
    reference = reproduce_reference(upstream, models)
    technical = check_rsna(models)
    result = {'status': 'numeric_preflight_completed_visual_review_pending',
              'reference_compatibility_passed': reference['compatibility_passed'],
              'current_fixture_vs_readme_passed': reference['passed'],
              'current_fixture_vs_readme_max_error': reference['max_absolute_error'],
              'historical_reference_max_error': reference.get('historical_reference_max_error'),
              'current_cli_max_error': reference.get('current_cli_max_error'),
              'rsna_fit_images_checked': technical['images'], 'final_test_results_opened': False,
              'rsna_accuracy_estimated': False, 'visual_review_required': True,
              'upstream_commit': UPSTREAM_COMMIT,
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}
    write_json(ROOT/'completed.json', result)
    print('GMIC PREFLIGHT COMPLETED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
