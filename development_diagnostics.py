"""Diagnose learning using reserved development patients only.

Tiny-set results measure memorization, not generalization. This standalone
launcher leaves the source of the completed pilot experiments unchanged.
"""
from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from mammo_reliability.audit import BREAST_KEY, environment, file_sha256, require, write_json
from mammo_reliability.model import BreastDataset, MammographyModel, collate_breasts, seed_everything
from mammo_reliability.pilot import run_pilot
from mammo_reliability.training import loader, source_fingerprint
from run_development import mounted_output


SETTINGS = {
    'purpose': 'development_only',
    'reason': 'Both initial pilots had weak ranking and high weighted loss.',
    'adaptive_development': True,
    'final_test_results_opened': False,
    'seed': 17,
    'tiny_positive_breasts': 8,
    'tiny_negative_breasts': 8,
    'tiny_optimizer_steps': 100,
    'tiny_learning_rate': 1e-5,
    'tiny_weight_decay': 0.0,
    'tiny_augmentation': False,
    'tiny_mixed_precision': False,
    'tiny_gate': {'maximum_final_to_initial_loss_ratio': 0.5, 'minimum_training_auc': 0.95},
    'linear_probe': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000, 'solver': 'lbfgs'},
    'stability_trial': 'stable_1024_lr1e5_acc32_v1',
    'stability_overrides': {'learning_rate': 1e-5, 'accumulation_steps': 32,
                            'epochs': 12, 'patience': 4},
    'interpretation': 'Joint optimization adjustment, not a single-factor ablation. No final method freeze.',
}


def select_tiny(fit, seed):
    ordered = fit.sort_values(BREAST_KEY)
    positives = ordered.loc[ordered.cancer == 1].drop_duplicates('patient_id').sample(
        SETTINGS['tiny_positive_breasts'], random_state=seed)
    negatives = ordered.loc[(ordered.cancer == 0) & ~ordered.patient_id.isin(positives.patient_id)]
    negatives = negatives.drop_duplicates('patient_id').sample(
        SETTINGS['tiny_negative_breasts'], random_state=seed + 1)
    tiny = pd.concat([positives, negatives]).sort_values(BREAST_KEY).reset_index(drop=True)
    require(tiny.patient_id.is_unique and len(tiny) == 16, 'Tiny fitting subset must use 16 distinct patients.')
    require(set(tiny.role) == {'fit'}, 'Tiny diagnostic must use fitting patients only.')
    return tiny


@torch.inference_mode()
def tiny_metrics(model, batches, device, step):
    model.eval()
    labels, logits = [], []
    for batch in batches:
        z, _ = model(batch['images'].to(device), batch['slots'].to(device), len(batch['labels']))
        labels.extend(batch['labels'].tolist())
        logits.extend(z.cpu().tolist())
    y, z = np.asarray(labels), np.asarray(logits)
    require(np.isfinite(z).all(), 'Nonfinite tiny-set predictions.')
    return {'optimizer_step': step, 'training_bce': float(np.mean(np.logaddexp(0, z) - y * z)),
            'training_auc': float(roc_auc_score(y, z)),
            'training_average_precision': float(average_precision_score(y, z)),
            'training_accuracy_at_zero_logit': float(np.mean((z >= 0) == y)),
            'logit_minimum': float(z.min()), 'logit_maximum': float(z.max())}


def memorization_diagnostic(manifest, fit, model_config, output, device):
    seed_everything(SETTINGS['seed'])
    tiny = select_tiny(fit, SETTINGS['seed'])
    tiny[BREAST_KEY + ['cancer', 'role']].to_csv(output / 'tiny_fit_breasts_private.csv', index=False)
    dataset = BreastDataset(manifest, tiny, augment=False,
                            height=model_config['height'], width=model_config['width'])
    # This small fixed set fits in host RAM, avoiding repeated file reads.
    batches = [collate_breasts([dataset[index]]) for index in range(len(dataset))]
    minima, maxima, deviations = [], [], []
    for batch in batches:
        pixels = batch['images']
        require(torch.isfinite(pixels).all().item(), 'Nonfinite model inputs.')
        minima.extend(pixels.amin(dim=(1, 2, 3)).tolist())
        maxima.extend(pixels.amax(dim=(1, 2, 3)).tolist())
        deviations.extend(pixels[:, 0].flatten(1).std(dim=1).tolist())
    require(min(deviations) > 0, 'Constant image in tiny diagnostic.')
    model = MammographyModel(pretrained=True).to(device)
    model.eval()
    amp_differences = []
    with torch.no_grad():
        for batch in batches:
            images, slots = batch['images'].to(device), batch['slots'].to(device)
            fp32, _ = model(images, slots, len(batch['labels']))
            with torch.autocast(device_type='cuda', enabled=True):
                mixed, _ = model(images, slots, len(batch['labels']))
            require(torch.isfinite(fp32).all().item() and torch.isfinite(mixed).all().item(),
                    'Nonfinite initial FP32 or mixed-precision output.')
            amp_differences.extend((fp32 - mixed).abs().cpu().tolist())
    numeric = {'normalized_input_minimum': min(minima), 'normalized_input_maximum': max(maxima),
               'minimum_per_image_first_channel_standard_deviation': min(deviations),
               'initial_fp32_amp_maximum_absolute_logit_difference': max(amp_differences),
               'breasts': len(tiny), 'images': len(minima), 'positive_breasts': int(tiny.cancer.sum())}
    write_json(output / 'input_numeric_checks.json', numeric)
    print('DIAGNOSTIC INPUT CHECKS', json.dumps(numeric), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['tiny_learning_rate'],
                                  weight_decay=SETTINGS['tiny_weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    history = [tiny_metrics(model, batches, device, 0)]
    print('MEMORIZATION DIAGNOSTIC', json.dumps(history[-1]), flush=True)
    largest_gradient_norm = 0.0
    started = time.monotonic()
    for step in range(1, SETTINGS['tiny_optimizer_steps'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            z, _ = model(batch['images'].to(device), batch['slots'].to(device), len(batch['labels']))
            loss = loss_fn(z, batch['labels'].to(device)) / len(batches)
            require(torch.isfinite(loss).item(), 'Nonfinite memorization loss.')
            loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        largest_gradient_norm = max(largest_gradient_norm, float(norm))
        optimizer.step()
        if step % 10 == 0:
            history.append(tiny_metrics(model, batches, device, step))
            pd.DataFrame(history).to_csv(output / 'memorization_learning_curve.csv', index=False)
            print('MEMORIZATION DIAGNOSTIC', json.dumps(history[-1]), flush=True)
    passed = (history[-1]['training_bce'] <= history[0]['training_bce'] *
              SETTINGS['tiny_gate']['maximum_final_to_initial_loss_ratio'] and
              history[-1]['training_auc'] >= SETTINGS['tiny_gate']['minimum_training_auc'])
    result = {'purpose': 'fitting-set engineering diagnostic, not generalization',
              'passed_declared_memorization_gate': bool(passed), 'initial': history[0], 'final': history[-1],
              'largest_preclip_gradient_norm': largest_gradient_norm, 'seconds': time.monotonic() - started,
              'selection_sha256': hashlib.sha256(tiny[BREAST_KEY + ['cancer']].to_csv(index=False).encode()).hexdigest()}
    write_json(output / 'memorization_result.json', result)
    print('MEMORIZATION RESULT', json.dumps(result), flush=True)
    del model, optimizer, batches
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def extract_features(manifest, breasts, model_config, device):
    seed_everything(SETTINGS['seed'])
    model = MammographyModel(pretrained=True).to(device)
    model.encoder.fc = nn.Identity()
    model.eval()
    dataset = BreastDataset(manifest, breasts, augment=False,
                            height=model_config['height'], width=model_config['width'])
    features, keys = [], []
    for i, batch in enumerate(loader(dataset, model_config, False)):
        slots = batch['slots'].to(device)
        values = model.encoder(batch['images'].to(device)).float()
        counts = torch.bincount(slots, minlength=2 * len(batch['labels']))
        require((counts > 0).all().item(), 'Feature pooling needs both views.')
        sums = torch.zeros((len(counts), values.shape[1]), device=device).index_add(0, slots, values)
        pooled = (sums / counts[:, None]).reshape(len(batch['labels']), 2, -1).mean(dim=1)
        require(torch.isfinite(pooled).all().item(), 'Nonfinite frozen ImageNet features.')
        features.append(pooled.cpu().numpy())
        keys.extend(zip(batch['patient_id'], batch['laterality']))
        if (i + 1) % 512 == 0:
            print('FROZEN FEATURE PROGRESS', json.dumps({'breasts_processed': len(keys), 'total': len(breasts)}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(features), pd.DataFrame(keys, columns=BREAST_KEY)


def linear_probe(manifest, labeled, model_config, output, device):
    started = time.monotonic()
    features, keys = extract_features(manifest, labeled, model_config, device)
    labels = keys.merge(labeled[BREAST_KEY + ['role', 'cancer']], on=BREAST_KEY, validate='one_to_one')
    require(keys.equals(labels[BREAST_KEY]), 'Feature-label ordering changed.')
    fit, tune = labels.role.eq('fit').to_numpy(), labels.role.eq('tune').to_numpy()
    require(set(labels.role) == {'fit', 'tune'}, 'Unexpected role in linear probe.')
    classifier = make_pipeline(StandardScaler(), LogisticRegression(**SETTINGS['linear_probe']))
    classifier.fit(features[fit], labels.loc[fit, 'cancer'])
    predictions = classifier.predict_proba(features)[:, 1]
    np.savez_compressed(output / 'imagenet_features_private.npz', features=features)
    labels['probability'] = predictions
    labels.to_csv(output / 'linear_probe_predictions_private.csv', index=False)
    scaler, lr = classifier.steps[0][1], classifier.steps[1][1]
    np.savez_compressed(output / 'linear_probe_parameters.npz', mean=scaler.mean_, scale=scaler.scale_,
                        coefficients=lr.coef_, intercept=lr.intercept_, classes=lr.classes_)
    result = {'purpose': 'development_only', 'features': int(features.shape[1]),
              'fit_breasts': int(fit.sum()), 'tune_breasts': int(tune.sum()),
              'tune_positive_breasts': int(labels.loc[tune, 'cancer'].sum()),
              'fit_average_precision': float(average_precision_score(labels.loc[fit, 'cancer'], predictions[fit])),
              'fit_roc_auc': float(roc_auc_score(labels.loc[fit, 'cancer'], predictions[fit])),
              'tune_average_precision': float(average_precision_score(labels.loc[tune, 'cancer'], predictions[tune])),
              'tune_roc_auc': float(roc_auc_score(labels.loc[tune, 'cancer'], predictions[tune])),
              'optimization_iterations': lr.n_iter_.tolist(), 'seconds': time.monotonic() - started}
    write_json(output / 'linear_probe_result.json', result)
    print('FROZEN FEATURE PROBE RESULT', json.dumps(result), flush=True)
    return result


def main():
    from mammo_reliability.audit import locate_dataset
    from mammo_reliability.cache_parts import assemble_parts
    from mammo_reliability.pipeline import initialize, require_pixel_audit
    root = Path('/kaggle/working/mammo_private_v0_2')
    config = json.loads(Path('config.json').read_text())
    data_root = locate_dataset()
    _, breasts, _, _ = initialize(data_root, root, config)
    parts = [mounted_output('03-rsna-lossless-cache-part-0', 'rsna_cache_part_0'),
             mounted_output('04-rsna-lossless-cache-part-1', 'rsna_cache_part_1')]
    expected = ['daf16f5f19161d06a00f1ccf56cdf3c3192712244ab81a0cba7366c80296fccc',
                '0cb17972ade50598829f8fa5c0969db9483e3816f4369934247242ba2dd2f87a']
    for path, digest in zip(parts, expected):
        require(file_sha256(path / 'manifest.csv') == digest, 'Base cache differs from the verified commit.')
    assemble_parts(parts, data_root, root / 'preprocessed', config['model']['height'], config['model']['width'])
    reference = require_pixel_audit(root, config)
    roles = pd.read_csv(root / 'pilot' / 'roles_private.csv')
    require((roles.cohort == 'pilot').all() and (roles.outer_fold == -1).all(), 'Nonpilot patient in diagnostics.')
    require(roles.patient_id.is_unique, 'Patient assigned twice.')
    labeled = breasts.merge(roles[['patient_id', 'role']], on='patient_id', validate='many_to_one')
    fit = labeled.loc[labeled.role == 'fit'].copy()
    require(set(fit.patient_id).isdisjoint(labeled.loc[labeled.role == 'tune', 'patient_id']), 'Patient leakage.')
    manifest = reference.loc[reference.patient_id.isin(roles.patient_id)].copy()
    require(len(labeled) == 4766 and int(fit.cancer.sum()) == 73 and
            int(labeled.loc[labeled.role == 'tune', 'cancer'].sum()) == 26, 'Diagnostic cohort changed.')
    require(torch.cuda.is_available(), 'Enable a GPU for diagnostics.')
    device = torch.device('cuda')
    output = root / 'pilot' / 'diagnostics_v1'
    output.mkdir(parents=True, exist_ok=True)
    plan = {**SETTINGS, 'source_sha256': source_fingerprint(),
            'diagnostic_script_sha256': file_sha256(Path(__file__)),
            'roles_file_sha256': file_sha256(root / 'pilot' / 'roles_private.csv'),
            'metadata_sha256': file_sha256(data_root / 'train.csv'), 'base_manifest_sha256': expected,
            'environment': environment(), 'gpu': torch.cuda.get_device_name(0)}
    write_json(output / 'diagnostic_plan.json', plan)
    print('DECLARED DEVELOPMENT DIAGNOSTIC PLAN', json.dumps(plan), flush=True)
    memory = memorization_diagnostic(manifest, fit, config['model'], output, device)
    probe = linear_probe(manifest, labeled, config['model'], output, device)
    if not memory['passed_declared_memorization_gate']:
        write_json(output / 'diagnostics_complete.json', {'memorization': memory, 'linear_probe': probe,
                   'stability_trial': 'not_run: tiny-set diagnostic needs investigation', 'final_test_results_opened': False})
        print('DIAGNOSTICS COMPLETE: memorization gate failed; investigate before a larger trial.', flush=True)
        return
    print('STARTING DEVELOPMENT TRIAL', SETTINGS['stability_trial'], flush=True)
    started = time.monotonic()
    stable = run_pilot(root, breasts, config, SETTINGS['stability_trial'], SETTINGS['stability_overrides'])
    stable['launcher_wall_seconds'] = time.monotonic() - started
    write_json(output / 'diagnostics_complete.json', {'memorization': memory, 'linear_probe': probe,
               'stability_trial': stable, 'final_test_results_opened': False})
    print('COMPLETED DEVELOPMENT TRIAL', json.dumps(stable), flush=True)
    print('DIAGNOSTICS COMPLETE. Final evaluation remains locked pending review.', flush=True)


if __name__ == '__main__':
    main()
