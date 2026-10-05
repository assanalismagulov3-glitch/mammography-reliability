"""One class-balanced development trial, with original source left unchanged."""
from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from balanced_sampling import balanced_epoch_indices, objective_scale
from mammo_reliability.audit import BREAST_KEY, environment, file_sha256, require, write_json
from mammo_reliability.model import BreastDataset, MammographyModel, collate_breasts, seed_everything
from mammo_reliability.training import predict, save_checkpoint, source_fingerprint
from run_development import mounted_output


SETTINGS = {
    'purpose': 'development_only', 'trial': 'balanced_1024_lr1e5_acc32_v1',
    'seed': 17, 'learning_rate': 1e-5, 'weight_decay': 1e-4,
    'group_size': 32, 'positive_per_group': 16, 'negative_per_group': 16,
    'sampling': 'uniform within each class, with replacement',
    'epochs': 12, 'patience': 4, 'gradient_clip_norm': 5.0,
    'mixed_precision': True, 'augmentation': 'original horizontal flips',
    'loss': 'unweighted BCE times 2*n_negative/n_fit; same expected objective as prior weighted BCE',
    'checkpoint_selection': 'maximum tuning average precision from probabilities',
    'adaptive_development': True, 'final_test_results_opened': False,
    'reference_notebook_version': 352083246,
    'interpretation': 'One seed and an adaptively reused tuning cohort; no confirmatory claim.',
}
EXPECTED_SOURCE = 'f4178761b59233ffe1048218d234624c22764c8129d11c48fe30821cbe079a7c'
EXPECTED_CHECKPOINT = 'c8a2e1ad8effc5fb9530fb9590184fad8ea83acab9263bca0b55d73787283371'
EXPECTED_ROLES = 'fc0308c63300be6849c7ac9b830d09480c2b62d64df4f1d036bab12548a7e15f'


def score_frame(frame):
    z = (frame.z_cc.to_numpy() + frame.z_mlo.to_numpy()) / 2
    y = frame.cancer.to_numpy()
    p = expit(z)
    result = {'breasts': len(frame), 'positive_breasts': int(y.sum()),
              'average_precision': float(average_precision_score(y, p)),
              'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
              'bce': float(np.mean(np.logaddexp(0, z) - y*z)),
              'class_logit_quantiles': {}, 'class_mean_bce': {}}
    for label in (0, 1):
        selected = z[y == label]
        if len(selected):
            result['class_logit_quantiles'][str(label)] = np.quantile(selected, [0, .1, .5, .9, 1]).tolist()
            result['class_mean_bce'][str(label)] = float(np.mean(np.logaddexp(0, selected) - label*selected))
    return result


def evaluate(model, manifest, breasts, config, device, path):
    data = BreastDataset(manifest, breasts, False, config['height'], config['width'])
    frame = predict(model, data, config, device).merge(
        breasts[BREAST_KEY + ['cancer', 'site_id']], on=BREAST_KEY, validate='one_to_one')
    require(len(frame) == len(breasts), 'Evaluation lost breast records.')
    frame.to_csv(path, index=False)
    summary = score_frame(frame)
    summary['by_site'] = {str(site): score_frame(group) for site, group in frame.groupby('site_id')}
    return summary


def train_balanced(manifest, fit, tune, config, device, output, signature):
    seed = SETTINGS['seed']
    seed_everything(seed)
    model = MammographyModel(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['learning_rate'],
                                 weight_decay=SETTINGS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=.5)
    scaler = torch.amp.GradScaler('cuda', enabled=SETTINGS['mixed_precision'])
    dataset = BreastDataset(manifest, fit, True, config['height'], config['width'])
    group_size = SETTINGS['group_size']
    factor = objective_scale(dataset.breasts.cancer)
    history, start_epoch, best_ap, stale = [], 0, -1., 0
    last = output / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        require(state['signature'] == signature, 'Changed trial cannot resume.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history, start_epoch, best_ap, stale = state['history'], state['next_epoch'], state['best_ap'], state['stale']
    for epoch in range(start_epoch, SETTINGS['epochs']):
        if stale >= SETTINGS['patience']:
            break
        started = time.monotonic()
        epoch_seed = seed + 1000 * epoch
        seed_everything(epoch_seed)
        indices = balanced_epoch_indices(dataset.breasts.cancer, group_size, epoch_seed)
        np.save(output / f'epoch_{epoch+1:02d}_fitting_indices_private.npy', indices)
        sampled_labels = dataset.breasts.cancer.to_numpy()[indices]
        require((sampled_labels.reshape(-1, group_size).sum(axis=1) == group_size // 2).all(),
                'An update does not contain equal classes.')
        batches = DataLoader(dataset, batch_size=1, sampler=indices.tolist(), num_workers=config['num_workers'],
                             collate_fn=collate_breasts, pin_memory=True, persistent_workers=False,
                             generator=torch.Generator().manual_seed(epoch_seed))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        class_loss, class_count, group_positive = np.zeros(2), np.zeros(2, dtype=int), 0
        norms, clipped, skipped, nonfinite = [], 0, 0, 0
        for i, batch in enumerate(batches):
            labels = batch['labels'].to(device)
            with torch.autocast('cuda', enabled=SETTINGS['mixed_precision']):
                z, _ = model(batch['images'].to(device, non_blocking=True), batch['slots'].to(device), 1)
                loss = nn.functional.binary_cross_entropy_with_logits(z, labels)
            require(torch.isfinite(loss).item(), 'Nonfinite training loss.')
            scaler.scale(loss * factor / group_size).backward()
            label = int(batch['labels'].item())
            class_loss[label] += float(loss.detach())
            class_count[label] += 1
            group_positive += label
            if (i+1) % group_size == 0:
                require(group_positive == group_size // 2, 'Loader changed accumulation order.')
                group_positive = 0
                scaler.unscale_(optimizer)
                norm = float(nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['gradient_clip_norm']))
                if np.isfinite(norm):
                    norms.append(norm)
                    clipped += int(norm > SETTINGS['gradient_clip_norm'])
                else:
                    nonfinite += 1
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                was_skipped = scaler.get_scale() < scale_before
                skipped += int(was_skipped)
                require(np.isfinite(norm) or was_skipped, 'Nonfinite gradient was not skipped.')
                optimizer.zero_grad(set_to_none=True)
            if (i+1) % 512 == 0:
                print('BALANCED PROGRESS', json.dumps({'epoch': epoch+1, 'breast_draws': i+1,
                                                      'total_draws': len(indices)}), flush=True)
        training_seconds = time.monotonic() - started
        tuning = evaluate(model, manifest, tune, config, device, output / 'latest_tune_predictions_private.csv')
        ap = tuning['average_precision']
        record = {'epoch': epoch+1, 'breast_draws': len(indices),
                  'unique_positive_breasts_seen': int(np.unique(indices[sampled_labels == 1]).size),
                  'unique_negative_breasts_seen': int(np.unique(indices[sampled_labels == 0]).size),
                  'optimizer_updates_attempted': len(indices)//group_size,
                  'optimizer_updates_skipped': skipped, 'nonfinite_gradient_updates': nonfinite,
                  'gradient_clipped_updates': clipped,
                  'finite_preclip_norm_quantiles': np.quantile(norms, [0, .5, .9, 1]).tolist() if norms else None,
                  'amp_scale': scaler.get_scale(),
                  'positive_mean_training_bce': float(class_loss[1]/class_count[1]),
                  'negative_mean_training_bce': float(class_loss[0]/class_count[0]),
                  'scaled_balanced_training_loss': float(class_loss.sum()/class_count.sum()*factor),
                  'tune_average_precision': ap, 'tune_roc_auc': tuning['roc_auc'],
                  'tune_class_logit_quantiles': tuning['class_logit_quantiles'],
                  'learning_rate': optimizer.param_groups[0]['lr'],
                  'training_seconds': training_seconds, 'epoch_seconds': time.monotonic()-started,
                  'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(device)/1024**3}
        history.append(record)
        print('BALANCED EPOCH', json.dumps(record), flush=True)
        if ap > best_ap:
            best_ap, stale = ap, 0
            save_checkpoint(output / 'best.pt', {'state_dict': model.state_dict(), 'epoch': epoch+1, 'seed': seed})
        else:
            stale += 1
        scheduler.step(ap)
        write_json(output / 'learning_curve.json', history)
        save_checkpoint(last, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                              'signature': signature, 'history': history, 'next_epoch': epoch+1,
                              'best_ap': best_ap, 'stale': stale})
    require((output / 'best.pt').exists(), 'No valid selected checkpoint.')
    selected = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(selected['state_dict'])
    fit_result = evaluate(model, manifest, fit, config, device, output / 'best_fit_predictions_private.csv')
    tune_result = evaluate(model, manifest, tune, config, device, output / 'best_tune_predictions_private.csv')
    require(np.isclose(tune_result['average_precision'], best_ap, atol=1e-7, rtol=0), 'Selected checkpoint did not reproduce.')
    result = {'status': 'training_completed', 'trial': SETTINGS['trial'], 'purpose': 'development_only',
              'best_epoch': selected['epoch'], 'epochs_run': len(history),
              'checkpoint_sha256': file_sha256(output / 'best.pt'), 'fit': fit_result, 'tune': tune_result,
              'final_test_results_opened': False}
    write_json(output / 'completed.json', result)
    print('BALANCED COMPLETED', json.dumps(result), flush=True)
    return result


def main():
    from mammo_reliability.audit import locate_dataset
    from mammo_reliability.cache_parts import assemble_parts
    from mammo_reliability.pipeline import initialize, require_pixel_audit
    require(source_fingerprint() == EXPECTED_SOURCE, 'Original audited package changed.')
    root = Path('/kaggle/working/mammo_private_v0_2')
    config = json.loads(Path('config.json').read_text())
    data_root = locate_dataset()
    _, breasts, _, _ = initialize(data_root, root, config)
    parts = [mounted_output('03-rsna-lossless-cache-part-0', 'rsna_cache_part_0'),
             mounted_output('04-rsna-lossless-cache-part-1', 'rsna_cache_part_1')]
    expected = ['daf16f5f19161d06a00f1ccf56cdf3c3192712244ab81a0cba7366c80296fccc',
                '0cb17972ade50598829f8fa5c0969db9483e3816f4369934247242ba2dd2f87a']
    for part, digest in zip(parts, expected):
        require(file_sha256(part / 'manifest.csv') == digest, 'Base manifest changed.')
    assemble_parts(parts, data_root, root/'preprocessed', 1024, 768)
    reference = require_pixel_audit(root, config)
    roles = pd.read_csv(root/'pilot'/'roles_private.csv')
    require(file_sha256(root/'pilot'/'roles_private.csv') == EXPECTED_ROLES, 'Development patients changed.')
    require(roles.patient_id.is_unique and set(roles.cohort) == {'pilot'} and (roles.outer_fold == -1).all(),
            'Nonpilot or duplicated patient.')
    labeled = breasts.merge(roles[['patient_id', 'role']], on='patient_id', validate='many_to_one')
    require(set(labeled.role) == {'fit', 'tune'}, 'Unexpected role.')
    fit, tune = [labeled.loc[labeled.role == role].copy() for role in ('fit', 'tune')]
    require(set(fit.patient_id).isdisjoint(tune.patient_id), 'Fitting/tuning patient leakage.')
    require(len(fit) == 3574 and fit.cancer.sum() == 73 and len(tune) == 1192 and tune.cancer.sum() == 26,
            'Unexpected development cohort.')
    manifest = reference.loc[reference.patient_id.isin(roles.patient_id)].copy()
    require(torch.cuda.is_available(), 'GPU required.')
    device = torch.device('cuda')
    previous = mounted_output('07-rsna-development-diagnostics', 'mammo_private_v0_2')
    reference_checkpoint = previous/'pilot'/'trials'/'stable_1024_lr1e5_acc32_v1'/'seed_17'/'best.pt'
    require(file_sha256(reference_checkpoint) == EXPECTED_CHECKPOINT, 'Reference checkpoint changed.')
    output = root/'pilot'/'sampling_trials'/SETTINGS['trial']
    output.mkdir(parents=True, exist_ok=True)
    fit[BREAST_KEY + ['cancer']].reset_index(drop=True).to_csv(output/'fit_index_mapping_private.csv', index=False)
    model_config = {**config['model'], 'learning_rate': 1e-5, 'accumulation_steps': 32, 'epochs': 12, 'patience': 4}
    signature = {**SETTINGS, 'source_sha256': source_fingerprint(),
                 'script_sha256': file_sha256(Path(__file__)), 'sampler_sha256': file_sha256(Path('balanced_sampling.py')),
                 'roles_sha256': EXPECTED_ROLES, 'metadata_sha256': file_sha256(data_root/'train.csv'),
                 'cache_manifests': expected, 'reference_checkpoint_sha256': EXPECTED_CHECKPOINT,
                 'model_config': model_config, 'environment': environment(),
                 'draws_per_epoch': int(np.ceil(len(fit)/32)*32),
                 'objective_scale': objective_scale(fit.cancer), 'gpu': torch.cuda.get_device_name(0)}
    signature_path = output/'declared_plan.json'
    if signature_path.exists():
        require(json.loads(signature_path.read_text()) == signature, 'Use a new trial for changed settings.')
    write_json(signature_path, signature)
    print('DECLARED BALANCED PLAN', json.dumps(signature), flush=True)
    if (output/'completed.json').exists():
        completed = json.loads((output/'completed.json').read_text())
        require(file_sha256(output/'best.pt') == completed['checkpoint_sha256'], 'Completed checkpoint changed.')
        print('BALANCED COMPLETED', json.dumps(completed), flush=True)
        return
    model = MammographyModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(reference_checkpoint, map_location=device, weights_only=True)['state_dict'])
    reference_tune = evaluate(model, manifest, tune, model_config, device, output/'reference_tune_predictions_private.csv')
    require(np.isclose(reference_tune['average_precision'], .03093795403496575, atol=1e-7, rtol=0),
            'Reference inference did not reproduce.')
    reference_fit = evaluate(model, manifest, fit, model_config, device, output/'reference_fit_predictions_private.csv')
    reference_result = {'fit': reference_fit, 'tune': reference_tune}
    write_json(output/'reference_evaluation.json', reference_result)
    print('REFERENCE CHECKPOINT EVALUATION', json.dumps(reference_result), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    train_balanced(manifest, fit, tune, model_config, device, output, signature)
    print('SAMPLING EXPERIMENT COMPLETE. Final evaluation remains locked.', flush=True)


if __name__ == '__main__':
    main()
