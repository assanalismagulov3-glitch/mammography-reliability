"""Join labels only after all five frozen inference shards are verified."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from fixed_gmic_evaluation import (IMAGE_KEY, PREDICTIONS, BREAST_KEY, pool_images,
    score_features, evaluate_scores, bootstrap_intervals)
from fixed_gmic_gate import verify_freeze
from fixed_gmic_prepare import sha, write_json, table_digest
from mammo_reliability.audit import require


def collect_shards(paths, bundle):
    require(len(paths) == 5, 'All five final shards are required before any outcome analysis.')
    manifest = json.loads((bundle/'cohort_manifest.json').read_text())
    freeze_hash = sha(bundle/'frozen_fixed_gmic.json')
    seen, frames, image_frames, environments, weights = set(), [], [], [], []
    for directory in paths:
        root = Path(directory)
        completed = json.loads((root/'completed.json').read_text())
        require(completed['status'] == 'label_free_final_shard_completed' and not completed['outcomes_evaluated'],
                'Expected completed label-free inference.')
        shard = completed['shard']
        require(shard in range(5) and shard not in seen, 'Duplicate/unexpected shard.')
        seen.add(shard)
        require(completed['freeze_sha256'] == freeze_hash, 'Shard used a different freeze.')
        for key, name in [('run_identity_sha256', 'run_identity.json'), ('images_sha256', 'image_identity_private.csv'),
                          ('predictions_sha256', 'image_predictions_private.csv'), ('features_sha256', 'breast_features_private.csv')]:
            require(completed[key] == sha(root/name), f'Shard file integrity failed: {name}.')
        identity = json.loads((root/'run_identity.json').read_text())
        require(identity['freeze_sha256'] == freeze_hash and identity['shard'] == shard, 'Run identity differs.')
        require(identity['fixed_models_sha256'] == sha(root/'fixed_models.json'), 'Model loading record changed.')
        require(json.loads((root/'reproduction_probe.json').read_text())['max_absolute_error'] <= 1e-6,
                'Development probe failed.')
        environments.append(identity['environment']); weights.append(identity['fixed_models_sha256'])
        images = pd.read_csv(root/'image_identity_private.csv')
        expected = manifest['shards'][shard]
        require(table_digest(images, IMAGE_KEY, ['image_id']) == expected['image_identity_sha256'], 'Shard image membership differs.')
        require(completed['images'] == expected['images'] and completed['breasts'] == expected['breasts'], 'Wrong shard size.')
        predicted = pd.read_csv(root/'image_predictions_private.csv')
        pooled = pool_images(images, predicted[['image_id']+PREDICTIONS])
        saved = pd.read_csv(root/'breast_features_private.csv').sort_values(BREAST_KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(pooled, saved, check_exact=False, rtol=1e-12, atol=1e-12)
        frames.append(pooled); image_frames.append(images)
    require(all(e == environments[0] for e in environments) and len(set(weights)) == 1,
            'Inference software/model environments differ between shards.')
    features = pd.concat(frames, ignore_index=True).sort_values(BREAST_KEY).reset_index(drop=True)
    images = pd.concat(image_frames, ignore_index=True)
    require(not features[BREAST_KEY].duplicated().any() and images.image_id.is_unique, 'Shards overlap.')
    require(len(features) == manifest['holdout_breasts'] and features.patient_id.nunique() == manifest['holdout_patients'],
            'Final breast/patient coverage incomplete.')
    require(len(images) == manifest['holdout_images'], 'Final image coverage incomplete.')
    return features, images


def main(bundle, shards, metadata, output):
    project = Path(__file__).resolve().parent
    bundle, output = Path(bundle), Path(output)
    verify_freeze(project, bundle)
    require(not (output/'final_results.json').exists(), 'Final results already exist; preserve the original evaluation.')
    config = json.loads((project/'fixed_gmic_final_config.json').read_text())
    features, images = collect_shards(shards, bundle)
    rule = json.loads((bundle/'calibration_rule_private.json').read_text())
    scored = score_features(features, rule)
    # Label boundary: every identity and prediction has already passed verification.
    require(sha(metadata) == config['provenance']['metadata_sha256'], 'Outcome metadata changed.')
    labels = pd.read_csv(metadata)
    selected = labels.loc[labels.image_id.isin(images.image_id)]
    for column in ['site_id', 'density', 'cancer']:
        require((selected.groupby(BREAST_KEY)[column].nunique(dropna=False) == 1).all(), f'Conflicting {column}.')
    outcomes = selected.groupby(BREAST_KEY)[['site_id', 'density', 'cancer']].first().reset_index()
    require(len(outcomes) == 19060 and int(outcomes.cancer.sum()) == 393, 'Unexpected final outcome coverage.')
    result = evaluate_scores(scored, outcomes, rule, config)
    dev = pd.read_csv(bundle/'development_features_private.csv')
    dev_outcomes = pd.read_csv(bundle/'development_outcomes_private.csv')
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output/'holdout_scored_private.csv', index=False)
    # Preserve the fixed point analysis before the longer bootstrap stage.
    write_json(output/'point_results.json', result)
    intervals, draws = bootstrap_intervals(dev, dev_outcomes, features, outcomes, rule, config)
    draws.to_csv(output/'bootstrap_draws.csv', index=False)
    result.update(status='final_fixed_model_evaluation_completed', bootstrap=intervals,
        freeze_sha256=sha(bundle/'frozen_fixed_gmic.json'), final_model_results_examined=True,
        no_human_reader_recovery_measured=True)
    write_json(output/'final_results.json', result)
    print('FINAL EVALUATION', json.dumps({'primary': result['primary'], 'bootstrap': intervals}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--shards', type=Path, nargs=5, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); main(a.bundle, a.shards, a.metadata, a.output)
