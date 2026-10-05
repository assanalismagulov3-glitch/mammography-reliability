"""Verify preserved development results, fit the final rule, bind cohort identities."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from fixed_gmic_evaluation import (BREAST_KEY, FEATURES, IMAGE_KEY, PREDICTIONS,
    METHODS, pool_images, fit_rule, score_features, select_review, evaluate_scores)
from gmic_operating_point import load_verified, fit_development_rule, apply_development_rule
from mammo_reliability.audit import require
from mammo_reliability.evaluation import review_at_budget, classification_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table_digest(frame, columns, order):
    """Canonical semantic identity; independent of browser CRLF/LF presentation."""
    text = frame[columns].sort_values(order).to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(text.encode()).hexdigest()


def cohort_record(patients, images, config, development_roles):
    patient_columns = ['patient_id', 'site_id', 'patient_cancer', 'cohort', 'outer_fold', 'pilot_role']
    require(set(patients) == set(patient_columns) and patients.patient_id.is_unique, 'Invalid original assignments.')
    require(set(images) == set(IMAGE_KEY) and images.image_id.is_unique, 'Invalid original image list.')
    require(len(patients) == 11913 and len(images) == 54668, 'Unexpected original coverage.')
    require(not patients.isna().any().any() and not images.isna().any().any(), 'Missing original identities.')
    require(set(patients.patient_id) == set(images.patient_id), 'Patient/image membership differs.')
    require(images.view.isin(['CC', 'MLO']).all(), 'Unexpected views.')
    require(images.laterality.isin(['L', 'R']).all(), 'Invalid breast side.')
    require((images.groupby(BREAST_KEY).view.nunique() == 2).all(), 'Missing CC or MLO.')
    require((images.groupby('patient_id').laterality.nunique() == 2).all(), 'Missing breast group.')
    dev = patients.loc[patients.cohort.eq('pilot')].copy()
    holdout = patients.loc[patients.cohort.eq('evaluation')].copy()
    require(len(dev) == 2383 and len(holdout) == 9530, 'Cohort roles changed.')
    require(set(dev.patient_id) == set(development_roles.patient_id), 'Reserved development membership changed.')
    expected = development_roles[patient_columns].sort_values('patient_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(dev[patient_columns].sort_values('patient_id').reset_index(drop=True), expected)
    require((dev.outer_fold == -1).all() and set(holdout.outer_fold) == set(range(5)), 'Invalid original folds.')
    require(holdout.pilot_role.eq('unused').all(), 'Unexpected holdout roles.')
    shards = []
    for fold, group in holdout.groupby('outer_fold'):
        image_group = images.loc[images.patient_id.isin(group.patient_id)]
        shards.append({'shard': int(fold), 'patients': len(group),
            'breasts': len(image_group[BREAST_KEY].drop_duplicates()), 'images': len(image_group),
            'patient_identity_sha256': table_digest(group, ['patient_id'], ['patient_id']),
            'image_identity_sha256': table_digest(image_group, IMAGE_KEY, ['image_id'])})
    return {'source_script_version_id': 352355871, 'metadata_sha256': config['provenance']['metadata_sha256'],
        'patient_table_sha256': table_digest(patients, patient_columns, ['patient_id']),
        'standard_image_table_sha256': table_digest(images, IMAGE_KEY, ['image_id']),
        'canonicalization': 'selected columns, ascending IDs, pandas CSV, LF line endings',
        'development_patients': len(dev), 'holdout_patients': len(holdout),
        'holdout_breasts': sum(s['breasts'] for s in shards), 'holdout_images': sum(s['images'] for s in shards),
        'shards': shards, 'review_budget_allocation': 'global, never per shard'}


def development_inputs(root):
    breasts, provenance = load_verified(root)
    images = pd.read_csv(root/'images_private.csv')
    predictions = pd.read_csv(root/'image_predictions_private.csv')
    pooled = pool_images(images[IMAGE_KEY], predictions[['image_id']+PREDICTIONS])
    original = breasts.sort_values(BREAST_KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(pooled, original[BREAST_KEY+FEATURES], check_exact=False, atol=1e-12, rtol=1e-12)
    for column in ['site_id', 'density', 'cancer']:
        require((images.groupby(BREAST_KEY)[column].nunique(dropna=False) == 1).all(), f'Conflicting {column}.')
    outcomes = images.groupby(BREAST_KEY)[['site_id', 'density', 'cancer']].first().reset_index()
    require(len(outcomes) == len(pooled), 'Missing outcome rows.')
    return breasts, pooled, outcomes, provenance


def verify_development_parity(breasts, config):
    # Use historical fit/tune roles, solely to verify equality with completed analysis.
    fit = breasts.loc[breasts.role.eq('fit')].reset_index(drop=True)
    tune = breasts.loc[breasts.role.eq('tune')].reset_index(drop=True)
    outcome = lambda f: f[BREAST_KEY+['site_id','cancer']].assign(density=None)[BREAST_KEY+['site_id','density','cancer']]
    cal, reference, threshold, raw = fit_development_rule(fit)
    old = apply_development_rule(cal, reference, tune, threshold)
    rule = fit_rule(fit[BREAST_KEY+FEATURES], outcome(fit))
    new = score_features(tune[BREAST_KEY+FEATURES], rule)
    require(abs(rule['raw_threshold']-raw) < 1e-14, 'Original threshold differs.')
    np.testing.assert_allclose(new[METHODS], old[METHODS], atol=1e-12, rtol=1e-12)
    result = evaluate_scores(new, outcome(tune), rule, config)
    expected = classification_metrics(tune.cancer, old.p_calibrated, threshold)
    require(result['classification'][0] == expected | {'group': 'pooled'}, 'Confusion/ranking metrics differ.')
    comparisons = 0
    for budget in config['review']['report_budgets']:
        for method in METHODS:
            old_result = review_at_budget(old, method, budget)
            actual = next(r for r in result['review'] if r['budget'] == budget and r['method'] == method)
            for key in ['negative_breasts', 'reviewed_breasts', 'false_negatives', 'false_negatives_flagged',
                        'fn_capture_fraction', 'review_yield', 'actual_review_fraction']:
                require(actual[key] == old_result[key], f'Original review mismatch: {key}.')
            comparisons += 1
    return {'pooling_reproduced': True, 'historical_tune_classification_reproduced': True,
        'review_comparisons_reproduced': comparisons, 'all_six_scores_reproduced': True,
        'historical_raw_cutoff': raw, 'historical_tune_false_negatives': expected['fn']}


def prepare(development_root, original_patients, original_images, original_experiment, output, config_path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    require(not (output/'frozen_fixed_gmic.json').exists(), 'Do not overwrite a frozen experiment.')
    config = json.loads(Path(config_path).read_text())
    experiment = json.loads(Path(original_experiment).read_text())
    require(experiment['metadata_sha256'] == config['provenance']['metadata_sha256'], 'Different original dataset.')
    breasts, features, outcomes, provenance = development_inputs(Path(development_root))
    parity = verify_development_parity(breasts, config)
    roles = pd.read_csv(Path(development_root)/'roles_private.csv')
    cohort = cohort_record(pd.read_csv(original_patients), pd.read_csv(original_images), config, roles)
    rule = fit_rule(features, outcomes, target=config['calibration']['empirical_sensitivity_target'])
    scored = score_features(features, rule, allow_calibration=True)
    # Only calibration fit diagnostics; do not label these independent validation.
    diagnostic = classification_metrics(outcomes.cancer, scored.p_calibrated, rule['calibrated_threshold'])
    result = {'status': 'calibration_prepared_not_frozen', 'final_test_results_opened': False,
        'development_provenance_verified': True, 'parity': parity,
        'calibration_fit_only': diagnostic,
        'raw_threshold': rule['raw_threshold'], 'calibrated_threshold': rule['calibrated_threshold'],
        'slope': rule['slope'], 'intercept': rule['intercept'],
        'reference_negative_breasts': rule['calibration_negative_breasts'],
        'cohort': cohort, 'environment': {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}}
    write_json(output/'calibration_rule_private.json', rule)
    write_json(output/'cohort_manifest.json', cohort)
    features.to_csv(output/'development_features_private.csv', index=False)
    outcomes.to_csv(output/'development_outcomes_private.csv', index=False)
    result['calibration_rule_sha256'] = sha(output/'calibration_rule_private.json')
    write_json(output/'preparation_result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--patients', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('fixed_gmic_final_config.json'))
    a = parser.parse_args()
    print(json.dumps(prepare(a.development, a.patients, a.images, a.experiment, a.output, a.config), indent=2))
