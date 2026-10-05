"""Development-only threshold and failure-review analysis of fixed GMIC scores.

Run from the project directory. No image loading, GPU, network, model selection,
or final-cohort inference occurs. See GMIC_OPERATING_POINT_PLAN.md.
"""
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
from sklearn.metrics import brier_score_loss, log_loss

from gmic_pilot_analysis import aggregate_predictions, summarize_development
from mammo_reliability.audit import BREAST_KEY, require
from mammo_reliability.evaluation import (
    MonotonePlatt, add_review_scores, classification_metrics,
    paired_cluster_bootstrap, review_at_budget, review_selection,
    sensitivity_threshold,
)

SOURCE_URL = 'https://www.kaggle.com/code/aseqwss/11-rsna-gmic-reserved-pilot/output?scriptVersionId=352364754'
EXPECTED_PLAN = '388b2f20d08e84850e8ae14b6d8e3ad9848da9bb4a76a061a48ea11c27a913ba'
EXPECTED_BREASTS = 'eb2bf94e1e752adb9cd6f96ec06898905acaf40815958f52e14a5a72df7bb906'
EXPECTED_ROLES = 'fc0308c63300be6849c7ac9b830d09480c2b62d64df4f1d036bab12548a7e15f'
METHODS = ['score_confidence', 'score_disagreement', 'score_uncertainty',
           'score_confidence_disagreement', 'score_confidence_uncertainty', 'score_combined']
BUDGETS = [0.01, 0.05, 0.10, 0.20]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified(root: Path):
    complete = json.loads((root / 'completed.json').read_text())
    require(complete['status'] == 'reserved_development_inference_completed', 'Incomplete inference.')
    require(complete['final_test_results_opened'] is False, 'Final results entered the source.')
    require(complete['declared_plan_sha256'] == EXPECTED_PLAN, 'Unexpected pilot plan.')
    require(complete['breast_predictions_sha256'] == EXPECTED_BREASTS, 'Unexpected scored cohort.')
    for key, name in [('declared_plan_sha256', 'declared_plan.json'),
                      ('image_predictions_sha256', 'image_predictions_private.csv'),
                      ('breast_predictions_sha256', 'breast_predictions_private.csv'),
                      ('summary_sha256', 'development_summary.json')]:
        require(digest(root / name) == complete[key], f'File integrity failed: {name}')
    require(digest(root / 'roles_private.csv') == EXPECTED_ROLES, 'Patient roles changed.')
    plan = json.loads((root / 'declared_plan.json').read_text())
    project = Path(__file__).resolve().parent
    for name, expected in plan['source_sha256'].items():
        require(digest(project / name) == expected, f'Inference source changed: {name}')
    roles = pd.read_csv(root / 'roles_private.csv')
    images = pd.read_csv(root / 'images_private.csv')
    predictions = pd.read_csv(root / 'image_predictions_private.csv')
    breasts = pd.read_csv(root / 'breast_predictions_private.csv')
    require(roles.patient_id.is_unique and len(roles) == 2383, 'Invalid role table.')
    require(set(roles.cohort) == {'pilot'}, 'Nonpilot patient in source.')
    require(set(images.patient_id) == set(roles.patient_id) == set(breasts.patient_id), 'Patient coverage differs.')
    expected_roles = roles.set_index('patient_id').role
    for frame in [images, breasts]:
        require((frame.role == frame.patient_id.map(expected_roles)).all(), 'Patient-role mismatch.')
    require(len(images) == 10881 and len(breasts) == 4766, 'Unexpected cohort size.')
    recomputed = aggregate_predictions(images, predictions)
    keys = BREAST_KEY
    ordered = breasts.sort_values(keys).reset_index(drop=True)
    recomputed = recomputed[ordered.columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered, recomputed, check_exact=False, atol=1e-12, rtol=1e-12)
    for role, size, positives in [('fit', 3574, 73), ('tune', 1192, 26)]:
        group = breasts.loc[breasts.role == role]
        require(len(group) == size and int(group.cancer.sum()) == positives, 'Unexpected role counts.')
    require(not set(breasts.loc[breasts.role == 'fit', 'patient_id']) &
            set(breasts.loc[breasts.role == 'tune', 'patient_id']), 'Patient leakage.')
    original = json.loads((root / 'development_summary.json').read_text())
    current = summarize_development(breasts)
    for a, b in zip(original['metrics'], current['metrics'], strict=True):
        for name in ['role', 'model', 'site', 'breasts', 'positive_breasts']:
            require(a[name] == b[name], 'Original metric group mismatch.')
        for name in ['average_precision', 'roc_auc']:
            require(abs(a[name] - b[name]) < 1e-12, 'Original ranking metric failed to reproduce.')
    return breasts, {'source': SOURCE_URL, 'complete': complete,
                     'input_sha256': {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
                     'pooling_and_ranking_reproduced': True, 'geometry': original['geometry']}


def fit_development_rule(fit):
    require(len(fit) > 0 and set(fit.role) == {'fit'}, 'Fit-only calibration is required.')
    calibrator = MonotonePlatt().fit(fit.p_raw, fit.cancer)
    reference = fit.copy()
    reference['p_calibrated'] = calibrator.predict(reference.p_raw)
    threshold = sensitivity_threshold(reference.p_calibrated, reference.cancer, target=0.90)
    raw_threshold = sensitivity_threshold(reference.p_raw, reference.cancer, target=0.90)
    require(np.array_equal(reference.p_raw >= raw_threshold, reference.p_calibrated >= threshold),
            'Calibration changed operating-point decisions.')
    return calibrator, reference, threshold, raw_threshold


def apply_development_rule(calibrator, reference, target, threshold):
    require(not set(reference.patient_id) & set(target.patient_id), 'Fit/tune patient overlap.')
    require(set(target.role) == {'tune'}, 'Evaluation is restricted to tune patients.')
    scored = target.copy()
    scored['p_calibrated'] = calibrator.predict(scored.p_raw)
    return add_review_scores(reference, scored, threshold)


def subgroup_reviews(scored, method, budget):
    """Subgroups inherit the one global selection; no subgroup budget refitting."""
    selected = review_selection(scored, method, budget)
    rows = []
    for site, group in scored.groupby('site_id'):
        chosen = selected.loc[group.index]
        negatives = group.loc[~group.predicted_positive]
        fn, caught = int(negatives.cancer.sum()), int(group.loc[chosen, 'cancer'].sum())
        k = int(chosen.sum())
        rows.append({'site_id': int(site), 'method': method, 'global_budget': budget,
                     'negative_breasts': len(negatives), 'false_negatives': fn,
                     'reviewed_breasts': k, 'false_negatives_flagged': caught,
                     'fn_capture_fraction': caught / fn if fn else None,
                     'within_site_review_fraction': k / len(negatives) if len(negatives) else None,
                     'selection': 'inherited from global cohort; no site-specific reallocation'})
    return rows


def analyze(breasts, repeats=2000):
    fit = breasts.loc[breasts.role == 'fit'].copy().reset_index(drop=True)
    tune = breasts.loc[breasts.role == 'tune'].copy().reset_index(drop=True)
    calibrator, reference, threshold, raw_threshold = fit_development_rule(fit)
    scored = apply_development_rule(calibrator, reference, tune, threshold)
    require(np.array_equal(scored.predicted_positive, scored.p_raw >= raw_threshold), 'Ranking/threshold mismatch.')
    metrics, calibration, distributions = [], [], []
    prior = float(fit.cancer.mean())
    for role, frame in [('fit', reference), ('tune', scored)]:
        groups = [('pooled', frame)] + [(int(site), group) for site, group in frame.groupby('site_id')]
        for site, group in groups:
            metrics.append({'role': role, 'site': site, **classification_metrics(group.cancer, group.p_calibrated, threshold)})
            for name, values in [('raw', group.p_raw), ('calibrated', group.p_calibrated),
                                 ('constant_fit_prevalence', np.full(len(group), prior))]:
                calibration.append({'role': role, 'site': site, 'score': name,
                    'brier': float(brier_score_loss(group.cancer, values)),
                    'log_loss': float(log_loss(group.cancer, values, labels=[0, 1]))})
            for column in ['p_raw', 'view_disagreement', 'ensemble_variance']:
                distributions.append({'role': role, 'site': site, 'column': column,
                    'quantiles': {str(q): float(group[column].quantile(q)) for q in [0, .1, .5, .9, .99, 1]}})
    review, site_review = [], []
    for budget in BUDGETS:
        for method in METHODS:
            row = review_at_budget(scored, method, budget)
            row['review_fraction_all_tune_breasts'] = row['reviewed_breasts'] / len(scored)
            review.append(row)
            site_review.extend(subgroup_reviews(scored, method, budget))
    # One development evaluation group, not an outer test fold. This constant
    # reuses the fold-budget bootstrap without splitting or double-counting it.
    bootstrap_frame = scored.assign(outer_fold=0)
    bootstrap = paired_cluster_bootstrap(bootstrap_frame, repeats=repeats)
    bootstrap['estimand'] = 'development tune-cohort combined-minus-cancer-score FN capture at a global 5% review budget'
    negative = scored.loc[~scored.predicted_positive]
    correlation = negative[['p_raw', 'view_disagreement', 'ensemble_variance']].corr(method='spearman')
    result = {'purpose': 'adaptive_development_operating_point_feasibility',
        'final_test_results_opened': False, 'target_sensitivity': .90,
        'calibration_fit_role': 'fit', 'evaluation_role': 'tune',
        'raw_threshold': raw_threshold, 'calibrated_threshold': threshold,
        'calibrator': {'slope': calibrator.slope, 'intercept': calibrator.intercept},
        'normalization_reference_breasts': int((reference.p_calibrated < threshold).sum()),
        'classification': metrics, 'calibration_comparison': calibration,
        'review': review, 'site_review': site_review, 'primary_development_bootstrap': bootstrap,
        'score_distributions': distributions,
        'negative_tune_spearman': correlation.to_dict(),
        'limitations': ['Adaptive development, not final performance or a novelty claim.',
            'Only 26 tune-positive breasts; subgroup and FN estimates are especially uncertain.',
            'Threshold uncertainty and model selection are not included in the conditional bootstrap.',
            'Flagged cases are simulated review candidates, not clinically recovered cancers.',
            'Density and hormone-receptor policies are not evaluated here.']}
    return result, scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('validation/gmic_operating_point'))
    parser.add_argument('--private-output', type=Path, default=Path('private_runs/gmic_operating_point'))
    args = parser.parse_args()
    breasts, provenance = load_verified(args.input)
    result, scored = analyze(breasts)
    result['provenance'] = provenance
    result['software'] = {'python': platform.python_version(), 'numpy': np.__version__,
                          'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
    result['analysis_sha256'] = digest(__file__)
    result['analysis_plan_sha256'] = digest(Path(__file__).with_name('GMIC_OPERATING_POINT_PLAN.md'))
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name in ['classification', 'calibration_comparison', 'review', 'site_review']:
        pd.DataFrame(result[name]).to_csv(args.output / f'{name}.csv', index=False)
    scored.to_csv(args.private_output / 'scored_tune_private.csv', index=False)
    for row in result['classification']:
        if row['role'] == 'tune': print('TUNE', json.dumps(row))
    for row in result['review']:
        if row['budget'] == .05: print('REVIEW_5_PERCENT', json.dumps(row))
    print('PAIRED_BOOTSTRAP', json.dumps(result['primary_development_bootstrap']))


if __name__ == '__main__':
    main()
