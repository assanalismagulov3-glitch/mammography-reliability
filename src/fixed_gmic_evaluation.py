"""Fixed-GMIC evaluation: labels are forbidden in pooling, scoring and selection.

No GPU or network dependency. The old trained-CNN evaluation remains unchanged.
All resampling retains patient clusters and a single global breast budget.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from mammo_reliability.audit import BREAST_KEY, require
from mammo_reliability.evaluation import (MonotonePlatt, checked_labels,
    classification_metrics, reference_percentile, sensitivity_threshold)

FEATURES = ['p_raw', 'view_disagreement', 'ensemble_variance']
METHODS = ['score_confidence', 'score_disagreement', 'score_uncertainty',
           'score_confidence_disagreement', 'score_confidence_uncertainty', 'score_combined']
IMAGE_KEY = ['image_id', 'patient_id', 'laterality', 'view']
PREDICTIONS = [f'm{m}_{c}' for m in range(1, 6) for c in ['benign', 'malignant']]


def identities(frame):
    require(set(BREAST_KEY).issubset(frame), 'Missing breast identity.')
    require(len(frame) > 0 and not frame[BREAST_KEY].isna().any().any(), 'Missing/empty identity.')
    require(not frame[BREAST_KEY].duplicated().any(), 'Duplicate breasts.')
    p = frame.patient_id.to_numpy(dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p == np.floor(p))).all(), 'Invalid patient IDs.')
    require(frame.laterality.isin(['L', 'R']).all(), 'Invalid laterality.')


def check_features(frame):
    identities(frame)
    require(set(frame) == set(BREAST_KEY + FEATURES), 'Only identity and three warning features allowed.')
    x = frame[FEATURES].to_numpy(dtype=float)
    require(np.isfinite(x).all(), 'Nonfinite features.')
    require(((x >= 0) & (x <= np.array([1, 1, .25]) + 1e-12)).all(), 'Features outside bounds.')


def pool_images(images, predictions):
    """Only image identifiers, view and numerical model outputs enter pooling."""
    require(set(images) == set(IMAGE_KEY), 'Image metadata must be label-free.')
    require(set(predictions) == set(['image_id'] + PREDICTIONS), 'Unexpected prediction columns.')
    require(len(images) > 0, 'Empty image cohort.')
    require(images.image_id.is_unique and predictions.image_id.is_unique, 'Duplicate images.')
    require(not images.isna().any().any(), 'Missing image identities.')
    require(set(images.image_id) == set(predictions.image_id), 'Incomplete prediction coverage.')
    require(images.view.isin(['CC', 'MLO']).all(), 'Nonstandard view.')
    require((images.groupby(BREAST_KEY).view.nunique() == 2).all(), 'Missing standard view.')
    x = predictions[PREDICTIONS].to_numpy(float)
    require(np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all(), 'Invalid image probabilities.')
    joined = images.merge(predictions, on='image_id', validate='one_to_one')
    columns = [f'm{m}_malignant' for m in range(1, 6)]
    views = joined.groupby(BREAST_KEY + ['view'])[columns].mean()
    cc, mlo = views.xs('CC', level='view'), views.xs('MLO', level='view')
    require(cc.index.equals(mlo.index), 'View identities differ.')
    member = (cc.to_numpy() + mlo.to_numpy()) / 2
    result = pd.DataFrame(index=cc.index)
    result['p_raw'] = member.mean(axis=1)
    result['view_disagreement'] = abs(cc.mean(axis=1) - mlo.mean(axis=1))
    result['ensemble_variance'] = member.var(axis=1, ddof=0)
    result = result.reset_index()
    check_features(result)
    return result


def join_outcomes(features, outcomes):
    """Explicit evaluation boundary; output must never enter the scorer."""
    identities(features)
    identities(outcomes)
    require(set(outcomes) == set(BREAST_KEY + ['site_id', 'density', 'cancer']), 'Invalid outcome schema.')
    checked_labels(outcomes.cancer, len(outcomes))
    require(outcomes.site_id.isin([1, 2]).all(), 'Missing or unexpected site.')
    require((outcomes.groupby('patient_id').site_id.nunique() == 1).all(), 'Patient crosses sites.')
    require(outcomes.density.dropna().isin(['A', 'B', 'C', 'D']).all(), 'Unexpected density.')
    require(len(features) == len(outcomes), 'Outcome coverage differs.')
    joined = features.merge(outcomes, on=BREAST_KEY, how='left', validate='one_to_one', indicator=True)
    require(joined['_merge'].eq('both').all(), 'Missing outcome identities.')
    return joined.drop(columns='_merge')


def weights_checked(weights, n):
    w = np.asarray(weights)
    require(w.shape == (n,) and np.isfinite(w).all(), 'Invalid weights.')
    require(((w >= 0) & (w == np.floor(w))).all() and w.sum() > 0, 'Empty/noninteger weights.')
    return w.astype(np.int64)


def fit_rule(features, outcomes, weights=None, target=.9):
    check_features(features)
    data = join_outcomes(features, outcomes)
    w = np.ones(len(data), dtype=np.int64) if weights is None else weights_checked(weights, len(data))
    # Literal repetition preserves the exact established optimizer/objective.
    # Only development data are repeated, at most a few thousand rows here.
    indices = np.repeat(np.arange(len(data)), w)
    p, y = data.p_raw.to_numpy()[indices], data.cancer.to_numpy()[indices]
    require(len(np.unique(y)) == 2, 'Calibration draw needs both classes.')
    cal = MonotonePlatt().fit(p, y)
    threshold = sensitivity_threshold(p, y, target)
    calibrated = cal.predict(p)
    cal_threshold = float(cal.predict([threshold])[0])
    require(np.array_equal(p >= threshold, calibrated >= cal_threshold), 'Calibration changes decisions.')
    require(np.array_equal(np.argsort(p, kind='stable'), np.argsort(calibrated, kind='stable')),
            'Calibration saturation/clipping changes score ordering.')
    negative = p < threshold
    require(negative.any(), 'No calibration negatives.')
    source = data.iloc[indices]
    refs = [calibrated[negative], source.view_disagreement.to_numpy()[negative],
            source.ensemble_variance.to_numpy()[negative]]
    return {'schema_version': '0.3', 'slope': cal.slope, 'intercept': cal.intercept,
            'raw_threshold': threshold, 'calibrated_threshold': cal_threshold,
            'target_sensitivity': target, 'reference_values': [np.sort(r).tolist() for r in refs],
            'calibration_patient_ids': sorted(int(v) for v in data.patient_id.unique()),
            'calibration_breasts': len(p), 'calibration_positive_breasts': int(y.sum()),
            'calibration_negative_breasts': int(negative.sum()),
            'calibration_detected_cancers': int(((y == 1) & ~negative).sum()),
            'development_prevalence': float(y.mean())}


def score_features(features, rule, allow_calibration=False):
    check_features(features)
    overlap = set(features.patient_id) & set(rule['calibration_patient_ids'])
    require(allow_calibration or not overlap, 'Calibration patients entered final scoring.')
    result = features.reset_index(drop=True).copy()
    cal = MonotonePlatt(rule['slope'], rule['intercept'])
    result['p_calibrated'] = cal.predict(result.p_raw)
    result['predicted_positive'] = result.p_raw >= rule['raw_threshold']
    require(np.array_equal(result.predicted_positive, result.p_calibrated >= rule['calibrated_threshold']),
            'Raw and calibrated decisions differ.')
    result[METHODS[0]] = result.p_calibrated
    result[METHODS[1]] = result.view_disagreement
    result[METHODS[2]] = result.ensemble_variance
    ranks = [reference_percentile(ref, result[col]) for ref, col in
             zip(rule['reference_values'], ['p_calibrated', 'view_disagreement', 'ensemble_variance'])]
    result[METHODS[3]] = (ranks[0] + ranks[1]) / 2
    result[METHODS[4]] = (ranks[0] + ranks[2]) / 2
    result[METHODS[5]] = sum(ranks) / 3
    return result


def select_review(scored, method, budget=.05, seed=20260909, weights=None):
    identities(scored)
    require(set(scored) == set(BREAST_KEY + FEATURES + ['p_calibrated', 'predicted_positive'] + METHODS),
            'Selection accepts only label-free scored features.')
    require(method in METHODS and 0 <= budget <= 1, 'Invalid method or budget.')
    require(scored.predicted_positive.isin([True, False]).all(), 'Invalid decisions.')
    scores = scored[method].to_numpy(float)
    require(np.isfinite(scores).all(), 'Nonfinite review scores.')
    w = np.ones(len(scored), np.int64) if weights is None else weights_checked(weights, len(scored))
    eligible = np.flatnonzero(~scored.predicted_positive.to_numpy(bool) & (w > 0))
    ties = np.array([hashlib.sha256(f'{seed}:{int(p)}:{l}'.encode()).hexdigest()
                    for p, l in scored[BREAST_KEY].itertuples(index=False, name=None)])
    order = eligible[np.lexsort((ties[eligible], -scores[eligible]))]
    k = math.floor(budget * int(w[eligible].sum()))
    preceding = np.cumsum(w[order]) - w[order]
    chosen = np.zeros(len(scored), np.int64)
    chosen[order] = np.minimum(w[order], np.maximum(0, k - preceding))
    require(int(chosen.sum()) == k, 'Global budget was not met.')
    return chosen


def review_counts(data, chosen, weights=None, mask=None):
    w = np.ones(len(data), np.int64) if weights is None else weights_checked(weights, len(data))
    if mask is not None:
        w = w * np.asarray(mask, bool)
        chosen = chosen * np.asarray(mask, bool)
    negative = ~data.predicted_positive.to_numpy(bool)
    y = data.cancer.to_numpy(int)
    n, fn, k, caught = int(w @ negative), int(w @ (negative * y)), int(chosen.sum()), int(chosen @ y)
    return {'negative_breasts': n, 'false_negatives': fn, 'reviewed_breasts': k,
            'false_negatives_flagged': caught, 'fn_capture_fraction': caught/fn if fn else None,
            'review_yield': caught/k if k else None, 'actual_review_fraction': k/n if n else 0.,
            'random_expected_fn_capture': k/n if n and fn else None,
            'unflagged_misses': fn-caught, 'residual_negative_cohort_risk': (fn-caught)/(n-k) if n > k else None}


def group_masks(data):
    yield 'pooled', np.ones(len(data), bool)
    for site in [1, 2]:
        yield f'site_{site}', data.site_id.eq(site).to_numpy()
    for name, values in [('AB', ['A', 'B']), ('CD', ['C', 'D'])]:
        yield f'site_1_density_{name}', (data.site_id.eq(1) & data.density.isin(values)).to_numpy()
    yield 'site_1_density_missing', (data.site_id.eq(1) & data.density.isna()).to_numpy()


def evaluate_scores(scored, outcomes, rule, config):
    data = join_outcomes(scored, outcomes)
    groups = [(name, mask) for name, mask in group_masks(data) if mask.any()]
    classification, calibration, bins = [], [], []
    for name, mask in groups:
        group = data.loc[mask]
        classification.append({'group': name, **classification_metrics(group.cancer,
            group.p_calibrated, rule['calibrated_threshold'])})
        for kind, p in [('raw', group.p_raw.to_numpy()), ('calibrated', group.p_calibrated.to_numpy()),
                        ('constant_development_prevalence', np.full(len(group), rule['development_prevalence']))]:
            calibration.append({'group': name, 'score': kind,
                'brier': float(brier_score_loss(group.cancer, p)),
                'log_loss': float(log_loss(group.cancer, p, labels=[0, 1]))})
            if kind == 'constant_development_prevalence': continue
            b = np.minimum((p*10).astype(int), 9)
            for i in range(10):
                m = b == i
                bins.append({'group': name, 'score': kind, 'bin': i, 'count': int(m.sum()),
                    'cancers': int(group.cancer.to_numpy()[m].sum()),
                    'mean_score': float(p[m].mean()) if m.any() else None})
    budgets = sorted(set(config['review']['report_budgets'] + config['review']['curve_budgets']))
    review, subgroups, overlap = [], [], []
    already_positive = set(data.loc[data.predicted_positive, 'patient_id'])
    for q in budgets:
        selections = {}
        for method in METHODS:
            selected = select_review(scored, method, q, config['review']['tie_seed'])
            selections[method] = selected
            patients = set(data.loc[selected.astype(bool), 'patient_id'])
            review.append({'method': method, 'budget': q, **review_counts(data, selected),
                'unique_selected_patients': len(patients),
                'selected_patients_with_other_breast_AI_positive': len(patients & already_positive)})
            if q in config['review']['report_budgets']:
                for group, mask in groups[1:]:
                    subgroups.append({'group': group, 'method': method, 'global_budget': q,
                                      **review_counts(data, selected, mask=mask)})
        a, b = selections['score_combined'], selections['score_confidence']
        common = (a > 0) & (b > 0)
        overlap.append({'budget': q, 'common_breasts': int(common.sum()),
                        'common_patients': len(set(data.loc[a > 0, 'patient_id']) & set(data.loc[b > 0, 'patient_id']))})
    primary = [r for r in review if r['budget'] == config['review']['primary_budget']]
    candidate = next(r for r in primary if r['method'] == 'score_combined')
    baseline = next(r for r in primary if r['method'] == 'score_confidence')
    fn = candidate['false_negatives']
    return {'classification': classification, 'calibration': calibration, 'calibration_bins': bins,
            'review': review, 'subgroup_review': subgroups, 'selection_overlap': overlap,
            'primary': {'candidate': candidate, 'baseline': baseline,
                'capture_difference': (candidate['false_negatives_flagged']-baseline['false_negatives_flagged'])/fn if fn else None}}


@dataclass
class PatientSampler:
    """Construct using identity/site only; labels cannot define strata."""
    frame: pd.DataFrame

    def __post_init__(self):
        identities(self.frame)
        require(set(self.frame) == set(BREAST_KEY + ['site_id']), 'Sampler accepts only identity and site.')
        require(self.frame.site_id.isin([1, 2]).all(), 'Invalid bootstrap site.')
        require((self.frame.groupby('patient_id').site_id.nunique() == 1).all(), 'Patient crosses sites.')
        self.patients, self.row_patient = np.unique(self.frame.patient_id, return_inverse=True)
        sites = self.frame.groupby('patient_id', sort=True).site_id.first().to_numpy()
        self.strata = [np.flatnonzero(sites == site) for site in sorted(np.unique(sites))]

    def draw(self, rng):
        counts = np.zeros(len(self.patients), np.int64)
        for idx in self.strata:
            counts[idx] = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)))
        return counts[self.row_patient]


def bootstrap_intervals(development, development_outcomes, holdout, holdout_outcomes, rule, config):
    check_features(development); check_features(holdout)
    require(not set(development.patient_id) & set(holdout.patient_id), 'Development/holdout overlap.')
    dev = join_outcomes(development, development_outcomes)
    test = join_outcomes(holdout, holdout_outcomes)
    dev_sampler = PatientSampler(dev[BREAST_KEY + ['site_id']])
    test_sampler = PatientSampler(test[BREAST_KEY + ['site_id']])
    setting = config['uncertainty']
    dev_rng = np.random.default_rng(setting['development_seed'])
    test_rng = np.random.default_rng(setting['holdout_seed'])
    fixed_scores = score_features(holdout, rule)
    records = []
    for draw in range(setting['bootstrap_repeats']):
        wd, wt = dev_sampler.draw(dev_rng), test_sampler.draw(test_rng)
        for mode in ['conditional', 'calibration_inclusive']:
            record = {'draw': draw, 'mode': mode, 'status': 'ok'}
            try:
                current_rule = rule if mode == 'conditional' else fit_rule(development, development_outcomes, wd,
                    config['calibration']['empirical_sensitivity_target'])
                scored = fixed_scores if mode == 'conditional' else score_features(holdout, current_rule)
                data = join_outcomes(scored, holdout_outcomes)
                a = select_review(scored, 'score_combined', config['review']['primary_budget'], config['review']['tie_seed'], wt)
                b = select_review(scored, 'score_confidence', config['review']['primary_budget'], config['review']['tie_seed'], wt)
                ra, rb = review_counts(data, a, wt), review_counts(data, b, wt)
                fn = ra['false_negatives']; positive = int(wt @ data.cancer.to_numpy(int))
                record.update(raw_threshold=current_rule['raw_threshold'], false_negatives=fn,
                    negative_breasts=ra['negative_breasts'], reviewed_breasts=ra['reviewed_breasts'],
                    flagged_combined=ra['false_negatives_flagged'], flagged_cancer_score=rb['false_negatives_flagged'],
                    sensitivity=(positive-fn)/positive if positive else None,
                    difference=(ra['false_negatives_flagged']-rb['false_negatives_flagged'])/fn if fn else None)
                if not fn: record['status'] = 'undefined_zero_FN'
            except (ValueError, RuntimeError, AssertionError) as exc:
                record.update(status='numerical_failure', error=str(exc), difference=None)
            records.append(record)
    table = pd.DataFrame(records)
    summaries = {}
    for mode, group in table.groupby('mode'):
        values = group.difference.dropna().to_numpy(float)
        failures = int(group.status.eq('numerical_failure').sum())
        summaries[mode] = {'requested': len(group), 'defined': len(values),
            'undefined_zero_FN': int(group.status.eq('undefined_zero_FN').sum()),
            'numerical_failures': failures, 'interval_valid': failures == 0 and len(values) > 0,
            'ci95': np.quantile(values, [.025, .975]).tolist() if len(values) and not failures else None}
    return summaries, table
