"""Patient-cluster resampling of the fixed-GMIC development operating point.

Only the already reserved pilot predictions are accepted by load_verified.
This script never changes model predictions or the original patient roles.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit

from gmic_operating_point import load_verified, digest
from mammo_reliability.audit import BREAST_KEY, require


def checked_weights(values, n):
    values = np.asarray(values)
    require(values.shape == (n,) and np.isfinite(values).all(), 'Invalid weight shape or values.')
    require(((values >= 0) & (values == np.floor(values))).all(), 'Weights must be nonnegative integers.')
    return values.astype(np.int64)


def weighted_sensitivity_threshold(p, y, weights, target=.9):
    p, y = np.asarray(p, float), np.asarray(y)
    require(p.ndim == 1 and y.shape == p.shape, 'Invalid score/label dimensions.')
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities.')
    require(np.isin(y, [0, 1]).all() and 0 < target <= 1, 'Invalid labels or sensitivity target.')
    weights = checked_weights(weights, len(p))
    positive = (y == 1) & (weights > 0)
    if not positive.any():
        return None
    order = np.argsort(-p[positive], kind='stable')
    scores, counts = p[positive][order], weights[positive][order]
    rank = math.ceil(target * int(counts.sum()))
    return float(scores[np.searchsorted(np.cumsum(counts), rank, side='left')])


def weighted_midrank(reference, weights, values):
    reference, values = np.asarray(reference, float), np.asarray(values, float)
    require(reference.ndim == values.ndim == 1 and np.isfinite(reference).all()
            and np.isfinite(values).all(), 'Invalid percentile values.')
    weights = checked_weights(weights, len(reference))
    require(weights.sum() > 0, 'Empty percentile reference.')
    order = np.argsort(reference, kind='stable')
    ref = reference[order]
    cumulative = np.r_[0, np.cumsum(weights[order])]
    left, right = np.searchsorted(ref, values, 'left'), np.searchsorted(ref, values, 'right')
    return (cumulative[left] + cumulative[right]) / (2 * int(weights.sum()))


@dataclass
class Cohort:
    frame: pd.DataFrame
    tie_seed: int = 20260909

    def __post_init__(self):
        self.frame = self.frame.sort_values(BREAST_KEY).reset_index(drop=True).copy()
        f = self.frame
        require(len(f) > 0 and not f[BREAST_KEY].duplicated().any(), 'Expected unique breast records.')
        require(set(f.role).issubset({'fit', 'tune'}), 'Nondevelopment role is forbidden.')
        require((f.groupby('patient_id').site_id.nunique() == 1).all(), 'Patient crosses sites.')
        require(np.isin(f.cancer, [0, 1]).all(), 'Invalid cancer labels.')
        self.columns = ['p_raw', 'view_disagreement', 'ensemble_variance']
        self.features = f[self.columns].to_numpy(dtype=float)
        require(np.isfinite(self.features).all(), 'Nonfinite features.')
        require(((self.features[:, 0] >= 0) & (self.features[:, 0] <= 1)).all(), 'Invalid probability.')
        self.p = self.features[:, 0]
        self.y = f.cancer.to_numpy(dtype=int)
        self.site = f.site_id.to_numpy()
        self.patient_ids, self.row_patient = np.unique(f.patient_id, return_inverse=True)
        self.patient_table = f.groupby('patient_id', sort=True).agg(
            site_id=('site_id', 'first'), patient_cancer=('cancer', 'max')).reset_index()
        require(np.array_equal(self.patient_ids, self.patient_table.patient_id), 'Patient ordering mismatch.')
        self.site_patient_indices = [np.flatnonzero(self.patient_table.site_id.to_numpy() == site)
                                     for site in sorted(self.patient_table.site_id.unique())]
        self.ties = np.array([hashlib.sha256(f'{self.tie_seed}:{p}:{l}'.encode()).hexdigest()
                              for p, l in f[BREAST_KEY].itertuples(index=False, name=None)])
        self.ones = np.ones(len(f), dtype=np.int64)

    def bootstrap_weights(self, rng):
        multiplicity = np.zeros(len(self.patient_ids), dtype=np.int64)
        for indices in self.site_patient_indices:
            n = len(indices)
            multiplicity[indices] = rng.multinomial(n, np.full(n, 1 / n))
        return multiplicity[self.row_patient]


def make_rule(calibration, target, weights, sensitivity=.9):
    require(not np.intersect1d(calibration.patient_ids, target.patient_ids).size,
            'Calibration/evaluation patient overlap.')
    weights = checked_weights(weights, len(calibration.frame))
    threshold = weighted_sensitivity_threshold(calibration.p, calibration.y, weights, sensitivity)
    if threshold is None:
        return None
    negative = calibration.p < threshold
    reference_weights = weights * negative
    if reference_weights.sum() == 0:
        return None
    combined = np.mean([weighted_midrank(calibration.features[:, j], reference_weights,
                                         target.features[:, j]) for j in range(3)], axis=0)
    return {'threshold': threshold, 'score_combined': combined, 'score_confidence': target.p.copy(),
            'predicted_negative': target.p < threshold,
            'calibration_breasts': int(weights.sum()),
            'calibration_positive_breasts': int(weights @ calibration.y),
            'normalization_breasts': int(reference_weights.sum()),
            'calibration_empirical_sensitivity': float(weights @ ((calibration.y == 1) &
                (calibration.p >= threshold)) / (weights @ calibration.y))}


def select_weighted_review(cohort, rule, weights, method, budget=.05):
    """Select floor(budget*N) bootstrap copies with label-free tie ordering."""
    require(0 <= budget <= 1, 'Invalid review budget.')
    weights = checked_weights(weights, len(cohort.frame))
    scores = np.asarray(rule[method], float)
    require(scores.shape == weights.shape and np.isfinite(scores).all(), 'Invalid review score.')
    negative = np.flatnonzero(rule['predicted_negative'] & (weights > 0))
    order = negative[np.lexsort((cohort.ties[negative], -scores[negative]))]
    k = math.floor(budget * int(weights[negative].sum()))
    ordered_weight = weights[order]
    preceding = np.cumsum(ordered_weight) - ordered_weight
    chosen = np.zeros(len(cohort.frame), dtype=np.int64)
    chosen[order] = np.minimum(ordered_weight, np.maximum(0, k - preceding))
    require(chosen.sum() == k, 'Review budget not met.')
    return chosen


def evaluate(cohort, rule, weights, stage, repeat, budget=.05):
    weights = checked_weights(weights, len(cohort.frame))
    selected = {name: select_weighted_review(cohort, rule, weights, name, budget)
                for name in ['score_confidence', 'score_combined']}
    groups = [('pooled', np.ones(len(weights), bool))]
    groups += [(str(site), cohort.site == site) for site in sorted(np.unique(cohort.site))]
    rows = []
    for site, mask in groups:
        w = weights * mask
        negative, cancer = rule['predicted_negative'], cohort.y == 1
        tp, fn = int(w @ (cancer & ~negative)), int(w @ (cancer & negative))
        tn, fp = int(w @ (~cancer & negative)), int(w @ (~cancer & ~negative))
        confidence = int((selected['score_confidence'] * mask) @ cohort.y)
        combined = int((selected['score_combined'] * mask) @ cohort.y)
        rows.append({'stage': stage, 'repeat': int(repeat), 'site': site,
            'threshold': rule['threshold'], 'calibration_breasts': rule['calibration_breasts'],
            'calibration_positive_breasts': rule['calibration_positive_breasts'],
            'calibration_empirical_sensitivity': rule['calibration_empirical_sensitivity'],
            'n_breasts': int(w.sum()), 'positive_breasts': tp + fn, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': tp / (tp + fn) if tp + fn else None,
            'specificity': tn / (tn + fp) if tn + fp else None,
            'ai_negative_breasts': tn + fn,
            'reviewed_confidence': int((selected['score_confidence'] * mask).sum()),
            'reviewed_combined': int((selected['score_combined'] * mask).sum()),
            'flagged_confidence': confidence, 'flagged_combined': combined,
            'capture_confidence': confidence / fn if fn else None,
            'capture_combined': combined / fn if fn else None,
            'capture_difference': (combined - confidence) / fn if fn else None})
    return rows


def alternative_partitions(frame, repeats=100, seed=20261028, evaluation_fraction=.25):
    patients = frame.groupby('patient_id', sort=True).agg(
        site_id=('site_id', 'first'), patient_cancer=('cancer', 'max')).reset_index()
    strata = patients.site_id.astype(str) + ':' + patients.patient_cancer.astype(str)
    splits = StratifiedShuffleSplit(n_splits=repeats, test_size=evaluation_fraction, random_state=seed)
    for fitting, evaluation in splits.split(patients, strata):
        calibration_ids = patients.iloc[fitting].patient_id.to_numpy()
        evaluation_ids = patients.iloc[evaluation].patient_id.to_numpy()
        require(not np.intersect1d(calibration_ids, evaluation_ids).size, 'Split patient overlap.')
        yield calibration_ids, evaluation_ids


def summarize(records, config):
    table = pd.DataFrame(records)
    metrics = ['threshold', 'calibration_positive_breasts', 'sensitivity', 'specificity',
               'ai_negative_breasts', 'fn', 'reviewed_confidence', 'reviewed_combined',
               'flagged_confidence', 'flagged_combined', 'capture_difference']
    summaries = []
    for (stage, site), group in table.groupby(['stage', 'site'], sort=False):
        if stage == 'observed':
            continue
        record = {'stage': stage, 'site': site, 'repetitions': len(group), 'metrics': {}}
        for metric in metrics:
            x = group[metric].dropna().to_numpy()
            record['metrics'][metric] = {'defined': len(x), 'undefined': int(group[metric].isna().sum()),
                'minimum': float(x.min()) if len(x) else None, 'maximum': float(x.max()) if len(x) else None,
                'median': float(np.median(x)) if len(x) else None,
                'p025': float(np.quantile(x, .025)) if len(x) else None,
                'p975': float(np.quantile(x, .975)) if len(x) else None}
        sens = group.sensitivity.dropna()
        record['sensitivity_target_reached_count'] = int((sens >= config['target_sensitivity']).sum())
        record['sensitivity_target_denominator'] = len(sens)
        valid = group.loc[group.fn > 0]
        difference = valid.flagged_combined - valid.flagged_confidence
        record['review_comparison'] = {'defined_draws': len(valid), 'zero_fn_draws': int((group.fn == 0).sum()),
            'combined_flags_more': int((difference > 0).sum()), 'equal': int((difference == 0).sum()),
            'combined_flags_fewer': int((difference < 0).sum())}
        summaries.append(record)
    original = table.loc[(table.stage == 'observed') & (table.site == 'pooled')].iloc[0]
    repeated = table.loc[(table.stage == 'alternative_split') & (table.site == 'pooled')]
    return {'groups': summaries, 'observed': table.loc[table.stage == 'observed'].to_dict('records'),
        'alternative_splits_at_or_below_original_sensitivity': int((repeated.sensitivity <= original.sensitivity).sum()),
        'alternative_split_count': len(repeated),
        'interpretation': {'fit_bootstrap': 'Empirical stability range on unchanged tune patients; not a performance CI.',
            'tune_bootstrap': 'Patient/site-stratified percentile distribution conditional on the original fixed rule.',
            'joint_bootstrap': 'Both calibration and evaluation samples vary; conditional on fixed development predictions and procedure.',
            'alternative_split': 'Overlapping development repetitions; empirical range, not a confidence interval or independent studies.'}}


def run(frame, config):
    calibration = Cohort(frame.loc[frame.role == 'fit'], config['review_tie_seed'])
    target = Cohort(frame.loc[frame.role == 'tune'], config['review_tie_seed'])
    original = make_rule(calibration, target, calibration.ones, config['target_sensitivity'])
    require(original is not None, 'Original rule undefined.')
    records = evaluate(target, original, target.ones, 'observed', 0, config['review_budget'])
    require(records[0]['tp'] == 18 and records[0]['fn'] == 8 and records[0]['reviewed_combined'] == 33,
            'Original confusion/workload did not reproduce.')
    require(records[0]['flagged_combined'] == 1 and records[0]['flagged_confidence'] == 0,
            'Original review comparison did not reproduce.')
    fit_rng = np.random.default_rng(config['calibration_bootstrap_seed'])
    tune_rng = np.random.default_rng(config['evaluation_bootstrap_seed'])
    skipped = []
    for repeat in range(config['bootstrap_repeats']):
        fitting_weights = calibration.bootstrap_weights(fit_rng)
        evaluation_weights = target.bootstrap_weights(tune_rng)
        records.extend(evaluate(target, original, evaluation_weights, 'tune_bootstrap', repeat, config['review_budget']))
        refitted = make_rule(calibration, target, fitting_weights, config['target_sensitivity'])
        if refitted is None:
            skipped.append({'repeat': repeat, 'reason': 'No calibration positives or no AI-negative normalization observations'})
            continue
        records.extend(evaluate(target, refitted, target.ones, 'fit_bootstrap', repeat, config['review_budget']))
        records.extend(evaluate(target, refitted, evaluation_weights, 'joint_bootstrap', repeat, config['review_budget']))
        if (repeat + 1) % 500 == 0:
            print('BOOTSTRAP_PROGRESS', repeat + 1, flush=True)
    patient_ids = np.sort(frame.patient_id.unique())
    membership = []
    for repeat, (fitting_ids, evaluation_ids) in enumerate(alternative_partitions(
            frame, config['alternative_splits'], config['alternative_split_seed'],
            config['alternative_split_evaluation_fraction'])):
        fitting = Cohort(frame.loc[frame.patient_id.isin(fitting_ids)], config['review_tie_seed'])
        evaluation = Cohort(frame.loc[frame.patient_id.isin(evaluation_ids)], config['review_tie_seed'])
        rule = make_rule(fitting, evaluation, fitting.ones, config['target_sensitivity'])
        require(rule is not None, 'Undefined rule in alternative split.')
        records.extend(evaluate(evaluation, rule, evaluation.ones, 'alternative_split', repeat, config['review_budget']))
        membership.append(np.isin(patient_ids, evaluation_ids))
    return records, summarize(records, config), skipped, patient_ids, np.asarray(membership)


def make_figure(table, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pooled = table.loc[table.site == 'pooled']
    original = pooled.loc[pooled.stage == 'observed'].iloc[0]
    fit = pooled.loc[pooled.stage == 'fit_bootstrap']
    split = pooled.loc[pooled.stage == 'alternative_split']
    joint = pooled.loc[pooled.stage == 'joint_bootstrap']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
        ax = axes[0, 0]
        ax.hist(fit.threshold, bins=24, color='#30638e', alpha=.85)
        ax.axvline(original.threshold, color='#b44b2f', linestyle='--', label='Original cutoff')
        ax.set(title='A  Cutoff varies with calibration patients', xlabel='Raw model-score cutoff', ylabel='Bootstrap draws')
        ax.legend(frameon=False)
        ax = axes[0, 1]
        counts = fit.tp.value_counts().sort_index()
        ax.bar(counts.index, counts.values, color='#30638e', width=.8)
        ax.axvline(24, color='#6c8439', linestyle=':', label='At least 24/26 needed for 90%')
        ax.set(title='B  Same 26 tune cancers, different cutoffs', xlabel='Cancer-positive breasts detected (out of 26)', ylabel='Bootstrap draws')
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, 0]
        ax.hist(split.sensitivity * 100, bins=np.arange(0, 105, 5), color='#30638e', alpha=.85)
        ax.axvline(original.sensitivity * 100, color='#b44b2f', linestyle='--', label='Original split: 69.2%')
        ax.axvline(90, color='#6c8439', linestyle=':', label='90% target')
        ax.set(title='C  100 alternative patient-separated splits', xlabel='Evaluation sensitivity (%)', ylabel='Development splits')
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, 1]
        values = joint.capture_difference.dropna().to_numpy() * 100
        ax.hist(values, bins=np.arange(-102.5, 105, 5), color='#30638e', alpha=.85)
        ax.axvline(0, color='#545454', linewidth=1)
        ax.axvline(original.capture_difference * 100, color='#b44b2f', linestyle='--', label='Original difference: +12.5 pp')
        ax.set(title='D  Both patient samples vary; 5% review budget', xlabel='Combined minus cancer-score FN capture (pp)', ylabel='Defined bootstrap draws')
        ax.legend(frameon=False, fontsize=8)
        fig.suptitle('Fixed GMIC ensemble: development stability diagnostics', fontsize=15)
        fig.get_layout_engine().set(rect=(0, .045, 1, .97))
        fig.text(.5, .012, 'Reserved development patients only. Resampling does not add independent cancers or establish clinical performance.',
                 ha='center', fontsize=9, color='#555555')
        fig.savefig(destination, dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('threshold_stability_config.json'))
    parser.add_argument('--output', type=Path, default=Path('validation/threshold_stability'))
    parser.add_argument('--private-output', type=Path, default=Path('private_runs/threshold_stability'))
    args = parser.parse_args()
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    require(config['final_test_results_opened'] is False, 'Final evaluation is forbidden.')
    frame, provenance = load_verified(args.input)
    records, summary, skipped, patients, splits = run(frame, config)
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(args.output/'resampling_draws.csv', index=False)
    np.savez_compressed(args.private_output/'split_assignments_private.npz',
                        patient_id=patients, evaluation_membership=splits)
    summary.update({'purpose': config['purpose'], 'config': config, 'skipped_calibration_draws': skipped,
        'provenance': provenance, 'source_sha256': digest(__file__), 'config_sha256': digest(args.config),
        'plan_sha256': digest(Path(__file__).with_name('THRESHOLD_STABILITY_PLAN.md')),
        'draws_sha256': digest(args.output/'resampling_draws.csv'),
        'split_assignments_sha256': digest(args.private_output/'split_assignments_private.npz'),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'original_roles_changed': False, 'model_predictions_changed': False,
        'final_test_results_opened': False, 'elapsed_seconds': time.monotonic() - started})
    (args.output/'threshold_stability_results.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    make_figure(table, args.output/'threshold_stability.png')
    for group in summary['groups']:
        if group['site'] == 'pooled':
            print('STABILITY_SUMMARY', json.dumps(group), flush=True)
    print('STABILITY_COMPLETED', json.dumps({'bootstrap_repeats': config['bootstrap_repeats'],
          'alternative_splits': config['alternative_splits'], 'aggregate_rows': len(table),
          'skipped_calibration_draws': len(skipped), 'final_test_results_opened': False}))


if __name__ == '__main__':
    main()
