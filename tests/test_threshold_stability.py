"""Patient isolation and weighted-to-expanded resampling checks on artificial data."""
import hashlib
import math
import unittest

import numpy as np
import pandas as pd

from mammo_reliability.evaluation import (
    add_review_scores, reference_percentile, review_selection, sensitivity_threshold,
)
from threshold_stability import (
    Cohort, alternative_partitions, evaluate, make_rule,
    select_weighted_review, weighted_midrank, weighted_sensitivity_threshold,
)


def fixture(offset=0, role='fit'):
    rows = []
    for patient in range(40):
        for side_index, side in enumerate(['L', 'R']):
            index = 2 * patient + side_index
            rows.append({'patient_id': offset + patient, 'laterality': side,
                         'site_id': 1 + patient // 20, 'role': role,
                         'cancer': int(patient % 4 == 0 and side == 'L'),
                         'p_raw': .005 + (index * 17 % 80) / 100,
                         'view_disagreement': (index * 7 % 12) / 20,
                         'ensemble_variance': (index * 3 % 7) / 100})
    return pd.DataFrame(rows)


class TestThresholdStability(unittest.TestCase):
    def test_weighted_threshold_matches_literal_duplicates_with_ties(self):
        p = np.array([.05, .05, .2, .7, .95])
        y = np.array([1, 1, 0, 1, 1])
        weights = np.array([2, 3, 0, 5, 1])
        for target in [.1, .5, .9, 1]:
            expected = sensitivity_threshold(np.repeat(p, weights), np.repeat(y, weights), target)
            self.assertEqual(weighted_sensitivity_threshold(p, y, weights, target), expected)
        self.assertIsNone(weighted_sensitivity_threshold(p, y, np.zeros(5)))
        with self.assertRaises(ValueError):
            weighted_sensitivity_threshold(p, y, [.5, 1, 1, 1, 1])

    def test_weighted_midrank_matches_expanded_reference(self):
        ref = np.array([.4, .1, .1, .7, .9])
        weights = np.array([2, 0, 3, 0, 1])
        values = np.array([0, .1, .3, .4, .7, .9, 1])
        np.testing.assert_allclose(weighted_midrank(ref, weights, values),
                                   reference_percentile(np.repeat(ref, weights), values),
                                   rtol=0, atol=0)
        with self.assertRaises(ValueError):
            weighted_midrank(ref, np.zeros(5), values)

    def test_bootstrap_clusters_breasts_and_preserves_site_patient_counts(self):
        cohort = Cohort(fixture())
        rng = np.random.default_rng(42)
        for _ in range(20):
            weights = cohort.bootstrap_weights(rng)
            f = cohort.frame.assign(weight=weights)
            self.assertTrue((f.groupby('patient_id').weight.nunique() == 1).all())
            self.assertEqual(f.drop_duplicates('patient_id').groupby('site_id').weight.sum().to_dict(),
                             {1: 20, 2: 20})

    def test_unweighted_rule_matches_original_evaluator(self):
        fit, tune = Cohort(fixture()), Cohort(fixture(100, 'tune'))
        rule = make_rule(fit, tune, fit.ones)
        reference = fit.frame.assign(p_calibrated=fit.p)
        evaluated = tune.frame.assign(p_calibrated=tune.p)
        expected = add_review_scores(reference, evaluated, rule['threshold'])
        np.testing.assert_allclose(rule['score_combined'], expected.score_combined, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(~rule['predicted_negative'], expected.predicted_positive)
        for method in ['score_confidence', 'score_combined']:
            np.testing.assert_array_equal(select_weighted_review(tune, rule, tune.ones, method, .35),
                                          review_selection(expected, method, .35).astype(int))

    def test_evaluation_labels_cannot_change_rule_or_review(self):
        fit, tune = Cohort(fixture()), Cohort(fixture(100, 'tune'))
        altered = Cohort(tune.frame.assign(cancer=1 - tune.y))
        a, b = make_rule(fit, tune, fit.ones), make_rule(fit, altered, fit.ones)
        self.assertEqual(a['threshold'], b['threshold'])
        for method in ['score_confidence', 'score_combined']:
            np.testing.assert_array_equal(a[method], b[method])
            np.testing.assert_array_equal(select_weighted_review(tune, a, tune.ones, method, .35),
                                          select_weighted_review(altered, b, altered.ones, method, .35))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            make_rule(fit, Cohort(fit.frame.assign(role='tune')), fit.ones)
        with self.assertRaisesRegex(ValueError, 'Nondevelopment'):
            Cohort(fixture(role='test'))

    def test_weighted_rule_and_review_match_literal_duplicate_rows(self):
        fit, tune = Cohort(fixture()), Cohort(fixture(100, 'tune'))
        fit_w = fit.bootstrap_weights(np.random.default_rng(121))
        tune_w = tune.bootstrap_weights(np.random.default_rng(322))
        expanded_fit = fit.frame.loc[fit.frame.index.repeat(fit_w)].reset_index(drop=True)
        expanded_tune = tune.frame.loc[tune.frame.index.repeat(tune_w)].copy()
        expanded_tune['source_row'] = expanded_tune.index
        expanded_tune = expanded_tune.reset_index(drop=True)
        expected_threshold = sensitivity_threshold(expanded_fit.p_raw, expanded_fit.cancer)
        expected = add_review_scores(expanded_fit.assign(p_calibrated=expanded_fit.p_raw),
                                     expanded_tune.assign(p_calibrated=expanded_tune.p_raw),
                                     expected_threshold)
        rule = make_rule(fit, tune, fit_w)
        self.assertEqual(rule['threshold'], expected_threshold)
        np.testing.assert_allclose(np.repeat(rule['score_combined'], tune_w), expected.score_combined,
                                   rtol=0, atol=1e-15)
        for method in ['score_confidence', 'score_combined']:
            negative = expected.loc[~expected.predicted_positive].copy()
            negative['tie'] = [hashlib.sha256(f'20260909:{p}:{l}'.encode()).hexdigest()
                               for p, l in negative[['patient_id', 'laterality']].itertuples(index=False, name=None)]
            ordered = negative.sort_values([method, 'tie'], ascending=[False, True], kind='stable')
            k = math.floor(.35 * len(negative))
            expected_counts = ordered.iloc[:k].source_row.value_counts().reindex(range(len(tune.frame)), fill_value=0)
            np.testing.assert_array_equal(select_weighted_review(tune, rule, tune_w, method, .35), expected_counts)
        metrics = evaluate(tune, rule, tune_w, 'test_fixture', 0, .35)
        for key in ['n_breasts', 'positive_breasts', 'tp', 'fn', 'tn', 'fp',
                    'reviewed_confidence', 'reviewed_combined', 'flagged_confidence', 'flagged_combined']:
            self.assertEqual(metrics[0][key], sum(group[key] for group in metrics[1:]))

    def test_alternative_partitions_are_disjoint_reproducible_and_preserve_input(self):
        frame = pd.concat([fixture(), fixture(100, 'tune')], ignore_index=True)
        original = frame.copy(deep=True)
        a = list(alternative_partitions(frame, repeats=3, seed=19))
        b = list(alternative_partitions(frame, repeats=3, seed=19))
        for (fit_a, tune_a), (fit_b, tune_b) in zip(a, b):
            self.assertFalse(set(fit_a) & set(tune_a))
            self.assertEqual(set(fit_a) | set(tune_a), set(frame.patient_id))
            self.assertEqual(len(tune_a), 20)
            np.testing.assert_array_equal(fit_a, fit_b)
            np.testing.assert_array_equal(tune_a, tune_b)
        pd.testing.assert_frame_equal(frame, original)


if __name__ == '__main__':
    unittest.main()
