import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_gmic_evaluation import (BREAST_KEY, FEATURES, METHODS, IMAGE_KEY, PREDICTIONS,
    PatientSampler, pool_images, fit_rule, score_features, select_review, evaluate_scores,
    bootstrap_intervals)
from mammo_reliability.evaluation import MonotonePlatt, sensitivity_threshold


def fixture(offset=0, n=40):
    rng = np.random.default_rng(73 + offset)
    frame = pd.DataFrame({'patient_id': np.repeat(np.arange(n)+offset, 2),
        'laterality': ['L', 'R'] * n, 'p_raw': rng.uniform(.001, .5, n*2),
        'view_disagreement': rng.uniform(0, .3, n*2), 'ensemble_variance': rng.uniform(0, .1, n*2)})
    outcomes = frame[BREAST_KEY].copy()
    outcomes['site_id'] = np.repeat(np.where(np.arange(n) % 2, 1, 2), 2)
    outcomes['density'] = np.where(outcomes.site_id == 1, 'C', None)
    outcomes['cancer'] = (np.arange(n*2) % 4 == 0).astype(int)
    return frame, outcomes


class FixedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dev, self.labels = fixture()
        self.test, self.outcomes = fixture(1000)
        self.rule = fit_rule(self.dev, self.labels)
        self.scored = score_features(self.test, self.rule)
        self.config = json.loads((Path(__file__).resolve().parents[1]/'fixed_gmic_final_config.json').read_text())

    def test_repeats_do_not_overweight_a_view(self):
        images = pd.DataFrame([[1, 7, 'L', 'CC'], [2, 7, 'L', 'CC'], [3, 7, 'L', 'MLO']], columns=IMAGE_KEY)
        pred = pd.DataFrame({'image_id': [1, 2, 3]})
        for col in PREDICTIONS: pred[col] = [.2, .4, .8]
        actual = pool_images(images, pred).iloc[0]
        self.assertAlmostEqual(actual.p_raw, .55)
        self.assertAlmostEqual(actual.view_disagreement, .5)
        self.assertAlmostEqual(actual.ensemble_variance, 0.)
        with self.assertRaises(ValueError): pool_images(images.iloc[:2], pred.iloc[:2])
        with self.assertRaises(ValueError): pool_images(images.assign(cancer=1), pred)

    def test_calibration_is_exact_existing_implementation(self):
        original = MonotonePlatt().fit(self.dev.p_raw, self.labels.cancer)
        self.assertEqual(self.rule['slope'], original.slope)
        self.assertEqual(self.rule['intercept'], original.intercept)
        self.assertEqual(self.rule['raw_threshold'], sensitivity_threshold(self.dev.p_raw, self.labels.cancer))

    def test_labels_and_extra_predictors_are_rejected(self):
        for col in ['cancer', 'BIRADS', 'site_id', 'density']:
            with self.assertRaises(ValueError): score_features(self.test.assign(**{col: 1}), self.rule)
            with self.assertRaises(ValueError): select_review(self.scored.assign(**{col: 1}), METHODS[0])

    def test_outcome_permutation_changes_results_not_selection(self):
        selected = select_review(self.scored, METHODS[5], .2)
        changed = self.outcomes.copy(); changed['cancer'] = 1 - changed.cancer
        a = evaluate_scores(self.scored, self.outcomes, self.rule, self.config)
        b = evaluate_scores(self.scored, changed, self.rule, self.config)
        np.testing.assert_array_equal(selected, select_review(self.scored, METHODS[5], .2))
        self.assertNotEqual(a['classification'][0]['n_positive'], b['classification'][0]['n_positive'])
        self.assertEqual(a['selection_overlap'], b['selection_overlap'])

    def test_patient_overlap_and_duplicate_identity_stop(self):
        with self.assertRaises(ValueError): score_features(self.dev, self.rule)
        with self.assertRaises(ValueError): score_features(pd.concat([self.test, self.test.iloc[[0]]]), self.rule)
        with self.assertRaises(ValueError): pool_images(pd.DataFrame(columns=IMAGE_KEY), pd.DataFrame(columns=['image_id']+PREDICTIONS))

    def test_ties_are_stable_under_row_reordering(self):
        tied = self.scored.copy(); tied[METHODS] = .5; tied['predicted_positive'] = False
        a = tied.loc[select_review(tied, METHODS[5], .13).astype(bool), BREAST_KEY]
        shuffled = tied.sample(frac=1, random_state=3).reset_index(drop=True)
        b = shuffled.loc[select_review(shuffled, METHODS[5], .13).astype(bool), BREAST_KEY]
        self.assertEqual(set(map(tuple, a.to_numpy())), set(map(tuple, b.to_numpy())))
        self.assertEqual(len(a), 10)

    def test_budget_is_global_and_subgroups_inherit_it(self):
        result = evaluate_scores(self.scored, self.outcomes, self.rule, self.config)
        pooled = next(r for r in result['review'] if r['method'] == METHODS[5] and r['budget'] == .05)
        bysite = [r for r in result['subgroup_review'] if r['method'] == METHODS[5] and r['global_budget'] == .05
                  and r['group'] in ['site_1', 'site_2']]
        self.assertEqual(sum(r['reviewed_breasts'] for r in bysite), pooled['reviewed_breasts'])
        self.assertEqual(sum(r['false_negatives_flagged'] for r in bysite), pooled['false_negatives_flagged'])

    def test_bootstrap_keeps_patients_and_site_counts(self):
        sampler = PatientSampler(self.outcomes[BREAST_KEY + ['site_id']])
        w = sampler.draw(np.random.default_rng(9))
        np.testing.assert_array_equal(w[::2], w[1::2])
        for site in [1, 2]: self.assertEqual(w[self.outcomes.site_id == site].sum(), int((self.outcomes.site_id == site).sum()))
        with self.assertRaises(ValueError): PatientSampler(self.outcomes)

    def test_weighted_review_matches_literal_duplicates(self):
        scored = self.scored.copy(); scored['predicted_positive'] = False; scored[METHODS[5]] = .5
        weights = np.tile([0, 1, 2, 3], len(scored)//4)
        actual = select_review(scored, METHODS[5], .137, weights=weights)
        import hashlib
        order = sorted(range(len(scored)), key=lambda i: hashlib.sha256(
            f'20260909:{int(scored.patient_id.iloc[i])}:{scored.laterality.iloc[i]}'.encode()).hexdigest())
        literal = [i for i in order for _ in range(weights[i])]
        selected = literal[:int(np.floor(.137*len(literal)))]; expected = np.bincount(selected, minlength=len(scored))
        np.testing.assert_array_equal(actual, expected)

    def test_weighted_calibration_matches_literal_data(self):
        w = np.repeat(np.tile([0, 1, 2, 1], 10), 2)
        rule = fit_rule(self.dev, self.labels, w)
        indices = np.repeat(np.arange(len(w)), w)
        cal = MonotonePlatt().fit(self.dev.p_raw.to_numpy()[indices], self.labels.cancer.to_numpy()[indices])
        self.assertEqual(rule['slope'], cal.slope)
        self.assertEqual(rule['raw_threshold'], sensitivity_threshold(self.dev.p_raw.to_numpy()[indices], self.labels.cancer.to_numpy()[indices]))

    def test_bootstrap_zero_fn_and_reproducibility(self):
        config = copy.deepcopy(self.config); config['uncertainty']['bootstrap_repeats'] = 6
        outcomes = self.outcomes.copy(); outcomes['cancer'] = 0
        a, rows = bootstrap_intervals(self.dev, self.labels, self.test, outcomes, self.rule, config)
        b, rows2 = bootstrap_intervals(self.dev, self.labels, self.test, outcomes, self.rule, config)
        self.assertEqual(a, b); pd.testing.assert_frame_equal(rows, rows2)
        for stats in a.values():
            self.assertEqual(stats['undefined_zero_FN'], 6)
            self.assertIsNone(stats['ci95'])
            self.assertEqual(stats['numerical_failures'], 0)


if __name__ == '__main__': unittest.main()
