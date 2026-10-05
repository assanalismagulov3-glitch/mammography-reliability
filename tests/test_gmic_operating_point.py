"""Isolation and global-budget regression checks; artificial data only."""
import numpy as np
import pandas as pd
import unittest
from gmic_operating_point import fit_development_rule, apply_development_rule, subgroup_reviews
from mammo_reliability.evaluation import review_at_budget, review_selection


def fixture():
    return pd.DataFrame({'patient_id': np.arange(20), 'laterality': ['L'] * 20,
        'site_id': [1] * 10 + [2] * 10, 'role': ['fit'] * 20,
        'cancer': [0, 1, 0, 1, 0] * 4,
        'p_raw': np.linspace(.001, .95, 20),
        'view_disagreement': np.linspace(.1, .7, 20),
        'ensemble_variance': np.linspace(.03, .01, 20)})


class TestOperatingPoint(unittest.TestCase):
    def test_tune_labels_cannot_change_scores_threshold_or_selections(self):
        fit = fixture()
        model, ref, threshold, raw = fit_development_rule(fit)
        tune = fixture().assign(patient_id=lambda f: f.patient_id + 100, role='tune')
        a = apply_development_rule(model, ref, tune, threshold)
        changed = tune.assign(cancer=1-tune.cancer)
        b = apply_development_rule(model, ref, changed, threshold)
        columns = ['p_calibrated', 'score_combined', 'predicted_positive']
        pd.testing.assert_frame_equal(a[columns], b[columns])
        pd.testing.assert_series_equal(review_selection(a, 'score_combined', .5),
                                       review_selection(b, 'score_combined', .5))
        assert np.array_equal(a.predicted_positive, a.p_raw >= raw)
        with self.assertRaisesRegex(ValueError, 'Fit-only'):
            fit_development_rule(tune)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            apply_development_rule(model, ref, fit.assign(role='tune'), threshold)

    def test_site_counts_inherit_global_selection_and_do_not_reallocate(self):
        f = fixture().assign(predicted_positive=False, score_combined=np.arange(20))
        rows = subgroup_reviews(f, 'score_combined', .2)
        assert [r['reviewed_breasts'] for r in rows] == [0, 4]
        overall = review_at_budget(f, 'score_combined', .2)
        for column in ['reviewed_breasts', 'false_negatives', 'false_negatives_flagged']:
            assert sum(r[column] for r in rows) == overall[column]


if __name__ == '__main__':
    unittest.main()
