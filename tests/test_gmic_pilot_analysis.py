"""Guard scientific pooling against repeat-view bias and cohort mismatch."""
import unittest
import numpy as np
import pandas as pd
from gmic_pilot_analysis import aggregate_predictions, PREDICTION_COLUMNS


def fixture():
    images = pd.DataFrame({'image_id': [10, 11, 12, 13, 14], 'patient_id': [1]*5,
                           'laterality': ['L']*5, 'view': ['CC']*4 + ['MLO'],
                           'site_id': [1]*5, 'role': ['fit']*5, 'cancer': [1]*5})
    predictions = pd.DataFrame({'image_id': [10, 11, 12, 13, 14]})
    for column in PREDICTION_COLUMNS:
        predictions[column] = [0.0]*4+[1.0] if column.endswith('malignant') else [0.9]*5
    return images, predictions


class PoolingTests(unittest.TestCase):
    def test_repeated_view_cannot_dominate_and_malignant_channel_is_used(self):
        images, predictions = fixture()
        result = aggregate_predictions(images, predictions.sample(frac=1, random_state=7)).iloc[0]
        self.assertEqual(result.p_raw, .5)  # Flat image mean would incorrectly be .2.
        self.assertEqual(result.p_cc, 0)
        self.assertEqual(result.p_mlo, 1)
        self.assertEqual(result.view_disagreement, 1)
        self.assertEqual(result.ensemble_variance, 0)

    def test_missing_extra_or_duplicate_predictions_fail(self):
        images, predictions = fixture()
        invalid = [predictions.iloc[:-1], pd.concat([predictions, predictions.iloc[:1]])]
        extra = predictions.copy(); extra.loc[0, 'image_id'] = 999
        invalid.append(extra)
        for frame in invalid:
            with self.assertRaises(ValueError): aggregate_predictions(images, frame)

    def test_missing_view_final_role_or_bad_probabilities_fail(self):
        images, predictions = fixture()
        for column, value in [('view', 'CC'), ('role', 'test')]:
            changed = images.copy(); changed[column] = value
            with self.assertRaises(ValueError): aggregate_predictions(changed, predictions)
        for value in [np.nan, -0.1, 1.1]:
            changed = predictions.copy(); changed.loc[0, 'm1_malignant'] = value
            with self.assertRaises(ValueError): aggregate_predictions(images, changed)


if __name__ == '__main__': unittest.main()
