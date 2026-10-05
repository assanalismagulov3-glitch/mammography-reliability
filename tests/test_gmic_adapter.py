"""Synthetic invariants for orientation and external artifact identity."""
from pathlib import Path
import hashlib
import tempfile
import unittest
import numpy as np

from gmic_adapter import digest_file, infer_horizontal_flip, filter_legacy_state


class AdapterTests(unittest.TestCase):
    def test_checkpoint_exception_cannot_hide_active_weight_errors(self):
        active = object()
        state = {'active.weight': active, 'shared_rep_filter.weight': object()}
        filtered, ignored = filter_legacy_state(state, ['active.weight'])
        self.assertIs(filtered['active.weight'], active)
        self.assertEqual(ignored, ['shared_rep_filter.weight'])
        self.assertIn('shared_rep_filter.weight', state)
        with self.assertRaises(ValueError): filter_legacy_state({'wrong.weight': active}, ['active.weight'])
        with self.assertRaises(ValueError): filter_legacy_state({}, ['active.weight'])

    def test_mirror_and_laterality_give_same_final_orientation(self):
        image = np.zeros((20, 30), dtype=np.uint16)
        image[2:18, :10] = 500
        image[5:15, 10:20] = 100
        for view in ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']:
            for mirrored in [False, True]:
                raw = np.fliplr(image) if mirrored else image
                flag, info = infer_horizontal_flip(raw, view)
                flip_needed = (flag == 'NO' and view.startswith('R-')) or (flag == 'YES' and view.startswith('L-'))
                normalized = np.fliplr(raw) if flip_needed else raw
                np.testing.assert_array_equal(normalized, image)
                gain_flag, gain_info = infer_horizontal_flip(raw.astype(float)*3, view)
                self.assertEqual(flag, gain_flag)
                self.assertAlmostEqual(info['edge_mass_contrast'], gain_info['edge_mass_contrast'])

    def test_invalid_and_ambiguous_images_are_not_silently_accepted_as_validated(self):
        for bad in [np.zeros((20, 20)), np.ones((20, 20)), np.full((20,20), np.nan), np.ones((4, 20))]:
            with self.assertRaises(ValueError): infer_horizontal_flip(bad, 'L-CC')
        symmetric = np.zeros((20,20)); symmetric[2:18, :] = 1
        _, info = infer_horizontal_flip(symmetric, 'R-MLO')
        self.assertEqual(info['edge_mass_contrast'], 0)
        self.assertTrue(info['orientation_requires_visual_review'])

    def test_git_content_identity_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture'
            path.write_bytes(b'abc')
            expected = hashlib.sha1(b'blob 3\0abc').hexdigest()
            self.assertEqual(digest_file(path, 'sha1', git_blob=True), expected)
            before = digest_file(path)
            path.write_bytes(b'abd')
            self.assertNotEqual(digest_file(path, 'sha1', git_blob=True), expected)
            self.assertNotEqual(digest_file(path), before)


if __name__ == '__main__': unittest.main()
