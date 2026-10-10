import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fixed_gmic_gate import SOURCE_FILES, RULE_FILES, source_hashes, freeze, verify_freeze
from fixed_gmic_prepare import sha, write_json, cohort_record
from fixed_gmic_final_analysis import collect_shards
from gmic_final_inference import checked_resume


class FixedGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project, self.bundle = self.root/'project', self.root/'bundle'
        for name in SOURCE_FILES:
            path = self.project/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('test fixture\n')
        self.bundle.mkdir()
        for name in RULE_FILES: (self.bundle/name).write_text('test fixture\n')
        cohort = {'holdout_images': 43787, 'holdout_patients': 9530}
        write_json(self.bundle/'cohort_manifest.json', cohort)
        write_json(self.bundle/'preparation_result.json', {'final_test_results_opened': False,
            'parity': {'review_comparisons_reproduced': 24}, 'cohort': cohort,
            'calibration_rule_sha256': sha(self.bundle/'calibration_rule_private.json')})
        self.checks = self.root/'checks.json'
        write_json(self.checks, {'passed': True, 'errors': 0, 'failures': 0, 'source_sha256': source_hashes(self.project)})

    def test_absent_freeze_and_incomplete_shards_block(self):
        with self.assertRaises(ValueError): verify_freeze(self.project, self.bundle)
        with self.assertRaises(ValueError): collect_shards([], self.bundle)

    def test_source_change_after_checks_blocks_freeze(self):
        (self.project/SOURCE_FILES[0]).write_text('changed')
        with self.assertRaises(ValueError): freeze(self.project, self.bundle, self.checks)

    def test_rule_tampering_is_detected(self):
        freeze(self.project, self.bundle, self.checks)
        verify_freeze(self.project, self.bundle)
        (self.bundle/'calibration_rule_private.json').write_text('changed')
        with self.assertRaises(ValueError): verify_freeze(self.project, self.bundle)

    def test_freeze_cannot_be_overwritten(self):
        freeze(self.project, self.bundle, self.checks)
        with self.assertRaises(ValueError): freeze(self.project, self.bundle, self.checks)

    def test_existing_predictions_block_retrospective_freeze(self):
        (self.bundle/'shard_0').mkdir()
        (self.bundle/'shard_0/image_predictions_private.csv').write_text('image_id\n')
        with self.assertRaises(ValueError): freeze(self.project, self.bundle, self.checks)

    def test_failed_checks_block_freeze(self):
        data = json.loads(self.checks.read_text()); data['passed'] = False
        write_json(self.checks, data)
        with self.assertRaises(ValueError): freeze(self.project, self.bundle, self.checks)

    def test_resume_rejects_a_wrong_run(self):
        resume = self.root/'resume'; resume.mkdir()
        write_json(resume/'run_identity.json', {'shard': 0})
        with self.assertRaises(ValueError): checked_resume(resume, [1, 2], ['image_id'], {'shard': 1})


if __name__ == '__main__': unittest.main()
