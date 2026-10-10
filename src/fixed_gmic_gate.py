"""Immutable fixed-model gate, separate from the historical trained-CNN gate."""
from pathlib import Path
from datetime import datetime, timezone
import json

from fixed_gmic_prepare import sha, write_json
from mammo_reliability.audit import require

SOURCE_FILES = [
    'fixed_gmic_evaluation.py', 'fixed_gmic_prepare.py', 'fixed_gmic_gate.py',
    'gmic_final_inference.py', 'fixed_gmic_final_analysis.py', 'build_fixed_gmic_notebooks.py',
    'tests/test_fixed_gmic_evaluation.py', 'tests/test_fixed_gmic_gate.py',
    'FIXED_GMIC_FINAL_PROTOCOL.md', 'fixed_gmic_final_config.json',
    'gmic_adapter.py', 'gmic_preflight.py', 'gmic_fullfield.py', 'gmic_fullfield_preflight.py',
    'gmic_pilot.py', 'gmic_pilot_analysis.py', 'gmic_operating_point.py',
    'gmic_upstream_manifest.json', 'gmic_historical_manifest.json', 'gmic_visual_review.json',
    'config.json', 'RELATED_WORK_DECISION_2026_09_28.md']
SOURCE_FILES += [f'mammo_reliability/{name}.py' for name in
    ['__init__', 'audit', 'cache_parts', 'evaluation', 'imaging', 'model', 'pilot', 'pipeline', 'study', 'training']]
RULE_FILES = ['calibration_rule_private.json', 'cohort_manifest.json', 'preparation_result.json',
              'development_features_private.csv', 'development_outcomes_private.csv']


def source_hashes(project):
    return {name: sha(Path(project)/name) for name in SOURCE_FILES}


def check_payload(project, bundle, checks):
    project, bundle = Path(project), Path(bundle)
    require(checks['passed'] is True and checks['failures'] == checks['errors'] == 0, 'Software checks failed.')
    require(checks['source_sha256'] == source_hashes(project), 'Source changed after checks.')
    result = json.loads((bundle/'preparation_result.json').read_text())
    require(result['final_test_results_opened'] is False, 'Final outcomes entered preparation.')
    require(result['parity']['review_comparisons_reproduced'] == 24, 'Historical parity incomplete.')
    require(result['calibration_rule_sha256'] == sha(bundle/'calibration_rule_private.json'), 'Calibration changed.')
    require(result['cohort'] == json.loads((bundle/'cohort_manifest.json').read_text()), 'Cohort manifest changed.')
    require(result['cohort']['holdout_images'] == 43787 and result['cohort']['holdout_patients'] == 9530,
            'Unexpected holdout coverage.')
    return result


def freeze(project, bundle, checks_path):
    project, bundle = Path(project), Path(bundle)
    path = bundle/'frozen_fixed_gmic.json'
    require(not path.exists(), 'A freeze already exists; never overwrite it.')
    require(not any(bundle.glob('shard_*/image_predictions_private.csv')), 'Predictions already exist; retrospective freeze forbidden.')
    checks = json.loads(Path(checks_path).read_text())
    check_payload(project, bundle, checks)
    record = {'schema_version': '0.3', 'status': 'frozen_for_runtime_verified_inference',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': source_hashes(project), 'artifact_sha256': {n: sha(bundle/n) for n in RULE_FILES},
        'software_checks_sha256': sha(checks_path), 'software_checks': checks,
        'final_model_results_examined_before_freeze': False,
        'runtime_requirements': ['exact mounted cohort identities', 'original dataset metadata checksum',
            'verified upstream model files', 'development image fixture within 1e-6',
            'complete five-shard coverage before final label join'],
        'legacy_trained_cnn_gate_modified': False}
    write_json(path, record)
    return record


def verify_freeze(project, bundle):
    project, bundle = Path(project), Path(bundle)
    path = bundle/'frozen_fixed_gmic.json'
    require(path.is_file(), 'Missing fixed-model freeze; inference is blocked.')
    record = json.loads(path.read_text())
    require(record['status'] == 'frozen_for_runtime_verified_inference', 'Invalid freeze status.')
    require(record['source_sha256'] == source_hashes(project), 'Frozen source/configuration changed.')
    require(record['artifact_sha256'] == {n: sha(bundle/n) for n in RULE_FILES}, 'Frozen rule/cohort changed.')
    check_payload(project, bundle, record['software_checks'])
    return record
