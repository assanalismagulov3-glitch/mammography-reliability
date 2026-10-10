"""Verify the compact package, included original source and saved results."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    provenance = json.loads((root / 'validation/release_provenance.json').read_text())
    layout = json.loads((root / 'validation/submission_manifest.json').read_text())
    frozen = provenance['frozen_source_sha256']
    included = layout['included_frozen_files']
    historical = layout['historical_only_frozen_files']
    if len(frozen) != 32 or set(included) & set(historical) or set(included) | set(historical) != set(frozen):
        raise ValueError('Frozen-source inventory is incomplete or overlapping.')
    if len(set(included.values())) != len(included):
        raise ValueError('Duplicate source destinations.')
    for original, current in included.items():
        if sha(root / current) != frozen[original]:
            raise ValueError(f'Included frozen source changed: {original}')
    manifest = json.loads((root / 'RELEASE_SHA256.json').read_text())['files']
    for name, expected in manifest.items():
        if sha(root / name) != expected:
            raise ValueError(f'Package file changed: {name}')
    result_path = root / 'validation/fixed_gmic_final_results_20261001.json'
    result = json.loads(result_path.read_text())
    if sha(result_path) != provenance['aggregate_result_sha256']:
        raise ValueError('Original aggregate result changed.')
    if result['freeze_sha256'] != provenance['freeze_sha256'] or result['final_model_results_examined'] is not True:
        raise ValueError('Original result/freeze identity does not agree.')
    print(f'Passed: {len(manifest)} package file digests; {len(included)} included frozen files unchanged.')
    print(f'{len(historical)} original context files remain at Git revision {layout["historical_commit"]}.')
    print('This check does not claim to rerun private inference or verify absent historical files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    verify(args.root.resolve())


if __name__ == '__main__':
    main()
