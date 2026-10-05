"""Verify released file digests and the untouched scientific source boundary."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    provenance = json.loads((root/'validation/release_provenance.json').read_text())
    frozen = provenance['frozen_source_sha256']
    assert len(frozen) == 32
    for name, expected in frozen.items():
        path = root/name
        assert path.is_file() and sha(path) == expected, f'Frozen source changed: {name}'
    checked = len(frozen)
    manifest = root/'RELEASE_SHA256.json'
    if manifest.exists():
        files = json.loads(manifest.read_text())['files']
        for name, expected in files.items():
            path = root/name
            assert path.is_file() and sha(path) == expected, f'Release file changed: {name}'
        checked = len(files)
    result = json.loads((root/'validation/fixed_gmic_final_results_20261001.json').read_text())
    assert result['freeze_sha256'] == provenance['freeze_sha256']
    assert result['final_model_results_examined'] is True
    print(f'Passed: {checked} file digests checked; all 32 frozen scientific files unchanged.')


if __name__ == '__main__':
    main()
