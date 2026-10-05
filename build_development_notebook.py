"""Build the private, two-resolution development run from the audited source."""
from pathlib import Path
import json

import nbformat as nbf


DRIVER = r'''
from pathlib import Path
import json
import time

import pandas as pd
import torch

from mammo_reliability.audit import file_sha256, locate_dataset, require, write_json
from mammo_reliability.cache_parts import assemble_parts
from mammo_reliability.pipeline import initialize, require_pixel_audit
from mammo_reliability.pilot import load_pilot_cache, run_pilot


def mounted_output(slug, folder):
    # Support the current owner-qualified notebook mount and the earlier layout.
    candidates = [Path('/kaggle/input/notebooks/aseqwss') / slug / folder,
                  Path('/kaggle/input') / slug / folder]
    matches = [p for p in candidates if p.is_dir()]
    require(len(matches) == 1, f'Attach the completed private notebook {slug}; expected one of {candidates}.')
    return matches[0]


def main():
    root = Path('/kaggle/working/mammo_private_v0_2')
    config = json.loads(Path('config.json').read_text())
    data_root = locate_dataset()
    _, breasts, _, _ = initialize(data_root, root, config)
    parts = [mounted_output('03-rsna-lossless-cache-part-0', 'rsna_cache_part_0'),
             mounted_output('04-rsna-lossless-cache-part-1', 'rsna_cache_part_1')]
    expected_hashes = {
        0: 'daf16f5f19161d06a00f1ccf56cdf3c3192712244ab81a0cba7366c80296fccc',
        1: '0cb17972ade50598829f8fa5c0969db9483e3816f4369934247242ba2dd2f87a',
    }
    for part, path in enumerate(parts):
        require(file_sha256(path / 'manifest.csv') == expected_hashes[part],
                'Attached base cache differs from the verified preprocessing commit.')
    assemble_parts(parts, data_root, root / 'preprocessed',
                   config['model']['height'], config['model']['width'])
    reference = require_pixel_audit(root, config)
    roles = pd.read_csv(root / 'pilot' / 'roles_private.csv')
    high_root = mounted_output('05-rsna-pilot-preparation', 'rsna_pilot_cache_1536')
    require(file_sha256(high_root / 'manifest.csv') ==
            'bb7dbc11a23b10c51c2abafdced4feb501c370603655d0228bedd3a8a6bcfec6',
            'Attached high-resolution cache differs from the verified preprocessing commit.')
    high_model = {**config['model'], 'height': 1536, 'width': 1152}
    high = load_pilot_cache(high_root, reference, roles, high_model,
                           file_sha256(data_root / 'train.csv'))
    validation = {
        'standard_images': len(reference),
        'pilot_images_at_1536': len(high),
        'pilot_patients': int(roles.patient_id.nunique()),
        'cross_patient_exact_pixel_duplicates': 0,
        'pilot_pixels_and_roles_match_full_audit': True,
        'final_test_predictions_opened': False,
        'verified_input_versions': {
            '03-rsna-lossless-cache-part-0': 348525456,
            '04-rsna-lossless-cache-part-1': 348526048,
            '05-rsna-pilot-preparation': 348537660,
        },
    }
    write_json(root / 'aggregates' / 'combined_cache_validation.json', validation)
    print('COMBINED CACHE VALIDATION', json.dumps(validation), flush=True)
    require(torch.cuda.is_available(), 'Select a Kaggle GPU before training.')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Only reserved development patients enter these two pilot trials.', flush=True)
    results = []
    for trial, overrides, cache in [
        ('baseline_1024_v1', {}, None),
        ('resolution_1536_v1', {'height': 1536, 'width': 1152}, high_root),
    ]:
        started = time.monotonic()
        print('STARTING DEVELOPMENT TRIAL', trial, flush=True)
        result = run_pilot(root, breasts, config, trial, overrides, cache)
        result['launcher_wall_seconds'] = time.monotonic() - started
        results.append(result)
        write_json(root / 'aggregates' / 'development_pilot_results.json', results)
        print('COMPLETED DEVELOPMENT TRIAL', json.dumps(result), flush=True)
    print('DEVELOPMENT RUN COMPLETE. Final evaluation remains locked pending review.', flush=True)


if __name__ == '__main__':
    main()
'''


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'config.json').read_text())
    cells = [nbf.v4.new_markdown_cell(
        '# Reserved development cohort: two image resolutions\n\n'
        'Keep this notebook private. Attach the official RSNA competition and version 1 outputs '
        'from private notebooks 03, 04, and 05. Select a GPU. This run verifies both base-cache '
        'parts together and the higher-resolution pilot, then trains the same seed on the same '
        'reserved fitting/tuning patients at 1024 × 768 and 1536 × 1152. Each trial uses the '
        'existing eight-epoch maximum and early-stopping rule. Final test evaluation stays locked. '
        'This notebook does not establish diagnostic performance or novelty.'
    ), nbf.v4.new_code_cell(
        'from pathlib import Path\nimport json\nimport subprocess\nimport sys\nimport importlib.util\n'
        "PROJECT = Path('/kaggle/working/mammo_project')\n"
        "(PROJECT / 'mammo_reliability').mkdir(parents=True, exist_ok=True)\n"
    )]
    for path in sorted((root / 'mammo_reliability').glob('*.py')):
        cells.append(nbf.v4.new_code_cell(
            f'# Audited source: {path.name}\n'
            f"(PROJECT / 'mammo_reliability' / {path.name!r}).write_text({path.read_text()!r})\n"
        ))
    cells.append(nbf.v4.new_code_cell(
        f'CONFIG = {config!r}\n'
        "(PROJECT / 'config.json').write_text(json.dumps(CONFIG, indent=2))\n"
        "packages = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.2,<3',\n"
        "            'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2',\n"
        "            'PIL': 'Pillow>=10,<13', 'pydicom': 'pydicom>=3,<4'}\n"
        "missing = [spec for module, spec in packages.items() if importlib.util.find_spec(module) is None]\n"
        "if missing:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing], check=True)\n"
    ))
    cells.append(nbf.v4.new_code_cell(
        f"(PROJECT / 'run_development.py').write_text({DRIVER!r})\n"
        "subprocess.run([sys.executable, '-u', 'run_development.py'], cwd=PROJECT, check=True)\n"
    ))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
    })
    nbf.validate(notebook)
    for index, cell in enumerate(cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'cell_{index}', 'exec')
    compile(DRIVER, 'run_development.py', 'exec')
    target = root / 'rsna_development_pilots.ipynb'
    nbf.write(notebook, target)
    print(target)


if __name__ == '__main__':
    main()
