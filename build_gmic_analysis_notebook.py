"""Build a CPU-only, private reproduction and technical-review notebook."""
from pathlib import Path
import json


def main():
    root = Path(__file__).resolve().parent
    names = ['mammo_reliability/__init__.py', 'mammo_reliability/audit.py',
             'mammo_reliability/evaluation.py', 'mammo_reliability/imaging.py',
             'gmic_pilot_analysis.py', 'gmic_operating_point.py', 'GMIC_OPERATING_POINT_PLAN.md',
             'gmic_orientation_review.py']
    names += list(json.loads((root/'private_runs/gmic_reserved_pilot/declared_plan.json').read_text())['source_sha256'])
    names = list(dict.fromkeys(names))
    code = "from pathlib import Path\nimport sys, subprocess, json\n"
    code += "PROJECT = Path('/kaggle/working/mammo_analysis_project')\nPROJECT.mkdir(exist_ok=True)\n"
    for name in names:
        source = (root/name).read_text()
        if name.endswith('.py'): compile(source, name, 'exec')
        code += f"target = PROJECT / {name!r}\ntarget.parent.mkdir(parents=True, exist_ok=True)\ntarget.write_text({source!r})\n"
    expected = json.loads((root/'validation/gmic_operating_point/results.json').read_text())
    reference = {key: expected[key] for key in ['raw_threshold', 'classification', 'review']}
    code += f"REFERENCE = {reference!r}\n"
    code += """
roots = [Path('/kaggle/input/notebooks/aseqwss/11-rsna-gmic-reserved-pilot/gmic_reserved_pilot'),
         Path('/kaggle/input/11-rsna-gmic-reserved-pilot/gmic_reserved_pilot')]
existing = [p for p in roots if (p/'completed.json').is_file()]
assert len(existing) == 1, 'Attach private notebook 11 version 2 output.'
SOURCE = existing[0]
OUT = Path('/kaggle/working/gmic_operating_point')
subprocess.run([sys.executable, '-u', 'gmic_operating_point.py', '--input', str(SOURCE),
                '--output', str(OUT), '--private-output', '/kaggle/working/gmic_analysis_private'],
               cwd=PROJECT, check=True)
result = json.loads((OUT/'results.json').read_text())
assert abs(result['raw_threshold']-REFERENCE['raw_threshold']) < 1e-12
for actual, expected in zip(result['classification'], REFERENCE['classification'], strict=True):
    for key in ['role', 'site', 'tp', 'fn', 'tn', 'fp']:
        assert actual[key] == expected[key], ('Confusion counts changed', key)
for actual, expected in zip(result['review'], REFERENCE['review'], strict=True):
    for key in ['method', 'budget', 'false_negatives', 'reviewed_breasts', 'false_negatives_flagged']:
        assert actual[key] == expected[key], ('Review selection result changed', key)
print('CPU_ANALYSIS_REPRODUCED', json.dumps({'classification_groups': 6, 'review_comparisons': 24,
        'final_test_results_opened': False}))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pydicom>=3,<4',
                'pylibjpeg>=2,<3', 'pylibjpeg-openjpeg>=2,<3', 'pylibjpeg-libjpeg>=2.1,<3'], check=True)
# Fresh process is essential: pydicom discovers decoder plugins at import time,
# before the optional installation above. Do not reuse a stale decoder registry.
subprocess.run([sys.executable, '-c',
                'import sys; from gmic_orientation_review import run; run(sys.argv[1], sys.argv[2])',
                str(SOURCE), '/kaggle/working/gmic_orientation_review'], cwd=PROJECT, check=True)
print('DEVELOPMENT_ANALYSIS_COMPLETE', json.dumps({'analysis_reproduced': True,
        'orientation_display_prepared': True, 'visual_review_pending': True,
        'final_test_results_opened': False}))
"""
    compile(code, 'gmic_analysis_notebook', 'exec')
    introduction = '# GMIC development analysis and orientation review\n\nKeep private. CPU only. '
    introduction += (
        'Attach notebook 11 version 2 and the official RSNA competition. '
        'Reproduce fixed development thresholds/review counts; display all three prior '
        'fit-image orientation flags. No model training, threshold retuning, final inference, '
        'or clinical image interpretation. See the embedded analysis plan.')
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'id': 'purpose', 'cell_type': 'markdown', 'metadata': {}, 'source': introduction},
                  {'id': 'analysis', 'cell_type': 'code', 'metadata': {}, 'source': code,
                   'execution_count': None, 'outputs': []}]}
    path = root/'rsna_gmic_operating_point.ipynb'
    path.write_text(json.dumps(nb, indent=1)+'\n')
    assert json.loads(path.read_text())['cells'][1]['source'] == code
    (root.parent/'gmic_analysis_notebook_code.txt').write_text(code)
    print('Notebook JSON roundtrip and Python syntax passed:', len(code), 'characters.')


if __name__ == '__main__': main()
