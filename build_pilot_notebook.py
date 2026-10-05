"""Prepare a CPU notebook for independent pilot assignment and higher resolution."""
from pathlib import Path
import json

import nbformat as nbf


def main():
    root = Path(__file__).resolve().parent
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell(
        '# Private RSNA development cohort and higher-resolution cache\n\n'
        'Reserve development patients before model results. Prepare 1536 × 1152 images from original DICOMs '
        'for those patients only. No model is trained by this notebook. '
        'PNG compression preserves the resized pixel values; resizing still removes spatial detail. '
        'Keep outputs private.')]
    notebook.cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\nimport json\nimport sys\nimport subprocess\n"
        "PROJECT = Path('/kaggle/working/pilot_project')\nPROJECT.mkdir(exist_ok=True)\n"
        "(PROJECT / 'mammo_reliability').mkdir(exist_ok=True)\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'pydicom>=3,<4', "
        "'pylibjpeg>=2,<3', 'pylibjpeg-openjpeg>=2,<3', 'pylibjpeg-libjpeg>=2,<3'], check=True)\n"))
    for path in sorted((root / 'mammo_reliability').glob('*.py')):
        notebook.cells.append(nbf.v4.new_code_cell(
            f'SOURCE = {path.read_text()!r}\n(PROJECT / "mammo_reliability" / {path.name!r}).write_text(SOURCE)\n'))
    config = json.loads((root / 'config.json').read_text())
    notebook.cells.append(nbf.v4.new_code_cell(
        f'CONFIG = {config!r}\n'
        "(PROJECT / 'config.json').write_text(json.dumps(CONFIG, indent=2))\n"
        "subprocess.run([sys.executable, '-m', 'mammo_reliability.pipeline', 'audit', "
        "'--config', str(PROJECT / 'config.json'), '--run-root', '/kaggle/working/study_v0_2'], cwd=PROJECT, check=True)\n"
        "print(Path('/kaggle/working/study_v0_2/aggregates/cohort_patient_counts.csv').read_text(), flush=True)\n"
        "print(Path('/kaggle/working/study_v0_2/aggregates/cohort_breast_counts.csv').read_text(), flush=True)\n"
        "subprocess.run([sys.executable, '-m', 'mammo_reliability.pilot', "
        "'--config', str(PROJECT / 'config.json'), '--output', '/kaggle/working/rsna_pilot_cache_1536', "
        "'--height', '1536', '--width', '1152', '--workers', '4'], cwd=PROJECT, check=True)\n"))
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
    nbf.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<pilot notebook>', 'exec')
    nbf.write(notebook, root / 'rsna_pilot_preparation.ipynb')


if __name__ == '__main__':
    main()
