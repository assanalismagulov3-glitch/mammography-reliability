"""Regenerate both standalone private cache notebooks from their source modules."""
from pathlib import Path
import nbformat as nbf


def main():
    root = Path(__file__).resolve().parent
    for part in (0, 1):
        notebook = nbf.v4.new_notebook()
        notebook.cells = [nbf.v4.new_markdown_cell(
            f'# Private RSNA lossless cache part {part + 1}/2\n\n'
            'Technical preprocessing only. Keep this notebook and its outputs private. '
            'No cancer-model evaluation is performed. Storage partitions are independent of evaluation folds.')]
        notebook.cells.append(nbf.v4.new_code_cell(
            "from pathlib import Path\nimport sys\nimport subprocess\n"
            "PROJECT = Path('/kaggle/working/cache_project')\nPROJECT.mkdir(exist_ok=True)\n"
            "(PROJECT / 'mammo_reliability').mkdir(exist_ok=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'pydicom>=3,<4', "
            "'pylibjpeg>=2,<3', 'pylibjpeg-openjpeg>=2,<3', 'pylibjpeg-libjpeg>=2,<3'], check=True)\n"))
        for filename in ('__init__.py', 'audit.py', 'imaging.py', 'cache_parts.py'):
            source = (root / 'mammo_reliability' / filename).read_text()
            notebook.cells.append(nbf.v4.new_code_cell(
                f'SOURCE = {source!r}\n(PROJECT / "mammo_reliability" / {filename!r}).write_text(SOURCE)\n'))
        notebook.cells.append(nbf.v4.new_code_cell(
            f"subprocess.run([sys.executable, '-m', 'mammo_reliability.cache_parts', 'build', "
            f"'--output', '/kaggle/working/rsna_cache_part_{part}', '--part', '{part}', "
            "'--n-parts', '2', '--workers', '4'], cwd=PROJECT, check=True)\n"))
        notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
        nbf.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, '<cache notebook>', 'exec')
        nbf.write(notebook, root / f'rsna_cache_part_{part}.ipynb')


if __name__ == '__main__':
    main()
