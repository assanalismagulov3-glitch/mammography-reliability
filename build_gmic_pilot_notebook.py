"""Build a gated private fixed-model development notebook."""
from pathlib import Path
import json
import nbformat as nbf


def main():
    root = Path(__file__).resolve().parent
    review = json.loads((root/'gmic_visual_review.json').read_text())
    if not review['engineering_visual_review_passed']:
        raise ValueError('Review real image geometry before creating an executable pilot.')
    notebook = nbf.read(root/'rsna_gmic_fullfield_preflight.ipynb', as_version=4)
    notebook.cells = notebook.cells[:-1]
    notebook.cells[0] = nbf.v4.new_markdown_cell(
        '# Fixed GMIC ensemble on reserved development patients\n\n'
        'Keep private. Attach the official RSNA competition, notebook 09 version 4, '
        'and the reviewed notebook 10 version recorded in gmic_visual_review.json. GPU required. '
        'Evaluate all five fixed external checkpoints on the existing pilot cohort only. '
        'This is an adapted transfer-feasibility study, not the final reliability experiment. '
        'No final patients, calibration, threshold selection, or density-effect testing. '
        'Patient predictions are private; only aggregate development metrics are logged.'
    )
    names = ['gmic_adapter.py', 'gmic_preflight.py', 'gmic_fullfield.py', 'gmic_fullfield_preflight.py',
             'gmic_pilot_analysis.py', 'gmic_pilot.py', 'gmic_upstream_manifest.json',
             'gmic_historical_manifest.json', 'gmic_visual_review.json']
    writes = ''
    for name in names:
        source = (root/name).read_text()
        if name.endswith('.py'): compile(source, name, 'exec')
        writes += f"(PROJECT / {name!r}).write_text({source!r})\n"
    writes += "subprocess.run([sys.executable,'-u','gmic_pilot.py'],cwd=PROJECT,check=True)\n"
    notebook.cells.append(nbf.v4.new_code_cell(writes))
    nbf.validate(notebook)
    code = '\n\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    compile(code, 'gmic_pilot_notebook', 'exec')
    nbf.write(notebook, root/'rsna_gmic_reserved_pilot.ipynb')
    (root.parent/'gmic_pilot_notebook_code.txt').write_text(code)
    print('GMIC pilot notebook schema and syntax passed:', len(code), 'characters.')


if __name__ == '__main__': main()
