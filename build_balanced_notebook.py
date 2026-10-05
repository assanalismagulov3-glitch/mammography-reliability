"""Package the standalone class-balanced pilot and unchanged audited source."""
from pathlib import Path
import nbformat as nbf
from build_development_notebook import DRIVER


def main():
    root = Path(__file__).resolve().parent
    notebook = nbf.read(root/'rsna_development_pilots.ipynb', as_version=4)
    notebook.cells = notebook.cells[:-1]
    notebook.cells[0] = nbf.v4.new_markdown_cell(
        '# Class-balanced development experiment\n\n'
        'Private. Attach official RSNA data and version 1 outputs from notebooks 03, 04, and 07. '
        'GPU required. Reproduce the selected reference checkpoint on the original pilot split, '
        'then test equal positive/negative accumulation groups with the same expected weighted '
        'objective, initialization seed, architecture, resolution, optimizer, and stopping rules. '
        'Sampling includes repeated breasts; tuning remains at natural prevalence. This is '
        'adaptive development, not final hypothesis evaluation. Final test patients stay locked.'
    )
    writes = f"(PROJECT / 'run_development.py').write_text({DRIVER!r})\n"
    for filename in ['balanced_sampling.py', 'development_balanced.py']:
        source = (root/filename).read_text()
        compile(source, filename, 'exec')
        writes += f"(PROJECT / {filename!r}).write_text({source!r})\n"
    writes += "subprocess.run([sys.executable, '-u', 'development_balanced.py'], cwd=PROJECT, check=True)\n"
    notebook.cells.append(nbf.v4.new_code_cell(writes))
    nbf.validate(notebook)
    code = '\n\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    compile(code, 'balanced_notebook', 'exec')
    nbf.write(notebook, root/'rsna_development_balanced.ipynb')
    print(f'Notebook schema and syntax passed: {len(code)} executable characters.')


if __name__ == '__main__':
    main()
