"""Build private diagnostics without changing the earlier model package."""
from pathlib import Path
import nbformat as nbf
from build_development_notebook import DRIVER


def main():
    root = Path(__file__).resolve().parent
    notebook = nbf.read(root / 'rsna_development_pilots.ipynb', as_version=4)
    notebook.cells = notebook.cells[:-1]
    notebook.cells[0] = nbf.v4.new_markdown_cell(
        '# Development diagnostics and one stability trial\n\n'
        'Keep private. Attach the official RSNA competition and verified version 1 outputs '
        'from notebooks 03 and 04. Use a GPU. Only reserved development patients are used. '
        'The 16-breast memorization check measures fitting behavior, not generalization. '
        'A frozen ImageNet linear probe uses the existing fit/tune split. One lower-learning-rate, '
        'larger-accumulation pilot runs only after the declared memorization gate passes. '
        'This is adaptive development following weak initial pilots. Final evaluation stays locked.'
    )
    script = (root / 'development_diagnostics.py').read_text()
    notebook.cells.append(nbf.v4.new_code_cell(
        f"(PROJECT / 'run_development.py').write_text({DRIVER!r})\n"
        f"(PROJECT / 'development_diagnostics.py').write_text({script!r})\n"
        "subprocess.run([sys.executable, '-u', 'development_diagnostics.py'], cwd=PROJECT, check=True)\n"
    ))
    nbf.validate(notebook)
    code = '\n\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    compile(code, 'diagnostics_notebook', 'exec')
    compile(script, 'development_diagnostics.py', 'exec')
    path = root / 'rsna_development_diagnostics.ipynb'
    nbf.write(notebook, path)
    print(f'{path}: {len(code)} executable characters; schema and syntax passed.')


if __name__ == '__main__':
    main()
