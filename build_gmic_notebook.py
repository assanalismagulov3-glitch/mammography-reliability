"""Build the private GMIC compatibility notebook without changing old models."""
from pathlib import Path
import json
import nbformat as nbf


def main():
    root = Path(__file__).resolve().parent
    notebook = nbf.read(root/'rsna_development_pilots.ipynb', as_version=4)
    notebook.cells = notebook.cells[:-1]
    notebook.cells[0] = nbf.v4.new_markdown_cell(
        '# GMIC reference reproduction and technical preflight\n\n'
        'Private. Attach the official RSNA competition only. GPU and Internet required. '
        'Fetch the inspected NYU source, five pretrained weights, and public author fixtures '
        'at one pinned commit; verify every Git blob. Reproduce the released default model’s '
        'reference outputs on historical fixtures and verify current CLI parity before processing reserved fitting images. '
        'Keep source/license/NOTICE unchanged. This is an engineering check, not cancer-model '
        'validation. Visual crop review is required before any full-cohort GMIC evaluation. '
        'No RSNA tuning or final test performance is calculated.'
    )
    notebook.cells.append(nbf.v4.new_code_cell(
        "dependencies = {'imageio':'imageio>=2.34,<3', 'h5py':'h5py>=3.10,<4',\n"
        "                'cv2':'opencv-python-headless>=4.9,<5', 'matplotlib':'matplotlib>=3.8,<4',\n"
        "                'pylibjpeg':'pylibjpeg>=2,<3', 'openjpeg':'pylibjpeg-openjpeg>=2,<3',\n"
        "                'libjpeg':'pylibjpeg-libjpeg>=2,<3'}\n"
        "missing = [spec for module,spec in dependencies.items() if importlib.util.find_spec(module) is None]\n"
        "if missing:\n"
        "    subprocess.run([sys.executable,'-m','pip','install','--quiet',*missing], check=True)\n"
    ))
    writes = ''
    for name in ['gmic_adapter.py', 'gmic_preflight.py', 'gmic_upstream_manifest.json', 'gmic_historical_manifest.json']:
        source = (root/name).read_text()
        if name.endswith('.py'): compile(source, name, 'exec')
        writes += f"(PROJECT / {name!r}).write_text({source!r})\n"
    writes += "subprocess.run([sys.executable,'-u','gmic_preflight.py'],cwd=PROJECT,check=True)\n"
    notebook.cells.append(nbf.v4.new_code_cell(writes))
    nbf.validate(notebook)
    code = '\n\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    compile(code, 'gmic_notebook', 'exec')
    nbf.write(notebook, root/'rsna_gmic_preflight.ipynb')
    (root.parent/'gmic_notebook_code.txt').write_text(code)
    print(f'Notebook schema and Python syntax passed: {len(code)} executable characters.')


if __name__ == '__main__': main()
