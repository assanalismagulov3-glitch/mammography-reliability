"""Build the private full-field adaptation check."""
from pathlib import Path
import nbformat as nbf


def main():
    root=Path(__file__).resolve().parent
    notebook=nbf.read(root/'rsna_development_pilots.ipynb',as_version=4)
    notebook.cells=notebook.cells[:-1]
    notebook.cells[0]=nbf.v4.new_markdown_cell(
        '# Full-field GMIC geometry check\n\n'
        'Private. Attach the official RSNA competition and notebook 09 version 4. GPU required. '
        'Reuse the verified fixed external models; inspect the same 20 reserved fitting images '
        'with full-image aspect-preserving downscaling and padding. This explicitly adapts the '
        'NYU preprocessing after the original crop excluded breast tissue. No cohort accuracy, '
        'tuning or final test evaluation occurs. Visual review must precede full pilot inference.'
    )
    original=nbf.read(root/'rsna_gmic_preflight.ipynb',as_version=4)
    notebook.cells.append(original.cells[-2])
    writes=''
    for name in ['gmic_adapter.py','gmic_preflight.py','gmic_fullfield.py','gmic_fullfield_preflight.py',
                 'gmic_upstream_manifest.json','gmic_historical_manifest.json']:
        source=(root/name).read_text()
        if name.endswith('.py'):compile(source,name,'exec')
        writes+=f"(PROJECT / {name!r}).write_text({source!r})\n"
    writes+="subprocess.run([sys.executable,'-u','gmic_fullfield_preflight.py'],cwd=PROJECT,check=True)\n"
    notebook.cells.append(nbf.v4.new_code_cell(writes))
    nbf.validate(notebook)
    code='\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
    compile(code,'fullfield_notebook','exec')
    nbf.write(notebook,root/'rsna_gmic_fullfield_preflight.ipynb')
    (root.parent/'gmic_fullfield_notebook_code.txt').write_text(code)
    print('Full-field notebook schema and syntax passed:',len(code),'characters.')


if __name__=='__main__':main()
