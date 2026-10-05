"""Build a standalone Kaggle notebook from the readable source package."""
from pathlib import Path
import json

import nbformat as nbf


def main():
    root = Path(__file__).resolve().parent
    notebook = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(
        "# RSNA mammography reliability workflow\n\n"
        "Continue the existing project: audit → image checks → development training → frozen evaluation. "
        "Keep this notebook **private** because execution writes patient-level tables and derived images.\n\n"
        "**No real-data experiment results are embedded in this file.** Local software tests used synthetic DICOMs. "
        "The initial stage is `audit`. Attach the official RSNA competition dataset before running it. "
        "The source and working protocol are in the accompanying project archive.\n\n"
        "Before final evaluation: inspect real preprocessing, run development-only pilots, check resolution and resources, "
        "and freeze the experimental settings. This is research software, not a clinical system."
    )]
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\nimport os\nimport sys\nimport json\nimport subprocess\nimport importlib.util\n\n"
        "PROJECT = Path(os.environ.get('MAMMO_PROJECT_ROOT', '/kaggle/working/mammo_project'))\n"
        "PROJECT.mkdir(parents=True, exist_ok=True)\n"
        "(PROJECT / 'mammo_reliability').mkdir(exist_ok=True)\n"
        "RUN_ROOT = Path('/kaggle/working/mammo_private_v0_2')\n"
        "print('Source directory:', PROJECT)\n"
        "print('Private experiment outputs:', RUN_ROOT)\n"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "The following cells materialize the source modules. Run them in order; "
        "they do not load the mammograms or start training. The same code is supplied as ordinary `.py` files."
    ))
    for path in sorted((root / "mammo_reliability").glob("*.py")):
        code = path.read_text()
        # Keep the source readable inside a raw string when it contains no conflicting delimiter.
        literal = "r'''" + code + "'''" if "'''" not in code else repr(code)
        cell = nbf.v4.new_code_cell(
            f"# Source module: {path.name}\n"
            f"SOURCE = {literal}\n"
            f"(PROJECT / 'mammo_reliability' / {path.name!r}).write_text(SOURCE)\n"
        )
        cell.metadata["jupyter"] = {"source_hidden": True}
        cells.append(cell)
    config = json.loads((root / "config.json").read_text())
    cells.append(nbf.v4.new_markdown_cell(
        "## Choose the execution stage\n\n"
        "`audit` reads metadata and reserves a separate development cohort. `prepare` performs the full pixel audit. "
        "`pilot` trains only reserved development patients. `freeze` records the completed development review. "
        "`train` fits one fold's three members after the freeze. `evaluate` opens that fold's held-out results. "
        "`summarize` requires all five evaluated folds. GPU training is needed for `pilot` and `train`.\n\n"
        "These configuration values are provisional until a real-data development pilot is reviewed. "
        "Changing them after starting a run requires a separate output directory. "
        "Attach the completed private cache parts and assemble them before training."
    ))
    cells.append(nbf.v4.new_code_cell(
        "STAGE = 'audit'\nFOLD = 0\n"
        "PILOT_TRIAL = 'baseline_1024_v1'\nPILOT_MODEL_OVERRIDES = {}\n"
        "PILOT_CACHE_ROOT = None\nDEVELOPMENT_REVIEW = None\n\n"
        f"CONFIG = {config!r}\n\n"
        "(PROJECT / 'config.json').write_text(json.dumps(CONFIG, indent=2) + '\\n')\n"
    ))
    cells.append(nbf.v4.new_code_cell(
        "# Audit dependencies; Torch is needed only for training and inference.\n"
        "packages = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.2,<3',\n"
        "            'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2',\n"
        "            'PIL': 'Pillow>=10,<13', 'pydicom': 'pydicom>=3,<4'}\n"
        "missing = [spec for module, spec in packages.items() if importlib.util.find_spec(module) is None]\n"
        "if missing:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing], check=True)\n"
        "if STAGE in {'prepare', 'pilot', 'train', 'evaluate'}:\n"
        "    decoders = {'pylibjpeg': 'pylibjpeg>=2,<3', 'openjpeg': 'pylibjpeg-openjpeg>=2,<3', 'libjpeg': 'pylibjpeg-libjpeg>=2,<3'}\n"
        "    missing_decoders = [spec for module, spec in decoders.items() if importlib.util.find_spec(module) is None]\n"
        "    if missing_decoders:\n"
        "        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing_decoders], check=True)\n"
        "if STAGE in {'pilot', 'freeze', 'train', 'evaluate'} and importlib.util.find_spec('torch') is None:\n"
        "    raise RuntimeError('PyTorch is missing. Select a suitable Kaggle GPU environment.')\n"
    ))
    cells.append(nbf.v4.new_code_cell(
        "# The subprocess gives every run a fresh Python import state.\n"
        "command = [sys.executable, '-m', 'mammo_reliability.pipeline', STAGE,\n"
        "           '--run-root', str(RUN_ROOT), '--config', str(PROJECT / 'config.json'),\n"
        "           '--fold', str(FOLD)]\n"
        "if STAGE == 'pilot':\n"
        "    pilot_config_path = PROJECT / 'pilot_model_overrides.json'\n"
        "    pilot_config_path.write_text(json.dumps(PILOT_MODEL_OVERRIDES))\n"
        "    command += ['--trial', PILOT_TRIAL, '--pilot-model-config', str(pilot_config_path)]\n"
        "    if PILOT_CACHE_ROOT is not None:\n"
        "        command += ['--pilot-cache-root', str(PILOT_CACHE_ROOT)]\n"
        "if STAGE == 'freeze':\n"
        "    if DEVELOPMENT_REVIEW is None:\n"
        "        raise ValueError('Provide the completed development review before freezing.')\n"
        "    command += ['--development-review', str(DEVELOPMENT_REVIEW)]\n"
        "subprocess.run(command, cwd=PROJECT, check=True)\n"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "Save the notebook and its outputs privately after completed stages. "
        "Checkpoints can resume interrupted runs only while their files remain available. "
        "Before a later public release, select source and aggregate outputs and remove patient-level records, images, and credentials. "
        "Record actual results and AI assistance in the project documentation."
    ))
    notebook.cells = cells
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}}
    nbf.validate(notebook)
    target = root / "rsna_reliability_workflow.ipynb"
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
