"""Create private notebooks with exact source bytes and the immutable rule bundle."""
from pathlib import Path
import argparse
import hashlib
import json

from fixed_gmic_gate import SOURCE_FILES, RULE_FILES, verify_freeze


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True),
              'id': hashlib.sha256(source.encode()).hexdigest()[:12]}
    if kind == 'code': result.update(execution_count=None, outputs=[])
    return result


def emit_notebook(path, title, setup, launch):
    value = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                     'language_info': {'name': 'python', 'version': '3.12'}},
        'cells': [cell('markdown', title), cell('code', setup), cell('code', launch)]}
    compile(setup+'\n'+launch, str(path), 'exec')
    path.write_text(json.dumps(value, indent=1)+'\n')


def main(bundle, output):
    root = Path(__file__).resolve().parent
    bundle, output = Path(bundle), Path(output)
    verify_freeze(root, bundle)
    output.mkdir(parents=True, exist_ok=True)
    setup = "from pathlib import Path\nimport json, subprocess, sys, importlib.util\nPROJECT=Path('/kaggle/working/fixed_gmic_project')\nBUNDLE=Path('/kaggle/working/fixed_gmic_bundle')\nPROJECT.mkdir(parents=True,exist_ok=True)\nBUNDLE.mkdir(parents=True,exist_ok=True)\n"
    for name in SOURCE_FILES:
        source = (root/name).read_text()
        if name.endswith('.py'): compile(source, name, 'exec')
        setup += f"(PROJECT/{name!r}).parent.mkdir(parents=True,exist_ok=True)\n(PROJECT/{name!r}).write_text({source!r})\n"
    for name in RULE_FILES+['frozen_fixed_gmic.json']:
        setup += f"(BUNDLE/{name!r}).write_text({(bundle/name).read_text()!r})\n"
    setup += "sys.path.insert(0,str(PROJECT))\n"
    common = '# Private fixed-GMIC evaluation\n\nKeep private: embedded calibration identifiers and patient-level outputs. No public release. '
    cpu = """
import unittest
import numpy as np
import pandas as pd
from fixed_gmic_gate import verify_freeze
from fixed_gmic_prepare import development_inputs, verify_development_parity, cohort_record, sha, write_json
from fixed_gmic_evaluation import score_features, BREAST_KEY, FEATURES
from mammo_reliability.audit import locate_dataset, require
verify_freeze(PROJECT,BUNDLE)
loader=unittest.TestLoader()
suite=unittest.TestSuite([loader.discover(str(PROJECT/'tests'),pattern=pattern) for pattern in
    ['test_fixed_gmic_evaluation.py','test_fixed_gmic_gate.py']])
result=unittest.TextTestRunner(verbosity=2).run(suite)
require(result.wasSuccessful(),'Kaggle software tests failed.')
def mounted(slug,child):
    candidates=[Path('/kaggle/input/notebooks/aseqwss')/slug/child,Path('/kaggle/input')/slug/child]
    found=[p for p in candidates if p.is_dir()]
    require(len(found)==1,f'Attach the exact private notebook output: {slug}')
    return found[0]
base=mounted('09-rsna-gmic-transfer-preflight','gmic_transfer_preflight')
development=mounted('11-rsna-gmic-reserved-pilot','gmic_reserved_pilot')
config=json.loads((PROJECT/'fixed_gmic_final_config.json').read_text())
require(sha(locate_dataset()/'train.csv')==config['provenance']['metadata_sha256'],'Metadata changed.')
import os
os.chdir(PROJECT)
breasts,features,outcomes,provenance=development_inputs(development)
parity=verify_development_parity(breasts,config)
cohort=cohort_record(pd.read_csv(base/'cohort_identity/patients_private.csv'),
    pd.read_csv(base/'cohort_identity/standard_images_private.csv'),config,
    pd.read_csv(development/'roles_private.csv'))
require(cohort==json.loads((BUNDLE/'cohort_manifest.json').read_text()),'Original cohort does not match.')
rule=json.loads((BUNDLE/'calibration_rule_private.json').read_text())
scored=score_features(features,rule,allow_calibration=True)
require(int((scored.predicted_positive & outcomes.cancer.eq(1)).sum())==90,'Frozen calibration decisions differ.')
record={'status':'fixed_gmic_cpu_verification_passed','tests_run':result.testsRun,
    'parity':parity,'cohort_matched':True,'calibration_rule_unchanged':True,
    'freeze_sha256':sha(BUNDLE/'frozen_fixed_gmic.json'),'final_inference_started':False,
    'final_model_outcomes_evaluated':False}
write_json(Path('/kaggle/working/fixed_gmic_cpu_verification.json'),record)
print('FIXED GMIC CPU VERIFIED',json.dumps(record),flush=True)
"""
    emit_notebook(output/'rsna_fixed_gmic_final_check.ipynb', common+
        'CPU verification only. Attach the official RSNA dataset, notebook 09 v4 and notebook 11 v2. No held-out pixels are scored.\n',setup,cpu)
    dependencies = """
dependencies={'pydicom':'pydicom>=3,<4','imageio':'imageio>=2.34,<3','h5py':'h5py>=3.10,<4',
 'cv2':'opencv-python-headless>=4.9,<5','matplotlib':'matplotlib>=3.8,<4',
 'pylibjpeg':'pylibjpeg>=2,<3','openjpeg':'pylibjpeg-openjpeg>=2,<3','libjpeg':'pylibjpeg-libjpeg>=2,<3'}
missing=[spec for module,spec in dependencies.items() if importlib.util.find_spec(module) is None]
if missing: subprocess.run([sys.executable,'-m','pip','install','--quiet',*missing],check=True)
"""
    for shard in range(5):
        launch = dependencies+"\nverification_candidates=[Path('/kaggle/input/notebooks/aseqwss/13-rsna-fixed-gmic-final-check/fixed_gmic_cpu_verification.json'),Path('/kaggle/input/13-rsna-fixed-gmic-final-check/fixed_gmic_cpu_verification.json')]\nverification=[p for p in verification_candidates if p.is_file()]\nassert len(verification)==1,'Attach the successful private notebook 13 output.'\n"
        launch += f"subprocess.run([sys.executable,'-u','gmic_final_inference.py','--bundle',str(BUNDLE),'--shard',{str(shard)!r},'--output','/kaggle/working/final_shard_{shard}','--verification',str(verification[0])],cwd=PROJECT,check=True)\n"
        emit_notebook(output/f'rsna_fixed_gmic_final_shard_{shard}.ipynb', common+
            f'Label-free inference shard {shard}/4. Attach official RSNA data, notebook 09 v4, notebook 10 v2 and successful notebook 13. GPU T4 required. '
            'Run only after the CPU verification has passed. No cancer outcomes are calculated; all five shards must be combined for one global evaluation.\n',setup,launch)
    report = {'notebooks': 6, 'source_files_embedded': len(SOURCE_FILES), 'freeze_verified': True,
              'syntax_compiled': True, 'patient_level_content_private': True}
    print(json.dumps(report))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); main(args.bundle,args.output)
