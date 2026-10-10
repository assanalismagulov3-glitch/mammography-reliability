"""Audit full-field GMIC adaptation on the exact original 20 fit images."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image

from gmic_adapter import digest_file, decode_native, filter_legacy_state, UPSTREAM_COMMIT
from gmic_fullfield import fullfield_array
from gmic_preflight import infer
from mammo_reliability.audit import locate_dataset, require, write_json, environment

OUTPUT = Path('/kaggle/working/gmic_fullfield_preflight')
EXPECTED_PLAN = '6f0aff552657aebfe4a4da73ddeacfdf29e62daaa3b590c34e7ef499949fdfc2'


def verified_preflight():
    candidates = [Path('/kaggle/input/notebooks/aseqwss/09-rsna-gmic-transfer-preflight/gmic_transfer_preflight'),
                  Path('/kaggle/input/09-rsna-gmic-transfer-preflight/gmic_transfer_preflight')]
    existing = [x for x in candidates if x.is_dir()]
    require(len(existing) == 1, 'Attach private notebook 09 version 4.')
    base = existing[0]
    require(digest_file(base/'declared_plan.json') == EXPECTED_PLAN, 'Wrong preflight version.')
    result = json.loads((base/'completed.json').read_text())
    require(result['reference_compatibility_passed'], 'Compatibility checks did not pass.')
    require(result['rsna_fit_images_checked'] == 20 and not result['final_test_results_opened'], 'Unexpected preflight.')
    manifest = json.loads(Path('gmic_upstream_manifest.json').read_text())
    require(manifest['commit'] == UPSTREAM_COMMIT, 'Changed upstream source.')
    upstream = base/'upstream'
    for name, blob in manifest['files'].items():
        require(digest_file(upstream/name, 'sha1', git_blob=True) == blob, f'Upstream file changed: {name}')
    return base, upstream


def load_fixed_models(upstream, output):
    sys.path.insert(0, str(upstream))
    from src.modeling.gmic import GMIC
    from src.constants import PERCENT_T_DICT
    models, records = [], []
    for index in range(1,6):
        params = {'device_type':'gpu','gpu_number':0,'cam_size':(46,30),'K':6,
                  'crop_shape':(256,256),'post_processing_dim':256,'num_classes':2,
                  'use_v1_global':False,'percent_t':PERCENT_T_DICT[str(index)]}
        model = GMIC(params)
        checkpoint = upstream/'models'/f'sample_model_{index}.p'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        active, ignored = filter_legacy_state(state, model.state_dict().keys())
        model.load_state_dict(active, strict=True)
        model.eval().to('cuda')
        for parameter in model.parameters(): parameter.requires_grad_(False)
        models.append(model)
        records.append({'model':index,'checkpoint_sha256':digest_file(checkpoint),'ignored_unused_keys':ignored})
    write_json(output/'fixed_models.json',records)
    return models


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(parents=True,exist_ok=True)
    require(torch.cuda.is_available(), 'GPU required.')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    base, upstream = verified_preflight()
    roles_path=base/'cohort_identity/pilot/roles_private.csv'
    require(digest_file(roles_path)=='fc0308c63300be6849c7ac9b830d09480c2b62d64df4f1d036bab12548a7e15f','Roles changed.')
    roles=pd.read_csv(roles_path)
    fit_ids=set(roles.loc[roles.role.eq('fit'),'patient_id'])
    sample=pd.read_csv(base/'engineering_sample_private.csv')
    require(len(sample)==20 and sample.image_id.is_unique and set(sample.patient_id).issubset(fit_ids), 'Wrong engineering sample.')
    declaration={'purpose':'full_field_geometry_repair_on_same_20_fit_images','preflight_plan_sha256':EXPECTED_PLAN,
                 'preprocessing':'full original field, horizontal orientation heuristic, downscale only to fit, left-aligned padding',
                 'target_shape':[2944,1920],'interpolation':'bilinear','image_extent_cropped':False,
                 'caution':'Downsampling can lose fine detail; this is an explicit RSNA adaptation, not exact NYU preprocessing.',
                 'frozen_models':5,'accuracy_calculated':False,'final_test_results_opened':False,
                 'environment':environment(),'script_sha256':digest_file(__file__),
                 'fullfield_sha256':digest_file('gmic_fullfield.py'),'adapter_sha256':digest_file('gmic_adapter.py')}
    write_json(OUTPUT/'declared_plan.json',declaration)
    print('FULLFIELD DECLARED PLAN',json.dumps(declaration),flush=True)
    models=load_fixed_models(upstream,OUTPUT)
    data_root=locate_dataset()
    require(digest_file(data_root/'train.csv')=='35408731b67b87272f85874ee3c4c9d5cea7a2c4fbcaeac3fbe415ba1a5817c1','Metadata changed.')
    records,thumbs=[],[]
    for i,row in enumerate(sample.itertuples(index=False),1):
        started=time.monotonic()
        raw,header=decode_native(data_root/'train_images'/str(row.patient_id)/f'{row.image_id}.dcm')
        view=f'{row.laterality}-{row.view}'
        array,canvas,geometry=fullfield_array(raw,view)
        predictions=infer(models,array)
        np.save(OUTPUT/f'sample_{i:02d}_predictions_private.npy',predictions)
        records.append({'sample':i,'site_id':int(row.site_id),'machine_id':int(row.machine_id),
                        'view':view,**header,**geometry,'seconds':time.monotonic()-started})
        pair=[]
        for img in [raw,canvas]:
            # Pillow cannot reduce every uint16 mode; float preserves display range.
            thumb=Image.fromarray(img.astype(np.float32));thumb.thumbnail((500,500));pair.append(np.asarray(thumb))
        thumbs.append(pair)
        print('FULLFIELD IMAGE',json.dumps(records[-1]),flush=True)
    for start in range(0,20,4):
        fig,axes=plt.subplots(4,2,figsize=(10,16),layout='constrained')
        for j in range(4):
            i=start+j
            for col in range(2):
                axes[j,col].imshow(thumbs[i][col],cmap='gray',aspect='equal')
                axes[j,col].axis('off')
                axes[j,col].set_title(f"Sample {i+1}, site {records[i]['site_id']}, {records[i]['view']}\n"
                                      +('Original presentation' if col==0 else 'Full-field model input'),fontsize=10)
        fig.suptitle('Private engineering review; no lesion or cancer interpretation')
        fig.savefig(OUTPUT/f'contact_sheet_{start//4+1:02d}_private.png',dpi=100)
        plt.close(fig)
    write_json(OUTPUT/'technical_images.json',records)
    result={'status':'numeric_fullfield_preflight_completed_visual_review_pending','images':20,
            'final_test_results_opened':False,'accuracy_calculated':False,'visual_review_required':True,
            'minimum_scale':min(r['scale'] for r in records),'maximum_scale':max(r['scale'] for r in records),
            'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'declared_plan_sha256':digest_file(OUTPUT/'declared_plan.json')}
    write_json(OUTPUT/'completed.json',result)
    print('FULLFIELD COMPLETED',json.dumps(result),flush=True)


if __name__=='__main__': main()
