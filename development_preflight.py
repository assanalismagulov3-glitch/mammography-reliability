"""Technical checks on fit-partition images; does not evaluate a cancer model."""
from pathlib import Path
import sys
import json
import io
import shutil
import time
import subprocess

PROJECT = Path('/kaggle/working/mammo_project')
RUN_ROOT = Path('/kaggle/working/mammo_private')
sys.path.insert(0, str(PROJECT))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                'pydicom>=3,<4', 'pylibjpeg>=2,<3',
                'pylibjpeg-openjpeg>=2,<3', 'pylibjpeg-libjpeg>=2,<3'], check=True)

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from mammo_reliability.audit import environment, locate_dataset, write_json
from mammo_reliability.imaging import decode_dicom

data_root = locate_dataset()
output = RUN_ROOT / 'technical_preflight'
output.mkdir(exist_ok=True)
aggregate_tables = {}
for name in ['density_availability', 'density_cancer', 'missing_columns', 'extra_images']:
    frame = pd.read_csv(RUN_ROOT / 'aggregates' / f'{name}.csv')
    aggregate_tables[name] = json.loads(frame.to_json(orient='records'))
    print('\n' + name + '\n' + frame.to_string(index=False), flush=True)

images = pd.read_csv(data_root / 'train.csv')
roles = pd.read_csv(RUN_ROOT / 'fold_0' / 'roles_private.csv')
fit = images[images.patient_id.isin(roles.loc[roles.role.eq('fit'), 'patient_id'])
             & images.view.isin(['CC', 'MLO'])].copy()
# Predefined engineering coverage, not selection based on model performance.
ordered = fit.sample(frac=1, random_state=20260909)
sample = pd.concat([
    ordered.groupby(['site_id', 'view', 'cancer'], sort=True).head(2),
    ordered.groupby('machine_id', sort=True).head(1),
]).drop_duplicates('image_id').reset_index(drop=True)
sample[['patient_id', 'image_id', 'site_id', 'machine_id', 'view', 'cancer']].to_csv(
    output / 'sample_private.csv', index=False)
records, arrays = [], []
for row in sample.itertuples(index=False):
    path = data_root / 'train_images' / str(row.patient_id) / f'{row.image_id}.dcm'
    header = pydicom.dcmread(path, stop_before_pixels=True)
    started = time.perf_counter()
    array, info = decode_dicom(path, 1024, 768)
    elapsed = time.perf_counter() - started
    stream = io.BytesIO()
    Image.fromarray(np.round(array * 65535).astype(np.uint16)).save(stream, format='PNG')
    records.append({'site_id': int(row.site_id), 'machine_id': int(row.machine_id),
                    'view': row.view, 'photometric': info['photometric'],
                    'transfer_syntax': str(header.file_meta.TransferSyntaxUID),
                    'height_original': info['height_original'],
                    'width_original': info['width_original'],
                    'voi_present': info['voi_present'], 'decode_seconds': elapsed,
                    'png_bytes': stream.getbuffer().nbytes,
                    'nonzero_fraction': float((array > 0).mean())})
    arrays.append(array)
    print(f'Decoded engineering sample {len(records)}/{len(sample)}', flush=True)

for start in range(0, len(arrays), 8):
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))
    for index, ax in enumerate(axes.flat, start=start):
        ax.axis('off')
        if index < len(arrays):
            item = records[index]
            ax.imshow(arrays[index], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Sample {index + 1}: site {item['site_id']} {item['view']}\n"
                         f"machine {item['machine_id']} {item['photometric']}", fontsize=9)
    fig.suptitle('Private technical preprocessing check; no lesion interpretation')
    fig.tight_layout()
    fig.savefig(output / f'contact_sheet_{start // 8 + 1}.png', dpi=130)
    plt.close(fig)

summary = {'purpose': 'technical_preflight_only_no_model_evaluation',
           'sampling_partition': 'outer_fold_0_fit', 'environment': environment(),
           'decoded_samples': len(records), 'image_checks': records,
           'aggregate_tables': aggregate_tables,
           'free_disk_gib': shutil.disk_usage(RUN_ROOT).free / 1024**3,
           'estimated_png_cache_gib': float(np.mean([r['png_bytes'] for r in records]))
               * int(images.view.isin(['CC', 'MLO']).sum()) / 1024**3,
           'sampling_note': 'Stratified engineering sample; cache estimate is approximate.'}
write_json(output / 'technical_preflight.json', summary)
print('\nPREFLIGHT_JSON_BEGIN\n' + json.dumps(summary, indent=2) + '\nPREFLIGHT_JSON_END', flush=True)
