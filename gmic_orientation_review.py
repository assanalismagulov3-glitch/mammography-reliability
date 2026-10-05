"""Generate private technical displays for every predeclared orientation flag."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from gmic_adapter import decode_native, digest_file
from gmic_fullfield import fullfield_array
from mammo_reliability.audit import locate_dataset, require


def run(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((source/'declared_plan.json').read_text())
    predictions = pd.read_csv(source/'image_predictions_private.csv')
    images = pd.read_csv(source/'images_private.csv')
    selected = predictions.loc[predictions.edge_mass_contrast < plan['orientation_low_contrast_flag']]
    selected = selected.merge(images, on='image_id', validate='one_to_one').sort_values('image_id')
    require(len(selected) == 3 and set(selected.role) == {'fit'}, 'Unexpected orientation-review scope.')
    # Selection uses the previously recorded image statistic, never labels.
    data_root = locate_dataset()
    sheet = Image.new('RGB', (1200, 2280), '#171b22')
    draw = ImageDraw.Draw(sheet)
    records = []
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        path = data_root/'train_images'/str(row.patient_id)/f'{row.image_id}.dcm'
        require(digest_file(path) == row.native_file_sha256, 'Native image bytes changed.')
        raw, header = decode_native(path)
        _, canvas, geometry = fullfield_array(raw, f'{row.laterality}-{row.view}')
        require(abs(geometry['edge_mass_contrast'] - row.edge_mass_contrast) < 1e-10,
                'Orientation statistic did not reproduce.')
        require(geometry['actual_flip_applied'] == row.actual_flip_applied, 'Orientation decision changed.')
        for col, (label, array) in enumerate([('Original full image', raw), ('Current full-field model input', canvas)]):
            # Display only: map the same 16-bit intensity scale to 8-bit and
            # preserve aspect ratio. No independent contrast enhancement/crop.
            display = Image.fromarray(np.clip(np.asarray(array)/65535*255, 0, 255).astype(np.uint8))
            display = ImageOps.contain(display, (570, 700))
            x, y = col*600+15, (index-1)*760
            draw.text((x, y+8), f'Flag {index} | {row.view} | {label}', fill='white')
            draw.text((x, y+25), f'Edge contrast {row.edge_mass_contrast:.4f}', fill='white')
            sheet.paste(display, (x+(570-display.width)//2, y+48))
        records.append({'flag_index': index, 'site_id': int(row.site_id), 'role': row.role,
                        'view': row.view, 'geometry': geometry, 'dicom': header})
    sheet.save(output/'orientation_flags_private.png')
    (output/'orientation_numeric_audit.json').write_text(json.dumps({
        'selection': 'All images below predeclared edge-mass contrast 0.2; labels not used',
        'images': len(records), 'visual_review_completed': False, 'records': records,
        'no_predictions_changed': True, 'final_test_results_opened': False}, indent=2)+'\n')
    print('ORIENTATION_REVIEW_PREPARED', json.dumps({'images': len(records), 'role': 'fit',
          'no_predictions_changed': True, 'visual_review_completed': False}))
