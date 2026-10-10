"""RSNA full-field adaptation for fixed GMIC weights; not NYU preprocessing."""
import numpy as np
from PIL import Image
from gmic_adapter import infer_horizontal_flip


def fullfield_array(raw, view, height=2944, width=1920):
    """Orient, fit the complete image extent, and pad without a crop window.

    Larger images are downsampled with preserved aspect ratio; smaller images
    are not enlarged. This avoids geometric exclusion but cannot guarantee
    preservation of small diagnostic details. No label enters this function.
    """
    if min(height, width) <= 0:
        raise ValueError('Invalid target dimensions.')
    flip_flag, orientation = infer_horizontal_flip(raw, view)
    flip = ((flip_flag == 'NO' and view.startswith('R-')) or
            (flip_flag == 'YES' and view.startswith('L-')))
    oriented = np.fliplr(raw) if flip else raw
    h, w = oriented.shape
    scale = min(1., height/h, width/w)
    new_h, new_w = max(1, round(h*scale)), max(1, round(w*scale))
    resized = np.asarray(Image.fromarray(oriented.astype(np.float32)).resize(
        (new_w, new_h), resample=Image.Resampling.BILINEAR))
    canvas = np.zeros((height, width), dtype=np.float32)
    # Chest wall on the left. Match the released CC/MLO padding convention.
    top = (height-new_h)//2 if view.endswith('CC') else 0
    canvas[top:top+new_h, :new_w] = resized
    processed = canvas.copy()
    processed -= processed.mean()
    processed /= max(float(processed.std()), 1e-5)
    if not np.isfinite(processed).all() or processed.std() < .99:
        raise ValueError('Invalid standardized full-field input.')
    details = {**orientation, 'horizontal_flip': flip_flag, 'actual_flip_applied': flip,
               'raw_shape': [h,w], 'resized_shape': [new_h,new_w],
               'canvas_shape': [height,width], 'scale': float(scale),
               'padding_top': top, 'padding_left': 0,
               'source_window': [0,h,0,w], 'geometric_cropping': False,
               'normalized_mean': float(processed.mean()), 'normalized_std': float(processed.std())}
    return processed, canvas, details
