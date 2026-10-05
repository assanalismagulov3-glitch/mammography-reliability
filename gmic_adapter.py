"""Engineering adapter for a pinned, unmodified NYU GMIC implementation.

The upstream implementation and weights retain their authorship and AGPLv3
license. This adapter is development code; it is not a clinical pipeline.
"""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request

import numpy as np

UPSTREAM_COMMIT = '594029efe6184bcca3a151fa1dd46c5fd620cb25'
HISTORICAL_FIXTURE_COMMIT = 'febe2431dbe6efd75bcb37ddea85dc48c0df0b07'


def filter_legacy_state(state, model_keys):
    """Permit one unused saved tensor, never arbitrary missing/extra weights.

    The pinned source has no shared_rep_filter module or forward-path use.
    Its official loader uses strict=False. Our narrower exception preserves
    strict loading for all live model parameters.
    """
    import copy
    extra = set(state) - set(model_keys)
    if extra - {'shared_rep_filter.weight'}:
        raise ValueError(f'Unexplained checkpoint tensors: {sorted(extra)}')
    missing = set(model_keys) - set(state)
    if any(not name.endswith('.num_batches_tracked') for name in missing):
        raise ValueError(f'Missing active checkpoint tensors: {sorted(missing)}')
    filtered = copy.copy(state)
    for name in extra:
        del filtered[name]
    return filtered, sorted(extra)


def digest_file(path, algorithm='sha256', git_blob=False):
    path = Path(path)
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def acquire_upstream(manifest_path, destination):
    """Check every file against the blob identity in the inspected Git tree."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['commit'] not in {UPSTREAM_COMMIT, HISTORICAL_FIXTURE_COMMIT}:
        raise ValueError('Unexpected GMIC revision.')
    destination = Path(destination)
    records = []
    for relative, blob in manifest['files'].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError('Invalid upstream path.')
        path = destination / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://raw.githubusercontent.com/nyukat/GMIC/{manifest['commit']}/{relative}"
            temporary = path.with_suffix(path.suffix + '.download')
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024*1024)
            if digest_file(temporary, 'sha1', git_blob=True) != blob:
                raise ValueError(f'Upstream content mismatch: {relative}')
            temporary.replace(path)
        if digest_file(path, 'sha1', git_blob=True) != blob:
            raise ValueError(f'Existing upstream content mismatch: {relative}')
        records.append({'path': relative, 'git_blob': blob, 'bytes': path.stat().st_size,
                        'sha256': digest_file(path)})
        print('GMIC VERIFIED FILE', relative, path.stat().st_size, flush=True)
    return records


def infer_horizontal_flip(image, view):
    """Provisional, label-free orientation heuristic requiring visual review.

Compare intensity mass in outer fifths of the original image. The brighter
side is the proposed chest-wall side. This is NOT a validated orientation
detector. Ambiguity is recorded, not used to exclude patients silently.
"""
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 5 or view not in ('L-CC', 'R-CC', 'L-MLO', 'R-MLO'):
        raise ValueError('Invalid image or standard view.')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() <= image.min():
        raise ValueError('Invalid image intensity range.')
    width = max(1, image.shape[1] // 5)
    left = float(image[:, :width].sum(dtype=np.float64))
    right = float(image[:, -width:].sum(dtype=np.float64))
    total = left + right
    if total <= 0:
        raise ValueError('No edge signal for orientation.')
    wall_left = left >= right
    usual_wall_left = view.startswith('L-')
    return ('NO' if wall_left == usual_wall_left else 'YES'), {
        'proposed_chest_wall_side': 'left' if wall_left else 'right',
        'edge_mass_contrast': abs(left-right)/total,
        'left_edge_mass': left, 'right_edge_mass': right,
        'orientation_requires_visual_review': True,
    }


def preprocess_array(image, view, horizontal_flip):
    """Use original crop, center, orientation, padding and normalization code."""
    from src.cropping.crop_mammogram import crop_img_from_largest_connected, image_orientation
    from src.optimal_centers.get_optimal_centers import extract_center
    from src.data_loading import loading, augmentations

    info = crop_img_from_largest_connected(
        image, image_orientation(horizontal_flip, view[0]), True, 100, 50, 1/3)
    top, bottom, left, right = map(int, info[0])
    crop = image[top:bottom, left:right]
    metadata = {'full_view': view, 'view': view[2:], 'side': view[0],
                'horizontal_flip': horizontal_flip, 'window_location': info[0],
                'rightmost_points': info[1], 'bottommost_points': info[2],
                'distance_from_starting_side': info[3]}
    center = extract_center(metadata, crop)
    oriented = loading.flip_image(crop.astype(np.float32), view, horizontal_flip)
    window, _ = augmentations.random_augmentation_best_center(
        oriented, (2944, 1920), np.random.RandomState(0), best_center=center, view=view)
    # Independently check the helper against the exact upstream inference entry.
    processed = loading.process_image(oriented, view, center)
    expected = window.copy()
    loading.standard_normalize_single_image(expected)
    if not np.array_equal(processed, expected):
        raise ValueError('Adapter changed upstream image processing.')
    if processed.shape != (2944, 1920) or not np.isfinite(processed).all() or processed.std() < .99:
        raise ValueError('Invalid processed GMIC input.')
    nonzero = max(int(np.count_nonzero(image)), 1)
    details = {'raw_shape': list(image.shape), 'crop_shape': list(crop.shape),
               'window_location': [top, bottom, left, right],
               'best_center': [int(x) for x in center],
               'distance_from_starting_side': int(info[3]),
               'nonzero_fraction_retained_after_crop': float(np.count_nonzero(crop)/nonzero),
               'nonzero_fraction_retained_in_window': float(np.count_nonzero(window)/nonzero),
               'horizontal_flip': horizontal_flip,
               'normalized_mean': float(processed.mean()), 'normalized_std': float(processed.std()),
               'processed_sha256': hashlib.sha256(processed.tobytes()).hexdigest()}
    return processed, window, details


def decode_native(path):
    """Keep native pixel dimensions; no reuse/upscaling of the small cache."""
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    from mammo_reliability.imaging import normalize_pixels
    ds = pydicom.dcmread(path)
    raw = ds.pixel_array
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or 'PresentationLUTSequence' in ds:
        raise ValueError('Unsupported DICOM presentation.')
    normalized = normalize_pixels(raw, apply_voi_lut(apply_modality_lut(raw, ds), ds),
                                  str(ds.PhotometricInterpretation), ds.get('PixelPaddingValue'),
                                  ds.get('PixelPaddingRangeLimit'))
    image = np.round(normalized*65535).astype(np.uint16)
    details = {'photometric': str(ds.PhotometricInterpretation),
               'transfer_syntax': str(ds.file_meta.TransferSyntaxUID),
               'patient_orientation': list(ds.get('PatientOrientation', [])),
               'pixel_spacing': [float(x) for x in ds.get('PixelSpacing', [])],
               'imager_pixel_spacing': [float(x) for x in ds.get('ImagerPixelSpacing', [])],
               'manufacturer': str(ds.get('Manufacturer', '')),
               'voi_present': bool('VOILUTSequence' in ds or 'WindowCenter' in ds)}
    return image, details
