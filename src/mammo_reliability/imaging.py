"""DICOM presentation processing, preserving the full field of view."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut

from .audit import STANDARD_VIEWS, file_sha256, require, write_json


def normalize_pixels(raw: np.ndarray, presented: np.ndarray, photometric: str,
                     padding_value=None, padding_limit=None) -> np.ndarray:
    require(raw.ndim == 2 and presented.shape == raw.shape, "Only two-dimensional grayscale images are supported.")
    require(photometric in ["MONOCHROME1", "MONOCHROME2"], "Unsupported photometric interpretation.")
    valid = np.isfinite(presented)
    if padding_value is not None:
        if padding_limit is None:
            valid &= raw != padding_value
        else:
            lower, upper = sorted([padding_value, padding_limit])
            valid &= ~((raw >= lower) & (raw <= upper))
    require(valid.any(), "No valid image pixels remain.")
    low, high = float(presented[valid].min()), float(presented[valid].max())
    require(high > low, "Uniform image after presentation processing.")
    result = ((presented.astype(np.float32) - low) / (high - low)).clip(0, 1)
    if photometric == "MONOCHROME1":
        result = 1 - result
    result[~valid] = 0
    return result.astype(np.float32)


def letterbox(array: np.ndarray, height: int = 1024, width: int = 768) -> np.ndarray:
    require(array.ndim == 2 and min(height, width) > 0, "Invalid image or target dimensions.")
    h, w = array.shape
    factor = min(height / h, width / w)
    new_h, new_w = max(1, round(h * factor)), max(1, round(w * factor))
    resized = np.asarray(Image.fromarray(array.astype(np.float32)).resize(
        (new_w, new_h), resample=Image.Resampling.BILINEAR))
    canvas = np.zeros((height, width), dtype=np.float32)
    y, x = (height - new_h) // 2, (width - new_w) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def decode_dicom(path: str | Path, height: int = 1024, width: int = 768) -> tuple[np.ndarray, dict]:
    ds = pydicom.dcmread(path)
    raw = ds.pixel_array
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    require(raw.ndim == 2, "Multiframe/color DICOM requires a separate preprocessing decision.")
    require("PresentationLUTSequence" not in ds, "Presentation LUT sequence requires image-specific review.")
    photo = str(ds.get("PhotometricInterpretation", ""))
    presented = apply_voi_lut(apply_modality_lut(raw, ds), ds)
    normalized = normalize_pixels(raw, presented, photo, ds.get("PixelPaddingValue"), ds.get("PixelPaddingRangeLimit"))
    digest = hashlib.sha256()
    digest.update(str((raw.shape, str(raw.dtype))).encode())
    digest.update(np.ascontiguousarray(raw).tobytes())
    info = {"height_original": int(raw.shape[0]), "width_original": int(raw.shape[1]),
            "photometric": photo, "decoded_pixel_sha256": digest.hexdigest(),
            "voi_present": bool("VOILUTSequence" in ds or "WindowCenter" in ds),
            "nonzero_fraction_after_normalization": float((normalized > 0).mean())}
    return letterbox(normalized, height, width), info


def check_cross_patient_duplicates(manifest: pd.DataFrame) -> None:
    require("decoded_pixel_sha256" in manifest, "Pixel duplicate audit was not performed.")
    shared = manifest.groupby("decoded_pixel_sha256")["patient_id"].nunique()
    require(not (shared > 1).any(), "Identical decoded images occur under different patients. Investigate/group them before training.")


def prepare_images(images: pd.DataFrame, data_root: str | Path, output: str | Path,
                   height: int = 1024, width: int = 768, cache_images: bool = True) -> pd.DataFrame:
    """Resume verified 16-bit PNG conversion, or audit pixels without caching.

    Never skip failed images. A stopped run can resume from the last manifest.
    An explicit disk check after the first 32 images prevents filling the volume.
    """
    data_root, output = Path(data_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(data_root / "train.csv")
    config = {"version": 1, "height": height, "width": width, "cache_images": cache_images, "metadata_sha256": source_hash}
    config_file, manifest_file = output / "preprocessing.json", output / "manifest.csv"
    if config_file.exists():
        require(json.loads(config_file.read_text()) == config, "Preprocessing settings/source changed; use a new output directory.")
    else:
        write_json(config_file, config)
    rows = images.loc[images["view"].isin(STANDARD_VIEWS)].sort_values(["patient_id", "image_id"])
    previous = pd.read_csv(manifest_file).to_dict("records") if manifest_file.exists() else []
    recorded = {int(r["image_id"]): r for r in previous}
    require(len(recorded) == len(previous), "Repeated image in preprocessing manifest.")
    require(set(recorded).issubset(set(rows["image_id"].astype(int))), "Manifest has unexpected images.")
    cached_sizes = []
    for row in rows.itertuples(index=False):
        image_id, patient_id = int(row.image_id), int(row.patient_id)
        if image_id in recorded:
            old = recorded[image_id]
            if cache_images:
                require(Path(old["cache_path"]).is_file(), "A recorded cached image is missing.")
                require(file_sha256(old["cache_path"]) == old["cache_sha256"], "Cached image hash mismatch.")
                cached_sizes.append(Path(old["cache_path"]).stat().st_size)
            continue
        source = data_root / "train_images" / str(patient_id) / f"{image_id}.dcm"
        array, info = decode_dicom(source, height, width)
        record = {"patient_id": patient_id, "image_id": image_id, "laterality": row.laterality,
                  "view": row.view, "source_path": str(source), **info}
        if cache_images:
            cache_path = output / "images" / str(patient_id) / f"{image_id}.png"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(array * 65535).astype(np.uint16)).save(cache_path)
            record.update(cache_path=str(cache_path), cache_sha256=file_sha256(cache_path))
            cached_sizes.append(cache_path.stat().st_size)
        recorded[image_id] = record
        if len(recorded) % 32 == 0 or len(recorded) == len(rows):
            temporary = manifest_file.with_suffix(".csv.tmp")
            pd.DataFrame(recorded.values()).to_csv(temporary, index=False)
            temporary.replace(manifest_file)
            print(f"Pixel audit: {len(recorded)}/{len(rows)} images", flush=True)
            if cache_images:
                estimate = float(np.mean(cached_sizes)) * (len(rows) - len(recorded)) * 1.25
                require(estimate + 2 * 1024**3 < shutil.disk_usage(output).free,
                        "Estimated PNG cache exceeds free disk space. Use a larger volume, or a NEW output directory with cache_images=False.")
    manifest = pd.DataFrame(recorded.values()).sort_values(["patient_id", "image_id"]).reset_index(drop=True)
    require(len(manifest) == len(rows), "Incomplete image manifest.")
    check_cross_patient_duplicates(manifest)
    write_json(output / "pixel_audit_complete.json", {"metadata_sha256": source_hash, "n_images": len(manifest),
                                                      "manifest_sha256": file_sha256(manifest_file),
                                                      "cross_patient_exact_pixel_duplicates": 0})
    return manifest
