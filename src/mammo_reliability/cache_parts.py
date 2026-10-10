"""Build private lossless cache parts, then validate and mount them together.

The partition is only for storage. It does not change patient evaluation folds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd

from .audit import STANDARD_VIEWS, file_sha256, locate_dataset, require, write_json
from .imaging import check_cross_patient_duplicates, prepare_images


def preprocessing_fingerprint() -> str:
    return file_sha256(Path(__file__).with_name('imaging.py'))


def _prepare_shard(task):
    images, data_root, folder, height, width = task
    return prepare_images(images, data_root, folder, height, width, True)


def build_part(data_root: Path, output: Path, part: int, n_parts: int = 2,
               workers: int = 4, height: int = 1024, width: int = 768) -> dict:
    require(n_parts > 0 and 0 <= part < n_parts and workers > 0, 'Invalid cache partition settings.')
    images = pd.read_csv(data_root / 'train.csv')
    images = images.loc[images['view'].isin(STANDARD_VIEWS)].copy()
    selected = images.loc[images['patient_id'] % n_parts == part].copy()
    require(len(selected) > 0, 'Empty cache part.')
    identity = {'version': 1, 'part': part, 'n_parts': n_parts, 'workers': workers,
                'height': height, 'width': width, 'metadata_sha256': file_sha256(data_root / 'train.csv'),
                'image_processing_source_sha256': preprocessing_fingerprint(),
                'n_images': len(selected), 'cache_format': '16-bit PNG after full-frame letterbox'}
    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / 'part_identity.json'
    if identity_file.exists():
        require(json.loads(identity_file.read_text()) == identity, 'Cache part settings or preprocessing changed.')
    else:
        write_json(identity_file, identity)
    tasks = []
    shard_id = (selected['patient_id'] // n_parts) % workers
    for shard in range(workers):
        subset = selected.loc[shard_id == shard]
        if len(subset):
            tasks.append((subset, data_root, output / f'shard_{shard}', height, width))
    print(f'Preparing private cache part {part + 1}/{n_parts}: {len(selected)} images; {len(tasks)} workers.', flush=True)
    if workers == 1:
        frames = [_prepare_shard(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_prepare_shard, tasks))
    manifest = pd.concat(frames, ignore_index=True).sort_values(['patient_id', 'image_id'])
    require(manifest['image_id'].is_unique, 'Repeated image in cache part.')
    require(set(manifest['image_id']) == set(selected['image_id']), 'Cache part image coverage differs from metadata.')
    check_cross_patient_duplicates(manifest)
    manifest['cache_relative_path'] = manifest['cache_path'].map(lambda p: str(Path(p).relative_to(output)))
    manifest.to_csv(output / 'manifest.csv', index=False)
    result = {**identity, 'manifest_sha256': file_sha256(output / 'manifest.csv'),
              'cross_patient_exact_pixel_duplicates_within_part': 0}
    write_json(output / 'part_complete.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def assemble_parts(part_roots: list[Path], data_root: Path, output: Path,
                   height: int = 1024, width: int = 768) -> pd.DataFrame:
    require(len(part_roots) > 0, 'Attach the completed private cache parts first.')
    metadata_hash = file_sha256(data_root / 'train.csv')
    frames, completions = [], []
    for root in part_roots:
        completion = json.loads((root / 'part_complete.json').read_text())
        require(completion['metadata_sha256'] == metadata_hash, 'A cache part uses different source metadata.')
        require(completion['height'] == height and completion['width'] == width, 'Cache part resolution mismatch.')
        require(completion['image_processing_source_sha256'] == preprocessing_fingerprint(), 'Cache preprocessing source mismatch.')
        require(completion['manifest_sha256'] == file_sha256(root / 'manifest.csv'), 'A cache part manifest changed.')
        frame = pd.read_csv(root / 'manifest.csv')
        require(len(frame) == completion['n_images'], 'Incomplete cache part.')
        require((frame['patient_id'] % completion['n_parts'] == completion['part']).all(), 'Cache contains the wrong patients.')
        def remount(relative):
            path = Path(relative)
            require(not path.is_absolute() and '..' not in path.parts, 'Unsafe relative cache path.')
            target = root / path
            require(target.is_file(), 'A cached image is missing from the mounted input.')
            return str(target)
        frame['cache_path'] = frame['cache_relative_path'].map(remount)
        frame['source_path'] = [str(data_root / 'train_images' / str(int(p)) / f'{int(i)}.dcm')
                                for p, i in zip(frame['patient_id'], frame['image_id'])]
        frames.append(frame)
        completions.append(completion)
    expected_parts = completions[0]['n_parts']
    require(all(c['n_parts'] == expected_parts for c in completions), 'Incompatible cache partition counts.')
    require(len(completions) == expected_parts and {c['part'] for c in completions} == set(range(expected_parts)),
            'Duplicate or missing cache parts.')
    manifest = pd.concat(frames, ignore_index=True).sort_values(['patient_id', 'image_id']).reset_index(drop=True)
    images = pd.read_csv(data_root / 'train.csv')
    expected = images.loc[images['view'].isin(STANDARD_VIEWS)]
    require(manifest['image_id'].is_unique and set(manifest['image_id']) == set(expected['image_id']),
            'Combined cache does not cover every standard-view image exactly once.')
    keys = ['patient_id', 'image_id', 'laterality', 'view']
    require(manifest[keys].sort_values('image_id').reset_index(drop=True).equals(
        expected[keys].sort_values('image_id').reset_index(drop=True)), 'Cache image identifiers or views mismatch metadata.')
    check_cross_patient_duplicates(manifest)
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / 'manifest.csv', index=False)
    write_json(output / 'preprocessing.json', {'version': 1, 'height': height, 'width': width,
               'cache_images': True, 'metadata_sha256': metadata_hash})
    write_json(output / 'pixel_audit_complete.json', {'metadata_sha256': metadata_hash,
               'n_images': len(manifest), 'manifest_sha256': file_sha256(output / 'manifest.csv'),
               'cross_patient_exact_pixel_duplicates': 0, 'cache_part_completions': completions})
    print(f'Validated and mounted {len(manifest)} images from {expected_parts} private parts.', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['build', 'assemble'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--n-parts', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--parts', type=Path, nargs='+')
    args = parser.parse_args()
    data_root = args.data_root or locate_dataset()
    if args.stage == 'build':
        build_part(data_root, args.output, args.part, args.n_parts, args.workers, args.height, args.width)
    else:
        assemble_parts(args.parts or [], data_root, args.output, args.height, args.width)


if __name__ == '__main__':
    main()
