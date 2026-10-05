"""CPU execution on artificial DICOMs; not an RSNA experiment."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from mammo_reliability.audit import audit_metadata, make_patient_folds, partition_outer_fold
from mammo_reliability.imaging import prepare_images
from mammo_reliability.model import BreastDataset, MammographyModel, pool_view_logits
from mammo_reliability.training import predict_members, train_member
from test_integrity import artificial_metadata


def test_view_pooling_is_equal_weighted_and_differentiable():
    values = torch.tensor([1.0, 3.0, 10.0], requires_grad=True)
    slots = torch.tensor([0, 0, 1])
    views = pool_view_logits(values, slots, 1)
    assert views.tolist() == [[2.0, 10.0]]
    breast_logit = views.mean()
    assert breast_logit.item() == 6.0
    breast_logit.backward()
    assert values.grad.tolist() == [0.25, 0.25, 0.5]
    duplicate = pool_view_logits(torch.tensor([1., 3., 1., 3., 10.]), torch.tensor([0, 0, 0, 0, 1]), 1)
    assert duplicate.mean().item() == breast_logit.item()


def test_resnet_forward_backward_and_frozen_batch_statistics():
    torch.set_num_threads(2)
    model = MammographyModel(pretrained=False).train()
    batch = torch.randn(3, 3, 32, 32)
    logits, views = model(batch, torch.tensor([0, 0, 1]), 1)
    assert logits.shape == (1,) and views.shape == (1, 2)
    logits.sum().backward()
    assert model.encoder.fc.weight.grad is not None
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))


def save_artificial_dicom(path: Path, image_id: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = ds.Columns = 32
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = np.random.default_rng(image_id).integers(0, 4096, (32, 32), dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_artificial_dicom_training_checkpoint_reuse_and_prediction(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    metadata = artificial_metadata(40)
    root = tmp_path / "artificial_data"
    root.mkdir()
    metadata.to_csv(root / "train.csv", index=False)
    for row in metadata.itertuples(index=False):
        save_artificial_dicom(root / "train_images" / str(row.patient_id) / f"{row.image_id}.dcm", row.image_id)
    manifest = prepare_images(metadata, root, tmp_path / "preprocessed", height=32, width=32)
    assert len(manifest) == len(metadata)
    resumed = prepare_images(metadata, root, tmp_path / "preprocessed", height=32, width=32)
    assert set(resumed.image_id) == set(manifest.image_id)
    breasts, _, _ = audit_metadata(metadata)
    roles = partition_outer_fold(make_patient_folds(breasts), 0)
    config = {"height": 32, "width": 32, "batch_size": 2, "num_workers": 0, "epochs": 2,
              "accumulation_steps": 2, "patience": 2, "learning_rate": 0.0001,
              "weight_decay": 0.0001, "pretrained": False, "require_gpu": False, "mixed_precision": False}
    folder = tmp_path / "fold_0"
    for seed in [17, 41]:
        out = folder / f"seed_{seed}"
        if seed == 17:
            import mammo_reliability.training as training_module
            original_save = training_module.save_checkpoint

            def interrupt_after_checkpoint(path, state):
                original_save(path, state)
                if path.name == "last.pt":
                    raise RuntimeError("SIMULATED_SESSION_INTERRUPTION")

            with monkeypatch.context() as patch:
                patch.setattr(training_module, "save_checkpoint", interrupt_after_checkpoint)
                with pytest.raises(RuntimeError, match="SIMULATED_SESSION_INTERRUPTION"):
                    train_member(manifest, breasts, roles, config, seed, out, "ARTIFICIAL_FIXTURE_ONLY")
        first = train_member(manifest, breasts, roles, config, seed, out, "ARTIFICIAL_FIXTURE_ONLY")
        second = train_member(manifest, breasts, roles, config, seed, out, "ARTIFICIAL_FIXTURE_ONLY")
        assert first == second and first["status"] == "training_completed"
    target = breasts.loc[breasts.patient_id.isin(roles.loc[roles.role == "test", "patient_id"])]
    prediction = predict_members(manifest, target, config, [17, 41], folder)
    assert len(prediction) == 2 * len(target)
    assert np.isfinite(prediction[["z_cc", "z_mlo"]].to_numpy()).all()
    changed_config = dict(config, learning_rate=0.002)
    with pytest.raises(ValueError, match="changed"):
        train_member(manifest, breasts, roles, changed_config, 17, folder / "seed_17", "ARTIFICIAL_FIXTURE_ONLY")
    # A validly encoded image with changed bytes must still be rejected.
    from PIL import Image
    first_image = manifest.iloc[0]
    Image.fromarray(np.zeros((32, 32), dtype=np.uint16)).save(first_image.cache_path)
    data = BreastDataset(manifest, breasts, False, 32, 32)
    with pytest.raises(ValueError, match="changed"):
        data[0]
