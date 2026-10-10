"""Shared image encoder with equal CC/MLO weighting and breast-level loss."""
from __future__ import annotations

from pathlib import Path
import hashlib
import io
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

from .audit import BREAST_KEY, require
from .imaging import decode_dicom


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some GPU reductions may still be nondeterministic; record warnings/environment.
    torch.use_deterministic_algorithms(True, warn_only=True)


class BreastDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, breasts: pd.DataFrame, augment: bool = False,
                 height: int = 1024, width: int = 768):
        self.breasts = breasts.reset_index(drop=True).copy()
        require(not self.breasts[BREAST_KEY].duplicated().any(), "Duplicate breasts in dataset.")
        self.groups = {key: g.sort_values("image_id") for key, g in manifest.groupby(BREAST_KEY)}
        require(set(map(tuple, self.breasts[BREAST_KEY].to_numpy())).issubset(self.groups), "A breast has no images.")
        self.augment, self.height, self.width = augment, height, width

    def __len__(self):
        return len(self.breasts)

    def __getitem__(self, index):
        breast = self.breasts.iloc[index]
        rows = self.groups[(breast.patient_id, breast.laterality)]
        require(set(rows["view"].unique()) == {"CC", "MLO"}, "Expected both standard views.")
        images, views = [], []
        for row in rows.itertuples(index=False):
            if hasattr(row, "cache_path") and pd.notna(row.cache_path):
                payload = Path(row.cache_path).read_bytes()
                require(hashlib.sha256(payload).hexdigest() == row.cache_sha256, "Cached pixels changed after the image audit.")
                array = np.asarray(Image.open(io.BytesIO(payload)), dtype=np.float32) / 65535.0
            else:
                array, _ = decode_dicom(row.source_path, self.height, self.width)
            require(array.shape == (self.height, self.width), "Cache shape does not match model configuration.")
            # Full image is retained. No random crop or vertical flip.
            if self.augment and random.random() < 0.5:
                array = np.fliplr(array)
            tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).repeat(3, 1, 1)
            tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            images.append(tensor)
            views.append(0 if row.view == "CC" else 1)
        return {"images": torch.stack(images), "views": torch.tensor(views, dtype=torch.long),
                "label": float(breast.cancer), "patient_id": int(breast.patient_id), "laterality": breast.laterality}


def collate_breasts(batch):
    slots = [item["views"] + 2 * index for index, item in enumerate(batch)]
    return {"images": torch.cat([item["images"] for item in batch]), "slots": torch.cat(slots),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
            "patient_id": [item["patient_id"] for item in batch],
            "laterality": [item["laterality"] for item in batch]}


def pool_view_logits(image_logits: torch.Tensor, slots: torch.Tensor, n_breasts: int) -> torch.Tensor:
    require(image_logits.ndim == 1 and len(image_logits) == len(slots), "Logit/group shape mismatch.")
    counts = torch.bincount(slots, minlength=2 * n_breasts)
    require(len(counts) == 2 * n_breasts and (counts > 0).all().item(), "A breast lacks a standard view.")
    sums = torch.zeros(2 * n_breasts, dtype=image_logits.dtype, device=image_logits.device)
    sums = sums.index_add(0, slots, image_logits)
    return (sums / counts.to(image_logits.dtype)).reshape(n_breasts, 2)


class MammographyModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        # Freeze running BN statistics for small batches; affine parameters remain trainable.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images, slots, n_breasts):
        image_logits = self.encoder(images).squeeze(1)
        view_logits = pool_view_logits(image_logits.float(), slots, n_breasts)
        # Extra CC images do not give CC more weight than MLO, or this breast more weight in the loss.
        return view_logits.mean(dim=1), view_logits
