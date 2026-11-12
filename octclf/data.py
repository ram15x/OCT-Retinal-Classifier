# data loading, uses the official train/val/test folders and all four classes

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_dataset_root(path: str | Path) -> Path:
    # find the folder that has train/ val/ test/ right inside it
    # (zips often extract into one extra nested folder)
    root = Path(path)
    candidates = [root, *[p for p in root.rglob("*") if p.is_dir()]]
    for cand in candidates:
        if all((cand / s).is_dir() for s in ("train", "val", "test")):
            return cand
    raise FileNotFoundError(f"No train/val/test folders found under {root}")


def train_transform() -> transforms.Compose:
    # scans are grayscale and the retina layers are always horizontal, so only mild
    # changes: no hue jitter, no vertical flip
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_dataset(root: Path, split: str, train: bool) -> datasets.ImageFolder:
    ds = datasets.ImageFolder(
        root / split, transform=train_transform() if train else eval_transform()
    )
    if tuple(ds.classes) != CLASSES:
        raise ValueError(f"Expected classes {CLASSES}, found {tuple(ds.classes)}")
    return ds


def class_counts(ds: datasets.ImageFolder) -> list[int]:
    counts = [0] * len(ds.classes)
    for _, y in ds.samples:
        counts[y] += 1
    return counts


def balanced_sampler(ds: datasets.ImageFolder, max_per_epoch: int | None = None):
    # sample every class about equally so the small ones are not drowned out
    counts = class_counts(ds)
    weights = [1.0 / counts[y] for _, y in ds.samples]
    n = max_per_epoch or len(ds)
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


def make_loader(ds, batch_size: int, *, sampler=None, shuffle=False, workers: int = 2):
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
