"""
Builds a unified 4-class dataset from CIFAR-10 and CIFAR-100.

Class mapping
─────────────
  Our label │ Name     │ Source
  ──────────┼──────────┼──────────────────────────────
     0      │ airplane │ CIFAR-10  class index 0
     1      │ bicycle  │ CIFAR-100 class index 8
     2      │ car      │ CIFAR-10  class index 1 (automobile)
     3      │ dog      │ CIFAR-10  class index 5

CIFAR-10 → 5 000 train / 1 000 test samples per class
CIFAR-100 → 500 train / 100 test samples for bicycle
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor

from config import (
    DATA_DIR,
    CIFAR10_WANTED,
    CIFAR100_BICYCLE_IDX,
    CIFAR100_BICYCLE_TARGET,
    CLASS_NAMES,
)


class FilteredCIFAR10(Dataset):
    """CIFAR-10 filtered to the classes we care about, with remapped labels."""

    def __init__(self, train: bool, transform: Optional[Callable] = None):
        raw = CIFAR10(root=DATA_DIR, train=train, download=True, transform=None)
        self.transform = transform

        # Build index list of samples that belong to our wanted classes
        targets = np.array(raw.targets)
        self.data: list[tuple] = []

        for cifar_idx, our_label in CIFAR10_WANTED.items():
            mask = np.where(targets == cifar_idx)[0]
            for i in mask:
                img, _ = raw[i]   # PIL Image, original label (ignored)
                self.data.append((img, our_label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class FilteredCIFAR100Bicycle(Dataset):
    """CIFAR-100 filtered to only the bicycle class."""

    def __init__(self, train: bool, transform: Optional[Callable] = None):
        raw = CIFAR100(root=DATA_DIR, train=train, download=True, transform=None)
        self.transform = transform

        # Verify the class index before using it — silent bug if wrong
        assert raw.classes[CIFAR100_BICYCLE_IDX] == "bicycle", (
            f"Expected 'bicycle' at CIFAR-100 index {CIFAR100_BICYCLE_IDX}, "
            f"got '{raw.classes[CIFAR100_BICYCLE_IDX]}'. "
            "Update CIFAR100_BICYCLE_IDX in config.py."
        )

        targets = np.array(raw.targets)
        mask = np.where(targets == CIFAR100_BICYCLE_IDX)[0]

        self.data: list[tuple] = []
        for i in mask:
            img, _ = raw[i]
            self.data.append((img, CIFAR100_BICYCLE_TARGET))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataset(train: bool, transform: Optional[Callable] = None) -> ConcatDataset:
    """Return the full merged dataset for a split."""
    cifar10_ds  = FilteredCIFAR10(train=train, transform=transform)
    bicycle_ds  = FilteredCIFAR100Bicycle(train=train, transform=transform)
    return ConcatDataset([cifar10_ds, bicycle_ds])


def build_weighted_sampler(dataset: ConcatDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that up-samples bicycle to balance batches.

    Collects all labels from the dataset (may take a few seconds on first run
    because it iterates without transforms).
    """
    all_labels: list[int] = []
    for ds in dataset.datasets:
        all_labels.extend([label for _, label in ds.data])

    class_counts = [0] * len(CLASS_NAMES)
    for lbl in all_labels:
        class_counts[lbl] += 1

    print("Class distribution:", dict(zip(CLASS_NAMES, class_counts)))

    # Weight per sample = 1 / frequency of its class
    sample_weights = [1.0 / class_counts[lbl] for lbl in all_labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(all_labels),
        replacement=True,
    )
    return sampler
