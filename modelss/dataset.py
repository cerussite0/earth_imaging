"""
Dataset loading and preprocessing for LULC segmentation.

Mirrors the dataloading logic from visualize_torch_dataset.ipynb:
  1. Loads the saved PyTorch TensorDataset from .pt files.
  2. All 7 bands (B2, B3, B4, B5, B6, B7, NDVI) are preserved as input channels.
  3. Labels are remapped from non-contiguous ESRI class IDs to contiguous indices.
  4. Input tensors are normalized per-channel (zero-mean, unit-variance).
  5. Data is split into train/val sets with reproducible random shuffling.
"""

import os
import random
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from config import (
    DATASET_PATH, METADATA_PATH, DEVICE,
    NUM_INPUT_CHANNELS, NUM_CLASSES,
    CLASS_TO_INDEX, NODATA_INDEX,
    BATCH_SIZE, TRAIN_SPLIT, RANDOM_SEED,
    PATCH_SIZE,
)


class LULCDataset(Dataset):
    """
    Custom Dataset wrapping the pre-built TensorDataset from prepare_dataset.py.

    Applies:
      - Per-channel normalization (zero-mean, unit-variance) on the 7-band input.
      - Label remapping from raw ESRI class IDs to contiguous [0..NUM_CLASSES-1].
    """

    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor,
                 channel_means: torch.Tensor = None,
                 channel_stds: torch.Tensor = None,
                 apply_augmentation: bool = False):
        """
        Parameters
        ----------
        x_tensor : Tensor of shape (N, 7, H, W), float32
            7-band input patches (B2, B3, B4, B5, B6, B7, NDVI).
        y_tensor : Tensor of shape (N, H, W), int64
            Raw ESRI LULC class labels.
        channel_means : Tensor of shape (7,), optional
            Per-channel means for normalization. Computed from data if None.
        channel_stds : Tensor of shape (7,), optional
            Per-channel stds for normalization. Computed from data if None.
        """
        super().__init__()
        self.x = x_tensor
        self.y = y_tensor

        # Compute or store normalization stats
        if channel_means is None or channel_stds is None:
            self.channel_means, self.channel_stds = self._compute_stats(x_tensor)
        else:
            self.channel_means = channel_means
            self.channel_stds = channel_stds

        self.apply_augmentation = apply_augmentation

        # Build the label remap lookup table
        self._build_remap_lut()

    def _compute_stats(self, x: torch.Tensor):
        """Compute per-channel mean and std across all samples."""
        # x shape: (N, C, H, W)
        means = x.mean(dim=(0, 2, 3))  # shape: (C,)
        stds = x.std(dim=(0, 2, 3))    # shape: (C,)
        # Avoid division by zero
        stds = torch.clamp(stds, min=1e-6)
        return means, stds

    def _build_remap_lut(self):
        """Create a lookup table for fast label remapping."""
        max_raw = max(CLASS_TO_INDEX.keys()) + 1
        self.remap_lut = torch.full((max_raw,), NODATA_INDEX, dtype=torch.long)
        for raw_id, idx in CLASS_TO_INDEX.items():
            if raw_id < max_raw:
                self.remap_lut[raw_id] = idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].clone()  # (C, H, W)
        y = self.y[idx].clone()  # (H, W)

        if self.apply_augmentation:
            # Random Horizontal Flip (50% chance)
            if random.random() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)
            # Random Vertical Flip (50% chance)
            if random.random() > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

        # Normalize: (x - mean) / std, broadcast over spatial dims
        mean = self.channel_means.view(-1, 1, 1)  # (C, 1, 1)
        std = self.channel_stds.view(-1, 1, 1)    # (C, 1, 1)
        x = (x - mean) / std

        # Remap labels to contiguous indices
        y = y.long()
        y = torch.clamp(y, min=0, max=len(self.remap_lut) - 1)
        y = self.remap_lut[y]

        return x, y


def load_raw_dataset():
    """
    Load the raw TensorDataset from disk (same logic as the notebook).

    Returns
    -------
    x_all : Tensor of shape (N, 7, H, W), float32
    y_all : Tensor of shape (N, H, W), int64
    metadata : list[dict]
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH}\n"
            f"Run prepare_dataset.py first."
        )

    print(f"  Loading dataset from: {DATASET_PATH}")
    torch_dataset = torch.load(DATASET_PATH, weights_only=False, map_location="cpu")

    # Load metadata if available
    metadata = []
    if os.path.exists(METADATA_PATH):
        metadata = torch.load(METADATA_PATH, weights_only=False, map_location="cpu")

    # Extract tensors from TensorDataset
    x_all = torch_dataset.tensors[0]  # (N, C, H, W)
    y_all = torch_dataset.tensors[1]  # (N, H, W)

    print(f"  Dataset loaded: {len(x_all)} samples")
    print(f"  Input shape:  {tuple(x_all.shape)}")
    print(f"  Label shape:  {tuple(y_all.shape)}")
    print(f"  Input dtype:  {x_all.dtype}")
    print(f"  Label dtype:  {y_all.dtype}")
    print(f"  Input range:  [{x_all.min():.2f}, {x_all.max():.2f}]")
    print(f"  Label unique: {torch.unique(y_all).tolist()}")
    print(f"  Metadata rows: {len(metadata)}")

    return x_all, y_all, metadata


def create_dataloaders():
    """
    Create train and validation DataLoaders.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    dataset_info : dict with normalization stats and class info
    """
    x_all, y_all, metadata = load_raw_dataset()

    # Verify channel count matches config
    assert x_all.shape[1] == NUM_INPUT_CHANNELS, (
        f"Expected {NUM_INPUT_CHANNELS} input channels, "
        f"got {x_all.shape[1]}. Check prepare_dataset.py."
    )

    # Compute normalization stats on the FULL dataset (before splitting)
    full_dataset = LULCDataset(x_all, y_all)
    channel_means = full_dataset.channel_means
    channel_stds = full_dataset.channel_stds

    print(f"\n  Channel normalization stats:")
    from config import BAND_NAMES
    for i, name in enumerate(BAND_NAMES):
        print(f"    {name}: mean={channel_means[i]:.2f}, std={channel_stds[i]:.2f}")

    # Split into train/val
    n = len(x_all)
    n_train = int(n * TRAIN_SPLIT)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(n, generator=generator)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create train and val datasets (sharing the same normalization stats)
    train_dataset = LULCDataset(
        x_all[train_indices], y_all[train_indices],
        channel_means=channel_means, channel_stds=channel_stds,
        apply_augmentation=True
    )
    val_dataset = LULCDataset(
        x_all[val_indices], y_all[val_indices],
        channel_means=channel_means, channel_stds=channel_stds,
    )

    print(f"\n  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    dataset_info = {
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "num_classes": NUM_CLASSES,
        "num_channels": NUM_INPUT_CHANNELS,
    }

    return train_loader, val_loader, dataset_info
