
import random
import os
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_DATASET_PATH, VAL_DATASET_PATH, NUM_INPUT_CHANNELS, NUM_CLASSES, CLASS_TO_INDEX, NODATA_INDEX, BATCH_SIZE, RANDOM_SEED

class LULCDataset(Dataset):

    def __init__(self, x_tensor, y_tensor, augment=False):
        super().__init__()
        self.x = x_tensor
        self.y = y_tensor
        self.augment = augment
        self._build_remap_lut()

    def _build_remap_lut(self):
        max_raw = (max(CLASS_TO_INDEX.keys()) + 1)
        self.remap_lut = torch.full((max_raw,), NODATA_INDEX, dtype=torch.long)
        for (raw_id, idx) in CLASS_TO_INDEX.items():
            if (raw_id < max_raw):
                self.remap_lut[raw_id] = idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].clone()
        y = self.y[idx].clone()
        if self.augment:
            if (random.random() > 0.5):
                x = TF.hflip(x)
                y = TF.hflip(y)
            if (random.random() > 0.5):
                x = TF.vflip(x)
                y = TF.vflip(y)
        y = y.long()
        y = torch.clamp(y, min=0, max=(len(self.remap_lut) - 1))
        y = self.remap_lut[y]
        return (x, y)

def _load_pt_file(path, label):
    ds = torch.load(path, weights_only=False, map_location='cpu')
    x = ds.tensors[0]
    y = ds.tensors[1]
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    if ((y.dim() == 4) and (y.shape[1] == 1)):
        y = y.squeeze(1)
    return (x, y)

def create_dataloaders():
    (x_train, y_train) = _load_pt_file(TRAIN_DATASET_PATH, 'train')
    (x_val, y_val) = _load_pt_file(VAL_DATASET_PATH, 'val')
    train_dataset = LULCDataset(x_train, y_train, augment=True)
    val_dataset = LULCDataset(x_val, y_val, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    dataset_info = {'n_train': len(train_dataset), 'n_val': len(val_dataset), 'num_classes': NUM_CLASSES, 'num_channels': NUM_INPUT_CHANNELS}
    return (train_loader, val_loader, dataset_info)
