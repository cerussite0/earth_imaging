"""
PyTorch Sliding Window Dataset for Landsat + ESRI

Dynamically slices a massive contiguous Landsat/ESRI tensor into matched
(X, y) patches compatible with native PyTorch DataLoaders.

X: 7-Channel Tensor (6 Landsat Bands + 1 NDVI)
y: 1-Channel Tensor (ESRI LULC Categorical classes)
"""

import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

class LandsatSlidingDataset(Dataset):
    """
    A dynamic memory-efficient Dataset that extracts spatial crops from 
    large contiguous remote sensing tensors.
    """
    
    def __init__(
        self, 
        tensor_path: str, 
        window_size: int = 128, 
        stride: int = 128, 
        min_valid: float = 0.5,
        transform=None
    ):
        super().__init__()
        self.tensor_path = tensor_path
        self.window_size = window_size
        self.stride = stride
        self.min_valid = min_valid
        self.transform = transform
        
        print(f"Loading master tensor dataset into memory from: {tensor_path}")
        master_data = torch.load(tensor_path, weights_only=True)
        
        # [6, H, W] bands and [1, H, W] NDVI -> [7, H, W] X
        print("Concatenating Bands and NDVI...")
        self.X = torch.cat([master_data['bands'], master_data['ndvi']], dim=0).float()
        
        # ESRI labels [1, H, W] -> y
        print("Loading ESRI labels...")
        self.y = master_data['esri'].long()
        
        # Validate shapes match
        self.channels, self.H, self.W = self.X.shape
        _, self.yH, self.yW = self.y.shape
        if self.H != self.yH or self.W != self.yW:
            raise RuntimeError(f"Shape Mismatch! X is {self.H}x{self.W} but y is {self.yH}x{self.yW}")

        print(f"Master Grid Resoluton: {self.H} × {self.W} pixels")
        print("Computing valid sliding window patches...")
        self._compute_offsets()

    def _compute_offsets(self):
        """
        Calculates all valid (row, col) patches in the huge continuous map.
        Skips patches that are primarily black padding borders.
        """
        self.offsets = []
        skipped = 0
        w = self.window_size
        s = self.stride
        
        # Sweep the grid
        for r in range(0, self.H - w + 1, s):
            for c in range(0, self.W - w + 1, s):
                # Inspect the first band to verify it's not all empty bounding-box padding
                # Since B2.TIF fill-value is 0.0 natively, 0 means padding
                patch_mask = self.X[0, r:r+w, c:c+w]
                valid_ratio = (patch_mask > 0).float().mean()
                
                if valid_ratio >= self.min_valid:
                    self.offsets.append((r, c))
                else:
                    skipped += 1
                    
        print(f"✓ Found {len(self.offsets)} completely valid {w}x{w} windows (skipped {skipped} due to padding bounds).")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        """
        Returns an isolated (X, y) patch formatted directly for Torch ML.
        X shape: (7, W, W)
        y shape: (1, W, W)  <-- Categorical integers
        """
        r, c = self.offsets[idx]
        w = self.window_size
        
        x_patch = self.X[:, r:r+w, c:c+w]
        y_patch = self.y[:, r:r+w, c:c+w]
        
        if self.transform is not None:
            x_patch, y_patch = self.transform(x_patch, y_patch)
            
        return x_patch, y_patch


# =====================================================================
# CLI Usage
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Construct and Test a PyTorch Dataset from the Master Tensors")
    parser.add_argument(
        "--tensor_path", type=str, 
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'torch_dataset', 'dataset_tensors.pt')),
        help="Path to the monolithic dataset_tensors.pt file"
    )
    parser.add_argument("--window_size", type=int, default=128, help="Window crop size (e.g., 128)")
    parser.add_argument("--stride", type=int, default=128, help="Window sliding step size")
    parser.add_argument("--min_valid", type=float, default=0.5, help="Min non-zero ratio required per tile (default 50%)")
    parser.add_argument(
        "--save_path", type=str, 
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'torch_dataset', 'torch_sliding_dataset.pt')),
        help="Path to physically save the overlapping dataset mapping"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print(" INITIALIZING DATASET")
    print("="*50)
    
    dataset = LandsatSlidingDataset(
        tensor_path=args.tensor_path,
        window_size=args.window_size,
        stride=args.stride,
        min_valid=args.min_valid
    )
    
    if len(dataset) == 0:
        print("\n✗ Dataset resulted in 0 valid patches. Exiting.")
        return

    print("\n" + "="*50)
    print(" SAVING COMPILED DATASET TO DISK")
    print("="*50)
    
    print(f"Iterating and stacking {len(dataset)} isolated patches into memory...")
    x_list, y_list = [], []
    for x_patch, y_patch in dataset:
        x_list.append(x_patch)
        y_list.append(y_patch)
        
    print("Consolidating contiguous X array...")
    X_full = torch.stack(x_list, dim=0)
    print("Consolidating contiguous y array...")
    Y_full = torch.stack(y_list, dim=0)
    
    print(f"Final X Shape: {X_full.shape} | {X_full.dtype}")
    print(f"Final y Shape: {Y_full.shape} | {Y_full.dtype}")
    
    print(f"\nCompressing and compiling to disk at: {args.save_path} ...")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    static_dataset = torch.utils.data.TensorDataset(X_full, Y_full)
    torch.save(static_dataset, args.save_path)
    
    final_size_mb = os.path.getsize(args.save_path) / (1024 * 1024)
    print(f"✓ Beautifully saved! Final disk size: {final_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
