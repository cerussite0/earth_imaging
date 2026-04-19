import os
import glob
import argparse
import rasterio
import torch
import numpy as np

def convert_tifs_to_tensor(input_dir, output_file="tensor_dataset.pt"):
    """
    Reads 6 band .tif files, an ndvi .tif, and an esri .tif from a directory
    and converts them into structured PyTorch tensors of shapes:
    - Bands: [6, H, W]
    - NDVI: [1, H, W]
    - ESRI: [1, H, W]
    All spatial dimensions (H, W) are strictly validated to be identical.
    """
    # Find all .tif and .TIF files in the directory
    import fnmatch
    all_files = os.listdir(input_dir)
    tif_files = [os.path.join(input_dir, f) for f in all_files if f.lower().endswith('.tif')]
    
    if not tif_files:
        print(f"Error: No .tif files found in '{input_dir}'")
        return
        
    ndvi_file = None
    esri_file = None
    band_files = []
    
    # Categorize files based on their names
    for f in tif_files:
        fname = os.path.basename(f).lower()
        if "ndvi" in fname:
            ndvi_file = f
        elif "esri" in fname or "lulc" in fname:
            esri_file = f
        else:
            band_files.append(f)
            
    # Sort band files to ensure consistent ordering (e.g., B2, B3, B4...)
    band_files = sorted(band_files)
    
    # Validations
    if len(band_files) != 6:
        print(f"Warning: Expected 6 band files, but found {len(band_files)}: {[os.path.basename(b) for b in band_files]}")
        print("Proceeding anyway, but please verify your input directory.")
        
    if not ndvi_file:
        raise FileNotFoundError(f"Could not find an NDVI tif file in {input_dir}")
    if not esri_file:
        raise FileNotFoundError(f"Could not find an ESRI tif file in {input_dir}")
        
    print(f"Found NDVI file: {os.path.basename(ndvi_file)}")
    print(f"Found ESRI file: {os.path.basename(esri_file)}")
    print(f"Found {len(band_files)} Band files.")
    
    def read_single_band_tif(filepath):
        with rasterio.open(filepath) as src:
            # src.read(1) reads the first band as a 2D array [H, W]
            return src.read(1)
            
    print("Reading files...")
    # Read NDVI and ESRI
    ndvi_array = read_single_band_tif(ndvi_file)
    esri_array = read_single_band_tif(esri_file)
    
    # Read the 6 spectral bands
    band_arrays = []
    for bf in band_files:
        band_arrays.append(read_single_band_tif(bf))
        
    # Shape Validation - ensuring all (H, W) match exactly
    H, W = ndvi_array.shape
    if esri_array.shape != (H, W):
        raise ValueError(f"Shape mismatch! NDVI is {H, W} but ESRI is {esri_array.shape}")
        
    for i, b_arr in enumerate(band_arrays):
        if b_arr.shape != (H, W):
            raise ValueError(f"Shape mismatch! Band '{os.path.basename(band_files[i])}' is {b_arr.shape}, expected {H, W}")
            
    print(f"All spatial dimensions match: (Height: {H}, Width: {W})")
    
    # Convert numpy arrays to PyTorch tensors
    # 1. Stack the bands to create [6, H, W]
    # np.stack creates [6, H, W], we copy() to ensure memory contiguity if rasterio locks it
    bands_tensor = torch.from_numpy(np.stack(band_arrays, axis=0).copy()).float()
    
    # 2. Add channel dimension to NDVI and ESRI to make them [1, H, W]
    ndvi_tensor = torch.from_numpy(ndvi_array.copy()).unsqueeze(0).float()
    
    # ESRI labels are usually discrete class integers, so it's common to cast them as Long/Int. 
    # If they are standard floats in your pipeline, change .long() to .float() here.
    esri_tensor = torch.from_numpy(esri_array.copy()).unsqueeze(0).long()
    
    print("\n====== TENSOR SHAPES ======")
    print(f"Bands Tensor: {bands_tensor.shape}")
    print(f"NDVI Tensor:  {ndvi_tensor.shape}")
    print(f"ESRI Tensor:  {esri_tensor.shape}")
    
    # Save to a standard PyTorch dictionary format
    dataset_dict = {
        "bands": bands_tensor,   # shape: [6, H, W]
        "ndvi": ndvi_tensor,     # shape: [1, H, W]
        "esri": esri_tensor      # shape: [1, H, W]
    }
    
    torch.save(dataset_dict, output_file)
    print(f"\nSuccessfully saved the tensor dataset to '{output_file}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Landsat/ESRI/NDVI .tif files into a PyTorch tensor dataset.")
    parser.add_argument("--input_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_tif', 'raw'), help="Directory containing the .tif files")
    parser.add_argument("--output_file", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'torch_dataset', 'dataset_tensors.pt'), help="Output path for the .pt file")
    
    args = parser.parse_args()
    convert_tifs_to_tensor(args.input_dir, args.output_file)
