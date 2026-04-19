# %% [markdown]
# # TIF File Analysis & Conversion Notebook
# This script is formatted with `# %%` blocks, meaning it can be run interactively in an IDE like VSCode as a Jupyter Notebook.

# %%
import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import torch
import numpy as np

# %%
# Define the path to your .tif file
# Replace this with the actual path to your downloaded Landsat TIF file
tif_path = 'sample.tif' 

# %%
if not os.path.exists(tif_path):
    print(f"Warning: The file '{tif_path}' does not exist.")
    print("Please replace 'sample.tif' with the correct path to your actual .tif file.")
else:
    # Open the .tif file using rasterio
    with rasterio.open(tif_path) as src:
        # Read the image data into a numpy array (reads all available bands)
        img_array = src.read()
        
        print("====== TIF FILE METADATA ======")
        print(f"File path: {tif_path}")
        print(f"Dimensions (Bands, Height, Width): {img_array.shape}")
        
        # Calculate number of pixels
        num_pixels_per_band = img_array.shape[1] * img_array.shape[2]
        total_pixels = img_array.size
        print(f"Number of pixels per band (Height x Width): {num_pixels_per_band:,}")
        print(f"Total number of pixels (across all bands): {total_pixels:,}")
        
        print("\n====== PYTORCH CONVERSION ======")
        # Convert to torch tensor
        # Note: We copy the array because rasterio's underlying read buffer can sometimes be read-only,
        # which PyTorch might complain about during tensor operations later.
        tensor_img = torch.from_numpy(img_array.copy())
        
        print(f"Successfully converted to PyTorch Tensor!")
        print(f"Tensor Shape: {tensor_img.shape}")
        print(f"Tensor Data Type: {tensor_img.dtype}")
        
        print("\n====== VISUALIZATION ======")
        # Visualize the image using matplotlib and rasterio's show wrapper
        plt.figure(figsize=(10, 10))
        
        # Handle visualization based on the number of bands
        if img_array.shape[0] >= 3:
            # If the image has 3 or more bands, plot the first 3 bands as an RGB composite.
            # Depending on the specific Landsat bands present, you may need to slice different channels
            # (e.g., if bands are B4, B3, B2 for true color, adjust the slicing accordingly).
            # We scale the image down to [0, 1] for matplotlib if the max value is > 255.
            plot_array = img_array[0:3, :, :].astype(float)
            if plot_array.max() > 255:
                 plot_array = plot_array / plot_array.max()
            
            show(plot_array, transform=src.transform, title="TIF Visualization (First 3 Bands)")
            
        elif img_array.shape[0] == 1:
            # Plot single-band (like an NDVI mask or a single Landsat band)
            show(img_array, transform=src.transform, cmap='viridis', title="TIF Visualization (Single Band)")
            
        else:
            show(img_array, transform=src.transform, title=f"TIF Visualization ({img_array.shape[0]} Bands)")
        
        plt.show()
