#!/usr/bin/env python3
import os
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def stretch_percentile(band, percentiles=(2, 98)):
    """Apply percentile stretching to make the image visible."""
    band = band.astype(np.float32)
    band[band == 0] = np.nan
    p_low, p_high = np.nanpercentile(band, percentiles)
    stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
    stretched[np.isnan(stretched)] = 0
    return stretched

def view_samples(num_samples=5):
    img_dir = os.path.join(config.DATA_DIR, "dataset", "images")
    lbl_dir = os.path.join(config.DATA_DIR, "dataset"
    , "labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Dataset directories not found: {img_dir}")
        return

    # Get a list of all tile files
    all_tiles = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")])
    if not all_tiles:
        print("No tiles found in the dataset.")
        return

    print(f"Found {len(all_tiles)} tiles. Selecting {min(num_samples, len(all_tiles))} to visualize...\n")
    
    # Pick random samples (use a fixed seed for reproducibility, or just random)
    random.seed(42)
    samples = random.sample(all_tiles, min(num_samples, len(all_tiles)))
    
    # Colormap setup for ESRI
    max_val = max(config.ESRI_CLASSES.keys())
    color_list = ["#000000"] * (max_val + 1)
    for val, (name, hex_color) in config.ESRI_CLASSES.items():
        color_list[val] = hex_color
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(np.arange(-0.5, max_val + 1.5, 1), cmap.N)
    
    legend_patches = [
        mpatches.Patch(color=hex_c, label=name)
        for val, (name, hex_c) in sorted(config.ESRI_CLASSES.items())
        if val != 0
    ]

    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 5 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for i, tile_name in enumerate(samples):
        img_path = os.path.join(img_dir, tile_name)
        lbl_path = os.path.join(lbl_dir, tile_name)
        
        # Read the Landsat image tile
        with rasterio.open(img_path) as src_img:
            # Band mapping: B2=idx 0, B3=idx 1, B4=idx 2, B5=idx 3, B6=idx 4, B7=idx 5
            # We want RGB: B4, B3, B2 (idx 2, 1, 0)
            img_data = src_img.read([3, 2, 1])  # 1-indexed for rasterio.read() -> B4, B3, B2
            img_h, img_w = src_img.height, src_img.width
            img_crs = src_img.crs
            print(f"[{tile_name}] Image : {img_h}×{img_w} pixels, shape={img_data.shape}, CRS={img_crs}, size={os.path.getsize(img_path)/1024:.1f} KB")

        # Stretch channels for RGB display
        rgb = np.dstack([
            stretch_percentile(img_data[0]),
            stretch_percentile(img_data[1]),
            stretch_percentile(img_data[2])
        ])

        # Read the ESRI label tile
        with rasterio.open(lbl_path) as src_lbl:
            lbl_data = src_lbl.read(1)
            lbl_h, lbl_w = src_lbl.height, src_lbl.width
            lbl_crs = src_lbl.crs
            print(f"[{tile_name}] Label : {lbl_h}×{lbl_w} pixels, shape={lbl_data.shape}, CRS={lbl_crs}, size={os.path.getsize(lbl_path)/1024:.1f} KB")

        print()

        # Plot RGB
        axes[i][0].imshow(rgb)
        axes[i][0].set_title(f"{tile_name} - True Color (B4/B3/B2)")
        axes[i][0].axis("off")

        # Plot ESRI LULC
        img = axes[i][1].imshow(lbl_data, cmap=cmap, norm=norm)
        axes[i][1].set_title(f"{tile_name} - ESRI LULC Labels")
        axes[i][1].axis("off")
        
        # Add legend to the first label plot
        if i == 0:
            axes[i][1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1),
                  loc="upper left", borderaxespad=0., title="Land Cover",
                  frameon=True)

    plt.tight_layout()
    out_path = os.path.join(config.DATA_DIR, "dataset", "preview_dataset_samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved visual preview of 5 tiles to {out_path}")

if __name__ == "__main__":
    view_samples(5)
