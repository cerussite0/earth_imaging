"""
Compute NDVI from the full-scene Landsat B4 (Red) and B5 (NIR) bands.
"""

import os
import numpy as np
import rasterio

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def compute_and_save_ndvi() -> None:
    """
    Load B4 (Red) and B5 (NIR), compute NDVI, and save.

    NDVI = (NIR − Red) / (NIR + Red)
    Output range: [−1, 1]   (NaN where input is 0 / no-data)
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    out_path = config.NDVI_PATH

    b4_path = config.band_path("B4")
    b5_path = config.band_path("B5")

    if os.path.exists(out_path):
        print(f"  ndvi.tif — already exists, skipping")
        return

    if not os.path.exists(b4_path) or not os.path.exists(b5_path):
        raise FileNotFoundError(
            "B4.TIF and B5.TIF must exist. Run download_bands.py first."
        )

    print("Loading B4 (Red)...")
    with rasterio.open(b4_path) as src:
        red = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    print("Loading B5 (NIR)...")
    with rasterio.open(b5_path) as src:
        nir = src.read(1).astype(np.float32)

    # Mask no-data (Landsat fill value = 0)
    red[red == 0] = np.nan
    nir[nir == 0] = np.nan

    print("Computing NDVI...")
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)

    print(f"  NDVI range: [{np.nanmin(ndvi):.4f}, {np.nanmax(ndvi):.4f}]")
    print(f"  Shape: {ndvi.shape[0]}×{ndvi.shape[1]}")

    # Save with the same profile as the bands
    ndvi_profile = profile.copy()
    ndvi_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    with rasterio.open(out_path, "w", **ndvi_profile) as dst:
        dst.write(ndvi.astype(np.float32), 1)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✓ Saved ndvi.tif ({size_kb:.0f} KB)")


if __name__ == "__main__":
    compute_and_save_ndvi()
