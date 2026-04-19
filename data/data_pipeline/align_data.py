"""
Align ESRI data to an identical pixel grid.

ESRI LULC (10m) is resampled to the 30m Landsat grid using nearest-neighbour 
interpolation to preserve class integers perfectly.
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def _get_reference_profile() -> dict:
    """
    Read the profile (CRS, transform, shape) from the unclipped master 
    Landsat band B4 to define the master pixel grid.
    """
    path = config.band_path("B4")
    if os.path.exists(path):
        with rasterio.open(path) as src:
            return src.profile.copy()

    raise FileNotFoundError(
        f"Landsat band {path} not found — run download_bands.py first."
    )


def align_esri_to_landsat() -> None:
    """
    Reproject the native-resolution ESRI LULC (10m) onto the exact pixel grid
    of the Landsat bands (30m).  Uses nearest-neighbour resampling so
    class integer values are preserved.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    src_path = config.ESRI_10M_PATH
    dst_path = config.ESRI_PATH

    if os.path.exists(dst_path):
        print(f"  esri_lulc.tif (aligned) — already exists, skipping")
        return

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"{src_path} not found — run download_esri.py first."
        )

    # Get the reference grid from Landsat band
    ref_profile = _get_reference_profile()
    ref_h = ref_profile["height"]
    ref_w = ref_profile["width"]

    print(f"  Reference grid: {ref_h}×{ref_w} "
          f"({ref_profile['crs']}, {ref_profile['transform'].a:.1f}m)")

    # Reproject ESRI onto the reference grid
    aligned = np.zeros((ref_h, ref_w), dtype=np.uint8)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )

    # Mask with Landsat reference valid data footprint to mimic exact satellite track orientation
    with rasterio.open(config.band_path("B4")) as ref:
        b4_data = ref.read(1)
        aligned[b4_data == 0] = 0

    # Save with the exact same profile as the Landsat bands
    out_profile = ref_profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=0)

    with rasterio.open(dst_path, "w", **out_profile) as dst:
        dst.write(aligned, 1)

    classes = np.unique(aligned)
    size_kb = os.path.getsize(dst_path) / 1024
    print(f"  ✓ Saved esri_lulc.tif ({ref_h}×{ref_w}, {size_kb:.0f} KB)")
    print(f"    Classes present: {classes}")


def verify_alignment() -> None:
    """
    Verify that all key files in the directory share the same shape, CRS,
    and transform. Prints a table for visual confirmation.
    """
    print("\n  Pixel-grid alignment check:")
    print(f"  {'File':20s}  {'Shape':16s}  {'CRS':12s}  {'Pixel size':10s}")
    print("  " + "─" * 65)

    files = []
    for band in config.BANDS:
        p = config.band_path(band)
        if os.path.exists(p):
            files.append((f"{band}.TIF", p))
    if os.path.exists(config.NDVI_PATH):
        files.append(("ndvi.tif", config.NDVI_PATH))
    if os.path.exists(config.ESRI_PATH):
        files.append(("esri_lulc.tif", config.ESRI_PATH))

    ref_shape = None
    ref_tf = None
    all_match = True

    for name, path in files:
        with rasterio.open(path) as src:
            shape = (src.height, src.width)
            crs = str(src.crs)
            px = f"{abs(src.transform.a):.1f}m"
            tf = src.transform

            match = "✓"
            if ref_shape is None:
                ref_shape = shape
                ref_tf = tf
            elif shape != ref_shape or tf != ref_tf:
                match = "✗ MISMATCH"
                all_match = False

            print(f"  {name:20s}  {shape[0]}×{shape[1]:<7d}  {crs:12s}  {px:10s}  {match}")

    if all_match:
        print("\n  ✓ All files are pixel-aligned. Ready for ML tensor generation.")
    else:
        print("\n  ✗ ALIGNMENT ERROR — some files have different grids!")


if __name__ == "__main__":
    align_esri_to_landsat()
    verify_alignment()
