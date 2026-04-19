#!/usr/bin/env python3
"""
Landsat Download Pipeline — Main Orchestrator
===============================================
Downloads Landsat bands B2–B7, downloads ESRI LULC,
aligns everything to the same pixel grid, and computes NDVI.

All pixel-aligned outputs go to Data_tif/raw/ with identical shape,
CRS, and transform — ready for segmentation model training.

Usage:
    python run_pipeline.py
"""

import os
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from download_bands import download_all_bands
from download_esri  import download_esri_lulc
from align_data     import align_esri_to_landsat, verify_alignment
from compute_ndvi   import compute_and_save_ndvi


def print_summary() -> None:
    """Print a summary table of all files in the data/ directory (recursive)."""
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Output directory : {config.DATA_DIR}")
    print(f"  AOI              : ({config.AOI['min_lat']}, {config.AOI['min_lon']}) → "
          f"({config.AOI['max_lat']}, {config.AOI['max_lon']})")
    print("-" * 60)

    total_bytes = 0
    for root, dirs, files in os.walk(config.DATA_DIR):
        for fname in sorted(files):
            if not fname.lower().endswith(('.tif', '.tiff')):
                continue
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            total_bytes += size
            size_str = f"{size / (1024*1024):.1f} MB" if size > 1024*1024 else f"{size / 1024:.0f} KB"
            rel = os.path.relpath(fpath, config.DATA_DIR)
            print(f"  {rel:35s}  {size_str:>10s}")

    print("-" * 60)
    print(f"  Total: {total_bytes / (1024*1024):.1f} MB")
    print("=" * 60)


def main() -> None:
    start = time.time()

    print("=" * 60)
    print("  LANDSAT DOWNLOAD PIPELINE")
    print("=" * 60)
    print(f"  AOI : lat [{config.AOI['min_lat']}, {config.AOI['max_lat']}]")
    print(f"        lon [{config.AOI['min_lon']}, {config.AOI['max_lon']}]")
    print(f"  Bands : {', '.join(config.BANDS)}")
    print("=" * 60 + "\n")

    # --- Step 1: Download Landsat bands (full scenes) ---
    print("━" * 60)
    print("  STEP 1 / 5 :  Download Landsat Bands (B2–B7)")
    print("━" * 60)
    entity_id = download_all_bands()

    # --- Step 2: Download ESRI LULC (native 10m) ---
    print("\n" + "━" * 60)
    print("  STEP 2 / 4 :  Download ESRI LULC Map (10m)")
    print("━" * 60)
    download_esri_lulc()

    # --- Step 3: Align ESRI to Landsat pixel grid ---
    print("\n" + "━" * 60)
    print("  STEP 3 / 4 :  Align ESRI to Landsat Grid (30m)")
    print("━" * 60)
    align_esri_to_landsat()

    # --- Step 4: Compute NDVI ---
    print("\n" + "━" * 60)
    print("  STEP 4 / 4 :  Compute NDVI")
    print("━" * 60)
    compute_and_save_ndvi()

    # --- Verification ---
    print("\n" + "━" * 60)
    print("  PIXEL ALIGNMENT VERIFICATION")
    print("━" * 60)
    verify_alignment()

    # --- Summary ---
    elapsed = time.time() - start
    print_summary()
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")
    print("  Run `python view_data.py` to visualize the downloaded data.\n")


if __name__ == "__main__":
    main()
