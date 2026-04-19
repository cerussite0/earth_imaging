"""
Download the ESRI 10m Land Use / Land Cover (LULC) map to perfectly overlap
the massive, unclipped Landsat scene boundary.

It parses the physical spatial boundaries of the previously downloaded 
Landsat B4.TIF to query the Planetary Computer STAC.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, Resampling
import pystac_client
import planetary_computer
import rioxarray
from pyproj import Transformer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def download_esri_lulc() -> None:
    """
    Search Planetary Computer for ESRI tiles spanning the entire Landsat 
    bounding box. Reproject them natively into a 10m grid overlaying exactly.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    out_path = config.ESRI_10M_PATH

    if os.path.exists(out_path):
        print(f"  esri_lulc_10m.tif — already exists, skipping")
        return

    # Extract bounds from Landsat tile dynamically
    b4_path = config.band_path("B4")
    if not os.path.exists(b4_path):
         raise FileNotFoundError(f"Landsat band {b4_path} must be downloaded first to calculate full bounds.")
         
    with rasterio.open(b4_path) as src:
        target_crs = src.crs
        src_bounds = src.bounds
        
    print(f"Resolving full geographic boundaries from {b4_path}...")
    t = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    # Bottom Left to Top Right
    min_lon, min_lat = t.transform(src_bounds.left, src_bounds.bottom)
    max_lon, max_lat = t.transform(src_bounds.right, src_bounds.top)
    
    # Pad to prevent edge drop-off
    landsat_bbox = [min_lon - 0.05, min_lat - 0.05, max_lon + 0.05, max_lat + 0.05]

    # 1. Connect to STAC
    print("\nConnecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # 2. Search for ALL tiles intersecting the Landsat Boundaries
    print(f"Searching collection '{config.ESRI_COLLECTION}' for 180km area...")
    search = catalog.search(
        collections=[config.ESRI_COLLECTION],
        bbox=landsat_bbox,
        datetime=config.ESRI_DATETIME,
    )
    items = list(search.item_collection())

    if not items:
        print("  ✗ Could not find ESRI LULC tiles for this massive target area.")
        return

    print(f"  ✓ Found {len(items)} consecutive tile(s) to fetch.")

    # 3. Determine the output grid spanning the 180km space at 10m resolution
    res = 10.0
    width  = int(np.ceil((src_bounds.right - src_bounds.left) / res))
    height = int(np.ceil((src_bounds.top - src_bounds.bottom) / res))
    dst_transform = transform_from_bounds(src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top, width, height)

    print(f"  Allocating massive base grid: {height}×{width} pixels @ 10m ({target_crs})")

    # 4. Create the output array (Can take ~400MB memory to instantiate)
    mosaic = np.zeros((height, width), dtype=np.uint8)

    for item in items:
        print(f"  Processing incoming tile {item.id}...")
        with rasterio.open(item.assets["data"].href) as src:
            temp = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=temp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                dst_nodata=0,
            )
            # Merge temp into mosaic, keeping existing data where temp is 0
            mosaic = np.where(temp > 0, temp, mosaic)

    # 5. Save the merged, projected 10m result
    profile = {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "width":     width,
        "height":    height,
        "count":     1,
        "crs":       target_crs,
        "transform": dst_transform,
        "nodata":    0,
    }

    print(f"  Writing {width}x{height} pixels to disk...")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic, 1)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✓ Saved esri_lulc_10m.tif ({height}×{width}, {size_kb:.0f} KB)")


if __name__ == "__main__":
    download_esri_lulc()
