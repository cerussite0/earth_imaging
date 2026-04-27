
import os, argparse, sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, Resampling
import pystac_client
import planetary_computer
from pyproj import Transformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def download_esri_lulc(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'esri_lulc_10m.tif')
    if os.path.exists(out_path):
        return
    b4_path = os.path.join(input_dir, 'B4.TIF')
    if (not os.path.exists(b4_path)):
        b4_path = os.path.join(input_dir, 'B4.tif')

    with rasterio.open(b4_path) as src:
        target_crs = src.crs
        src_bounds = src.bounds
    t = Transformer.from_crs(target_crs, 'EPSG:4326', always_xy=True)
    (min_lon, min_lat) = t.transform(src_bounds.left, src_bounds.bottom)
    (max_lon, max_lat) = t.transform(src_bounds.right, src_bounds.top)
    bbox = [(min_lon - 0.05), (min_lat - 0.05), (max_lon + 0.05), (max_lat + 0.05)]
    catalog = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[config.ESRI_COLLECTION], bbox=bbox, datetime=config.ESRI_DATETIME)
    items = list(search.item_collection())
    if (not items):
        print('No ESRI tiles found.')
        return
    print(f'Found {len(items)} tile(s)')
    res = 10.0
    width = int(np.ceil(((src_bounds.right - src_bounds.left) / res)))
    height = int(np.ceil(((src_bounds.top - src_bounds.bottom) / res)))
    dst_transform = transform_from_bounds(src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top, width, height)
    mosaic = np.zeros((height, width), dtype=np.uint8)
    for item in items:
        with rasterio.open(item.assets['data'].href) as src:
            temp = np.zeros((height, width), dtype=np.uint8)
            reproject(source=rasterio.band(src, 1), destination=temp, src_transform=src.transform, src_crs=src.crs, dst_transform=dst_transform, dst_crs=target_crs, resampling=Resampling.nearest, dst_nodata=0)
            mosaic = np.where((temp > 0), temp, mosaic)
    profile = {'driver': 'GTiff', 'dtype': 'uint8', 'width': width, 'height': height, 'count': 1, 'crs': target_crs, 'transform': dst_transform, 'nodata': 0}
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mosaic, 1)
    print(f'Saved esri_lulc_10m.tif ({height}x{width})')
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    download_esri_lulc(args.input_dir, args.output_dir)
