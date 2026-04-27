
import os, argparse
import numpy as np
import rasterio

def compute_and_save_ndvi(input_dir, out_path):
    os.makedirs((os.path.dirname(out_path) or '.'), exist_ok=True)
    b4_path = os.path.join(input_dir, 'B4.TIF')
    b5_path = os.path.join(input_dir, 'B5.TIF')
    if (not os.path.exists(b4_path)):
        b4_path = os.path.join(input_dir, 'B4.tif')
    if (not os.path.exists(b5_path)):
        b5_path = os.path.join(input_dir, 'B5.tif')
    if os.path.exists(out_path):
        return
    with rasterio.open(b4_path) as src:
        red = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    with rasterio.open(b5_path) as src:
        nir = src.read(1).astype(np.float32)
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = ((nir - red) / (nir + red))
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    ndvi_profile = profile.copy()
    ndvi_profile.update(dtype=rasterio.float32, count=1, nodata=0.0)
    with rasterio.open(out_path, 'w', **ndvi_profile) as dst:
        dst.write(ndvi.astype(np.float32), 1)
    print(f'Saved NDVI [{ndvi.min():.4f}, {ndvi.max():.4f}]')
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    compute_and_save_ndvi(args.input_dir, args.output_file)
