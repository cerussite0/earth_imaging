
import os, argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def _get_reference_profile(b4_path):
    with rasterio.open(b4_path) as src:
        return src.profile.copy()

def align_esri_to_landsat(input_dir, output_file):
    os.makedirs((os.path.dirname(output_file) or '.'), exist_ok=True)
    src_path = os.path.join(input_dir, 'esri_lulc_10m.tif')
    b4_path = os.path.join(input_dir, 'B4.TIF')
    if (not os.path.exists(b4_path)):
        b4_path = os.path.join(input_dir, 'B4.tif')
    if os.path.exists(output_file):
        return
    ref = _get_reference_profile(b4_path)
    (ref_h, ref_w) = (ref['height'], ref['width'])
    aligned = np.zeros((ref_h, ref_w), dtype=np.uint8)
    with rasterio.open(src_path) as src:
        reproject(source=rasterio.band(src, 1), destination=aligned, src_transform=src.transform, src_crs=src.crs, dst_transform=ref['transform'], dst_crs=ref['crs'], resampling=Resampling.nearest)
    with rasterio.open(b4_path) as f:
        aligned[f.read(1) == 0] = 0
    out_profile = ref.copy()
    out_profile.update(dtype='uint8', count=1, nodata=0)
    with rasterio.open(output_file, 'w', **out_profile) as dst:
        dst.write(aligned, 1)
    print(f'Saved aligned ESRI ({ref_h}x{ref_w})')

def verify_alignment(input_dir, dst_path):
    files = []
    for f in os.listdir(input_dir):
        if (f.casefold().endswith('.tif') and ('10m' not in f.casefold())):
            files.append((f, os.path.join(input_dir, f)))
    if os.path.exists(dst_path):
        if (not any(((f[0] == os.path.basename(dst_path)) for f in files))):
            files.append((os.path.basename(dst_path), dst_path))
    (ref_shape, ref_tf) = (None, None)
    all_ok = True
    for (name, path) in files:
        with rasterio.open(path) as src:
            shape = (src.height, src.width)
            tf = src.transform
            if (ref_shape is None):
                (ref_shape, ref_tf) = (shape, tf)
            elif ((shape != ref_shape) or (tf != ref_tf)):
                print(f'  MISMATCH: {name}')
                all_ok = False
    if all_ok:
        print('All files pixel-aligned.')
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    align_esri_to_landsat(args.input_dir, args.output_file)
    verify_alignment(args.input_dir, args.output_file)
