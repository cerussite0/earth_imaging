
import os, argparse
import numpy as np
import rasterio
from rasterio.windows import Window

def find_largest_inscribed_square(mask):
    (H, W) = mask.shape
    dp = np.zeros((H, W), dtype=np.int32)
    dp[0, :] = mask[0, :].astype(np.int32)
    dp[:, 0] = mask[:, 0].astype(np.int32)
    for i in range(1, H):
        row_mask = mask[i, 1:]
        dp[i, 1:] = np.where(row_mask, (np.minimum(np.minimum(dp[i - 1, 1:], dp[i, :-1]), dp[i - 1, :-1]) + 1), 0)
    side = int(dp.max())
    br = int(np.argmax(dp))
    (br_row, br_col) = divmod(br, W)
    return (((br_row - side) + 1), ((br_col - side) + 1), side)

def clip_raster(src_path, dst_path, row_off, col_off, size):
    with rasterio.open(src_path) as src:
        window = Window(col_off=col_off, row_off=row_off, width=size, height=size)
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(width=size, height=size, transform=src.window_transform(window))
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data)
    return (os.path.getsize(dst_path) / 1024)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mask_file', type=str, default=None)
    parser.add_argument('--pixel_size', type=float, default=30.0)
    args = parser.parse_args()
    standalone = False
    if (args.mask_file and os.path.isfile(args.mask_file)):
        mask_path = args.mask_file
        if (os.path.dirname(os.path.abspath(mask_path)) != os.path.abspath(args.input_dir)):
            standalone = True
    else:
        mask_path = os.path.join(args.input_dir, 'B4.TIF')
        if (not os.path.exists(mask_path)):
            mask_path = os.path.join(args.input_dir, 'B4.tif')
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
    mask = (mask_data > 0)
    (row_start, col_start, side) = find_largest_inscribed_square(mask)
    px = args.pixel_size
    print(f'Found {side}x{side} square at ({row_start}, {col_start}) ({((side * px) / 1000):.1f} x {((side * px) / 1000):.1f} km)')
    os.makedirs(args.output_dir, exist_ok=True)
    if standalone:
        files = [os.path.basename(mask_path)]
        src_paths = [mask_path]
    else:
        files = sorted((f for f in os.listdir(args.input_dir) if f.casefold().endswith('.tif')))
        src_paths = [os.path.join(args.input_dir, f) for f in files]
    for (name, src_path) in zip(files, src_paths):
        dst = os.path.join(args.output_dir, name)
        kb = clip_raster(src_path, dst, row_start, col_start, side)
        print(f'  {name}: {kb:,.0f} KB')
    print(f'Clipped {len(files)} file(s) to {side}x{side}')
if (__name__ == '__main__'):
    main()
