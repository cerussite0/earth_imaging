
import os, argparse
import rasterio
import torch
import numpy as np

def convert_esri_only(input_file, output_file):
    with rasterio.open(input_file) as src:
        data = src.read(1)
    esri_tensor = torch.from_numpy(data.copy()).unsqueeze(0).long()
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save({'esri': esri_tensor}, output_file)
    print(f'Saved ESRI tensor {esri_tensor.shape} to {output_file}')

def convert_tifs_to_tensor(input_dir, output_file='tensor_dataset.pt'):
    all_files = os.listdir(input_dir)
    tif_files = [os.path.join(input_dir, f) for f in all_files if f.lower().endswith('.tif')]
    (ndvi_file, esri_file, band_files) = (None, None, [])
    for f in tif_files:
        fname = os.path.basename(f).lower()
        if ('ndvi' in fname):
            ndvi_file = f
        elif (('esri' in fname) or ('lulc' in fname)):
            esri_file = f
        else:
            band_files.append(f)
    band_files = sorted(band_files)

    def read_band(path):
        with rasterio.open(path) as src:
            return src.read(1)
    ndvi_array = read_band(ndvi_file)
    esri_array = read_band(esri_file)
    band_arrays = [read_band(f) for f in band_files]
    (H, W) = ndvi_array.shape
    bands_tensor = torch.from_numpy(np.stack(band_arrays, axis=0).copy()).float()
    for i in range(bands_tensor.shape[0]):
        band = bands_tensor[i]
        valid = band[band > 0].numpy()
        if (len(valid) == 0):
            valid = band.flatten().numpy()
        (p2, p98) = (np.percentile(valid, 2), np.percentile(valid, 98))
        bands_tensor[i] = torch.clamp(((band - p2) / ((p98 - p2) + 1e-09)), 0.0, 1.0)
    ndvi_tensor = torch.from_numpy(ndvi_array.copy()).unsqueeze(0).float()
    esri_tensor = torch.from_numpy(esri_array.copy()).unsqueeze(0).long()
    dataset_dict = {'bands': bands_tensor, 'ndvi': ndvi_tensor, 'esri': esri_tensor}
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(dataset_dict, output_file)
    print(f'Saved: bands {bands_tensor.shape}, ndvi {ndvi_tensor.shape}, esri {esri_tensor.shape}')
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--esri_only', type=str, default=None)
    args = parser.parse_args()
    if args.esri_only:
        convert_esri_only(args.esri_only, args.output_file)
    else:
        if (not args.input_dir):
            parser.error('--input_dir required when not using --esri_only')
        convert_tifs_to_tensor(args.input_dir, args.output_file)
