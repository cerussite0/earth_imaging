
import os, time, argparse, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from download_bands import download_all_bands
from download_esri import download_esri_lulc
from align_data import align_esri_to_landsat, verify_alignment
from compute_ndvi import compute_and_save_ndvi

def main(output_dir):
    start = time.time()
    print(f'Output: {output_dir}')
    print(f"AOI: lat [{config.AOI['min_lat']}, {config.AOI['max_lat']}]")
    print(f"     lon [{config.AOI['min_lon']}, {config.AOI['max_lon']}]")
    print(f'''Bands: {', '.join(config.BANDS)}
''')
    entity_id = download_all_bands(output_dir)
    download_esri_lulc(output_dir, output_dir)
    aligned_esri = os.path.join(output_dir, 'esri_lulc.tif')
    align_esri_to_landsat(output_dir, aligned_esri)
    ndvi_path = os.path.join(output_dir, 'ndvi.tif')
    compute_and_save_ndvi(output_dir, ndvi_path)
    verify_alignment(output_dir, aligned_esri)
    elapsed = (time.time() - start)
    print(f'''
Pipeline complete in {elapsed:.1f}s''')
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args.output_dir)
