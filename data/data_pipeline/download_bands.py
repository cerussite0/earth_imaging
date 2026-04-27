
import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import usgs_api

def download_all_bands(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    api_key = usgs_api.login(config.USGS_API_URL, config.USGS_USERNAME, config.USGS_TOKEN)
    scenes = usgs_api.search_scenes(api_url=config.USGS_API_URL, api_key=api_key, dataset=config.DATASET_NAME, aoi=config.AOI, max_cloud=config.MAX_CLOUD, max_results=config.MAX_RESULTS, sort_field=config.SORT_FIELD, sort_direction=config.SORT_DIRECTION)

    best = scenes[0]
    entity_id = best['entityId']
    print(f"Selected: {entity_id} (cloud: {best.get('cloudCover')}%)")
    urls = usgs_api.get_download_urls(api_url=config.USGS_API_URL, api_key=api_key, dataset=config.DATASET_NAME, entity_id=entity_id, bands=config.BANDS, bundle_id=config.BUNDLE_ID)
    for item in urls:
        (band, url) = (item['band'], item['url'])
        out = os.path.join(output_dir, f'{band}.TIF')
        if os.path.exists(out):
            continue
        print(f'  Downloading {band}.TIF...', end='  ')
        usgs_api.download_file(url, out)
    return entity_id
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    download_all_bands(args.output_dir)
