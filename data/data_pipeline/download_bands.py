"""
Download Landsat spectral bands (B2–B7) for the configured AOI.
Uses the USGS M2M API via the helpers in usgs_api.py.
"""

import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import usgs_api


def download_all_bands() -> str:
    """
    Authenticate, search for the clearest scene, and download bands B2–B7.

    Returns
    -------
    str
        The entity ID of the selected scene.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # 1. Login
    api_key = usgs_api.login(
        config.USGS_API_URL, config.USGS_USERNAME, config.USGS_TOKEN,
    )

    # 2. Search — no date filter, lowest cloud cover first
    scenes = usgs_api.search_scenes(
        api_url=config.USGS_API_URL,
        api_key=api_key,
        dataset=config.DATASET_NAME,
        aoi=config.AOI,
        max_cloud=config.MAX_CLOUD,
        max_results=config.MAX_RESULTS,
        sort_field=config.SORT_FIELD,
        sort_direction=config.SORT_DIRECTION,
    )

    if not scenes:
        raise RuntimeError("No scenes found for the given AOI and cloud filter.")

    # Select the clearest scene (first result, already sorted ASC by cloud cover)
    best_scene = scenes[0]
    entity_id  = best_scene["entityId"]
    print(f"\n→ Selected scene: {entity_id} "
          f"(cloud: {best_scene.get('cloudCover')}%)\n")

    # 3. Resolve download URLs
    urls = usgs_api.get_download_urls(
        api_url=config.USGS_API_URL,
        api_key=api_key,
        dataset=config.DATASET_NAME,
        entity_id=entity_id,
        bands=config.BANDS,
        bundle_id=config.BUNDLE_ID,
    )

    # 4. Download each band
    print(f"\nDownloading {len(urls)} bands to {config.DATA_DIR}/\n")
    for item in urls:
        band = item["band"]
        url  = item["url"]
        out  = config.band_path(band)

        if os.path.exists(out):
            print(f"  {band}.TIF — already exists, skipping")
            continue

        print(f"  {band}.TIF — downloading...", end="  ")
        usgs_api.download_file(url, out)

    print(f"\n✓ All bands downloaded for scene {entity_id}")
    return entity_id


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_all_bands()
