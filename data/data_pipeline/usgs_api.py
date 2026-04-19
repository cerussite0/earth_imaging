"""
USGS M2M API helpers.
Handles authentication, scene search, download-option resolution, and file download.
"""

import json
import sys
import requests


def login(api_url: str, username: str, token: str) -> str:
    """
    Authenticate with the USGS M2M API using a login-token.

    Returns
    -------
    str
        The session API key used for subsequent requests.
    """
    url = api_url + "login-token"
    payload = {"username": username, "token": token}

    print("Logging in to USGS M2M API...")
    response = requests.post(
        url, json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    try:
        if response.status_code != 200:
            print(f"HTTP Error {response.status_code}")
            sys.exit(1)

        output = response.json()
        if output.get("errorCode") is not None:
            print(f"API Error: {output['errorCode']} — {output.get('errorMessage', '')}")
            sys.exit(1)

        api_key = output["data"]
        print(f"  ✓ Authenticated (key: {api_key[:12]}...)")
        return api_key

    finally:
        response.close()


def search_scenes(api_url: str, api_key: str, dataset: str,
                  aoi: dict, max_cloud: int, max_results: int,
                  sort_field: str = "cloudCover",
                  sort_direction: str = "ASC") -> list:
    """
    Search for scenes matching the AOI bounding box and cloud-cover filter.
    No date restriction — returns the all-time clearest scenes.

    Parameters
    ----------
    aoi : dict
        Must have keys: min_lat, min_lon, max_lat, max_lon.

    Returns
    -------
    list[dict]
        Scene result dicts (entityId, cloudCover, publishDate, …).
    """
    url = api_url + "scene-search"
    headers = {"X-Auth-Token": api_key, "Content-Type": "application/json"}

    payload = {
        "datasetName": dataset,
        "maxResults": max_results,
        "startingNumber": 1,
        "metadataType": "full",
        "sortField": sort_field,
        "sortDirection": sort_direction,
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {
                    "latitude":  aoi["min_lat"],
                    "longitude": aoi["min_lon"],
                },
                "upperRight": {
                    "latitude":  aoi["max_lat"],
                    "longitude": aoi["max_lon"],
                },
            },
            "cloudCoverFilter": {
                "min": 0,
                "max": max_cloud,
                "includeUnknown": True,
            },
        },
    }

    print(f"Searching for scenes in '{dataset}' (cloud ≤ {max_cloud}%)...")
    response = requests.post(url, headers=headers, json=payload, timeout=60)

    try:
        if response.status_code != 200:
            print(f"HTTP Error {response.status_code}")
            sys.exit(1)

        output = response.json()
        if output.get("errorCode") is not None:
            print(f"API Error: {output['errorCode']} — {output.get('errorMessage', '')}")
            sys.exit(1)

        total_hits = output["data"]["totalHits"]
        results    = output["data"]["results"]

        print(f"  ✓ {total_hits} total scenes found, showing top {len(results)}:\n")
        for i, scene in enumerate(results, 1):
            eid   = scene.get("entityId")
            cc    = scene.get("cloudCover")
            date  = scene.get("publishDate")
            print(f"  {i:02d}. {eid}  |  Cloud: {cc}%  |  Date: {date}")

        return results

    finally:
        response.close()


def get_download_urls(api_url: str, api_key: str, dataset: str,
                      entity_id: str, bands: list,
                      bundle_id: str) -> list:
    """
    Resolve direct download URLs for the specified bands of a scene.

    Returns
    -------
    list[dict]
        Each dict has keys: band (str), url (str).
    """
    headers = {"X-Auth-Token": api_key, "Content-Type": "application/json"}

    # Step 1 — get download options
    print(f"\nFetching download options for {entity_id}...")
    options_payload = {"datasetName": dataset, "entityIds": [entity_id]}
    resp = requests.post(
        api_url + "download-options",
        headers=headers, json=options_payload, timeout=60,
    )
    options_data = resp.json()["data"]
    resp.close()

    # Step 2 — collect product IDs for the requested bands
    files_to_download = []
    for item in options_data:
        if item["id"] != bundle_id:
            continue
        for f in item.get("secondaryDownloads", []):
            if not f.get("available"):
                continue
            display_id = f.get("displayId", "")
            if any(b in display_id for b in bands):
                files_to_download.append({
                    "entityId":  f["entityId"],
                    "productId": f["id"],
                })

    if not files_to_download:
        print("  ✗ No instantly-available files found (may be in cold storage).")
        sys.exit(1)

    print(f"  ✓ {len(files_to_download)} band files matched")

    # Step 3 — request download URLs
    print("Requesting direct download URLs...")
    dl_payload = {
        "downloads": files_to_download,
        "label": "landsat_pipeline_download",
    }
    resp = requests.post(
        api_url + "download-request",
        headers=headers, json=dl_payload, timeout=60,
    )
    dl_data = resp.json()
    resp.close()

    # Step 4 — map each URL back to its band name
    result = []
    for dl in dl_data["data"]["availableDownloads"]:
        url = dl["url"]
        matched_band = None
        for b in bands:
            if f"{b}.TIF" in url:
                matched_band = b
                break
        if matched_band:
            result.append({"band": matched_band, "url": url})

    print(f"  ✓ {len(result)} download URLs resolved")
    return result


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> None:
    """Stream-download a file to disk."""
    r = requests.get(url, stream=True, timeout=120)
    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)

    size_mb = downloaded / (1024 * 1024)
    print(f"    Saved ({size_mb:.1f} MB)")
    r.close()
