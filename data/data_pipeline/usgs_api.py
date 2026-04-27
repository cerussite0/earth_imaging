
import os
import sys
import requests

def login(api_url, username, token):
    resp = requests.post((api_url + 'login-token'), json={'username': username, 'token': token}, headers={'Content-Type': 'application/json'}, timeout=30)
    data = resp.json()
    return data['data']

def search_scenes(api_url, api_key, dataset, aoi, max_cloud, max_results, sort_field='cloudCover', sort_direction='ASC'):
    payload = {'datasetName': dataset, 'maxResults': max_results, 'startingNumber': 1, 'metadataType': 'full', 'sortField': sort_field, 'sortDirection': sort_direction, 'sceneFilter': {'spatialFilter': {'filterType': 'mbr', 'lowerLeft': {'latitude': aoi['min_lat'], 'longitude': aoi['min_lon']}, 'upperRight': {'latitude': aoi['max_lat'], 'longitude': aoi['max_lon']}}, 'cloudCoverFilter': {'min': 0, 'max': max_cloud, 'includeUnknown': True}}}
    resp = requests.post((api_url + 'scene-search'), headers={'X-Auth-Token': api_key, 'Content-Type': 'application/json'}, json=payload, timeout=60)
    data = resp.json()
    results = data['data']['results']
    print(f"Found {data['data']['totalHits']} scenes, showing top {len(results)}")
    return results

def get_download_urls(api_url, api_key, dataset, entity_id, bands, bundle_id):
    headers = {'X-Auth-Token': api_key, 'Content-Type': 'application/json'}
    resp = requests.post((api_url + 'download-options'), headers=headers, json={'datasetName': dataset, 'entityIds': [entity_id]}, timeout=60)
    options = resp.json()['data']
    resp.close()
    files = []
    for item in options:
        if (item['id'] != bundle_id):
            continue
        for f in item.get('secondaryDownloads', []):
            if (not f.get('available')):
                continue
            if any(((b in f.get('displayId', '')) for b in bands)):
                files.append({'entityId': f['entityId'], 'productId': f['id']})
    resp = requests.post((api_url + 'download-request'), headers=headers, json={'downloads': files, 'label': 'landsat_download'}, timeout=60)
    dl_data = resp.json()
    resp.close()
    result = []
    for dl in dl_data['data']['availableDownloads']:
        url = dl['url']
        for b in bands:
            if (f'{b}.TIF' in url):
                result.append({'band': b, 'url': url})
                break
    return result

def download_file(url, output_path, chunk_size=8192):
    r = requests.get(url, stream=True, timeout=120)
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    size_mb = ((os.path.getsize(output_path) / (1024 * 1024)) if os.path.exists(output_path) else 0)
    print(f'({size_mb:.1f} MB)')
    r.close()
