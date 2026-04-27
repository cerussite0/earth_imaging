
import os
USGS_USERNAME = 'Rishi_677'
USGS_TOKEN = '5Fdntjd2uEgeVHXQ@nEv0M@za847T!11MhPKLKH80VDTd1sRMufUV1@s1_r6qS6w'
TOP_LEFT = (45.243651, (- 74.900375))
BOTTOM_RIGHT = (43.623781, (- 72.636509))
AOI = {'min_lat': min(TOP_LEFT[0], BOTTOM_RIGHT[0]), 'max_lat': max(TOP_LEFT[0], BOTTOM_RIGHT[0]), 'min_lon': min(TOP_LEFT[1], BOTTOM_RIGHT[1]), 'max_lon': max(TOP_LEFT[1], BOTTOM_RIGHT[1])}
AOI_CENTER_LAT = ((AOI['min_lat'] + AOI['max_lat']) / 2)
AOI_CENTER_LON = ((AOI['min_lon'] + AOI['max_lon']) / 2)
AOI_BBOX = [AOI['min_lon'], AOI['min_lat'], AOI['max_lon'], AOI['max_lat']]
DATASET_NAME = 'landsat_ot_c2_l2'
MAX_CLOUD = 20
MAX_RESULTS = 20
SORT_FIELD = 'cloudCover'
SORT_DIRECTION = 'ASC'
BUNDLE_ID = '5e83d14fec7cae84'
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
ESRI_COLLECTION = 'io-lulc-9-class'
ESRI_DATETIME = '2023-01-01/2023-12-31'
ESRI_CLASSES = {0: ('No Data', '#000000'), 1: ('Water', '#1A5BAB'), 2: ('Trees', '#358221'), 4: ('Flooded Vegetation', '#87D19E'), 5: ('Crops', '#FFDB5C'), 7: ('Built Area', '#ED022A'), 8: ('Bare Ground', '#EDE9E4'), 9: ('Snow/Ice', '#F2FAFF'), 10: ('Clouds', '#C8C8C8'), 11: ('Rangeland', '#C6D79B')}
USGS_API_URL = 'https://m2m.cr.usgs.gov/api/api/json/stable/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data_tif/data', 'raw')
ESRI_10M_PATH = os.path.join(DATA_DIR, 'esri_lulc_10m.tif')
ESRI_PATH = os.path.join(DATA_DIR, 'esri_lulc.tif')
NDVI_PATH = os.path.join(DATA_DIR, 'ndvi.tif')
_utm_zone = (int(((AOI_CENTER_LON + 180) / 6)) + 1)
_epsg_prefix = (326 if (AOI_CENTER_LAT >= 0) else 327)
TARGET_CRS = f'EPSG:{_epsg_prefix}{_utm_zone:02d}'

def band_path(band_name):
    return os.path.join(DATA_DIR, f'{band_name}.TIF')

def clipped_band_path(band_name):
    return os.path.join(BASE_DIR, 'Data_tif/test', 'clipped', f'{band_name}.TIF')
