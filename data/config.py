"""
Configuration for the Landsat Download Pipeline.
Edit the values below to change the study area, credentials, or output paths.
"""

import os

# =============================================================================
# USGS EarthExplorer Credentials
# =============================================================================
USGS_USERNAME = "Rishi_677"
USGS_TOKEN    = "5Fdntjd2uEgeVHXQ@nEv0M@za847T!11MhPKLKH80VDTd1sRMufUV1@s1_r6qS6w"

# =============================================================================
# Area of Interest (AOI) — WGS84 decimal degrees
# Used explicitly to *search* the USGS catalog for the overlapping Landsat scene. 
# Regardless of size, it will trigger a full ~185x180km scene download. 
# =============================================================================
TOP_LEFT     = (-19.772621, -53.030725)      # (Latitude, Longitude)
BOTTOM_RIGHT = (-24.264177, -48.186007)      # (Latitude, Longitude)

# Derived variables for backward compatibility with USGS/STAC search libraries
AOI = {
    "min_lat": min(TOP_LEFT[0], BOTTOM_RIGHT[0]),
    "max_lat": max(TOP_LEFT[0], BOTTOM_RIGHT[0]),
    "min_lon": min(TOP_LEFT[1], BOTTOM_RIGHT[1]),
    "max_lon": max(TOP_LEFT[1], BOTTOM_RIGHT[1]),
}

# Derived center + offset 
AOI_CENTER_LAT = (AOI["min_lat"] + AOI["max_lat"]) / 2
AOI_CENTER_LON = (AOI["min_lon"] + AOI["max_lon"]) / 2
AOI_BBOX = [AOI["min_lon"], AOI["min_lat"], AOI["max_lon"], AOI["max_lat"]]

# =============================================================================
# Landsat Scene Search Parameters
# =============================================================================
DATASET_NAME   = "landsat_ot_c2_l2"   # Landsat Collection 2, Level 2
MAX_CLOUD      = 20                    # Maximum cloud cover (%)
MAX_RESULTS    = 20                    # Number of candidate scenes to retrieve
SORT_FIELD     = "cloudCover"          # Sort by cloud cover
SORT_DIRECTION = "ASC"                 # Ascending = clearest first

# Bundle ID for "instantly available" downloads (no cold-storage wait)
BUNDLE_ID = "5e83d14fec7cae84"

# =============================================================================
# Bands to Download
# =============================================================================
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
#  B2 = Blue        (0.45–0.51 µm)
#  B3 = Green       (0.53–0.59 µm)
#  B4 = Red         (0.64–0.67 µm)
#  B5 = NIR         (0.85–0.88 µm)
#  B6 = SWIR-1      (1.57–1.65 µm)
#  B7 = SWIR-2      (2.11–2.29 µm)

# =============================================================================
# ESRI LULC Parameters
# =============================================================================
ESRI_COLLECTION = "io-lulc-9-class"
ESRI_DATETIME   = "2023-01-01/2023-12-31"  # Latest full-year classification

ESRI_CLASSES = {
    0:  ("No Data",            "#000000"),
    1:  ("Water",              "#1A5BAB"),
    2:  ("Trees",              "#358221"),
    4:  ("Flooded Vegetation", "#87D19E"),
    5:  ("Crops",              "#FFDB5C"),
    7:  ("Built Area",         "#ED022A"),
    8:  ("Bare Ground",        "#EDE9E4"),
    9:  ("Snow/Ice",           "#F2FAFF"),
    10: ("Clouds",             "#C8C8C8"),
    11: ("Rangeland",          "#C6D79B"),
}

# =============================================================================
# USGS M2M API
# =============================================================================
USGS_API_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"

# =============================================================================
# Output Paths
#
# data/             — Full Landsat scenes + raw ESRI at native 10m
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "Data_tif", "raw")

# Raw downloads
ESRI_10M_PATH = os.path.join(DATA_DIR, "esri_lulc_10m.tif")  # Native 10m

# Pixel-aligned outputs (all share the same grid natively out of DATA_DIR)
ESRI_PATH = os.path.join(DATA_DIR, "esri_lulc.tif")  # Resampled to 30m Landsat grid
NDVI_PATH = os.path.join(DATA_DIR, "ndvi.tif")

# CRS — auto-detect UTM zone from the AOI center longitude
_utm_zone = int((AOI_CENTER_LON + 180) / 6) + 1
_epsg_prefix = 326 if AOI_CENTER_LAT >= 0 else 327  # N vs S hemisphere
TARGET_CRS = f"EPSG:{_epsg_prefix}{_utm_zone:02d}"

def band_path(band_name: str) -> str:
    """Return the full path for a full-scene band (e.g. 'B2' → '.../data/B2.TIF')."""
    return os.path.join(DATA_DIR, f"{band_name}.TIF")
