# Landsat Download & ML Processing Pipeline

This repository automates the extraction, alignment, and packaging of Landsat 8/9 Level-2 imagery, corresponding ESRI Land Use/Land Cover datasets, and derived vegetation indices (NDVI) into ML-ready PyTorch tensors.

## Repository Structure

The project was refactored to explicitly separate standard geospatial data operations from machine learning workflows.

```text
landsat_download/
├── config.py                     # Central configuration (AOI, paths, credentials)
├── README.md                     # Documentation
├── requirements.txt              # PyPI dependencies
│
├── data_pipeline/                # Data Acquisition & Alignment 
│   ├── run_pipeline.py           # Master orchestration script
│   ├── download_bands.py         # USGS M2M API downloader
│   ├── clip_bands.py             # Geospatial AOI clipping
│   ├── download_esri.py          # ESRI LULC 10m tile STAC fetcher
│   ├── align_data.py             # 30m nearest-neighbor resampling & reprojection
│   ├── compute_ndvi.py           # (NIR - Red) / (NIR + Red) generation
│   └── usgs_api.py               # USGS M2M communication library
│
├── tensor_and_vis/               # Pytorch Ops & Visualizations
│   ├── converting_tif_to_tensor.py # Tensor compilation 
│   ├── analyze_tif.py            # Local tif metadata extraction
│   ├── prepare_dataset.py        # ML Dataset compilation logics
│   ├── view_dataset_samples.py   # Tensor sampling scripts
│   ├── view_data.py              # Visualizations
│   └── trial.ipynb               # Prototyping notebook
│
├── Data_tif/                     # Master TIF storage
│   ├── raw/                      # Unclipped, full-scene downloaded variables
│   └── clipped/                  # Perfectly grid-aligned, ready-to-process maps
│
└── torch_dataset/                # PyTorch exported datasets
    └── dataset_tensors.pt        # Resulting [C, H, W] tensor dictionaries
```

## Setup

It is highly recommended to explicitly activate your virtual environment:
```bash
# E.g.
source ../.venv/bin/activate
pip install -r requirements.txt
```

## Quick Start: Full Pipeline

To execute the entire sequential pipeline (downloading through tensor packaging), ensure you have set up your USGS credentials in `config.py`, then run:

```bash
python -m data_pipeline.run_pipeline
```

*(You can also optionally run individual scripts like `python data_pipeline/clip_bands.py` in isolation).*

## Pipeline Steps Explained

### 1. Configuration (`config.py`)
Centralized variables handling Authentication, AOI boundaries (Latitude/Longitude bounding boxes), expected Cloud Cover thresholds, and specific Landsat Bands (e.g. B2-B7).

### Phase 1: Pipeline (`data_pipeline/`)
- **`download_bands.py`**: Interacts with USGS M2M API to pull Level-2 surface reflectance bulk imagery into `Data_tif/raw/`.
- **`clip_bands.py`**: Uses `rasterio` to slice gigantic bounding boxes exactly down to the predefined AOI, exporting them to `Data_tif/clipped/`.
- **`download_esri.py`**: Communicates with the Microsoft Planetary Computer STAC catalog, pulling 10-meter ESRI LULC masks for the designated year.
- **`align_data.py`**: Enforces strict spatial integrity. It evaluates the Landsat transformations and resamples external layers (ESRI) onto the exact same 30m grid.
- **`compute_ndvi.py`**: Extracts the Red and NIR components and mathematically derives the vegetation index bounding map.

### Phase 2: ML Conversion (`tensor_and_vis/`)
- **`converting_tif_to_tensor.py`**: Rapidly crawls through `Data_tif/clipped/` pulling 6 discrete band maps, 1 NDVI map, and 1 ESRI categorical map. It perfectly stacks these into multi-dimensional arrays, wrapping them as PyTorch sensors configured natively as `[C, H, W]`. Output is saved entirely to `torch_dataset/dataset_tensors.pt`.
```bash
python tensor_and_vis/converting_tif_to_tensor.py
```
