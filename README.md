# 🌍 LULC Segmentation — Landsat 8 Deep Learning Pipeline

Semantic segmentation of **Land Use / Land Cover (LULC)** from multispectral Landsat 8 imagery using encoder-decoder CNNs (ResNet-18, EfficientNet-B0, MobileNetV2). The pipeline covers the full lifecycle: satellite data acquisition → spatial harmonization → patch generation → model training → hyperparameter optimization → evaluation & visualization.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Acquisition Pipeline](#data-acquisition-pipeline)
  - [1. Download Landsat Bands](#1-download-landsat-bands)
  - [2. Download ESRI LULC Labels](#2-download-esri-lulc-labels)
  - [3. Spatial Harmonization](#3-spatial-harmonization)
  - [4. Compute NDVI](#4-compute-ndvi)
  - [Running the Full Pipeline](#running-the-full-pipeline)
- [Tensor Preparation](#tensor-preparation)
  - [TIF → Tensor Conversion](#tif--tensor-conversion)
  - [Sliding Window Dataset](#sliding-window-dataset)
  - [Train / Validation Split](#train--validation-split)
- [Model Training](#model-training)
  - [Baseline CNN Training](#baseline-cnn-training)
  - [Architecture Ablation](#architecture-ablation)
  - [Hyperparameter Optimization (Optuna)](#hyperparameter-optimization-optuna)
- [Evaluation & Visualization](#evaluation--visualization)
- [Configuration Reference](#configuration-reference)
- [Spectral Band Reference](#spectral-band-reference)
- [LULC Class Definitions](#lulc-class-definitions)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project performs pixel-level land cover classification on a region covering parts of São Paulo State, Brazil, using 30 m Landsat 8 multispectral imagery and 10 m ESRI LULC labels. The workflow includes:

1. **Data Acquisition** — Automated download of Landsat 8 bands (B2–B7) via the USGS M2M API and ESRI LULC maps via Planetary Computer.
2. **Spatial Harmonization** — Nearest-neighbour resampling of 10 m ESRI labels onto the 30 m Landsat pixel grid.
3. **Feature Engineering** — NDVI computation from Red (B4) and NIR (B5) bands; concatenation into a 7-channel input tensor.
4. **Patch Generation** — Sliding window extraction of 128×128 patches with per-channel 2nd–98th percentile min-max normalization.
5. **Model Training** — MobileNetV2-style baseline CNN and UNet encoder-decoder architectures with inverse-root-frequency class weighting, data augmentation, AdamW optimizer, and cosine annealing.
6. **Ablation & HPO** — Architecture comparison across three backbones and Bayesian hyperparameter search via Optuna.

---

## Repository Structure

```
dat_submission/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                              # Data acquisition & preprocessing
│   ├── config.py                      # AOI coordinates, USGS credentials, output paths
│   ├── commands.sh                    # Quick-reference shell commands
│   │
│   ├── data_pipeline/                 # Stage 1: Raw data download & alignment
│   │   ├── run_pipeline.py            # Orchestrates all 4 download steps
│   │   ├── download_bands.py          # USGS M2M API → Landsat B2–B7 GeoTIFFs
│   │   ├── download_esri.py           # Planetary Computer → ESRI LULC 10 m
│   │   ├── align_data.py              # Nearest-neighbour resample to 30 m grid
│   │   ├── compute_ndvi.py            # (NIR − Red) / (NIR + Red) → ndvi.tif
│   │   ├── clip_to_square.py          # Optional AOI clipping utility
│   │   └── usgs_api.py                # USGS M2M REST communication
│   │
│   ├── tensor_and_vis/                # Stage 2: Tensor conversion & visualization
│   │   ├── converting_tif_to_tensor.py  # TIF stack → PyTorch .pt dictionary
│   │   ├── prepare_dataset.py         # Sliding-window patch extraction + normalization
│   │   ├── split_master_tensor.py     # Spatial train/val split (80/20 along width)
│   │   ├── compute_global_stats.py    # Per-channel statistics
│   │   ├── analyze_tif.py             # GeoTIFF metadata inspector
│   │   ├── view_data.py               # Band preview & true-color composites
│   │   ├── view_dataset_samples.py    # Patch grid visualizer
│   │   └── plot_rgb_split.py          # Train/val boundary overlay
│   │
│   ├── Data_tif/                      # (git-ignored) Raw GeoTIFF storage
│   │   ├── raw/                       #   Full-scene Landsat + aligned ESRI + NDVI
│   │   └── clipped/                   #   AOI-clipped variants
│   │
│   └── torch_dataset/                 # (git-ignored) Exported PyTorch tensors
│       ├── dataset_tensors.pt         #   Master [C, H, W] tensor dictionary
│       ├── train_dataset_sliding.pt   #   Training patches
│       └── val_dataset_sliding.pt     #   Validation patches
│
└── models/                            # Model training & evaluation
    ├── config.py                      # Training hyperparameters & class definitions
    ├── model.py                       # MobileNetV2-style baseline architecture
    ├── dataset.py                     # Dataset loader with augmentation & label remapping
    ├── metrics.py                     # OA, mIoU, F1, confusion matrix
    ├── train.py                       # Main training loop with early stopping
    ├── eval_vis.py                    # Evaluation + side-by-side prediction visualizations
    ├── visualize_dataset.py           # Sanity-check dataset viewer
    ├── visualize_rgb.py               # RGB + mask overlay viewer
    │
    ├── ablations/                     # Architecture comparison & HPO
    │   ├── architectures.py           # Self-contained ResNet18, EfficientNet-B0,
    │   │                              #   MobileNetV2 UNet encoder-decoder implementations
    │   ├── archi_ablation.py          # 3-backbone ablation runner (100 epochs each)
    │   ├── hpo_efficientnet.py        # Optuna HPO for EfficientNet-B0
    │   ├── plot_comparisons.py        # Ablation curve plotter
    │   └── plot_resnet18_results.py   # ResNet-18 prediction grid generator
    │
    └── outputs/                       # (git-ignored) Checkpoints & training logs
        ├── checkpoints/
        └── logs/
```

> **Note:** All `.tif`, `.pt`, `.pth` files and `__pycache__/` directories are git-ignored. You must regenerate the data and train models from scratch using the instructions below.

---

## Requirements

- **Python** ≥ 3.9
- **CUDA** GPU (strongly recommended for training; ≥ 8 GB VRAM)
- A free [USGS EarthExplorer account](https://ers.cr.usgs.gov/register) for Landsat downloads

### Python Dependencies

| Package | Purpose |
|---------|---------|
| `torch` ≥ 2.0 | Model training & tensor operations |
| `rasterio` ≥ 1.3 | GeoTIFF I/O |
| `pyproj` ≥ 3.4 | Coordinate system transformations |
| `numpy` ≥ 1.24 | Array computation |
| `matplotlib` ≥ 3.7 | Visualization |
| `pystac-client` ≥ 0.7 | STAC API access (ESRI LULC) |
| `planetary-computer` ≥ 1.0 | Microsoft Planetary Computer auth |
| `rioxarray` ≥ 0.15 | Raster xarray integration |
| `requests` ≥ 2.28 | HTTP requests (USGS M2M API) |
| `optuna` | Hyperparameter optimization (ablation only) |
| `torchvision` | MobileNetV2 base model + transforms |
| `tqdm` | Progress bars |

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd dat_submission

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install additional packages for ablation / HPO (not in requirements.txt)
pip install optuna torchvision tqdm
```

---

## Data Acquisition Pipeline

All data acquisition scripts live in `data/data_pipeline/`. Before running, configure your credentials and study area.

### Configuration

Edit `data/config.py`:

```python
# USGS EarthExplorer credentials
USGS_USERNAME = "your_username"
USGS_TOKEN    = "your_api_token"

# Area of Interest — WGS84 decimal degrees
TOP_LEFT     = (45.243651, -74.900375)    # (Latitude, Longitude)
BOTTOM_RIGHT = (43.623781, -72.636509)    # (Latitude, Longitude)
```

> **Important:** The USGS API downloads entire WRS-2 scenes (~185×180 km) regardless of AOI size. The coordinates are only used to locate the correct scene.

### Running the Full Pipeline

```bash
cd data
python data_pipeline/run_pipeline.py --output_dir Data_tif/raw
```

This runs all 4 steps sequentially (~5 min depending on bandwidth):

| Step | Script | Output |
|------|--------|--------|
| 1 | `download_bands.py` | `B2.TIF` … `B7.TIF` (Landsat bands) |
| 2 | `download_esri.py` | `esri_lulc_10m.tif` (10 m ESRI labels) |
| 3 | `align_data.py` | `esri_lulc.tif` (resampled to 30 m grid) |
| 4 | `compute_ndvi.py` | `ndvi.tif` (vegetation index) |

### Running Steps Individually

```bash
cd data/data_pipeline

# Step 1: Download Landsat bands B2–B7
python download_bands.py

# Step 2: Download ESRI LULC at native 10 m resolution
python download_esri.py

# Step 3: Align ESRI labels to the 30 m Landsat pixel grid
python align_data.py

# Step 4: Compute NDVI from B4 (Red) and B5 (NIR)
python compute_ndvi.py
```

---

## Tensor Preparation

After the data pipeline completes, convert the raw GeoTIFFs into training-ready PyTorch tensors.

### TIF → Tensor Conversion

```bash
cd data
python tensor_and_vis/converting_tif_to_tensor.py \
    --input_dir Data_tif/raw \
    --output_file torch_dataset/dataset_tensors.pt
```

**Output:** A `.pt` dictionary with three keys:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `bands` | `[6, H, W]` | uint16 | B2, B3, B4, B5, B6, B7 |
| `ndvi` | `[1, H, W]` | float32 | Vegetation index (−1 to 1) |
| `esri` | `[1, H, W]` | uint8 | LULC class IDs |

### Sliding Window Dataset

Extract 128×128 patches with per-channel 2nd–98th percentile normalization:

```bash
python tensor_and_vis/prepare_dataset.py \
    --tensor_path torch_dataset/dataset_tensors.pt \
    --window_size 128 \
    --stride 128 \
    --min_valid 0.5 \
    --save_path torch_dataset/torch_sliding_dataset.pt
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--tensor_path` | `torch_dataset/dataset_tensors.pt` | Input tensor dictionary |
| `--window_size` | `128` | Patch spatial size (pixels) |
| `--stride` | `128` | Sliding step; set < window_size for overlap |
| `--min_valid` | `0.5` | Minimum fraction of non-padding pixels to keep a patch |
| `--save_path` | `torch_dataset/torch_sliding_dataset.pt` | Output TensorDataset |

**Output tensor shapes:**
```
X: [N, 7, 128, 128]   float32   (B2, B3, B4, B5, B6, B7, NDVI)
y: [N, 1, 128, 128]   int64     (ESRI LULC class IDs)
```

### Train / Validation Split

A spatial split (80/20 along the image width) is performed to prevent spatial data leakage:

```bash
python tensor_and_vis/split_master_tensor.py
```

This produces:
- `torch_dataset/train_dataset_sliding.pt`
- `torch_dataset/val_dataset_sliding.pt`

---

## Model Training

All training scripts live in `models/`. Before training, ensure the dataset `.pt` files are in place and update the paths in `models/config.py` if needed:

```python
# models/config.py — update these to point to your tensor files
TRAIN_DATASET_PATH = "/path/to/train_dataset_sliding.pt"
VAL_DATASET_PATH   = "/path/to/val_dataset_sliding.pt"
```

### Baseline CNN Training

Train the MobileNetV2-style encoder-decoder baseline:

```bash
cd models
python train.py
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--weight-decay` | `1e-4` | AdamW L2 regularization |
| `--patience` | `15` | Early stopping patience (epochs without mIoU improvement) |
| `--resume` | `None` | Path to checkpoint `.pth` to resume training from |

**Training details:**
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** Cosine annealing (η_min=1e-6)
- **Loss:** Cross-entropy with inverse-root-frequency class weights
- **Augmentation:** Random horizontal and vertical flips (p=0.5 each)
- **Early stopping:** Monitored on validation mIoU (patience=15)
- **Checkpointing:** Best model saved to `outputs/checkpoints/best_model.pth`

### Architecture Ablation

Compare three UNet encoder-decoder backbones (ResNet-18, EfficientNet-B0, MobileNetV2), each trained for 100 epochs with Focal Loss:

```bash
cd models/ablations
python archi_ablation.py --epochs 100
```

Each architecture is trained sequentially with architecture-specific batch sizes to maximize GPU utilization. Results are saved to `ablations/logs/<arch>_history.json` and best checkpoints to `ablations/checkpoints/<arch>_best.pth`.

### Hyperparameter Optimization (Optuna)

Bayesian HPO for EfficientNet-B0 using Tree-structured Parzen Estimator:

```bash
cd models/ablations
python hpo_efficientnet.py --epochs 30 --n_trials 20
```

**Search space:**

| Hyperparameter | Range | Scale |
|----------------|-------|-------|
| Learning rate | [1e-5, 1e-2] | Log-uniform |
| Weight decay | [1e-6, 1e-2] | Log-uniform |
| Batch size | {64, 128, 192, 224} | Categorical |
| Dropout rate | [0.0, 0.5] | Uniform |

Median pruning is applied after a 5-epoch warmup. Best parameters are saved to `ablations/logs/best_hpo_params.json`.

---

## Evaluation & Visualization

### Full Evaluation with Prediction Maps

```bash
cd models
python eval_vis.py
```

This loads the best checkpoint, evaluates on the full validation set, prints per-class IoU/F1/OA metrics, and saves a side-by-side visualization (`outputs/validation_vis.png`) showing:

| Panel 1 | Panel 2 | Panel 3 |
|---------|---------|---------|
| Input RGB (B4, B3, B2) | Ground Truth | Model Prediction |

### Dataset Sanity Checks

```bash
# Visualize RGB + LULC ground truth side-by-side
cd models
python visualize_dataset.py

# Visualize RGB with mask overlay
python visualize_rgb.py
```

### Ablation Curve Plots

```bash
cd models/ablations
python plot_comparisons.py            # All 3 architectures on one plot
python plot_resnet18_results.py       # ResNet-18 prediction grid
```

---

## Configuration Reference

### `data/config.py` — Data Pipeline

| Variable | Description |
|----------|-------------|
| `USGS_USERNAME` / `USGS_TOKEN` | USGS EarthExplorer API credentials |
| `TOP_LEFT` / `BOTTOM_RIGHT` | AOI bounding box (lat, lon) |
| `BANDS` | Landsat bands to download (`["B2"..."B7"]`) |
| `MAX_CLOUD` | Maximum cloud cover percentage for scene search |
| `ESRI_COLLECTION` | STAC collection ID for ESRI LULC |
| `TARGET_CRS` | Auto-detected UTM EPSG code from AOI center |

### `models/config.py` — Training

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 100 | Maximum training epochs |
| `LEARNING_RATE` | 1e-3 | Initial learning rate |
| `WEIGHT_DECAY` | 1e-4 | AdamW regularization |
| `EARLY_STOPPING_PATIENCE` | 15 | Epochs without improvement before stopping |
| `NUM_INPUT_CHANNELS` | 7 | 6 spectral bands + NDVI |
| `NUM_CLASSES` | 10 | ESRI LULC class count |
| `ENCODER_CHANNELS` | [32, 64, 128, 256] | Encoder feature widths |
| `EXPANSION_RATIO` | 6 | MobileNetV2 inverted residual expansion |

---

## Spectral Band Reference

| Index | Band | Wavelength (µm) | Description |
|-------|------|------------------|-------------|
| 0 | B2 — Blue | 0.452 – 0.512 | Deep water, sediment, urban surfaces |
| 1 | B3 — Green | 0.533 – 0.590 | Peak vegetation reflectance |
| 2 | B4 — Red | 0.636 – 0.673 | Chlorophyll absorption (NDVI denominator) |
| 3 | B5 — NIR | 0.851 – 0.879 | Healthy vegetation reflectance (NDVI numerator) |
| 4 | B6 — SWIR-1 | 1.566 – 1.651 | Soil/vegetation moisture content |
| 5 | B7 — SWIR-2 | 2.107 – 2.294 | Dry vegetation, bare rock, geology |
| 6 | NDVI | Derived | (B5 − B4) / (B5 + B4), range [−1, 1] |

---

## LULC Class Definitions

Classes follow the [ESRI 10 m Land Use/Land Cover](https://www.arcgis.com/home/item.html?id=d6642f59a9cc4c6c86b8f4c184d2e8d0) schema.

| ID | Class | Color | Description |
|----|-------|-------|-------------|
| 0 | No Data | ⬛ `#000000` | Invalid / masked pixels (ignored in training) |
| 1 | Water | 🟦 `#1A5BAB` | Open water bodies |
| 2 | Trees | 🟩 `#358221` | Tree canopy cover |
| 4 | Flooded Vegetation | 🟢 `#87D19E` | Wetland / flooded plant cover |
| 5 | Crops | 🟨 `#FFDB5C` | Cultivated agricultural land |
| 7 | Built Area | 🟥 `#ED022A` | Urban / built-up surfaces |
| 8 | Bare Ground | ⬜ `#EDE9E4` | Exposed soil, sand, rock |
| 9 | Snow/Ice | 🤍 `#F2FAFF` | Snow and ice cover |
| 10 | Clouds | ☁️ `#C8C8C8` | Cloud-contaminated pixels |
| 11 | Rangeland | 🟫 `#C6D79B` | Grass, shrub, low vegetation |

> Non-contiguous raw IDs (0, 1, 2, 4, 5, 7, 8, 9, 10, 11) are remapped to contiguous indices [0…9] during training.

---

## Results

Best validation mIoU across architectures (100-epoch ablation):

| Model | mIoU |
|-------|------|
| **EfficientNet-B0** | **0.8624** |
| ResNet-18 | 0.8424 |
| MobileNet-V2 | 0.8370 |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{chauhan2026lulc,
  title     = {LULC Segmentation},
  author    = {Chauhan, Aditya and Tomar, Akshat and Srivastava, Arman and Rao, Kartikeya and Sharma, Yug},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  address   = {Seoul, South Korea}
}
```

---

## License

This project is for academic and research purposes.
