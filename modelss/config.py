"""
Training Configuration for LULC Segmentation Model.

Centralizes all hyperparameters, dataset paths, class definitions,
and training settings in one place.
"""

import os
import torch

# =============================================================================
# Device
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Dataset Paths
# =============================================================================
# Base directory of the landsat_download project
LANDSAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "landsat_download")

# Dataset file paths
WORK_DIR = os.path.dirname(LANDSAT_DIR)
_primary_dataset = os.path.join(LANDSAT_DIR, "big_data", "dataset", "torch_sliding_dataset.pt")
_fallback_dataset = os.path.join(WORK_DIR, "data", "torch_dataset", "torch_sliding_dataset.pt")
DATASET_PATH = _primary_dataset if os.path.exists(_primary_dataset) else _fallback_dataset

_primary_metadata = os.path.join(LANDSAT_DIR, "big_data", "dataset", "torch_sliding_dataset_metadata.pt")
_fallback_metadata = os.path.join(WORK_DIR, "data", "torch_dataset", "torch_sliding_dataset_metadata.pt")
METADATA_PATH = _primary_metadata if os.path.exists(_primary_metadata) else _fallback_metadata

# =============================================================================
# Band Configuration
# =============================================================================
# Input bands: B2 (Blue), B3 (Green), B4 (Red), B5 (NIR), B6 (SWIR-1), B7 (SWIR-2), NDVI
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "NDVI"]
NUM_INPUT_CHANNELS = len(BAND_NAMES)  # 7

# =============================================================================
# ESRI LULC Class Definitions
# =============================================================================
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

# The raw class IDs are not contiguous (0,1,2,4,5,7,8,9,10,11).
# We remap them to contiguous indices for the model.
RAW_CLASS_IDS = sorted(ESRI_CLASSES.keys())  # [0, 1, 2, 4, 5, 7, 8, 9, 10, 11]
NUM_CLASSES = len(RAW_CLASS_IDS)  # 10

# Mapping from raw class ID -> contiguous index [0..NUM_CLASSES-1]
CLASS_TO_INDEX = {raw_id: idx for idx, raw_id in enumerate(RAW_CLASS_IDS)}
# Mapping from contiguous index -> raw class ID
INDEX_TO_CLASS = {idx: raw_id for raw_id, idx in CLASS_TO_INDEX.items()}

# Human-readable class names (indexed by contiguous index)
CLASS_NAMES = [ESRI_CLASSES[raw_id][0] for raw_id in RAW_CLASS_IDS]
CLASS_COLORS = [ESRI_CLASSES[raw_id][1] for raw_id in RAW_CLASS_IDS]

# The "No Data" class index (in the contiguous scheme)
NODATA_INDEX = CLASS_TO_INDEX[0]

# =============================================================================
# Model Hyperparameters
# =============================================================================
# Patch size (spatial dimensions of input)
PATCH_SIZE = 32

# MobileNetV2-style expansion ratio for inverted residual blocks
EXPANSION_RATIO = 6

# Channel progression for the encoder backbone
ENCODER_CHANNELS = [32, 64, 128, 256]

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.8          # 80% train, 20% val
RANDOM_SEED = 42

# Learning rate scheduler
LR_SCHEDULER = "cosine"     # "cosine" or "step"
LR_STEP_SIZE = 30           # for StepLR
LR_GAMMA = 0.1              # for StepLR

# Early stopping
EARLY_STOPPING_PATIENCE = 15

# =============================================================================
# Output / Checkpointing
# =============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
