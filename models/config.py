
import os
import torch
DEVICE = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
LANDSAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'landsat_download')
WORK_DIR = os.path.dirname(LANDSAT_DIR)
TORCH_DATASET_DIR = os.path.join(WORK_DIR, 'data', 'Tensors')
TRAIN_DATASET_PATH = os.path.join(TORCH_DATASET_DIR, 'sliced_dataset_val.pt')
VAL_DATASET_PATH = os.path.join(TORCH_DATASET_DIR, 'sliced_dataset_val.pt')
BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI']
NUM_INPUT_CHANNELS = len(BAND_NAMES)
ESRI_CLASSES = {0: ('No Data', '#000000'), 1: ('Water', '#1A5BAB'), 2: ('Trees', '#358221'), 4: ('Flooded Vegetation', '#87D19E'), 5: ('Crops', '#FFDB5C'), 7: ('Built Area', '#ED022A'), 8: ('Bare Ground', '#EDE9E4'), 9: ('Snow/Ice', '#F2FAFF'), 10: ('Clouds', '#C8C8C8'), 11: ('Rangeland', '#C6D79B')}
RAW_CLASS_IDS = sorted(ESRI_CLASSES.keys())
NUM_CLASSES = len(RAW_CLASS_IDS)
CLASS_TO_INDEX = {raw_id: idx for (idx, raw_id) in enumerate(RAW_CLASS_IDS)}
INDEX_TO_CLASS = {idx: raw_id for (raw_id, idx) in CLASS_TO_INDEX.items()}
CLASS_NAMES = [ESRI_CLASSES[raw_id][0] for raw_id in RAW_CLASS_IDS]
CLASS_COLORS = [ESRI_CLASSES[raw_id][1] for raw_id in RAW_CLASS_IDS]
NODATA_INDEX = CLASS_TO_INDEX[0]
EXPANSION_RATIO = 6
ENCODER_CHANNELS = [32, 64, 128, 256]
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
RANDOM_SEED = 42
LR_SCHEDULER = 'cosine'
LR_STEP_SIZE = 30
LR_GAMMA = 0.1
EARLY_STOPPING_PATIENCE = 15
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
