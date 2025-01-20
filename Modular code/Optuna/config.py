import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data_all")
TRAIN_PATH_HALF = os.path.join(PROJECT_ROOT, "data_labels/new_labels/splits/train_half.csv")
TRAIN_PATH_QUARTER = os.path.join(PROJECT_ROOT, "data_labels/new_labels/splits/train_quarter.csv")
TRAIN_PATH_TEN = os.path.join(PROJECT_ROOT, "data_labels/new_labels/splits/train_ten.csv")
TRAIN_PATH_FULL = os.path.join(PROJECT_ROOT, "data_labels/new_labels/train.csv")
VAL_PATH = os.path.join(PROJECT_ROOT, "data_labels/new_labels/validation.csv")
LOCAL_DINO_MODEL_PATH = '/home/hpc/iwfa/iwfa054h/.cache/torch/hub/facebookresearch_dinov2_main'
