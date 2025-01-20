# config.py
import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data_all")
TRAIN_PATH_HALF = os.path.join(PROJECT_ROOT, "data_labels/new_labels/splits/train_half.csv")
TRAIN_PATH_QUARTER = os.path.join(PROJECT_ROOT, "data_labels/new_labels/splits/train_quarter.csv")
TRAIN_PATH_TEN = os.path.join(PROJECT_ROOT, "data_labels/new_labels/splits/train_ten.csv")
TRAIN_PATH_FULL = os.path.join(PROJECT_ROOT, "data_labels/new_labels/train.csv")

IMAGE_DIR_10_25_SD = os.path.join(PROJECT_ROOT, "Augmented_data/25%_data")
IMAGE_DIR_10_50_SD = os.path.join(PROJECT_ROOT, "Augmented_data/50%_data_img2img")
IMAGE_DIR_25_50_SD = os.path.join(PROJECT_ROOT, "Augmented_data/25-50_SD_img2img")
IMAGE_DIR_50_100_SD = os.path.join(PROJECT_ROOT, "Augmented_data/50-100_SD_img2img")
IMAGE_DIR_100_200_SD = os.path.join(PROJECT_ROOT, "Augmented_data/100-200_SD_img2img")

TRAIN_PATH_10_25_SD = os.path.join(PROJECT_ROOT, "data_labels/augmentation_img2img/10_25_augmented.csv")
TRAIN_PATH_10_50_SD = os.path.join(PROJECT_ROOT, "data_labels/augmentation_img2img/10_50_augmented.csv")
TRAIN_PATH_25_50_SD = os.path.join(PROJECT_ROOT, "data_labels/augmentation_img2img/25_50_augmented.csv")
TRAIN_PATH_50_100_SD = os.path.join(PROJECT_ROOT, "data_labels/augmentation_img2img/50_100_augmented.csv")
TRAIN_PATH_100_200_SD = os.path.join(PROJECT_ROOT, "data_labels/augmentation_img2img/100_200_augmented.csv")

IMAGE_DIR_10_25_Dreambooth = os.path.join(PROJECT_ROOT, "Augmented_data/25%_training_data_using_dreambooth_resized")
TRAIN_PATH_10_25_Dreambooth = os.path.join(PROJECT_ROOT, "data_labels/augmentation_txt2img/train_ten_25_text2img_dreambooth.csv")

VAL_PATH = os.path.join(PROJECT_ROOT, "data_labels/new_labels/validation.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data_labels/new_labels/test.csv")
LOCAL_DINO_MODEL_PATH = '/home/hpc/iwfa/iwfa054h/.cache/torch/hub/facebookresearch_dinov2_main'
