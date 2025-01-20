from config import TRAIN_PATH_HALF,TRAIN_PATH_QUARTER, TRAIN_PATH_TEN, TRAIN_PATH_FULL, VAL_PATH, IMAGE_DIR, LOCAL_DINO_MODEL_PATH
from optimization import run_optimization
import pandas as pd

if __name__ == "__main__":
    # choose dataset and model for your study
    dataset = TRAIN_PATH_TEN
    MODEL_TYPE = "efficientnet"  # change to efficientnet or dino
    STUDY_NAME = f"{MODEL_TYPE}_optuna_study"
    STORAGE_URL = f"sqlite:///{MODEL_TYPE}_optuna.db"
    
    train_df = pd.read_csv(dataset)
    y_columns = train_df.drop(columns=["image", "binary_NOK"]).columns  # Adjust columns as needed

    # Run optimization with configuration settings
    run_optimization(
        study_name=STUDY_NAME,
        storage_url=STORAGE_URL,
        train_path=dataset,
        val_path=VAL_PATH,
        image_dir=IMAGE_DIR,
        y_columns=y_columns,
        model_type=MODEL_TYPE,
        local_model_path=LOCAL_DINO_MODEL_PATH if MODEL_TYPE == "dino" else None
    )
