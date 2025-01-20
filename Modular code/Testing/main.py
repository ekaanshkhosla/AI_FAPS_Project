import torch
import pandas as pd
import numpy as np
from config import (
    TRAIN_PATH_HALF, TRAIN_PATH_QUARTER, TRAIN_PATH_TEN, TRAIN_PATH_FULL, VAL_PATH, TEST_PATH, LOCAL_DINO_MODEL_PATH,
    IMAGE_DIR_10_25_SD, IMAGE_DIR_10_50_SD, IMAGE_DIR_25_50_SD, IMAGE_DIR_50_100_SD, IMAGE_DIR_100_200_SD,
    TRAIN_PATH_10_25_SD, TRAIN_PATH_10_50_SD, TRAIN_PATH_25_50_SD, TRAIN_PATH_50_100_SD, TRAIN_PATH_100_200_SD,
    IMAGE_DIR_10_25_Dreambooth, TRAIN_PATH_10_25_Dreambooth
)
from model_dino import CustomDINONormModel
from model_efficientnet import define_efficientnet
from train_and_test import train, test
from data_loader import get_data_loaders

# Model selection and configuration
model_type = "DINO"  # Choose between "DINO" and "EfficientNet"
train_path = TRAIN_PATH_10_25_SD
train_image_dir = IMAGE_DIR_10_25_SD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run training and testing 10 times
num_runs = 10
f1_scores = []

for i in range(num_runs):
    print(f"\nRun {i + 1}/{num_runs}")

    # Initialize model and hyperparameters based on model type
    if model_type == "DINO":
        # Define DINO-specific hyperparameters
        DINO_HYPERPARAMS = {
            'image_size': 224,
            'batch_size': 4,
            'learning_rate': 7.5e-07,
            'fc_units': 1024,
            'fc_units_2': 512,
            'num_classes': 3
        }
        dino_model = torch.hub.load(LOCAL_DINO_MODEL_PATH, 'dinov2_vitl14', source='local')
        model = CustomDINONormModel(
            dino_model, DINO_HYPERPARAMS['fc_units'], 
            DINO_HYPERPARAMS['fc_units_2'], DINO_HYPERPARAMS['num_classes']
        ).to(device)
        hyperparams = DINO_HYPERPARAMS

    else:
        # Define EfficientNet-specific hyperparameters
        EFFICIENTNET_HYPERPARAMS = {
            'image_size': 224,
            'batch_size': 8,
            'learning_rate': 7.5e-06,
            'fc_units': 256,
            'dropout_rate': 0,
            'layer_freeze': 'features.0.1.bias',
            'num_classes': 3
        }
        model = define_efficientnet(
            EFFICIENTNET_HYPERPARAMS['layer_freeze'], EFFICIENTNET_HYPERPARAMS['fc_units'],
            EFFICIENTNET_HYPERPARAMS['dropout_rate'], EFFICIENTNET_HYPERPARAMS['num_classes']
        ).to(device)
        hyperparams = EFFICIENTNET_HYPERPARAMS

    # Prepare data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_path, train_image_dir, VAL_PATH, TEST_PATH, hyperparams['batch_size'], hyperparams['image_size']
    )

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define model save path for each run
    model_path = f"{model_type}_best_model_run_{i+1}.pth"

    # Train and evaluate
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5, model_path=model_path)
    model.load_state_dict(torch.load(model_path))
    test_f1 = test(model, test_loader, device)
    
    # Log the F1 score
    f1_scores.append(test_f1)
    print(f"Run {i + 1} Test F1-score: {test_f1:.4f}")

# Create a DataFrame with the F1-scores for each run
results_df = pd.DataFrame({
    'Run': list(range(1, num_runs + 1)),
    'F1-Score': f1_scores
})

# Save the results to a CSV file
results_df.to_csv(f"{model_type}_f1_scores.csv", index=False)
print(f"\nResults saved to {model_type}_f1_scores.csv")
