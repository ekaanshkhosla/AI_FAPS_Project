import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import cv2
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import numpy as np
import optuna
from sklearn.metrics import f1_score
from optuna.exceptions import TrialPruned
import random

# Set working directory and paths 
working_folder = os.path.abspath("")
image_dir = os.path.join(working_folder, "data_all")

train_df = pd.read_csv('data_labels/new_labels/splits/train_half.csv')
val_df = pd.read_csv('data_labels/new_labels/validation.csv')

y_columns = train_df.drop(columns = ["image", "binary_NOK"]).columns

#Set CustomImageDataset
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, y_columns, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.y_columns = y_columns
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = f"{self.image_dir}/{self.dataframe.iloc[idx, 0]}"
        image = Image.open(img_name).convert('L')
        image = image.convert('RGB')
        labels = torch.tensor(self.dataframe.iloc[idx][self.y_columns].to_numpy().astype('float32'))

        if self.transform:
            image = self.transform(image)

        return image, labels
    

# print hyperparameters used in the current trail      
def print_hyperparameters(image_size, batch_size, learning_rate, fc_units, fc_units_2):
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate:.10f}")
    print(f"Fully connected units 1: {fc_units}")
    print(f"Fully connected units 2: {fc_units_2}")
    
    
# Add classification head on DINOv2      
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, fc_units, fc_units_2):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(1024, fc_units),
            nn.LayerNorm(fc_units),
            nn.Linear(fc_units, fc_units_2),
            nn.ReLU(),
            nn.Linear(fc_units_2, 3),
        )

    def forward(self, x):
        x = self.dino_model(x)
        x = self.classifier(x)
        return x
    

# initialize DINOv2 model    
def define_model(fc_units, fc_units_2):
    
    local_model_path = '/home/hpc/iwfa/iwfa054h/.cache/torch/hub/facebookresearch_dinov2_main'
    dino_model = torch.hub.load(local_model_path, 'dinov2_vitl14', source='local')
    model = CustomDINONormModel(dino_model, fc_units, fc_units_2)
    
    return model



# Use SQLite file for storing trails
storage_url = "sqlite:///DinoV2_optuna_winding_large_half.db"
study_name = "DinoV2_optuna_winding_large_half"
study = optuna.create_study(study_name=study_name, storage=storage_url, direction="maximize", load_if_exists=True)


# Define objective function
def objective(trial):
    
    # Hyperparameters
    image_size = trial.suggest_categorical('image_size', [224])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
                                                                0.0025, 0.00025, 0.000025, 0.0000025, 0.00000025,
                                                                0.005, 0.0005, 0.00005, 0.000005, 0.0000005,
                                                                0.0075, 0.00075, 0.000075, 0.0000075, 0.00000075,])
    fc_units = trial.suggest_categorical('fc_units', [128, 256, 512, 1024, 2048])
    fc_units_2 = trial.suggest_categorical('fc_units_2', [128, 256, 512, 1024, 2048])

    
    num_classes = 3

    print("====================",f"Training of trial number:{trial.number}","====================")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print_hyperparameters(image_size, batch_size, learning_rate, fc_units, fc_units_2)

    train_dataset = CustomImageDataset(train_df, image_dir, y_columns, transform=transform)
    val_dataset = CustomImageDataset(val_df, image_dir, y_columns, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = define_model(fc_units, fc_units_2)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 100
    patience = 10

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        best_val_f1 = study.best_trial.value
    else:
        best_val_f1 = 0

    print("Best F1-score on Validation data until now:", best_val_f1)
    epochs_no_improve = 0
    trail_best = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        all_labels = []
        all_predictions = []
        running_loss = 0.0
        
        for inputs, labels in (train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs)).detach()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        model.eval()
        all_preds, all_labels = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in (val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training F1-score: {train_f1:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation F1-score: {val_f1:.4f}')

        trial.report(val_f1, epoch)
        if trial.should_prune():
            print("Best F1-score till now on Validation data:", best_val_f1)
            raise TrialPruned()

        if val_f1 < 0.1:
            print("Very bad trail")
            break

        if val_f1 > trail_best:
            trail_best = val_f1
            epochs_no_improve = 0
            print("Best F1-score till now on the current trail on Validation data:", trail_best)
            if(val_f1 > best_val_f1):
                best_val_f1 = val_f1
                print("Best F1-score till now on Validation data:", best_val_f1)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                print("Best F1-score till now on Validation data:", best_val_f1)
                break
    
    return best_val_f1


# printing best trail unitl now
completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
if(len(completed_trials) > 0):
    best_trial = study.best_trial
    print("Best trial's number: ", best_trial.number)
    print(f"Best score: {best_trial.value}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")
        
        
 # Defining how many trails to run         
total_trails_to_run = 100
completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
num_trials_completed = len(completed_trials)
pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
num_pruned_trials = len(pruned_trials)
print(f"Number of trials completed: {num_trials_completed}")
print(f"Number of pruned trials: {num_pruned_trials}")
print(f"Total number of trails completed: {num_trials_completed + num_pruned_trials}")
trials_to_run = max(0, total_trails_to_run - (num_trials_completed + num_pruned_trials))
print(f"Number of trials to run: {trials_to_run}")

study.optimize(objective, trials_to_run)

# printing best hyper parameters
best_trial = study.best_trial
print("Best trial's number: ", best_trial.number)
print(f"Best score: {best_trial.value}")
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")