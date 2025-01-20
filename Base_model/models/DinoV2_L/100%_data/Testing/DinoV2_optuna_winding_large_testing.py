import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import json
import glob
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import sys
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import optuna
from sklearn.metrics import f1_score
from optuna.exceptions import TrialPruned
import random

# Set working directory and paths
working_folder = os.path.abspath("")
image_dir = os.path.join(working_folder, "data_all")

train_df = pd.read_csv('data_labels/new_labels/train.csv')
test_df = pd.read_csv('data_labels/new_labels/test.csv')
val_df = pd.read_csv('data_labels/new_labels/validation.csv')

y_columns = train_df.drop(columns = ["image", "binary_NOK"]).columns

# Add classification head on DINOv2
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.dino_model(x)
        x = self.classifier(x)
        return x
    

test_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    
    
train_dataset = CustomImageDataset(train_df, image_dir, y_columns, transform=test_transform)
val_dataset = CustomImageDataset(val_df, image_dir, y_columns, transform=test_transform)
test_dataset = CustomImageDataset(test_df, image_dir, y_columns, transform=test_transform)
    
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

all_scores = []

# Training loop and repeating the testing for 10 times so that we get Average, Max and Min score of the model performance
for i in range(0, 10):
    
    local_model_path = '/home/hpc/iwfa/iwfa054h/.cache/torch/hub/facebookresearch_dinov2_main'
    dino_model = torch.hub.load(local_model_path, 'dinov2_vitl14', source='local')
    model = CustomDINONormModel(dino_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-06)
    loss_function = nn.BCEWithLogitsLoss()

    epochs_no_improve = 0
    num_epochs = 5
    best_val_f1 = 0
    best_model_path = "DinoV2_optuna_winding_large_testing_best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        all_labels = []
        all_predictions = []
        running_loss = 0.0
        
        for inputs, labels in (train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(torch.sigmoid(outputs)).detach()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
        
        epoch_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        model.eval()
        all_preds = []
        all_labels = []
        val_running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.round(torch.sigmoid(outputs))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training F1-score: {train_f1:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation F1-score: {val_f1:.4f}')

        # Save the model checkpoint on the epoch where validation score is highest
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)        

    # load the model for testing
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
        
    all_preds_test = []
    all_labels_test = []
        
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs))
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())
            
    test_f1 = f1_score(all_labels_test, all_preds_test, average='macro')
    print(f'Test F1-score: {test_f1:.4f}')
    all_scores.append(test_f1)
        
# document all the scores generated         
pd.DataFrame(all_scores).to_csv("DinoV2_optuna_winding_large_testing.csv")