import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import sys
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
image_dir_train = os.path.join(working_folder, "Augmented_data/25%_training_data_using_dreambooth_resized")

train_df = pd.read_csv('data_labels/augmentation_txt2img/train_ten_25_text2img_dreambooth.csv')
test_df = pd.read_csv('data_labels/new_labels/test.csv')
val_df = pd.read_csv('data_labels/new_labels/validation.csv')

y_columns = train_df.drop(columns = ["image", "binary_NOK"]).columns

# Best hyper parameters obtained by Optuna study
best_hyperparameters = {
    'image_size': 224,  
    'batch_size': 8,  
    'learning_rate': 2.5e-05,  
    'fc_units': 256, 
    'dropout_rate': 0,  
    'layer_freeze': 'features.1.3.block.0.1.bias',
    'num_classes': 3  
}

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
        
 
# Define efficientnet_v2_l model  
def define_model(layer_freeze_upto, fc_units, dropout_rate, num_classes):
    
    model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

    cutoff_reached = False
    for name, param in model.named_parameters():
        if not cutoff_reached:
            if name == layer_freeze_upto:
                cutoff_reached = True
            param.requires_grad = False
        else:
            param.requires_grad = True
        
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, fc_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(fc_units, num_classes),
    )
    
    return model
    
    

test_transform = transforms.Compose([
    transforms.Resize((best_hyperparameters['image_size'], best_hyperparameters['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
    
train_dataset = CustomImageDataset(train_df, image_dir_train, y_columns, transform=test_transform)
val_dataset = CustomImageDataset(val_df, image_dir, y_columns, transform=test_transform)
test_dataset = CustomImageDataset(test_df, image_dir, y_columns, transform=test_transform)
    
train_loader = DataLoader(train_dataset, batch_size=best_hyperparameters['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=best_hyperparameters['batch_size'], num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=best_hyperparameters['batch_size'], num_workers=4)


all_scores = []

# Training loop and repeating the testing for 10 times so that we get Average, Max and Min score of the model performance
for i in range(0, 10):
            
    model = define_model(best_hyperparameters['layer_freeze'], best_hyperparameters['fc_units'], best_hyperparameters['dropout_rate'], best_hyperparameters['num_classes'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparameters['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    epochs_no_improve = 0
    num_epochs = 5
    best_val_f1 = 0
    best_model_path = "EfficientNetV2_optuna_winding_large_testing_aug_10_25_dreambooth_all_defects_best_model.pth"
    
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
                loss = criterion(outputs, labels)
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
    all_scores.append(test_f1)
    print(f'Test F1-score: {test_f1:.4f}')

    
# document all the scores generated    
pd.DataFrame(all_scores).to_csv("EfficientNetV2_optuna_winding_large_testing_aug_10_25_dreambooth_all_defects.csv")