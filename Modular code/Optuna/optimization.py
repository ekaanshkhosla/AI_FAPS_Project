import optuna
from trainer import Trainer
from model_dino import define_dino_model
from model_efficientnet import define_efficientnet_model
from data_loader import create_dataloaders
import torch
import torch.optim as optim
import torch.nn as nn
from optuna.exceptions import TrialPruned

# Shared training function
def train_model(trial, model, train_path, val_path, image_dir, y_columns, image_size, batch_size, learning_rate):
    # Create data loaders with dynamic image size and batch size
    train_loader, val_loader = create_dataloaders(
        train_path, val_path, image_dir, y_columns, image_size, batch_size
    )
    
    

    # Set up optimizer, criterion, and device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, optimizer, criterion, device)

    # Early stopping configuration
    best_val_f1, patience, epochs_no_improve = 0, 10, 0

    for epoch in range(100):
        train_loss, train_f1 = trainer.train_epoch(train_loader)
        val_loss, val_f1 = trainer.evaluate(val_loader)
        
        print(f'Epoch {epoch+1}/{100}, Training Loss: {train_loss:.4f}, Training F1-score: {train_f1:.4f}, Validation Loss: {val_loss:.4f}, Validation F1-score: {val_f1:.4f}')
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise TrialPruned()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_val_f1

# print hyperparameters used in the current trail    
def print_hyperparameters_dino(image_size, batch_size, learning_rate, fc_units, fc_units_2):
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate:.10f}")
    print(f"Fully connected units: {fc_units}")
    print(f"Fully connected units: {fc_units_2}")
    
def print_hyperparameters_efficientnet(image_size, batch_size, learning_rate, fc_units, dropout_rate, layer_freeze_upto):
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate:.6f}")
    print(f"Fully connected layer: {fc_units}")
    print(f"Dropout rate: {dropout_rate:.6f}")
    print(f"Layer Freeze Upto: {layer_freeze_upto}")    


# DINOv2-specific objective function
def dino_objective(trial, train_path, val_path, image_dir, y_columns, local_model_path):
    # Hyperparameters
    image_size = trial.suggest_categorical('image_size', [224])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
                                                                0.0025, 0.00025, 0.000025, 0.0000025, 0.00000025,
                                                                0.005, 0.0005, 0.00005, 0.000005, 0.0000005,
                                                                0.0075, 0.00075, 0.000075, 0.0000075, 0.00000075,])
    fc_units = trial.suggest_categorical('fc_units', [128, 256, 512, 1024, 2048])
    fc_units_2 = trial.suggest_categorical('fc_units_2', [128, 256, 512, 1024, 2048])
    
    print("====================",f"Training of trial number:{trial.number}","====================")
    print_hyperparameters_dino(image_size, batch_size, learning_rate, fc_units, fc_units_2)

    # Define the DINOv2 model
    model = define_dino_model(fc_units, fc_units_2, local_model_path, num_classes=3)
    return train_model(trial, model, train_path, val_path, image_dir, y_columns, image_size, batch_size, learning_rate)

# EfficientNetV2-specific objective function
def efficientnet_objective(trial, train_path, val_path, image_dir, y_columns):
    # Hyperparameters
    image_size = trial.suggest_categorical('image_size', [224])
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
                                                                0.0025, 0.00025, 0.000025, 0.0000025, 0.00000025,
                                                                0.005, 0.0005, 0.00005, 0.000005, 0.0000005,
                                                                0.0075, 0.00075, 0.000075, 0.0000075, 0.00000075,])
    fc_units = trial.suggest_categorical('fc_units', [256, 512, 1024])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0, 0.5])
    layer_freeze_upto = trial.suggest_categorical('layer_freeze_upto', [
                                                                        'features.4.9.block.3.1.bias',
                                                                        'features.3.6.block.1.1.bias', 
                                                                        'features.2.6.block.1.1.bias',
                                                                        'features.1.3.block.0.1.bias',
                                                                        'features.0.1.bias'])

    print("====================",f"Training of trial number:{trial.number}","====================")
    print_hyperparameters_efficientnet(image_size, batch_size, learning_rate, fc_units, dropout_rate, layer_freeze_upto)
    
    # Define the EfficientNet model
    model = define_efficientnet_model(layer_freeze_upto, fc_units, dropout_rate, num_classes=3)
    return train_model(trial, model, train_path, val_path, image_dir, y_columns, image_size, batch_size, learning_rate)

# Run optimization function
def run_optimization(study_name, storage_url, train_path, val_path, image_dir, y_columns, model_type="dino", local_model_path=None):
    study = optuna.create_study(study_name=study_name, storage=storage_url, direction="maximize", load_if_exists=True)
    
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if(len(completed_trials) > 0):
        best_trial = study.best_trial
        print("Best trial's number: ", best_trial.number)
        print(f"Best score: {best_trial.value}")
        print("Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"{key}: {value}")
    
    total_trails_to_run = 50
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    num_trials_completed = len(completed_trials)
    pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
    num_pruned_trials = len(pruned_trials)
    print(f"Number of trials completed: {num_trials_completed}")
    print(f"Number of pruned trials: {num_pruned_trials}")
    print(f"Total number of trails completed: {num_trials_completed + num_pruned_trials}")
    trials_to_run = max(0, total_trails_to_run - (num_trials_completed + num_pruned_trials))
    print(f"Number of trials to run: {trials_to_run}")
    
    
    if model_type == "dino":
        study.optimize(lambda trial: dino_objective(trial, train_path, val_path, image_dir, y_columns, local_model_path), n_trials=trials_to_run)
    elif model_type == "efficientnet":
        study.optimize(lambda trial: efficientnet_objective(trial, train_path, val_path, image_dir, y_columns), n_trials=trials_to_run)
    
    print("Best trial's number:", study.best_trial.number)
    print("Best score:", study.best_trial.value)
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
