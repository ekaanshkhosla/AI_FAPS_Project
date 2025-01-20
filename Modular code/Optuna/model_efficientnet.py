import torch
import torch.nn as nn
from torchvision import models

def define_efficientnet_model(layer_freeze_upto, fc_units, dropout_rate, num_classes):
    model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    
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
        nn.Linear(fc_units, num_classes)
    )
    return model
