# model_dino.py
import torch.nn as nn
import torch

class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, fc_units1, fc_units2, num_classes):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(1024, fc_units1),
            nn.LayerNorm(fc_units1),
            nn.Linear(fc_units1, fc_units2),
            nn.ReLU(),
            nn.Linear(fc_units2, num_classes),
        )

    def forward(self, x):
        x = self.dino_model(x)
        x = self.classifier(x)
        return x
