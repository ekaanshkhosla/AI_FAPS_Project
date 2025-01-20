import torch
import torch.nn as nn

class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, fc_units, fc_units_2, num_classes):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(1024, fc_units),
            nn.LayerNorm(fc_units),
            nn.Linear(fc_units, fc_units_2),
            nn.ReLU(),
            nn.Linear(fc_units_2, num_classes)
        )

    def forward(self, x):
        x = self.dino_model(x)
        x = self.classifier(x)
        return x

def define_dino_model(fc_units, fc_units_2, local_model_path, num_classes):
    dino_model = torch.hub.load(local_model_path, 'dinov2_vitl14', source='local')
    return CustomDINONormModel(dino_model, fc_units, fc_units_2, num_classes)
