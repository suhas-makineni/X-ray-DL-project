import torch.nn as nn
from torchvision.models import resnet18


class MultiTaskXRayModel(nn.Module):
    def __init__(self, num_diseases):
        super().__init__()

        self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.disease_head = nn.Linear(in_features, num_diseases)
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        disease_logits = self.disease_head(features)
        gender_logits = self.gender_head(features)
        return disease_logits, gender_logits
