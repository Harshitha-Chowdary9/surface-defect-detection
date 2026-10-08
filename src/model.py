"""Models: a small CNN trained from scratch and a ResNet50 transfer-learning option."""
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "cnn":
        return SmallCNN(num_classes)
    if name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"unknown model {name}")
