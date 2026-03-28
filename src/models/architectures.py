"""
architectures.py -- Architectures de modeles CNN.

Contient :
- PneumoniaCNN : CNN baseline a 3 blocs convolutifs (from scratch)
- PneumoniaResNet18 : Transfer learning avec ResNet18 pre-entraine ImageNet
- PneumoniaDenseNet121 : Transfer learning avec DenseNet121 pre-entraine ImageNet
- PneumoniaEfficientNet : Transfer learning avec EfficientNet-B0 pre-entraine ImageNet
"""

import torch
import torch.nn as nn
from torchvision import models

from ..config import DEFAULT_DROPOUT_RATE


class PneumoniaCNN(nn.Module):
    """CNN baseline pour la classification binaire Normal vs Pneumonia.

    Architecture :
        - Bloc 1 : Conv(32) + BatchNorm + ReLU + MaxPool  (224 -> 112)
        - Bloc 2 : Conv(64) + BatchNorm + ReLU + MaxPool  (112 -> 56)
        - Bloc 3 : Conv(128) + BatchNorm + ReLU + MaxPool (56 -> 28)
        - Flatten -> Dense(128) + Dropout -> Dense(1)

    La sortie est un logit brut (pas de sigmoid) -- on utilise BCEWithLogitsLoss.
    """

    def __init__(self, dropout_rate: float = DEFAULT_DROPOUT_RATE):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

    @property
    def last_conv_layer(self) -> nn.Module:
        """Couche cible pour Grad-CAM."""
        return self.block3[0]


class PneumoniaResNet18(nn.Module):
    """Transfer learning avec ResNet18 pre-entraine sur ImageNet.

    Strategie :
    - On gele les couches convolutives pre-entrainees (feature extractor)
    - On remplace la derniere couche fully-connected par un classifieur
      a 2 couches : Linear(in, 256) -> ReLU -> Dropout -> Linear(256, 1)
    - Option fine_tune=True pour degeler les dernieres couches
    """

    def __init__(self, dropout_rate: float = DEFAULT_DROPOUT_RATE, fine_tune: bool = False):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Geler toutes les couches
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Degeler les dernieres couches si fine_tune
        if fine_tune:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # Remplacer le classifieur
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def last_conv_layer(self) -> nn.Module:
        """Couche cible pour Grad-CAM (derniere couche conv de ResNet)."""
        return self.backbone.layer4[-1].conv2


class PneumoniaDenseNet121(nn.Module):
    """Transfer learning avec DenseNet121 pre-entraine sur ImageNet.

    DenseNet121 est souvent utilise en imagerie medicale (ex. CheXNet).
    Classifieur a 2 couches : Linear(in, 256) -> ReLU -> Dropout -> Linear(256, 1).
    """

    def __init__(self, dropout_rate: float = DEFAULT_DROPOUT_RATE, fine_tune: bool = False):
        super().__init__()

        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Geler toutes les couches
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Degeler le dernier dense block si fine_tune
        if fine_tune:
            for param in self.backbone.features.denseblock4.parameters():
                param.requires_grad = True

        # Remplacer le classifieur
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def last_conv_layer(self) -> nn.Module:
        """Couche cible pour Grad-CAM."""
        return self.backbone.features.denseblock4


class PneumoniaEfficientNet(nn.Module):
    """Transfer learning avec EfficientNet-B0 pre-entraine sur ImageNet.

    EfficientNet utilise des compound scaling pour un bon ratio
    performance/parametres. Architecture legere et performante.
    Classifieur a 2 couches : Linear(in, 256) -> ReLU -> Dropout -> Linear(256, 1).
    """

    def __init__(self, dropout_rate: float = DEFAULT_DROPOUT_RATE, fine_tune: bool = False):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Geler toutes les couches
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Degeler les derniers blocs si fine_tune
        if fine_tune:
            for param in self.backbone.features[-2:].parameters():
                param.requires_grad = True

        # Remplacer le classifieur
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def last_conv_layer(self) -> nn.Module:
        """Couche cible pour Grad-CAM (dernier bloc conv d'EfficientNet)."""
        return self.backbone.features[-1]
