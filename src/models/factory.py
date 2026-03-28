"""
factory.py -- Factory function pour instancier les modeles par nom.

Usage:
    from src.models import create_model
    model = create_model("resnet18", fine_tune=True)
"""

import torch
import torch.nn as nn

from ..config import get_device, DEFAULT_DROPOUT_RATE
from .architectures import (
    PneumoniaCNN,
    PneumoniaResNet18,
    PneumoniaDenseNet121,
    PneumoniaEfficientNet,
)

_MODELS = {
    "cnn_baseline": PneumoniaCNN,
    "resnet18": PneumoniaResNet18,
    "densenet121": PneumoniaDenseNet121,
    "efficientnet": PneumoniaEfficientNet,
}


def create_model(
    name: str = "cnn_baseline",
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    fine_tune: bool = False,
    device: torch.device | None = None,
) -> nn.Module:
    """Cree un modele par son nom.

    Args:
        name: "cnn_baseline", "resnet18", "densenet121", ou "efficientnet"
        dropout_rate: Taux de dropout
        fine_tune: Degeler les dernieres couches (transfer learning uniquement)
        device: Device cible (auto-detecte si None)

    Returns:
        Modele instancie et place sur le device
    """
    if device is None:
        device = get_device()

    if name not in _MODELS:
        raise ValueError(f"Modele inconnu: {name}. Choix: {list(_MODELS.keys())}")

    cls = _MODELS[name]
    if name == "cnn_baseline":
        model = cls(dropout_rate=dropout_rate)
    else:
        model = cls(dropout_rate=dropout_rate, fine_tune=fine_tune)

    model = model.to(device)

    # Stats
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modele : {name}")
    print(f"   Parametres totaux :       {total:,}")
    print(f"   Parametres entrainables : {trainable:,}")

    return model
