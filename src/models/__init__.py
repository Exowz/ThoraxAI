"""
models -- Package des architectures CNN.

Re-exporte tout pour compatibilite avec les imports existants :
    from src.models import create_model, PneumoniaCNN, ...
"""

from .architectures import (
    PneumoniaCNN,
    PneumoniaResNet18,
    PneumoniaDenseNet121,
    PneumoniaEfficientNet,
)
from .factory import create_model, _MODELS

__all__ = [
    "PneumoniaCNN",
    "PneumoniaResNet18",
    "PneumoniaDenseNet121",
    "PneumoniaEfficientNet",
    "create_model",
    "_MODELS",
]
