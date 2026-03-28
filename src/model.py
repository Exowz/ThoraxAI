"""
model.py -- Shim de retrocompatibilite.

Toute la logique a ete deplacee dans src/models/.
Ce fichier re-exporte tout pour que `from src.model import X` fonctionne.
"""

from .models import (  # noqa: F401
    PneumoniaCNN,
    PneumoniaResNet18,
    PneumoniaDenseNet121,
    PneumoniaEfficientNet,
    create_model,
    _MODELS,
)
