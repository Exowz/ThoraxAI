"""
dataset.py -- Shim de retrocompatibilite.

Toute la logique a ete deplacee dans src/data/.
Ce fichier re-exporte tout pour que `from src.dataset import X` fonctionne.
"""

from .data import (  # noqa: F401
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_no_augment_transform,
    get_train_transform,
    get_strong_augment_transform,
    get_eval_transform,
    SubsetWithTransform,
    create_dataloaders,
)
