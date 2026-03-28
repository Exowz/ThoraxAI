"""
train -- Package d'entrainement.

Re-exporte tout pour compatibilite avec les imports existants :
    from src.train import train_model, EarlyStopping, get_device, ...
"""

from ..config import get_device  # noqa: F401 -- re-export pour compatibilite

from .early_stopping import EarlyStopping
from .loop import train_one_epoch, validate, train_model
from .kfold import train_model_kfold

__all__ = [
    "get_device",
    "EarlyStopping",
    "train_one_epoch",
    "validate",
    "train_model",
    "train_model_kfold",
]
