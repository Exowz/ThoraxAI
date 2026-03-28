"""
config.py -- Constantes centralisees du projet.

Regroupe toutes les valeurs partagees entre les modules
(normalisation, hyperparametres par defaut, chemins, seed, device).
"""

import torch


# Normalisation ImageNet (standard, meme pour grayscale converti en RGB)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Taille des images (entree des modeles)
IMG_SIZE = 224

# Graine aleatoire pour reproductibilite
SEED = 42

# Hyperparametres par defaut
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT_RATE = 0.5
DEFAULT_PATIENCE = 7
DEFAULT_MIN_DELTA = 0.001
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_NUM_WORKERS = 2
DEFAULT_NUM_EPOCHS = 30

# Chemins
CHECKPOINT_DIR = "outputs/checkpoints"
RESULTS_DIR = "outputs"


def get_device() -> torch.device:
    """Detecte le meilleur device disponible (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
