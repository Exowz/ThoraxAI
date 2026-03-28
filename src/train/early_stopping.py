"""
early_stopping.py -- Mecanisme d'arret premature.

Surveille la val_loss et arrete l'entrainement si elle ne s'ameliore
plus suffisamment pendant un nombre donne d'epoques.
"""

from ..config import DEFAULT_PATIENCE, DEFAULT_MIN_DELTA


class EarlyStopping:
    """Early stopping avec min_delta.

    Arrete l'entrainement si la val_loss ne s'ameliore pas d'au moins
    min_delta pendant patience epoques consecutives.

    Args:
        patience: Nombre d'epoques sans amelioration avant l'arret
        min_delta: Amelioration minimale requise pour considerer un progres
    """

    def __init__(
        self,
        patience: int = DEFAULT_PATIENCE,
        min_delta: float = DEFAULT_MIN_DELTA,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
        self._epoch = 0

    def __call__(self, val_loss: float) -> bool:
        """Retourne True si on doit arreter l'entrainement."""
        self._epoch += 1
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = self._epoch
            return False
        self.counter += 1
        return self.counter >= self.patience

    @property
    def improved(self) -> bool:
        """True si le dernier appel a enregistre une amelioration."""
        return self.counter == 0

    def reset(self):
        """Reinitialise l'etat (utile entre les folds du K-fold)."""
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
        self._epoch = 0
