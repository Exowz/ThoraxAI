"""
kfold.py -- Entrainement avec validation croisee K-fold stratifiee.

Utilise StratifiedKFold de scikit-learn pour garantir la repartition
des classes dans chaque fold, puis entraine un nouveau modele par fold.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score

from ..config import (
    SEED,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_PATIENCE,
    DEFAULT_MIN_DELTA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_NUM_WORKERS,
)
from .loop import train_model


def train_model_kfold(
    model_name: str,
    full_dataset,
    device: torch.device,
    class_weights: dict[int, float],
    k: int = 5,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    fine_tune: bool = False,
) -> dict:
    """Entrainement avec validation croisee K-fold stratifiee.

    Args:
        model_name: Nom du modele a creer via create_model
        full_dataset: Dataset complet (ImageFolder)
        device: Device (cuda/mps/cpu)
        class_weights: Poids des classes {0: w0, 1: w1}
        k: Nombre de folds
        num_epochs: Nombre max d'epoques par fold
        learning_rate: Learning rate initial
        weight_decay: Regularisation L2
        patience: Patience pour l'early stopping
        min_delta: Amelioration minimale pour l'early stopping
        batch_size: Taille des batchs
        dropout_rate: Taux de dropout
        fine_tune: Degeler les dernieres couches (transfer learning)

    Returns:
        Dict avec les metriques moyennes +/- ecart-type par fold :
        {accuracy, f1, recall, auc_roc} chacun avec _mean, _std, et _folds
    """
    from ..models import create_model

    labels = np.array([s[1] for s in full_dataset.samples])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    fold_metrics = {
        "accuracy": [],
        "f1": [],
        "recall": [],
        "auc_roc": [],
    }

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{k}")
        print(f"{'='*60}")

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=DEFAULT_NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler,
                                num_workers=DEFAULT_NUM_WORKERS, pin_memory=True)

        # Nouveau modele pour chaque fold
        model = create_model(model_name, dropout_rate=dropout_rate,
                             fine_tune=fine_tune, device=device)

        # Entrainer
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_weights=class_weights,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            checkpoint_dir=f"outputs/checkpoints/kfold_fold{fold_idx + 1}",
            model_name=f"{model_name}_fold{fold_idx + 1}",
        )

        # Evaluer sur le fold de validation
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, batch_labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_labels.extend(batch_labels.numpy())
                all_probs.extend(probs)

        y_true = np.array(all_labels)
        y_probs = np.array(all_probs)
        y_pred = (y_probs > 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        auc_val = roc_auc_score(y_true, y_probs)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["f1"].append(f1)
        fold_metrics["recall"].append(rec)
        fold_metrics["auc_roc"].append(auc_val)

        print(f"Fold {fold_idx + 1} -- Acc: {acc:.4f}, F1: {f1:.4f}, "
              f"Recall: {rec:.4f}, AUC-ROC: {auc_val:.4f}")

    # Resultats agreges
    results = {}
    print(f"\n{'='*60}")
    print(f"Resultats K-Fold ({k} folds)")
    print(f"{'='*60}")
    for metric_name, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        results[f"{metric_name}_mean"] = mean_val
        results[f"{metric_name}_std"] = std_val
        results[f"{metric_name}_folds"] = values
        print(f"   {metric_name}: {mean_val:.4f} +/- {std_val:.4f}")

    return results
