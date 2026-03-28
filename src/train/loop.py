"""
loop.py -- Boucles d'entrainement et de validation.

Contient :
- train_one_epoch : une passe d'entrainement avec tqdm
- validate : evaluation sur un DataLoader (val ou test)
- train_model : boucle complete avec early stopping, lr scheduling, checkpoints
"""

import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

from ..config import (
    DEFAULT_NUM_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_PATIENCE,
    DEFAULT_MIN_DELTA,
    CHECKPOINT_DIR,
)
from .early_stopping import EarlyStopping


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Entraine le modele sur une epoque.

    Returns:
        (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evalue le modele sur un DataLoader (val ou test).

    Returns:
        (epoch_loss, epoch_accuracy, auc_roc)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels_f = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels_f)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        correct += (predicted == labels_f).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy().flatten())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # AUC-ROC (peut echouer si une seule classe presente dans le batch)
    try:
        epoch_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
    except ValueError:
        epoch_auc = 0.0

    return epoch_loss, epoch_acc, epoch_auc


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_weights: dict[int, float],
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    model_name: str = "model",
) -> dict:
    """Entraine un modele avec early stopping et lr scheduling.

    Args:
        model: Modele a entrainer
        train_loader: DataLoader d'entrainement
        val_loader: DataLoader de validation
        device: Device (cuda/mps/cpu)
        class_weights: Poids des classes {0: w0, 1: w1}
        num_epochs: Nombre max d'epoques
        learning_rate: Learning rate initial
        weight_decay: Regularisation L2
        patience: Patience pour l'early stopping
        min_delta: Amelioration minimale pour l'early stopping
        checkpoint_dir: Dossier de sauvegarde
        model_name: Nom du modele (pour le fichier checkpoint)

    Returns:
        Dictionnaire d'historique {train_loss, val_loss, train_acc, val_acc, val_auc, lr}
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Loss avec pos_weight pour le desequilibre
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimiseur -- on n'entraine que les parametres requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # Scheduler : reduit le lr quand la val_loss stagne
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # Historique
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_auc": [],
        "lr": [],
    }

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    t0 = time.time()

    print(f"Debut de l'entrainement de '{model_name}' ({num_epochs} epoques max, patience={patience}, min_delta={min_delta})")
    print(f"   pos_weight={pos_weight.item():.3f}, lr={learning_rate}, wd={weight_decay}")
    print("=" * 80)

    for epoch in range(num_epochs):
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        except RuntimeError as e:
            if device.type == "mps" and ("not currently supported" in str(e) or "not implemented" in str(e)):
                warnings.warn(
                    f"Operation non supportee sur MPS: {e}\n"
                    "Fallback vers CPU. Relancez avec device='cpu' pour eviter ce warning.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                device = torch.device("cpu")
                model = model.to(device)
                pos_weight = pos_weight.to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = optim.Adam(trainable_params, lr=optimizer.param_groups[0]["lr"],
                                       weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=3,
                )
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
            else:
                raise

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch + 1:2d}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(val_loss)

        # Early stopping + checkpoint
        should_stop = early_stopping(val_loss)
        if early_stopping.improved:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "history": history,
                    "training_time": time.time() - t0,
                },
                checkpoint_dir / f"best_{model_name}.pt",
            )
            print(f"   Meilleur modele sauvegarde (val_loss: {val_loss:.4f})")

        if should_stop:
            print(f"\nEarly stopping a l'epoque {epoch + 1}")
            break

    elapsed = time.time() - t0
    print("=" * 80)
    print(f"Termine en {elapsed:.0f}s. Meilleur : epoque {early_stopping.best_epoch} "
          f"(val_loss: {early_stopping.best_loss:.4f})")

    history["training_time"] = elapsed
    return history
