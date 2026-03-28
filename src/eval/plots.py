"""
plots.py -- Visualisations des metriques et erreurs.

Contient :
- plot_confusion_matrix : matrice de confusion (heatmap seaborn)
- plot_roc_pr_curves : courbes ROC et Precision-Recall
- optimize_threshold : F1-score en fonction du seuil de decision
- show_errors : affichage des faux negatifs / faux positifs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)

from ..config import IMAGENET_MEAN, IMAGENET_STD


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Matrice de confusion",
    save_path: str | None = None,
) -> None:
    """Trace et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 18, "fontweight": "bold"},
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Prediction", fontsize=13, fontweight="bold")
    ax.set_ylabel("Reel", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title_suffix: str = "",
    save_path: str | None = None,
) -> tuple[float, float]:
    """Trace les courbes ROC et Precision-Recall.

    Returns:
        (auc_roc, auc_pr)
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    precision_c, recall_c, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_c, precision_c)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr, tpr, color="#378ADD", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#B4B2A9", linestyle="--", linewidth=1)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#378ADD")
    axes[0].set_xlabel("Taux de faux positifs")
    axes[0].set_ylabel("Taux de vrais positifs")
    axes[0].set_title("Courbe ROC", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recall_c, precision_c, color="#1D9E75", linewidth=2,
                 label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Courbe Precision-Recall", fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Courbes d'evaluation {title_suffix}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return roc_auc, pr_auc


def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str | None = None,
) -> float:
    """Trouve le seuil optimal (maximise le F1-score).

    Returns:
        Seuil optimal
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]

    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, f1_scores, color="#378ADD", linewidth=2)
    plt.axvline(optimal_thresh, color="#D85A30", linestyle="--",
                label=f"Optimal: {optimal_thresh:.2f} (F1={optimal_f1:.4f})")
    plt.axvline(0.5, color="#888780", linestyle=":", label="Defaut: 0.50")
    plt.xlabel("Seuil de decision")
    plt.ylabel("F1-score")
    plt.title("F1-score en fonction du seuil", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Seuil optimal : {optimal_thresh:.2f} (F1={optimal_f1:.4f})")
    return optimal_thresh


def show_errors(
    dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    error_type: str = "FN",
    n: int = 4,
) -> None:
    """Affiche des exemples de faux negatifs ou faux positifs."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    if error_type == "FN":
        indices = np.where((y_true == 1) & (y_pred == 0))[0]
        title = "Faux negatifs (pneumonies manquees)"
        color = "#E24B4A"
    else:
        indices = np.where((y_true == 0) & (y_pred == 1))[0]
        title = "Faux positifs (fausses alertes)"
        color = "#BA7517"

    n = min(n, len(indices))
    if n == 0:
        print(f"Aucun {error_type} trouve !")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, idx in enumerate(indices[:n]):
        img, label = dataset[idx]
        img_show = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img_show)
        axes[i].set_title(
            f"P(pneu)={y_probs[idx]:.3f}\nVrai: {class_names[y_true[idx]]}",
            fontsize=11, color=color, fontweight="bold",
        )
        axes[i].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", color=color)
    plt.tight_layout()
    plt.show()
