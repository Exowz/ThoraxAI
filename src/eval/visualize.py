"""
visualize.py -- Visualisations Grad-CAM et Integrated Gradients.

Contient :
- show_gradcam : affiche les cartes Grad-CAM (3 lignes : image, heatmap, superposition)
- show_gradcam_errors : affiche les erreurs (FN/FP) avec Grad-CAM
- show_gradcam_comparison : compare Grad-CAM vs Grad-CAM++
- show_integrated_gradients : visualise les attributions IG
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2

from ..config import IMAGENET_MEAN, IMAGENET_STD
from .gradcam import GradCAM, GradCAMPlusPlus, IntegratedGradients


def show_gradcam(
    model: nn.Module,
    dataset,
    indices: list[int],
    class_names: list[str],
    device: torch.device,
    img_size: int = 224,
    save_path: str | None = None,
) -> None:
    """Affiche les cartes Grad-CAM pour plusieurs images.

    3 lignes : image originale, heatmap seule, superposition.
    """
    grad_cam = GradCAM(model, model.last_conv_layer)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    n = len(indices)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_show = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        # Grad-CAM
        input_t = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        cam = grad_cam.generate(input_t)
        cam_resized = cv2.resize(cam, (img_size, img_size))

        # Prediction
        with torch.no_grad():
            prob = torch.sigmoid(model(img_tensor.unsqueeze(0).to(device))).item()

        color = "#1D9E75" if class_names[label] == "NORMAL" else "#D85A30"

        # Ligne 1 : image originale
        axes[0, i].imshow(img_show)
        axes[0, i].set_title(
            f"{class_names[label]}\nP(pneu)={prob:.3f}",
            fontsize=11, color=color, fontweight="bold",
        )
        axes[0, i].axis("off")

        # Ligne 2 : heatmap seule
        axes[1, i].imshow(cam_resized, cmap="jet")
        axes[1, i].set_title("Heatmap Grad-CAM", fontsize=11, fontweight="bold")
        axes[1, i].axis("off")

        # Ligne 3 : superposition
        axes[2, i].imshow(img_show)
        axes[2, i].imshow(cam_resized, cmap="jet", alpha=0.4)
        axes[2, i].set_title("Superposition", fontsize=11, fontweight="bold")
        axes[2, i].axis("off")

    plt.suptitle("Interpretabilite -- Grad-CAM", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def show_gradcam_errors(
    model: nn.Module,
    dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    device: torch.device,
    error_type: str = "FN",
    n: int = 4,
    img_size: int = 224,
    save_path: str | None = None,
) -> None:
    """Affiche les erreurs (FN ou FP) avec Grad-CAM.

    Pour chaque erreur, affiche 3 colonnes : image originale, heatmap seule, superposition.
    """
    if error_type == "FN":
        indices = np.where((y_true == 1) & (y_pred == 0))[0]
        title = "Faux negatifs avec Grad-CAM"
        title_color = "#E24B4A"
    else:
        indices = np.where((y_true == 0) & (y_pred == 1))[0]
        title = "Faux positifs avec Grad-CAM"
        title_color = "#BA7517"

    n = min(n, len(indices))
    if n == 0:
        print(f"Aucun {error_type} trouve !")
        return

    grad_cam = GradCAM(model, model.last_conv_layer)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for i, idx in enumerate(indices[:n]):
        img_tensor, label = dataset[idx]
        img_show = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        input_t = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        cam = grad_cam.generate(input_t)
        cam_resized = cv2.resize(cam, (img_size, img_size))

        prob = y_probs[idx]
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]

        # Col 1 : image originale
        axes[i, 0].imshow(img_show)
        axes[i, 0].set_title(
            f"Vrai: {true_label} | Predit: {pred_label}\nP(pneu)={prob:.3f}",
            fontsize=10, fontweight="bold", color=title_color,
        )
        axes[i, 0].axis("off")

        # Col 2 : heatmap seule
        axes[i, 1].imshow(cam_resized, cmap="jet")
        axes[i, 1].set_title("Heatmap Grad-CAM", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # Col 3 : superposition
        axes[i, 2].imshow(img_show)
        axes[i, 2].imshow(cam_resized, cmap="jet", alpha=0.4)
        axes[i, 2].set_title("Superposition", fontsize=10, fontweight="bold")
        axes[i, 2].axis("off")

    plt.suptitle(title, fontsize=15, fontweight="bold", color=title_color, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def show_gradcam_comparison(
    model: nn.Module,
    dataset,
    indices: list[int],
    class_names: list[str],
    device: torch.device,
    img_size: int = 224,
    save_path: str | None = None,
) -> None:
    """Affiche la comparaison Grad-CAM vs Grad-CAM++ sur plusieurs images.

    3 lignes : image originale, Grad-CAM, Grad-CAM++.
    """
    grad_cam = GradCAM(model, model.last_conv_layer)
    grad_cam_pp = GradCAMPlusPlus(model, model.last_conv_layer)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    n = len(indices)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    row_labels = ["Image originale", "Grad-CAM", "Grad-CAM++"]

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_show = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        input_t = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        cam = grad_cam.generate(input_t)
        cam_resized = cv2.resize(cam, (img_size, img_size))

        input_t2 = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        cam_pp = grad_cam_pp.generate(input_t2)
        cam_pp_resized = cv2.resize(cam_pp, (img_size, img_size))

        with torch.no_grad():
            prob = torch.sigmoid(model(img_tensor.unsqueeze(0).to(device))).item()

        color = "#1D9E75" if class_names[label] == "NORMAL" else "#D85A30"

        axes[0, i].imshow(img_show)
        axes[0, i].set_title(
            f"{class_names[label]}\nP(pneu)={prob:.3f}",
            fontsize=11, color=color, fontweight="bold",
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(img_show)
        axes[1, i].imshow(cam_resized, cmap="jet", alpha=0.4)
        axes[1, i].set_title("Grad-CAM", fontsize=11, fontweight="bold")
        axes[1, i].axis("off")

        axes[2, i].imshow(img_show)
        axes[2, i].imshow(cam_pp_resized, cmap="jet", alpha=0.4)
        axes[2, i].set_title("Grad-CAM++", fontsize=11, fontweight="bold")
        axes[2, i].axis("off")

    for row_idx, label_text in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label_text, fontsize=12, fontweight="bold", rotation=90, labelpad=10)

    plt.suptitle("Comparaison Grad-CAM vs Grad-CAM++", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def show_integrated_gradients(
    model: nn.Module,
    dataset,
    indices: list[int],
    class_names: list[str],
    device: torch.device,
    img_size: int = 224,
    n_steps: int = 50,
    save_path: str | None = None,
) -> None:
    """Visualise les attributions Integrated Gradients sous forme de heatmap."""
    ig = IntegratedGradients(model)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_show = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        input_t = img_tensor.unsqueeze(0).to(device)
        attr_map = ig.compute(input_t, n_steps=n_steps)

        with torch.no_grad():
            prob = torch.sigmoid(model(input_t)).item()

        color = "#1D9E75" if class_names[label] == "NORMAL" else "#D85A30"

        axes[0, i].imshow(img_show)
        axes[0, i].set_title(
            f"{class_names[label]}\nP(pneu)={prob:.3f}",
            fontsize=11, color=color, fontweight="bold",
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(img_show)
        axes[1, i].imshow(attr_map, cmap="hot", alpha=0.5)
        axes[1, i].set_title("Integrated Gradients", fontsize=11, fontweight="bold")
        axes[1, i].axis("off")

    plt.suptitle("Interpretabilite -- Integrated Gradients", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
