"""
metrics.py -- Predictions, metriques, comparaison de modeles et export JSON.

Contient :
- get_predictions : recupere labels/probs/predictions depuis un DataLoader
- print_classification_report : affiche le rapport complet
- _compute_model_metrics : calcule toutes les metriques pour un modele
- compare_models : compare N modeles (tableau + graphiques)
- export_results : sauvegarde les resultats en JSON
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    f1_score,
    accuracy_score,
)


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recupere labels, probabilites et predictions binaires.

    Returns:
        (y_true, y_probs, y_pred) -- arrays numpy
    """
    model.eval()
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = (y_probs > 0.5).astype(int)

    return y_true, y_probs, y_pred


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """Affiche le rapport de classification complet."""
    print("=" * 60)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Matrice de confusion :")
    print(f"   TN={tn}  FP={fp}")
    print(f"   FN={fn}  TP={tp}")
    print(f"\n   Sensibilite (Recall) : {tp / (tp + fn):.4f}")
    print(f"   Specificite :          {tn / (tn + fp):.4f}")


def _compute_model_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> dict:
    """Calcule toutes les metriques pour un modele."""
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_pred = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(rec_c, prec_c)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc,
        "auc_pr": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compare_models(
    results: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Compare les performances de plusieurs modeles.

    Affiche :
    - Tableau texte avec Accuracy, Recall, Specificite, F1, AUC-ROC + diff vs baseline
    - Courbes ROC comparees
    - Courbes de training comparees
    - Bar chart groupe des metriques

    Args:
        results: {model_name: {"y_true": ..., "y_probs": ..., "history": ...}}
    """
    if not results:
        print("Attention : aucun modele a comparer (results est vide).")
        return

    colors = ["#378ADD", "#D85A30", "#1D9E75", "#9B59B6"]
    model_names = list(results.keys())

    # Calculer les metriques pour chaque modele
    metrics_by_model = {}
    for name, res in results.items():
        metrics_by_model[name] = _compute_model_metrics(res["y_true"], res["y_probs"])

    # Metriques du baseline pour la colonne diff
    baseline_name = "cnn_baseline" if "cnn_baseline" in metrics_by_model else model_names[0]
    baseline_metrics = metrics_by_model[baseline_name]

    # Tableau comparatif
    print("=" * 90)
    print("COMPARAISON DES MODELES")
    print("=" * 90)
    header_metrics = ["Accuracy", "Recall", "Specif.", "F1", "AUC-ROC", "Diff vs baseline"]
    print(f"{'Modele':<18} " + " ".join(f"{m:>12}" for m in header_metrics))
    print("-" * 90)

    for name in model_names:
        m = metrics_by_model[name]
        diff = m["auc_roc"] - baseline_metrics["auc_roc"]
        diff_str = f"{diff:+.4f}" if name != baseline_name else "  (ref)"
        print(
            f"{name:<18} "
            f"{m['accuracy']:>12.4f} "
            f"{m['recall']:>12.4f} "
            f"{m['specificity']:>12.4f} "
            f"{m['f1']:>12.4f} "
            f"{m['auc_roc']:>12.4f} "
            f"{diff_str:>12}"
        )

    print("-" * 90)

    # --- Graphiques ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Courbes ROC comparees
    for idx, name in enumerate(model_names):
        y_true, y_probs = results[name]["y_true"], results[name]["y_probs"]
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        c = colors[idx % len(colors)]
        axes[0].plot(fpr, tpr, color=c, linewidth=2, label=f"{name} (AUC={roc_auc:.4f})")

    axes[0].plot([0, 1], [0, 1], color="#B4B2A9", linestyle="--", linewidth=1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("Courbes ROC comparees", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 2. Courbes de training comparees
    has_history = False
    for idx, name in enumerate(model_names):
        res = results[name]
        if "history" in res and res["history"] and "val_loss" in res["history"] and len(res["history"]["val_loss"]) > 0:
            h = res["history"]
            c = colors[idx % len(colors)]
            epochs = range(1, len(h["val_loss"]) + 1)
            axes[1].plot(epochs, h["val_loss"], color=c, linewidth=2, label=f"{name}")
            has_history = True

    axes[1].set_xlabel("Epoque")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Convergence comparee (val loss)", fontweight="bold")
    if has_history:
        axes[1].legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "Pas d'historique disponible",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=12, color="#888780")
    axes[1].grid(True, alpha=0.3)

    # 3. Bar chart groupe
    bar_metrics = ["accuracy", "recall", "specificity", "f1", "auc_roc"]
    bar_labels = ["Accuracy", "Recall", "Specificite", "F1", "AUC-ROC"]
    x = np.arange(len(bar_metrics))
    width = 0.8 / len(model_names)

    for idx, name in enumerate(model_names):
        m = metrics_by_model[name]
        values = [m[k] for k in bar_metrics]
        offset = (idx - len(model_names) / 2 + 0.5) * width
        bars = axes[2].bar(x + offset, values, width, label=name, color=colors[idx % len(colors)], alpha=0.85)
        for bar, val in zip(bars, values):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(bar_labels, fontsize=10)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("Metriques comparees", fontweight="bold")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Comparaison des modeles", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def export_results(
    results: dict[str, dict],
    output_path: str | Path = "outputs/results.json",
    kfold_results: dict | None = None,
    config: dict | None = None,
) -> None:
    """Sauvegarde tous les resultats numeriques en JSON.

    Args:
        results: {model_name: {"y_true": ..., "y_probs": ..., "history": ...}}
        output_path: Chemin du fichier JSON de sortie
        kfold_results: Resultats du K-fold (optionnel)
        config: Hyperparametres utilises (optionnel)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export = {"models": {}}

    for name, res in results.items():
        m = _compute_model_metrics(res["y_true"], res["y_probs"])

        # Seuil optimal
        y_true_arr = np.asarray(res["y_true"])
        y_probs_arr = np.asarray(res["y_probs"])
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = [f1_score(y_true_arr, (y_probs_arr >= t).astype(int)) for t in thresholds]
        optimal_thresh = float(thresholds[np.argmax(f1_scores)])

        # Nombre d'epoques
        history = res.get("history", {})
        epochs_trained = len(history.get("val_loss", []))

        export["models"][name] = {
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "specificity": round(m["specificity"], 4),
            "f1": round(m["f1"], 4),
            "auc_roc": round(m["auc_roc"], 4),
            "auc_pr": round(m["auc_pr"], 4),
            "optimal_threshold": round(optimal_thresh, 4),
            "confusion_matrix": m["confusion_matrix"],
            "epochs_trained": epochs_trained,
        }

    if kfold_results is not None:
        export["kfold"] = {}
        for metric_name in ["accuracy", "f1", "recall", "auc_roc"]:
            mean_key = f"{metric_name}_mean"
            std_key = f"{metric_name}_std"
            if mean_key in kfold_results and std_key in kfold_results:
                export["kfold"][metric_name] = {
                    "mean": round(kfold_results[mean_key], 4),
                    "std": round(kfold_results[std_key], 4),
                }

    if config is not None:
        export["config"] = config

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=4, ensure_ascii=False)

    print(f"Resultats exportes vers {output_path}")
