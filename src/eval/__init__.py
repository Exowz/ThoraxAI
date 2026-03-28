"""
eval -- Package d'evaluation, metriques et interpretabilite.

Re-exporte tout pour compatibilite avec les imports existants :
    from src.eval import GradCAM, get_predictions, compare_models, ...
"""

from .gradcam import GradCAM, GradCAMPlusPlus, IntegratedGradients
from .metrics import (
    get_predictions,
    print_classification_report,
    _compute_model_metrics,
    compare_models,
    export_results,
)
from .plots import (
    plot_confusion_matrix,
    plot_roc_pr_curves,
    optimize_threshold,
    show_errors,
)
from .visualize import (
    show_gradcam,
    show_gradcam_errors,
    show_gradcam_comparison,
    show_integrated_gradients,
)

__all__ = [
    # gradcam
    "GradCAM",
    "GradCAMPlusPlus",
    "IntegratedGradients",
    # metrics
    "get_predictions",
    "print_classification_report",
    "_compute_model_metrics",
    "compare_models",
    "export_results",
    # plots
    "plot_confusion_matrix",
    "plot_roc_pr_curves",
    "optimize_threshold",
    "show_errors",
    # visualize
    "show_gradcam",
    "show_gradcam_errors",
    "show_gradcam_comparison",
    "show_integrated_gradients",
]
