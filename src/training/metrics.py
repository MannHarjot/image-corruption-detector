"""Evaluation metrics for the corruption classification task.

Provides per-class precision/recall/F1, overall accuracy, macro F1, and
confusion matrix generation using scikit-learn.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Human-readable names for each class index. If ``None``,
            integer indices are used as keys.

    Returns:
        Dictionary with keys:
            - ``"accuracy"`` (float): Overall accuracy in [0, 1].
            - ``"macro_f1"`` (float): Macro-averaged F1 score.
            - ``"per_class"`` (dict): Per-class dict mapping class name to
              ``{"precision": float, "recall": float, "f1": float, "support": int}``.
            - ``"confusion_matrix"`` (np.ndarray): Confusion matrix [n_classes, n_classes].

    Example:
        >>> metrics = compute_metrics(y_true, y_pred, class_names=["clean", "blur"])
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    num_classes = len(precision)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def classification_report_dict(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Tuple[str, Dict]:
    """Return scikit-learn classification report as both string and dict.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Names for each class index.

    Returns:
        Tuple of (formatted_string, report_dict).

    Example:
        >>> report_str, report_dict = classification_report_dict(y_true, y_pred)
        >>> print(report_str)
    """
    target_names = class_names if class_names else None

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return report_str, report_dict


def format_metrics_table(metrics: Dict, class_names: Optional[List[str]] = None) -> str:
    """Format a metrics dict as a readable table string.

    Args:
        metrics: Output of :func:`compute_metrics`.
        class_names: Optional list of class name strings.

    Returns:
        Multi-line string with accuracy, macro F1, and per-class table.
    """
    lines = [
        f"Overall Accuracy : {metrics['accuracy']:.4f}",
        f"Macro F1         : {metrics['macro_f1']:.4f}",
        "",
        f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 64,
    ]
    for cls, vals in metrics["per_class"].items():
        lines.append(
            f"{cls:<20} {vals['precision']:>10.4f} {vals['recall']:>10.4f} "
            f"{vals['f1']:>10.4f} {vals['support']:>10d}"
        )
    return "\n".join(lines)
