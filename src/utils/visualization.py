"""Publication-quality visualization utilities.

All functions save output PNGs to disk and return the output path.
Nothing is displayed interactively — suitable for headless server execution.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global style
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def plot_training_curves(
    history: Union[dict, str, Path],
    output_path: Path = Path("outputs/training_curves.png"),
) -> Path:
    """Plot training and validation loss + accuracy curves.

    Args:
        history: Training history dict (keys: ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``, ``lr``) **or** path to a
            ``training_history.json`` file.
        output_path: Destination PNG file.

    Returns:
        Resolved path to the saved PNG.

    Example:
        >>> plot_training_curves("outputs/training_history.json")
    """
    if not isinstance(history, dict):
        with open(history, "r", encoding="utf-8") as fh:
            history = json.load(fh)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss subplot
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation", linewidth=2, linestyle="--")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy subplot
    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Validation", linewidth=2, linestyle="--")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # LR subplot
    axes[2].semilogy(epochs, history["lr"], color="tab:green", linewidth=2)
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate (log scale)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", output_path)
    return output_path


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: Path = Path("outputs/confusion_matrix.png"),
    normalize: bool = True,
) -> Path:
    """Plot a labelled confusion matrix heatmap.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Human-readable names for each class index.
        output_path: Destination PNG file.
        normalize: If ``True``, normalize counts by true-class totals (recall view).

    Returns:
        Resolved path to the saved PNG.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix (Recall)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix (Counts)"

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)
    return output_path


def plot_sample_predictions(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    output_path: Path = Path("outputs/sample_predictions.png"),
    max_samples: int = 16,
) -> Path:
    """Plot a grid of sample images with true and predicted labels.

    Args:
        images: List of uint8 RGB NumPy arrays (H, W, 3).
        true_labels: Ground-truth class name for each image.
        pred_labels: Predicted class name for each image.
        confidences: Softmax confidence for each prediction.
        output_path: Destination PNG file.
        max_samples: Maximum number of images to show (truncated if larger).

    Returns:
        Resolved path to the saved PNG.
    """
    n = min(len(images), max_samples)
    images = images[:n]
    true_labels = true_labels[:n]
    pred_labels = pred_labels[:n]
    confidences = confidences[:n]

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3 + 0.5))
    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    axes = np.array(axes).reshape(-1)  # flatten regardless of shape

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(images[i])
            correct = true_labels[i] == pred_labels[i]
            color = "green" if correct else "red"
            ax.set_title(
                f"True: {true_labels[i]}\nPred: {pred_labels[i]} ({confidences[i]:.1%})",
                fontsize=8,
                color=color,
            )
        ax.axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sample predictions grid saved to %s", output_path)
    return output_path


def plot_corruption_examples(
    clean_img: np.ndarray,
    corrupted_versions: List[Tuple[str, int, np.ndarray]],
    output_path: Path = Path("outputs/corruption_examples.png"),
) -> Path:
    """Show side-by-side examples of each corruption type.

    Args:
        clean_img: Original clean uint8 RGB NumPy array (H, W, 3).
        corrupted_versions: List of tuples (corruption_type, severity, image_array).
        output_path: Destination PNG file.

    Returns:
        Resolved path to the saved PNG.

    Example:
        >>> from src.data.corruption_pipeline import apply_corruption
        >>> corrupted = [
        ...     ("gaussian_blur", 2, apply_corruption(img, "gaussian_blur", 2)),
        ...     ("gaussian_noise", 2, apply_corruption(img, "gaussian_noise", 2)),
        ... ]
        >>> plot_corruption_examples(img, corrupted)
    """
    n_cols = 1 + len(corrupted_versions)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))
    fig.suptitle("Corruption Examples", fontsize=14, fontweight="bold")

    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(clean_img)
    axes[0].set_title("Clean", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for ax, (corruption_type, severity, img_arr) in zip(axes[1:], corrupted_versions):
        ax.imshow(img_arr)
        ax.set_title(f"{corruption_type}\n(sev={severity})", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Corruption examples saved to %s", output_path)
    return output_path


def plot_class_distribution(
    class_counts: dict,
    output_path: Path = Path("outputs/class_distribution.png"),
    title: str = "Dataset Class Distribution",
) -> Path:
    """Bar chart of sample counts per class.

    Args:
        class_counts: Dict mapping class name -> count.
        output_path: Destination PNG file.
        title: Chart title.

    Returns:
        Resolved path to the saved PNG.
    """
    names = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names, counts, color=sns.color_palette("tab10", len(names)))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Corruption Type")
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Class distribution chart saved to %s", output_path)
    return output_path
