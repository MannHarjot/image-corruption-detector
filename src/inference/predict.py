"""Single-image and batch inference with confidence scores.

Provides a clean API for loading a trained checkpoint and running predictions
on individual images or entire directories.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import CLASS_NAMES, IDX_TO_CLASS, get_default_transforms
from src.models.resnet_classifier import CorruptionClassifier, get_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> CorruptionClassifier:
    """Load a CorruptionClassifier from a saved checkpoint file.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint saved by the Trainer.
        device: Target device. Auto-detects CUDA/MPS/CPU if ``None``.

    Returns:
        Loaded model in evaluation mode, moved to *device*.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.

    Example:
        >>> model = load_model_from_checkpoint("checkpoints/best_model.pt")
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, map_location=device)
    cfg = state.get("config", {})
    model_cfg = cfg.get("model", {})

    model = get_model(
        num_classes=model_cfg.get("num_classes", len(CLASS_NAMES)),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        dropout=model_cfg.get("dropout", 0.3),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        device=device,
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    logger.info("Model loaded from %s (epoch=%s)", checkpoint_path, state.get("epoch"))
    return model


def predict_single(
    image_path: Union[str, Path],
    model: CorruptionClassifier,
    transform=None,
    device: Optional[torch.device] = None,
    image_size: int = 224,
) -> Dict:
    """Run inference on a single image and return detailed predictions.

    Args:
        image_path: Path to the input image file.
        model: A trained :class:`~src.models.resnet_classifier.CorruptionClassifier`
            in evaluation mode.
        transform: Optional torchvision transform pipeline. Defaults to the
            standard validation/test transform (resize + normalize).
        device: Compute device. Uses the model's device if ``None``.
        image_size: Image resize target (used only when *transform* is ``None``).

    Returns:
        Dictionary with keys:
            - ``"filepath"`` (str): Absolute path to the input image.
            - ``"predicted_class"`` (str): Top-1 predicted class name.
            - ``"predicted_index"`` (int): Top-1 class index.
            - ``"confidence"`` (float): Softmax probability of top-1 prediction.
            - ``"top3"`` (list[dict]): Top-3 predictions, each with
              ``"class"``, ``"index"``, and ``"confidence"``.

    Raises:
        FileNotFoundError: If *image_path* does not exist.

    Example:
        >>> result = predict_single("photo.jpg", model)
        >>> print(result["predicted_class"], result["confidence"])
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if device is None:
        device = next(model.parameters()).device

    if transform is None:
        transform = get_default_transforms(image_size=image_size, split="test")

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)                          # (1, num_classes)
        probs = F.softmax(logits, dim=1).squeeze(0)    # (num_classes,)

    top3_probs, top3_idx = probs.topk(3)
    top3 = [
        {
            "class": IDX_TO_CLASS[int(idx)],
            "index": int(idx),
            "confidence": float(prob),
        }
        for idx, prob in zip(top3_idx.tolist(), top3_probs.tolist())
    ]

    return {
        "filepath": str(image_path.resolve()),
        "predicted_class": top3[0]["class"],
        "predicted_index": top3[0]["index"],
        "confidence": top3[0]["confidence"],
        "top3": top3,
    }


def predict_batch(
    directory_path: Union[str, Path],
    model: CorruptionClassifier,
    transform=None,
    device: Optional[torch.device] = None,
    image_size: int = 224,
    recursive: bool = False,
) -> List[Dict]:
    """Run inference on all images in a directory.

    Args:
        directory_path: Path to a directory containing image files.
        model: Trained :class:`~src.models.resnet_classifier.CorruptionClassifier`.
        transform: Optional torchvision transform. Uses default if ``None``.
        device: Compute device. Auto-detected if ``None``.
        image_size: Target image size for default transform.
        recursive: If ``True``, search subdirectories recursively.

    Returns:
        List of prediction dicts (one per image), same format as
        :func:`predict_single`.

    Raises:
        NotADirectoryError: If *directory_path* is not a directory.
    """
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    glob_pattern = "**/*" if recursive else "*"
    image_paths = sorted(
        p for p in directory_path.glob(glob_pattern)
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    if not image_paths:
        logger.warning("No images found in %s", directory_path)
        return []

    logger.info("Running batch inference on %d images ...", len(image_paths))
    results = []
    for path in image_paths:
        try:
            result = predict_single(path, model, transform, device, image_size)
            results.append(result)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", path, exc)

    logger.info("Batch inference complete. %d results.", len(results))
    return results


def save_predictions(
    results: List[Dict],
    output_path: Union[str, Path],
    fmt: str = "json",
) -> None:
    """Save prediction results to disk.

    Args:
        results: List of prediction dicts from :func:`predict_single` or
            :func:`predict_batch`.
        output_path: Destination file path.
        fmt: Output format — ``"json"`` (default) or ``"csv"``.

    Raises:
        ValueError: If *fmt* is not ``"json"`` or ``"csv"``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    elif fmt == "csv":
        import csv
        fieldnames = ["filepath", "predicted_class", "predicted_index", "confidence"]
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    else:
        raise ValueError(f"Unsupported format {fmt!r}. Choose 'json' or 'csv'.")

    logger.info("Predictions saved to %s (%s)", output_path, fmt)
