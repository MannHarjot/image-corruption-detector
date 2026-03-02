"""Generate the training dataset from CIFAR-10 base images.

Downloads CIFAR-10 via torchvision, applies the corruption pipeline to each
clean image, saves everything to disk in a structured directory, and writes a
metadata CSV.

Resulting layout::

    data/
    ├── train/
    │   ├── clean/
    │   ├── gaussian_blur/
    │   ├── gaussian_noise/
    │   ├── salt_pepper/
    │   ├── jpeg_artifacts/
    │   ├── brightness/
    │   └── contrast/
    ├── val/   (same structure)
    └── test/  (same structure)

Metadata CSV columns:
    filepath, label, corruption_type, severity, split
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

from src.data.corruption_pipeline import apply_corruption, list_corruption_types
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Label index mapping — deterministic and fixed
CLASS_NAMES: List[str] = [
    "clean",
    "gaussian_blur",
    "gaussian_noise",
    "salt_pepper",
    "jpeg_artifacts",
    "brightness",
    "contrast",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _load_config(config_path: Path) -> dict:
    """Load YAML config and return as dict.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _download_cifar10(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Download CIFAR-10 and return (images, labels) arrays.

    Images are returned as uint8 arrays of shape (N, 32, 32, 3) in RGB order.

    Args:
        root: Directory where torchvision will store the dataset.

    Returns:
        Tuple of (images array, labels array).
    """
    import torchvision.datasets as dsets

    logger.info("Downloading / loading CIFAR-10 from %s", root)
    train_ds = dsets.CIFAR10(root=str(root), train=True, download=True)
    test_ds = dsets.CIFAR10(root=str(root), train=False, download=True)

    train_images = np.array(train_ds.data, dtype=np.uint8)  # (50000, 32, 32, 3)
    test_images = np.array(test_ds.data, dtype=np.uint8)    # (10000, 32, 32, 3)
    images = np.concatenate([train_images, test_images], axis=0)
    labels = np.concatenate([
        np.array(train_ds.targets),
        np.array(test_ds.targets),
    ])
    logger.info("Loaded %d CIFAR-10 images", len(images))
    return images, labels


def _train_val_test_split(
    n: int,
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create shuffled index arrays for train/val/test splits.

    Args:
        n: Total number of samples.
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (train_indices, val_indices, test_indices).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        indices[:n_train],
        indices[n_train: n_train + n_val],
        indices[n_train + n_val:],
    )


def generate_dataset(
    config_path: Path = Path("config/default_config.yaml"),
    output_root: Optional[Path] = None,
    seed: int = 42,
) -> Path:
    """Generate the full corruption dataset from CIFAR-10.

    For each clean CIFAR-10 image (up to ``samples_per_class`` per corruption
    class), the function applies the corresponding corruption at each of the
    configured severity levels and saves the result as a PNG file.

    Args:
        config_path: Path to the YAML configuration file.
        output_root: Root directory for output. Defaults to config ``data_dir``.
        seed: Random seed used for train/val/test split.

    Returns:
        Path to the root data directory containing the generated dataset.
    """
    cfg = _load_config(config_path)
    data_cfg = cfg["data"]
    path_cfg = cfg["paths"]

    if output_root is None:
        output_root = Path(path_cfg["data_dir"])

    corruption_types: List[str] = data_cfg["corruption_types"]
    severities: List[int] = data_cfg["severities"]
    samples_per_class: int = data_cfg["samples_per_class"]
    train_frac: float = data_cfg["train_split"]
    val_frac: float = data_cfg["val_split"]

    # Only use the *non-clean* corruptions here; clean images are handled separately
    non_clean = [c for c in corruption_types if c != "clean"]

    # Download CIFAR-10
    raw_cache = output_root / "_cifar10_raw"
    images, _ = _download_cifar10(raw_cache)
    total_images = len(images)

    # Limit base pool
    pool_size = min(samples_per_class, total_images)
    rng = np.random.default_rng(seed)
    base_indices = rng.choice(total_images, size=pool_size, replace=False)
    base_images = images[base_indices]

    # Build metadata rows
    metadata_rows: List[dict] = []
    splits = ["train", "val", "test"]
    train_idx, val_idx, test_idx = _train_val_test_split(
        pool_size, train_frac, val_frac, seed
    )
    split_map = {
        **{i: "train" for i in train_idx},
        **{i: "val" for i in val_idx},
        **{i: "test" for i in test_idx},
    }

    # ------------------------------------------------------------------
    # Save clean images
    # ------------------------------------------------------------------
    logger.info("Generating clean images (%d samples)...", pool_size)
    for local_idx, img_arr in enumerate(base_images):
        split = split_map[local_idx]
        out_dir = output_root / split / "clean"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"clean_{local_idx:05d}.png"
        filepath = out_dir / filename
        Image.fromarray(img_arr).save(filepath)
        metadata_rows.append(
            {
                "filepath": str(filepath),
                "label": CLASS_TO_IDX["clean"],
                "corruption_type": "clean",
                "severity": 0,
                "split": split,
            }
        )

    # ------------------------------------------------------------------
    # Save corrupted images
    # ------------------------------------------------------------------
    for corruption_type in non_clean:
        logger.info("Generating %s corruptions...", corruption_type)
        for severity in severities:
            for local_idx, img_arr in enumerate(base_images):
                split = split_map[local_idx]
                out_dir = output_root / split / corruption_type
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{corruption_type}_s{severity}_{local_idx:05d}.png"
                filepath = out_dir / filename
                corrupted = apply_corruption(img_arr, corruption_type, severity)
                Image.fromarray(corrupted).save(filepath)
                metadata_rows.append(
                    {
                        "filepath": str(filepath),
                        "label": CLASS_TO_IDX[corruption_type],
                        "corruption_type": corruption_type,
                        "severity": severity,
                        "split": split,
                    }
                )

    # ------------------------------------------------------------------
    # Write metadata CSV
    # ------------------------------------------------------------------
    csv_path = output_root / "metadata.csv"
    fieldnames = ["filepath", "label", "corruption_type", "severity", "split"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    logger.info(
        "Dataset generation complete. %d total samples saved to %s",
        len(metadata_rows),
        output_root,
    )
    _log_split_stats(metadata_rows)
    return output_root


def _log_split_stats(rows: List[dict]) -> None:
    """Log per-split sample counts.

    Args:
        rows: List of metadata row dicts.
    """
    from collections import Counter

    split_counts = Counter(r["split"] for r in rows)
    for split, count in sorted(split_counts.items()):
        logger.info("  %s: %d samples", split, count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate image corruption dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output directory (default: from config)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_dataset(config_path=args.config, output_root=args.output, seed=args.seed)
