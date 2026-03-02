"""CLI entry point for training the image corruption classifier.

Usage examples::

    python scripts/train.py
    python scripts/train.py --config config/default_config.yaml --epochs 30
    python scripts/train.py --batch-size 32 --lr 0.0005 --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ResNet-18 image corruption classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default_config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="Override batch size"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cpu', 'cuda', 'mps'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Override output directory for history and logs",
    )
    return parser.parse_args()


def main() -> None:
    """Train the model end-to-end."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load and optionally patch config
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = str(args.output_dir)

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lazy imports (keep startup fast)
    from src.data.dataset import CLASS_NAMES, create_dataloaders
    from src.models.resnet_classifier import get_model
    from src.training.trainer import Trainer

    data_cfg = cfg["data"]
    path_cfg = cfg["paths"]
    model_cfg = cfg["model"]

    metadata_csv = Path(path_cfg["data_dir"]) / "metadata.csv"
    if not metadata_csv.exists():
        print(
            f"ERROR: Dataset not found at {metadata_csv}.\n"
            "Run `python src/data/generate_dataset.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading dataset from {metadata_csv} ...")
    loaders = create_dataloaders(
        metadata_csv=metadata_csv,
        batch_size=cfg["training"]["batch_size"],
        image_size=data_cfg["image_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"] and device.type == "cuda",
        normalize_mean=tuple(data_cfg["normalize_mean"]),
        normalize_std=tuple(data_cfg["normalize_std"]),
    )

    print("Building model ...")
    model = get_model(
        num_classes=model_cfg["num_classes"],
        freeze_backbone=model_cfg["freeze_backbone"],
        dropout=model_cfg["dropout"],
        hidden_dim=model_cfg["hidden_dim"],
        device=device,
    )

    output_dir = Path(path_cfg["output_dir"])
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=cfg,
        output_dir=output_dir,
        device=device,
    )

    print("Starting training ...")
    history = trainer.train()

    print("\nRunning final evaluation on test set ...")
    best_ckpt = Path(path_cfg["checkpoint_dir"]) / "best_model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    _, test_metrics = trainer.evaluate(loaders["test"], class_names=CLASS_NAMES)
    print(f"\nTest accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1 : {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
