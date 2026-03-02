"""CLI entry point for running inference on new images.

Usage::

    # Single image
    python scripts/predict.py --model-path checkpoints/best_model.pt \\
        --image-path photo.jpg

    # Directory of images, output as CSV
    python scripts/predict.py --model-path checkpoints/best_model.pt \\
        --input-dir my_images/ --output-format csv \\
        --output-path outputs/predictions.csv
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corruption classification inference on images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        dest="model_path",
        help="Path to saved .pt checkpoint",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image-path",
        type=Path,
        dest="image_path",
        help="Path to a single image file",
    )
    source.add_argument(
        "--input-dir",
        type=Path,
        dest="input_dir",
        help="Directory of image files for batch inference",
    )

    parser.add_argument(
        "--output-format",
        choices=["json", "csv"],
        default="json",
        dest="output_format",
        help="Output file format",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/predictions.json"),
        dest="output_path",
        help="Destination path for results file",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override: 'cpu', 'cuda', 'mps'"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search input-dir recursively for images"
    )
    return parser.parse_args()


def main() -> None:
    """Run inference and display / save results."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = _parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    from src.inference.predict import (
        load_model_from_checkpoint,
        predict_batch,
        predict_single,
        save_predictions,
    )

    print(f"Loading model from {args.model_path} ...")
    model = load_model_from_checkpoint(args.model_path, device=device)

    if args.image_path is not None:
        result = predict_single(args.image_path, model, device=device)
        results = [result]
        print(f"\nFile      : {result['filepath']}")
        print(f"Prediction: {result['predicted_class']} ({result['confidence']:.1%})")
        print("Top-3:")
        for entry in result["top3"]:
            print(f"  {entry['class']:<20} {entry['confidence']:.4f}")
    else:
        results = predict_batch(
            args.input_dir,
            model,
            device=device,
            recursive=args.recursive,
        )
        print(f"\nProcessed {len(results)} images.")

    save_predictions(results, args.output_path, fmt=args.output_format)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
