"""Image Corruption Detector — Interactive Demo

Gradio application for Hugging Face Spaces deployment.
Entry point: this file must be named app.py and live at the repo root.

Tabs:
  1. Corruption Explorer  — apply all 6 corruptions to any uploaded image
  2. Corruption Classifier — predict the corruption type using ResNet-18
"""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = [
    "clean",
    "gaussian_blur",
    "gaussian_noise",
    "salt_pepper",
    "jpeg_artifacts",
    "brightness",
    "contrast",
]
CLASS_DISPLAY: List[str] = [
    "Clean",
    "Gaussian Blur",
    "Gaussian Noise",
    "Salt & Pepper",
    "JPEG Artifacts",
    "Brightness Shift",
    "Contrast Reduction",
]
CLASS_TO_DISPLAY: Dict[str, str] = dict(zip(CLASS_NAMES, CLASS_DISPLAY))

CORRUPTION_DISPLAY_NAMES: Dict[str, str] = {
    k: v for k, v in CLASS_TO_DISPLAY.items() if k != "clean"
}

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SEVERITY_DESCRIPTIONS = {
    1: "Mild — subtle corruption, barely noticeable",
    2: "Moderate — clearly visible degradation",
    3: "Severe — heavy corruption, significant quality loss",
}

# ─────────────────────────────────────────────────────────────────────────────
# Corruption pipeline (self-contained — no src/ import needed)
# ─────────────────────────────────────────────────────────────────────────────

_BLUR_KERNELS = {1: 5, 2: 9, 3: 15}
_NOISE_SIGMAS = {1: 10.0, 2: 25.0, 3: 50.0}
_SP_DENSITIES = {1: 0.02, 2: 0.05, 3: 0.10}
_JPEG_QUALITY = {1: 20, 2: 10, 3: 5}
_BRIGHT_DELTA = {1: 40, 2: 80, 3: 120}
_CONTRAST_FAC = {1: 0.70, 2: 0.40, 3: 0.15}


def _u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _corrupt_gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = _BLUR_KERNELS[severity]
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)


def _corrupt_gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    noise = np.random.default_rng().normal(0.0, _NOISE_SIGMAS[severity], img.shape)
    return _u8(img.astype(np.float32) + noise)


def _corrupt_salt_pepper(img: np.ndarray, severity: int) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    n = int(_SP_DENSITIES[severity] / 2 * h * w)
    rng = np.random.default_rng()
    out[rng.integers(0, h, n), rng.integers(0, w, n)] = 255
    out[rng.integers(0, h, n), rng.integers(0, w, n)] = 0
    return out


def _corrupt_jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=_JPEG_QUALITY[severity])
    buf.seek(0)
    return np.array(Image.open(buf))


def _corrupt_brightness(img: np.ndarray, severity: int) -> np.ndarray:
    sign = 1 if severity % 2 == 1 else -1
    return _u8(img.astype(np.int16) + sign * _BRIGHT_DELTA[severity])


def _corrupt_contrast(img: np.ndarray, severity: int) -> np.ndarray:
    return _u8(128.0 + _CONTRAST_FAC[severity] * (img.astype(np.float32) - 128.0))


CORRUPTION_FNS = {
    "gaussian_blur": _corrupt_gaussian_blur,
    "gaussian_noise": _corrupt_gaussian_noise,
    "salt_pepper": _corrupt_salt_pepper,
    "jpeg_artifacts": _corrupt_jpeg,
    "brightness": _corrupt_brightness,
    "contrast": _corrupt_contrast,
}


def apply_all_corruptions(
    img_arr: np.ndarray, severity: int
) -> List[Tuple[Image.Image, str]]:
    """Apply every corruption type at *severity* and return (PIL Image, label) pairs."""
    results = []
    for key, fn in CORRUPTION_FNS.items():
        corrupted = fn(img_arr, severity)
        label = f"{CORRUPTION_DISPLAY_NAMES[key]}  (severity {severity})"
        results.append((Image.fromarray(corrupted), label))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

class CorruptionClassifier(nn.Module):
    """ResNet-18 with a two-layer classification head (matches training code)."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.3) -> None:
        super().__init__()
        resnet = models.resnet18(weights=None)
        feature_dim = resnet.fc.in_features  # 512
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (tries local paths, then HF Hub)
# ─────────────────────────────────────────────────────────────────────────────

_model: Optional[CorruptionClassifier] = None
_device = torch.device("cpu")


def _load_model() -> Optional[CorruptionClassifier]:
    """Attempt to load model weights from local paths or HF Hub."""
    local_candidates = [
        Path("best_model.pt"),
        Path("checkpoints/best_model.pt"),
    ]

    for path in local_candidates:
        if path.exists():
            try:
                state = torch.load(path, map_location=_device, weights_only=False)
                state_dict = state.get("model_state_dict", state)
                m = CorruptionClassifier()
                m.load_state_dict(state_dict)
                m.eval()
                print(f"[app] Model loaded from {path}")
                return m
            except Exception as exc:
                print(f"[app] Failed to load {path}: {exc}")

    # Try HF Hub if MODEL_REPO_ID env var is set
    repo_id = os.environ.get("MODEL_REPO_ID", "").strip()
    if repo_id:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo_id, filename="best_model.pt")
            state = torch.load(path, map_location=_device, weights_only=False)
            state_dict = state.get("model_state_dict", state)
            m = CorruptionClassifier()
            m.load_state_dict(state_dict)
            m.eval()
            print(f"[app] Model loaded from HF Hub: {repo_id}")
            return m
        except Exception as exc:
            print(f"[app] HF Hub load failed: {exc}")

    print("[app] No model weights found — classifier tab will show a warning.")
    return None


_model = _load_model()
MODEL_AVAILABLE = _model is not None

# ─────────────────────────────────────────────────────────────────────────────
# Gradio callback functions
# ─────────────────────────────────────────────────────────────────────────────

def explorer_fn(
    pil_image: Optional[Image.Image],
    severity: int,
) -> List[Tuple[Image.Image, str]]:
    """Return corrupted versions of the uploaded image for all 6 types."""
    if pil_image is None:
        return []
    img_arr = np.array(pil_image.convert("RGB"))
    return apply_all_corruptions(img_arr, int(severity))


def classifier_fn(
    pil_image: Optional[Image.Image],
) -> Dict[str, float]:
    """Return softmax probability dict for each class."""
    if pil_image is None or _model is None:
        return {d: 0.0 for d in CLASS_DISPLAY}
    img_rgb = pil_image.convert("RGB")
    tensor = IMAGENET_TRANSFORM(img_rgb).unsqueeze(0).to(_device)
    with torch.inference_mode():
        probs = F.softmax(_model(tensor), dim=1).squeeze(0)
    return {CLASS_DISPLAY[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# ─────────────────────────────────────────────────────────────────────────────
# UI strings
# ─────────────────────────────────────────────────────────────────────────────

HEADER_HTML = """
<div style="text-align:center; padding:24px 0 12px;">
  <h1 style="font-size:2.2rem; font-weight:800; margin:0; letter-spacing:-0.5px;">
    🔍 Image Corruption Detector
  </h1>
  <p style="font-size:1.05rem; color:#555; margin:8px 0 14px;">
    ResNet-18 transfer learning &nbsp;·&nbsp; 7-class corruption classification &nbsp;·&nbsp; PyTorch
  </p>
  <div style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap;">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white"/>
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"/>
    <img src="https://img.shields.io/badge/License-MIT-22c55e"/>
  </div>
</div>
"""

NO_MODEL_BANNER = """
<div style="background:#fff7ed; border:1px solid #f97316; border-radius:10px;
            padding:14px 18px; margin:0 0 16px; font-size:0.95rem;">
  ⚠️ <b>Classifier unavailable</b> — model weights (<code>best_model.pt</code>) not found.<br>
  Train the model locally with <code>python scripts/train.py</code>, then upload
  <code>best_model.pt</code> to this Space (Settings → Files) or set the
  <code>MODEL_REPO_ID</code> environment variable to your HF model repo.
</div>
"""

EXPLORER_DESCRIPTION = """
Upload **any image** to instantly see how all 6 corruption types look at your chosen severity level.
No model required — this runs entirely on the OpenCV corruption pipeline.

| Severity | Description |
|---|---|
| **1** | Mild — subtle, barely noticeable |
| **2** | Moderate — clearly visible degradation |
| **3** | Severe — heavy corruption, significant loss |
"""

CLASSIFIER_DESCRIPTION = """
Upload an image and the **ResNet-18 classifier** predicts which corruption type is present
(or classifies it as clean). Try uploading one of the corrupted outputs from the Explorer tab!

*The model was fine-tuned on CIFAR-10 images corrupted with the same pipeline shown above.*
"""

FOOTER_MD = """
---
<div style="text-align:center; font-size:0.88rem; color:#888; padding:8px 0;">
  Built with PyTorch · OpenCV · Gradio &nbsp;|&nbsp;
  <a href="https://github.com/MannHarjot/image-corruption-detector" target="_blank">GitHub</a>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Gradio interface
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    title="Image Corruption Detector",
    css="""
        .gradio-container { max-width: 1100px !important; }
        .tab-nav button { font-size: 1rem !important; font-weight: 600; }
        footer { display: none !important; }
    """,
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Tab 1: Corruption Explorer ─────────────────────────────────────
        with gr.Tab("🎨  Corruption Explorer"):
            gr.Markdown(EXPLORER_DESCRIPTION)

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    explorer_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        sources=["upload", "webcam", "clipboard"],
                        height=260,
                    )
                    severity_slider = gr.Slider(
                        minimum=1, maximum=3, step=1, value=2,
                        label="Corruption Severity",
                    )
                    explore_btn = gr.Button(
                        "▶  Apply All Corruptions", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    explorer_gallery = gr.Gallery(
                        label="Corrupted Outputs  (6 types)",
                        columns=3,
                        rows=2,
                        height=440,
                        object_fit="contain",
                        show_label=True,
                        show_download_button=True,
                    )

            explore_btn.click(
                fn=explorer_fn,
                inputs=[explorer_input, severity_slider],
                outputs=explorer_gallery,
            )
            explorer_input.change(
                fn=explorer_fn,
                inputs=[explorer_input, severity_slider],
                outputs=explorer_gallery,
            )
            severity_slider.change(
                fn=explorer_fn,
                inputs=[explorer_input, severity_slider],
                outputs=explorer_gallery,
            )

        # ── Tab 2: Corruption Classifier ───────────────────────────────────
        with gr.Tab("🤖  Corruption Classifier"):
            if not MODEL_AVAILABLE:
                gr.HTML(NO_MODEL_BANNER)

            gr.Markdown(CLASSIFIER_DESCRIPTION)

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    classifier_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        sources=["upload", "webcam", "clipboard"],
                        height=260,
                    )
                    classify_btn = gr.Button(
                        "▶  Classify Image",
                        variant="primary" if MODEL_AVAILABLE else "secondary",
                        size="lg",
                        interactive=MODEL_AVAILABLE,
                    )

                with gr.Column(scale=1):
                    classifier_output = gr.Label(
                        label="Corruption Type — Confidence Scores",
                        num_top_classes=7,
                    )
                    gr.Markdown("""
                    **How to read:** The bar lengths show the model's confidence for each
                    class. A well-trained model should show a dominant bar for the
                    correct corruption type.
                    """)

            classify_btn.click(
                fn=classifier_fn,
                inputs=classifier_input,
                outputs=classifier_output,
            )
            classifier_input.change(
                fn=classifier_fn,
                inputs=classifier_input,
                outputs=classifier_output,
            )

    gr.HTML(FOOTER_MD)


if __name__ == "__main__":
    demo.launch(show_error=True)
