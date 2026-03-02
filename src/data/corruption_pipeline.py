"""Image corruption functions using OpenCV and NumPy.

Each function accepts a uint8 NumPy array in HWC (height, width, channels)
RGB format and returns a corrupted image of the same shape and dtype.
Severity levels 1-3 map to mild, moderate, and severe corruption respectively.
"""

import io
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Severity parameter tables
# ---------------------------------------------------------------------------
_BLUR_KERNELS: Dict[int, int] = {1: 5, 2: 9, 3: 15}
_NOISE_SIGMAS: Dict[int, float] = {1: 10.0, 2: 25.0, 3: 50.0}
_SALT_PEPPER_DENSITIES: Dict[int, float] = {1: 0.02, 2: 0.05, 3: 0.10}
_JPEG_QUALITIES: Dict[int, int] = {1: 20, 2: 10, 3: 5}
_BRIGHTNESS_DELTAS: Dict[int, int] = {1: 40, 2: 80, 3: 120}
_CONTRAST_FACTORS: Dict[int, float] = {1: 0.7, 2: 0.4, 3: 0.15}


def _validate_severity(severity: int) -> None:
    """Raise ValueError if severity is not in {1, 2, 3}.

    Args:
        severity: Severity level to validate.

    Raises:
        ValueError: If severity is outside the valid range.
    """
    if severity not in (1, 2, 3):
        raise ValueError(f"severity must be 1, 2, or 3; got {severity!r}")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip and cast image to uint8.

    Args:
        image: Input array (any numeric dtype).

    Returns:
        uint8 array clipped to [0, 255].
    """
    return np.clip(image, 0, 255).astype(np.uint8)


def apply_gaussian_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Apply Gaussian blur to simulate out-of-focus or motion blur.

    Args:
        image: Input image as uint8 HWC NumPy array (RGB).
        severity: Blur strength — 1 (kernel 5), 2 (kernel 9), 3 (kernel 15).

    Returns:
        Blurred image with the same shape and dtype as *image*.

    Example:
        >>> blurred = apply_gaussian_blur(img, severity=2)
    """
    _validate_severity(severity)
    k = _BLUR_KERNELS[severity]
    return cv2.GaussianBlur(image, (k, k), sigmaX=0)


def apply_gaussian_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Add additive Gaussian (white) noise to the image.

    Args:
        image: Input image as uint8 HWC NumPy array (RGB).
        severity: Noise strength — 1 (sigma=10), 2 (sigma=25), 3 (sigma=50).

    Returns:
        Noisy image with the same shape and dtype as *image*.
    """
    _validate_severity(severity)
    sigma = _NOISE_SIGMAS[severity]
    rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    return _to_uint8(image.astype(np.float32) + noise)


def apply_salt_pepper_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Apply salt-and-pepper impulse noise to the image.

    Args:
        image: Input image as uint8 HWC NumPy array (RGB).
        severity: Density of corrupted pixels — 1 (2%), 2 (5%), 3 (10%).

    Returns:
        Image with randomly zeroed (pepper) and maxed (salt) pixels,
        same shape and dtype as *image*.
    """
    _validate_severity(severity)
    density = _SALT_PEPPER_DENSITIES[severity]
    corrupted = image.copy()
    rng = np.random.default_rng()
    h, w = image.shape[:2]
    total_pixels = h * w

    num_salt = int(density / 2 * total_pixels)
    salt_coords = (
        rng.integers(0, h, num_salt),
        rng.integers(0, w, num_salt),
    )
    corrupted[salt_coords] = 255

    num_pepper = int(density / 2 * total_pixels)
    pepper_coords = (
        rng.integers(0, h, num_pepper),
        rng.integers(0, w, num_pepper),
    )
    corrupted[pepper_coords] = 0

    return corrupted


def apply_jpeg_artifacts(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulate JPEG compression artifacts by encoding/decoding at low quality.

    Args:
        image: Input image as uint8 HWC NumPy array (RGB).
        severity: Compression level — 1 (quality=20), 2 (quality=10), 3 (quality=5).

    Returns:
        Compressed-and-decompressed image with JPEG blocking artifacts,
        same shape and dtype as *image*.
    """
    _validate_severity(severity)
    quality = _JPEG_QUALITIES[severity]
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))


def apply_brightness_shift(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Shift image brightness to simulate over- or under-exposure.

    Severity 1 and 3 increase brightness; severity 2 decreases it,
    producing a balanced mix across the dataset.

    Args:
        image: Input image as uint8 HWC NumPy array (RGB).
        severity: Magnitude — 1 (+40), 2 (-80), 3 (+120).

    Returns:
        Brightness-shifted image clipped to [0, 255].
    """
    _validate_severity(severity)
    delta = _BRIGHTNESS_DELTAS[severity]
    sign = 1 if severity % 2 == 1 else -1
    return _to_uint8(image.astype(np.int16) + sign * delta)


def apply_contrast_reduction(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Reduce image contrast by scaling pixel values toward mid-gray (128).

    Args:
        image: Input image as uint8 HWC NumPy array (RGB).
        severity: Degree of contrast reduction — 1 (0.7x), 2 (0.4x), 3 (0.15x).

    Returns:
        Contrast-reduced image with same shape and dtype as *image*.
    """
    _validate_severity(severity)
    factor = _CONTRAST_FACTORS[severity]
    img_float = image.astype(np.float32)
    reduced = 128.0 + factor * (img_float - 128.0)
    return _to_uint8(reduced)


def apply_corruption(
    image: np.ndarray,
    corruption_type: str,
    severity: int = 1,
) -> np.ndarray:
    """Apply a named corruption to an image at the given severity.

    This is the primary entry-point for the corruption pipeline.
    ``"clean"`` is a no-op and returns a copy of the original image.

    Args:
        image: Input image as uint8 HWC NumPy array (RGB).
        corruption_type: One of ``"clean"``, ``"gaussian_blur"``,
            ``"gaussian_noise"``, ``"salt_pepper"``, ``"jpeg_artifacts"``,
            ``"brightness"``, ``"contrast"``.
        severity: Corruption severity level (1-3).

    Returns:
        Corrupted (or clean copy) image with same shape and dtype as *image*.

    Raises:
        ValueError: If *corruption_type* is not recognised.

    Example:
        >>> corrupted = apply_corruption(img, "gaussian_noise", severity=3)
    """
    if corruption_type == "clean":
        return image.copy()
    if corruption_type not in CORRUPTION_REGISTRY:
        valid = ", ".join(["clean"] + sorted(CORRUPTION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown corruption type {corruption_type!r}. Valid types: {valid}"
        )
    return CORRUPTION_REGISTRY[corruption_type](image, severity)


def list_corruption_types() -> List[str]:
    """Return all supported corruption type names including ``"clean"``.

    Returns:
        List of corruption type strings, ``"clean"`` first then alphabetical.
    """
    return ["clean"] + sorted(CORRUPTION_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Registry — maps corruption_type string -> function
# ---------------------------------------------------------------------------
CORRUPTION_REGISTRY: Dict[str, callable] = {
    "gaussian_blur": apply_gaussian_blur,
    "gaussian_noise": apply_gaussian_noise,
    "salt_pepper": apply_salt_pepper_noise,
    "jpeg_artifacts": apply_jpeg_artifacts,
    "brightness": apply_brightness_shift,
    "contrast": apply_contrast_reduction,
}
