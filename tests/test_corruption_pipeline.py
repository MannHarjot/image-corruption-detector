"""Unit tests for src/data/corruption_pipeline.py.

Tests verify that each corruption function:
  - Returns an array with the same shape as the input.
  - Returns a uint8 dtype array.
  - Actually modifies the image (i.e., the result differs from the original).
  - Handles all three severity levels without raising.
"""

import numpy as np
import pytest

from src.data.corruption_pipeline import (
    apply_brightness_shift,
    apply_contrast_reduction,
    apply_corruption,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_artifacts,
    apply_salt_pepper_noise,
    list_corruption_types,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=42)

@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a deterministic 64x64 RGB uint8 image."""
    return RNG.integers(30, 220, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def flat_image() -> np.ndarray:
    """Return a flat mid-gray image useful for contrast/brightness tests."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_output(original: np.ndarray, result: np.ndarray) -> None:
    """Assert shape, dtype, and that the result differs from the original."""
    assert result.shape == original.shape, (
        f"Shape mismatch: expected {original.shape}, got {result.shape}"
    )
    assert result.dtype == np.uint8, (
        f"dtype mismatch: expected uint8, got {result.dtype}"
    )
    assert not np.array_equal(result, original), (
        "Corruption had no effect — output is identical to input"
    )


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

class TestGaussianBlur:
    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_shape_dtype_severity(self, sample_image, severity):
        result = apply_gaussian_blur(sample_image, severity)
        _assert_valid_output(sample_image, result)

    def test_invalid_severity(self, sample_image):
        with pytest.raises(ValueError, match="severity"):
            apply_gaussian_blur(sample_image, severity=0)

    def test_higher_severity_more_blur(self, sample_image):
        """Higher severity should produce greater deviation from the original."""
        diff1 = np.abs(apply_gaussian_blur(sample_image, 1).astype(float) - sample_image.astype(float)).mean()
        diff3 = np.abs(apply_gaussian_blur(sample_image, 3).astype(float) - sample_image.astype(float)).mean()
        assert diff3 >= diff1, "Severity 3 should blur more than severity 1"


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------

class TestGaussianNoise:
    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_shape_dtype_severity(self, sample_image, severity):
        result = apply_gaussian_noise(sample_image, severity)
        _assert_valid_output(sample_image, result)

    def test_invalid_severity(self, sample_image):
        with pytest.raises(ValueError, match="severity"):
            apply_gaussian_noise(sample_image, severity=4)

    def test_values_in_valid_range(self, sample_image):
        result = apply_gaussian_noise(sample_image, severity=3)
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Salt-and-pepper noise
# ---------------------------------------------------------------------------

class TestSaltPepperNoise:
    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_shape_dtype_severity(self, sample_image, severity):
        result = apply_salt_pepper_noise(sample_image, severity)
        _assert_valid_output(sample_image, result)

    def test_salt_pixels_are_255(self, sample_image):
        result = apply_salt_pepper_noise(sample_image, severity=3)
        assert np.any(result == 255), "Salt pixels (255) should be present"

    def test_pepper_pixels_are_0(self, sample_image):
        result = apply_salt_pepper_noise(sample_image, severity=3)
        assert np.any(result == 0), "Pepper pixels (0) should be present"

    def test_higher_severity_more_corrupted_pixels(self, sample_image):
        result1 = apply_salt_pepper_noise(sample_image, 1)
        result3 = apply_salt_pepper_noise(sample_image, 3)
        diff1 = np.sum(result1 != sample_image)
        diff3 = np.sum(result3 != sample_image)
        assert diff3 >= diff1


# ---------------------------------------------------------------------------
# JPEG artifacts
# ---------------------------------------------------------------------------

class TestJpegArtifacts:
    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_shape_dtype_severity(self, sample_image, severity):
        result = apply_jpeg_artifacts(sample_image, severity)
        _assert_valid_output(sample_image, result)

    def test_values_in_valid_range(self, sample_image):
        result = apply_jpeg_artifacts(sample_image, severity=3)
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Brightness shift
# ---------------------------------------------------------------------------

class TestBrightnessShift:
    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_shape_dtype_severity(self, sample_image, severity):
        result = apply_brightness_shift(sample_image, severity)
        _assert_valid_output(sample_image, result)

    def test_severity1_increases_brightness(self, flat_image):
        result = apply_brightness_shift(flat_image, severity=1)
        assert result.mean() > flat_image.mean()

    def test_severity2_decreases_brightness(self, flat_image):
        result = apply_brightness_shift(flat_image, severity=2)
        assert result.mean() < flat_image.mean()

    def test_values_clamped(self, sample_image):
        result = apply_brightness_shift(sample_image, severity=3)
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Contrast reduction
# ---------------------------------------------------------------------------

class TestContrastReduction:
    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_shape_dtype_severity(self, sample_image, severity):
        result = apply_contrast_reduction(sample_image, severity)
        _assert_valid_output(sample_image, result)

    def test_reduces_std(self, sample_image):
        """Contrast reduction should lower the standard deviation of pixel values."""
        result = apply_contrast_reduction(sample_image, severity=2)
        assert result.std() < sample_image.std()

    def test_severity3_most_reduced(self, sample_image):
        std1 = apply_contrast_reduction(sample_image, 1).std()
        std3 = apply_contrast_reduction(sample_image, 3).std()
        assert std3 <= std1


# ---------------------------------------------------------------------------
# apply_corruption dispatcher
# ---------------------------------------------------------------------------

class TestApplyCorruption:
    def test_clean_returns_copy(self, sample_image):
        result = apply_corruption(sample_image, "clean", severity=1)
        assert np.array_equal(result, sample_image)
        assert result is not sample_image  # must be a copy

    @pytest.mark.parametrize("corruption_type", [
        "gaussian_blur", "gaussian_noise", "salt_pepper",
        "jpeg_artifacts", "brightness", "contrast",
    ])
    def test_all_types_return_valid_output(self, sample_image, corruption_type):
        result = apply_corruption(sample_image, corruption_type, severity=1)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_unknown_type_raises(self, sample_image):
        with pytest.raises(ValueError, match="Unknown corruption type"):
            apply_corruption(sample_image, "unknown_corruption", severity=1)


# ---------------------------------------------------------------------------
# list_corruption_types
# ---------------------------------------------------------------------------

class TestListCorruptionTypes:
    def test_returns_list(self):
        types = list_corruption_types()
        assert isinstance(types, list)

    def test_clean_is_first(self):
        assert list_corruption_types()[0] == "clean"

    def test_contains_all_expected(self):
        types = list_corruption_types()
        expected = {
            "clean", "gaussian_blur", "gaussian_noise", "salt_pepper",
            "jpeg_artifacts", "brightness", "contrast",
        }
        assert expected == set(types)

    def test_has_seven_types(self):
        assert len(list_corruption_types()) == 7
