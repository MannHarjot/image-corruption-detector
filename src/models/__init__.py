"""Model definitions for corruption classification."""

from .resnet_classifier import CorruptionClassifier, get_model

__all__ = ["CorruptionClassifier", "get_model"]
