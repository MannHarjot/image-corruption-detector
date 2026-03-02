"""ResNet-18 transfer learning model for image corruption classification.

Strategy:
  - Load pretrained ResNet-18 weights (ImageNet).
  - Optionally freeze early layers (conv1, layer1, layer2).
  - Fine-tune layer3 and layer4.
  - Replace the final FC layer with a two-layer classification head:
      512 -> 256 (ReLU, Dropout) -> num_classes.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CorruptionClassifier(nn.Module):
    """ResNet-18 backbone with a custom two-layer classification head.

    The backbone is loaded with ImageNet pretrained weights. Early convolutional
    layers (``conv1``, ``bn1``, ``layer1``, ``layer2``) are optionally frozen so
    that only ``layer3``, ``layer4``, and the new head are updated during training.

    Args:
        num_classes: Number of output classes (default: 7).
        freeze_backbone: If ``True``, freeze conv1 through layer2 and only
            fine-tune layer3, layer4, and the classification head.
        dropout: Dropout probability applied in the classification head.
        hidden_dim: Size of the intermediate hidden layer in the head.

    Attributes:
        backbone: The ResNet-18 feature extractor (minus the original FC layer).
        classifier: Two-layer MLP head replacing ResNet's default FC.

    Example:
        >>> model = CorruptionClassifier(num_classes=7, freeze_backbone=True)
        >>> out = model(torch.randn(4, 3, 224, 224))
        >>> out.shape
        torch.Size([4, 7])
    """

    def __init__(
        self,
        num_classes: int = 7,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # Load pretrained ResNet-18
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Remove the original classification head; keep feature extractor
        # ResNet-18 feature dim = 512
        feature_dim = resnet.fc.in_features  # 512

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        # Custom classification head: 512 -> 256 -> num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_backbone:
            self._freeze_early_layers(resnet)

        self._log_parameter_summary()

    def _freeze_early_layers(self, resnet: models.ResNet) -> None:
        """Freeze conv1, bn1, layer1, and layer2 of the backbone.

        Args:
            resnet: The original ResNet-18 module (used to identify layers).
        """
        # Layers in backbone sequential order: conv1(0), bn1(1), relu(2),
        # maxpool(3), layer1(4), layer2(5), layer3(6), layer4(7), avgpool(8)
        freeze_up_to = 6  # freeze indices 0-5 (layer3 and later are trainable)
        for i, child in enumerate(self.backbone.children()):
            if i < freeze_up_to:
                for param in child.parameters():
                    param.requires_grad = False
        logger.info(
            "Backbone frozen: conv1, bn1, layer1, layer2 are non-trainable. "
            "layer3, layer4, and classifier head are trainable."
        )

    def _log_parameter_summary(self) -> None:
        """Log total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Model parameters: total=%s, trainable=%s (%.1f%%)",
            f"{total:,}",
            f"{trainable:,}",
            100.0 * trainable / total,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Logit tensor of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 512-dim backbone feature vector (before the head).

        Useful for embedding visualization (e.g. t-SNE).

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Feature tensor of shape (B, 512).
        """
        features = self.backbone(x)
        return features.flatten(start_dim=1)


def get_model(
    num_classes: int = 7,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
    hidden_dim: int = 256,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> CorruptionClassifier:
    """Factory function to create (and optionally restore) a CorruptionClassifier.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze early backbone layers.
        dropout: Dropout probability in the classification head.
        hidden_dim: Hidden layer size in the classification head.
        checkpoint_path: If provided, load model weights from this ``.pt`` file.
        device: Target device. Defaults to CUDA if available, else CPU.

    Returns:
        Ready-to-use :class:`CorruptionClassifier` moved to *device*.

    Example:
        >>> model = get_model(num_classes=7, freeze_backbone=True)
        >>> model = get_model(checkpoint_path="checkpoints/best_model.pt")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = CorruptionClassifier(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        hidden_dim=hidden_dim,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        # Support checkpoints saved as {"model_state_dict": ...} or raw state dict
        state_dict: Dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        logger.info("Loaded weights from checkpoint: %s", checkpoint_path)

    model.to(device)
    return model
