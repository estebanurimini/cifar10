"""OwnVGG — CIFAR10-adapted VGG implementation (no pretrained weights).

CIFAR10 adaptation: the first two max-pool layers are removed because
32×32 images are already very small. This preserves spatial resolution
and gives better accuracy compared to standard VGG with 5 pooling layers.

Architecture only — no training code. Import and use with the training framework.
"""

import torch
import torch.nn as nn

from typing import Literal

# ---------------------------------------------------------------------------
# VGG config dictionary — adapted for CIFAR10 by removing first 2 max-pools
# Standard VGG pools 5 times (32→16→8→4→2→1).
# CIFAR VGG pools 3 times  (32→16→8→4) for better small-image accuracy.
# ---------------------------------------------------------------------------

VGG_CIFAR_CONFIGS = {
    "vgg11_bn": {
        "conv_cfg": [
            64,
            128,
            "M",
            256, 256,
            "M",
            512, 512,
            "M",
            512, 512,
        ],
    },
    "vgg16_bn": {
        "conv_cfg": [
            64, 64,
            128, 128,
            "M",
            256, 256, 256,
            "M",
            512, 512, 512,
            "M",
            512, 512, 512,
        ],
    },
}


def _make_vgg_layers(cfg: list, batch_norm: bool = True) -> nn.Sequential:
    """Build convolutional feature extractor from VGG config.

    Args:
        cfg: List where int = out_channels, "M" = max-pool.
        batch_norm: Whether to insert BatchNorm after each conv.

    Returns:
        A ``nn.Sequential`` feature extractor.
    """
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class OwnVGG(nn.Module):
    """VGG for CIFAR10 (own implementation, no pretrained weights).

    Adapted for small images (32×32):
        - Removes first 2 max-pool layers (standard VGG pools 5× → pools 3×).
        - BatchNorm after every convolutional layer.
        - Classifier with dropout for regularization.

    Args:
        variant: One of ``"vgg11_bn"`` or ``"vgg16_bn"``.
        num_classes: Number of output classes.
        dropout: Dropout probability in the classifier.
    """

    def __init__(
        self,
        variant: Literal["vgg11_bn", "vgg16_bn"] = "vgg16_bn",
        num_classes: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        if variant not in VGG_CIFAR_CONFIGS:
            raise ValueError(
                f"Unknown VGG variant '{variant}'. "
                f"Available: {list(VGG_CIFAR_CONFIGS)}"
            )

        self.variant = variant
        cfg = VGG_CIFAR_CONFIGS[variant]["conv_cfg"]

        # Feature extractor
        self.features = _make_vgg_layers(cfg, batch_norm=True)

        # Compute the spatial size after conv layers
        # Input: 32×32. With the CIFAR-adapted config (3 max-pools):
        #   32 → (after M) 16 → (after M) 8 → (after M) 4
        # Final channels = last non-M entry in cfg
        last_channels = [v for v in cfg if v != "M"][-1]
        self.final_spatial = 4  # 32 / (2^3) — three max-pools

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(last_channels * self.final_spatial * self.final_spatial, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x