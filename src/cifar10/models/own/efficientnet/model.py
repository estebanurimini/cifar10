"""OwnEfficientNetV2 — EfficientNet-V2 from scratch for CIFAR10 with 32×32 inputs.

A simplified EfficientNetV2-style architecture designed for CIFAR10,
trained from scratch (no pretrained weights). Uses FusedMBConv in early
stages and MBConv+SE in later stages, following the V2 design philosophy.

Key differences from torchvision's EfficientNetV2:
    - Stride-1 stem (preserves 32×32 resolution)
    - Fewer stages with gentler downsampling trajectory
    - Smaller channel widths and fewer blocks per stage
    - Zero-initialized projection BN for better gradient flow
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar10.models.own.blocks import drop_path


# =========================================================
# Squeeze-and-Excitation
# =========================================================

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block with reduction ratio 4.

    Args:
        in_channels: Number of input channels.
        reduced_channels: Number of channels in the bottleneck
            (``in_channels // reduction_ratio``).
    """

    def __init__(self, in_channels: int, reduced_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global Average Pooling (spatial squeeze)
        y = x.mean(dim=(2, 3), keepdim=True)
        # Bottleneck
        y = F.silu(self.fc1(y))
        # Gating
        y = torch.sigmoid(self.fc2(y))
        # Scale
        return x * y


# =========================================================
# FusedMBConv Block (EfficientNet-V2 innovation)
# =========================================================

class FusedMBConv(nn.Module):
    """Fused Mobile Inverted Bottleneck block (EfficientNet-V2).

    Replaces the expand → depthwise → project pattern with a single
    3×3 conv → 1×1 proj, which is more efficient for early stages.

        1. Fused expand: 3×3 conv (with optional stride)
        2. Project: 1×1 pointwise back to ``out_channels`` (no activation)

    Residual connection is added when ``stride == 1`` and
    ``in_channels == out_channels``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        expand_ratio: Expansion ratio for the fused conv.
        stride: Spatial stride (1 or 2).
        survival_prob: Stochastic depth survival probability for DropPath.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int = 1,
        survival_prob: float = 1.0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.survival_prob = survival_prob

        expanded_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []

        if expand_ratio == 1:
            # No expansion: single 3×3 conv → BN → SiLU
            layers.extend([
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=1, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            ])
        else:
            # Fused expand: 3×3 conv → BN → SiLU
            if stride == 2:
                layers.append(nn.ZeroPad2d(1))
                conv_stride = 2
                conv_padding = 0
            else:
                conv_stride = 1
                conv_padding = 1

            layers.extend([
                nn.Conv2d(
                    in_channels, expanded_channels,
                    kernel_size=3, stride=conv_stride,
                    padding=conv_padding, bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True),
            ])

            # Project: 1×1 conv → BN (no activation)
            layers.extend([
                nn.Conv2d(
                    expanded_channels, out_channels,
                    kernel_size=1, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            ])

        self.block = nn.Sequential(*layers)
        # Store reference to the projection BN for zero-initialization
        self._projection_bn = layers[-1] if expand_ratio > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)

        if self.use_residual:
            if self.survival_prob < 1.0 and self.training:
                out = drop_path(out, 1.0 - self.survival_prob, self.training)
            return out + x

        return out


# =========================================================
# MBConv Block (standard inverted bottleneck)
# =========================================================

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck block with SE and stochastic depth.

    The block follows the MobileNetV2/EfficientNet design:

        1. Expand (1×1 pointwise, ``expand_ratio * in_channels``).
        2. Depthwise convolution (3×3, ``groups=expanded_channels``).
        3. Squeeze-and-Excitation.
        4. Project (1×1 pointwise back to ``out_channels``, no activation).

    Residual connection is added when ``stride == 1`` and
    ``in_channels == out_channels``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        expand_ratio: Expansion ratio for the inverted bottleneck.
        stride: Spatial stride (1 or 2).
        se: If True, add Squeeze-and-Excitation after depthwise conv.
        survival_prob: Stochastic depth survival probability for DropPath.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int = 1,
        se: bool = True,
        survival_prob: float = 1.0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.survival_prob = survival_prob

        expanded_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []

        # --- Pointwise Expansion (1×1) ---
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True),
            ])
        else:
            expanded_channels = in_channels

        # --- Depthwise Convolution (3×3) ---
        if stride == 2:
            layers.append(nn.ZeroPad2d(1))
            conv_stride = 2
            conv_padding = 0
        else:
            conv_stride = 1
            conv_padding = 1

        layers.extend([
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                stride=conv_stride,
                padding=conv_padding,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
        ])

        # --- Squeeze-and-Excitation ---
        if se:
            reduced_channels = max(1, expanded_channels // 4)
            layers.append(SqueezeExcitation(expanded_channels, reduced_channels))

        # --- Pointwise Projection (1×1, no activation) ---
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)
        # Store reference to the projection BN for zero-initialization
        self._projection_bn = layers[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)

        if self.use_residual:
            if self.survival_prob < 1.0 and self.training:
                out = drop_path(out, 1.0 - self.survival_prob, self.training)
            return out + x

        return out


# =========================================================
# OwnEfficientNetV2 (from scratch)
# =========================================================

class OwnEfficientNetV2(nn.Module):
    """EfficientNet-V2 for CIFAR10 trained from scratch on 32×32 images.

    Uses FusedMBConv in early stages and MBConv+SE in later stages,
    following the EfficientNet-V2 design philosophy. The architecture is
    simplified for CIFAR10 with fewer parameters than the ImageNet variant.

    Args:
        num_classes: Number of output classes (default 10).
        dropout_rate: Dropout rate on the classification head (default 0.2).
        stochastic_depth_prob: Max stochastic depth probability (default 0.0,
            disabled). Set to e.g. 0.2 to enable.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Stem: 3×3 stride-1 conv (preserves 32×32 resolution)
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Stages
        #
        # Each entry is (num_layers, in_ch, out_ch, stride, expand_ratio, block_type)
        # block_type: 'fused' for FusedMBConv, 'mbconv' for MBConv+SE
        # ------------------------------------------------------------------
        stage_specs: list[tuple[int, int, int, int, int, str]] = [
            (1, 32, 32, 1, 1, 'fused'),    # Stage 0 — FusedMBConv, no expansion
            (2, 32, 48, 2, 4, 'fused'),    # Stage 1 — FusedMBConv
            (2, 48, 80, 2, 4, 'mbconv'),   # Stage 2 — MBConv+SE
            (3, 80, 112, 2, 4, 'mbconv'),  # Stage 3 — MBConv+SE
            (2, 112, 144, 1, 6, 'mbconv'), # Stage 4 — MBConv+SE
            (2, 144, 192, 2, 6, 'mbconv'), # Stage 5 — MBConv+SE
        ]

        # Compute linear stochastic depth schedule across all blocks
        total_blocks = sum(s[0] for s in stage_specs)

        stages: list[nn.Module] = []
        in_ch = stage_specs[0][1]  # 32 — matches stem output
        block_idx = 0
        for num_layers, _, out_ch, stride, expand_ratio, block_type in stage_specs:
            for i in range(num_layers):
                # First layer in each stage may have stride > 1 or channel change
                layer_stride = stride if i == 0 else 1
                layer_in_ch = in_ch if i == 0 else out_ch

                # Linear survival probability: block 0 = 1.0, last block = 1 - max_drop_rate
                if total_blocks > 1:
                    survival_prob = 1.0 - (block_idx / (total_blocks - 1)) * stochastic_depth_prob
                else:
                    survival_prob = 1.0

                if block_type == 'fused':
                    stages.append(FusedMBConv(
                        in_channels=layer_in_ch,
                        out_channels=out_ch,
                        expand_ratio=expand_ratio,
                        stride=layer_stride,
                        survival_prob=survival_prob,
                    ))
                else:
                    stages.append(MBConv(
                        in_channels=layer_in_ch,
                        out_channels=out_ch,
                        expand_ratio=expand_ratio,
                        stride=layer_stride,
                        se=True,
                        survival_prob=survival_prob,
                    ))
                block_idx += 1

            in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        self.head_conv = nn.Sequential(
            nn.Conv2d(192, 768, kernel_size=1, bias=False),
            nn.BatchNorm2d(768),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(768, num_classes, bias=False)

        # ------------------------------------------------------------------
        # Weight initialization
        # ------------------------------------------------------------------
        self._initialize_weights()
        self._zero_init_projection_bns()

    def _initialize_weights(self) -> None:
        """Initialize conv weights with He normal and BN as identity."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _zero_init_projection_bns(self) -> None:
        """Zero-initialize the last BN gamma in each residual block.

        Modern residual architectures initialize the final BN in each block
        to zero, which improves gradient flow early in training.
        """
        for m in self.modules():
            if hasattr(m, '_projection_bn') and m._projection_bn is not None:
                nn.init.zeros_(m._projection_bn.weight)

    def _find_norm_parameter_names(self) -> set[str]:
        """Collect full parameter names that belong to normalization layers.

        Inspects the module hierarchy to identify BatchNorm parameters
        by module type, not by name string.
        """
        norm_names: set[str] = set()
        for mod_name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for p_name in module.state_dict():
                    full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    norm_names.add(full_name)
        return norm_names

    def _get_param_groups(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.02,
        norm_weight_decay: float = 0.0,
        bias_weight_decay: float = 0.0,
    ) -> list[dict]:
        """Build parameter groups for AdamW with no weight decay on
        BatchNorm parameters and biases.

        Args:
            lr: Learning rate for all groups.
            weight_decay: Weight decay for weight-containing parameters
                (conv and linear weights).
            norm_weight_decay: Weight decay for BatchNorm parameters
                (typically 0.0).
            bias_weight_decay: Weight decay for bias terms (typically 0.0).

        Returns:
            List of param groups suitable for ``torch.optim.AdamW``.
        """
        # Build set of norm parameter names (only once)
        if not hasattr(self, "_norm_param_names"):
            self._norm_param_names = self._find_norm_parameter_names()

        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude BatchNorm weight/bias and all bias terms from WD
            if name in self._norm_param_names or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": norm_weight_decay},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x