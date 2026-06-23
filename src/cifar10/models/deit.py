"""DeiT (Data-efficient Image Transformers) model for CIFAR10.

Architecture only — no training code. Import and use with the training
framework's DistillationTrainer.
"""

import torch
import torch.nn as nn

from cifar10.models.blocks import TransformerBlock


class ConvStemPatchEmbedding(nn.Module):
    """Overlapping convolution stem for patch embedding.

    Gracefully downsamples 32x32 -> 8x8 spatial grid using two strided
    convolutions, providing better low-level feature extraction than a
    linear patch projection.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        # Logical patch_size=4 maintains the 64 token sequence length
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class DeiT(nn.Module):
    """DeiT (Data-efficient Image Transformer) for small images.

    Uses a convolutional stem, distillation token, and stochastic depth.
    During training returns ``(cls_logits, dist_logits)``.
    During evaluation returns the average of the two heads.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = ConvStemPatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.norm(x)

        cls_out = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])

        if self.training:
            return cls_out, dist_out
        # During inference, average the predictions of the two tokens
        return (cls_out + dist_out) / 2