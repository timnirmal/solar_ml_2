from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class ConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "tiny",  # tiny|small|base|large
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_until: Optional[str] = None,  # e.g., "features.6" or None
    ) -> None:
        super().__init__()

        if variant == "tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            embed_dim = 768
        elif variant == "small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_small(weights=weights)
            embed_dim = 768
        elif variant == "base":
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_base(weights=weights)
            embed_dim = 1024
        elif variant == "large":
            weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_large(weights=weights)
            embed_dim = 1536
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")

        # Keep features; replace classifier head
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        if freeze_until is not None:
            self._freeze_until(freeze_until)

    def _freeze_until(self, tag: str) -> None:
        should_freeze = True
        for name, module in self.features.named_children():
            if should_freeze:
                for p in module.parameters():
                    p.requires_grad = False
            if tag in name:
                should_freeze = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.head(x)
        return logits



