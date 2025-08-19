from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from .layers.se import SqueezeExcitation


class ResidualMLPHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        hidden = max(in_features // 2, 256)
        self.norm = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.do = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, in_features)
        self.out = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.do(x)
        x = self.fc2(x)
        x = x + identity
        logits = self.out(x)
        return logits


class InceptionV3SEClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_until: Optional[str] = "mixed5",
    ) -> None:
        super().__init__()

        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None, aux_logits=False)
        # Input size for inception_v3 is 299x299
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )

        # Channel size after last inception block is 2048
        self.se = SqueezeExcitation(2048, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.head = ResidualMLPHead(2048, num_classes=num_classes, dropout=dropout)

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
        x = self.se(x)
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.head(x)
        return logits



