from __future__ import annotations
import torch
from torch import nn


class CompressConv(nn.Module):
    def __init__(
        self,
        n_feature: int,
    ) -> None:
        super().__init__()

        self.conv_compress = nn.Conv2d(
            in_channels=n_feature, out_channels=1, kernel_size=3, padding=1
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv_compress(x)
        return x
