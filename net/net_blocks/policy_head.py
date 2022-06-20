from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from variables import GRID_SIZE


class PolicyHead(nn.Module):
    def __init__(
        self,
        n_feature: int,
    ) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.conv = nn.Conv2d(in_channels=n_feature, out_channels=2, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(2)
        self.fcl = nn.Linear(GRID_SIZE ** 2 * 2, GRID_SIZE ** 2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        s = self.batch_norm(x)
        x = F.relu(s)
        x = x.reshape((-1, GRID_SIZE ** 2 * 2))  # double grid to vector
        x = self.fcl(x)
        x = F.softmax(x, dim=1)  # to get probability distribution
        x = x.reshape(-1, GRID_SIZE, GRID_SIZE)  # vector to grid
        return x
