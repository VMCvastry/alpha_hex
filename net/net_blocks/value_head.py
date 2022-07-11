from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from variables import GRID_SIZE


class ValueHead(nn.Module):
    def __init__(self, n_feature: int, hidden_layer: int) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.conv = nn.Conv2d(in_channels=n_feature, out_channels=1, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.fcl1 = nn.Linear(GRID_SIZE ** 2, hidden_layer)
        self.fcl2 = nn.Linear(hidden_layer, 1)

        # tests
        self.conv_replace_fcl = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=GRID_SIZE
        )
        self.fclT = nn.Linear(GRID_SIZE * GRID_SIZE, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        s = self.batch_norm(x)
        x = F.relu(s)
        x = x.view(-1, GRID_SIZE * GRID_SIZE)  # grid to vector
        x = self.fcl1(x)
        x = F.relu(x)
        x = self.fcl2(x)
        x = torch.tanh(x)
        return x
