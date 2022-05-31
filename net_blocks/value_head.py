from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from logger import logging


class ValueHead(nn.Module):
    def __init__(self, n_feature: int, hidden_layer: int) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.conv = nn.Conv2d(in_channels=n_feature, out_channels=1, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.fcl1 = nn.Linear(3 * 3, hidden_layer)
        self.fcl2 = nn.Linear(hidden_layer, 1)

        # tests
        self.conv_replace_fcl = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.fclT = nn.Linear(3 * 3, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        s = self.batch_norm(x)
        x = F.relu(s)
        # x = x.view((-1, 9))  # grid to vector
        # x = x.view(-1, 9)
        # x = self.fcl1(x)
        # x = F.relu(x)
        # x = self.fcl2(x)
        x = self.conv_replace_fcl(x)  # TODO WRONG should be the fcl but it doesnt train
        x = torch.tanh(x)
        # x = self.fclT(x)
        return x
