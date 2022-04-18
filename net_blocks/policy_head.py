import torch
from torch import nn
from torch.nn import functional as F


class PolicyHead(nn.Module):
    def __init__(
        self,
        n_feature: int,
    ) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.conv = nn.Conv2d(in_channels=n_feature, out_channels=2, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(2)
        self.fcl = nn.Linear(3 * 3 * 2, 3 * 3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        s = self.batch_norm(x)
        x = F.relu(s)
        x = x.reshape((-1, 18))  # double grid to vector
        x = self.fcl(x)
        x = F.softmax(x, dim=1)  # to get probability distribution
        x = x.reshape(-1, 3, 3)  # vector to grid
        return x
