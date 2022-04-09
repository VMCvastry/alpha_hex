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
        print(x.size())
        s = self.batch_norm(x)
        x = F.relu(x)
        print(x.size())
        x = self.fcl(x)
        print(x.size())

        return x
