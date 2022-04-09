import torch
from torch import nn
from torch.nn import functional as F


class Convolution(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_feature: int,
    ) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.input_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=n_feature, kernel_size=3, padding=1
        )
        self.batch_norm = nn.BatchNorm2d(n_feature)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        :returns: predictions with size [batch, output_size]
        """
        x = self.input_conv(x)
        s = self.batch_norm(x)
        x = F.relu(x)
        return x
