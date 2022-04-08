import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_feature: int,
    ) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=n_feature, kernel_size=3, padding=1
        )
        self.conv_compress = nn.Conv2d(
            in_channels=n_feature, out_channels=1, kernel_size=3, padding=1
        )
        # self.fc1 = nn.Linear(n_feature * 5 * 5, 10)
        # self.fc2 = nn.Linear(10, 10)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        :returns: predictions with size [batch, output_size]
        """
        x = self.conv1(x)
        x = self.conv_compress(x)
        # x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # x = x.view(x.shape[0], -1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return x
