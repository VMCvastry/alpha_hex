import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


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


class NET(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_feature: int,
    ) -> None:
        super().__init__()
        layers = []
        super().__init__()
        self.n_feature = n_feature
        self.input_conv = Convolution(input_channels, n_feature)
        self.residual = ResidualBlock(n_feature)
        self.conv_compress = CompressConv(n_feature)
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
        x = self.input_conv(x)
        x = self.residual(x)
        x = self.conv_compress(x)

        x = F.max_pool2d(x, kernel_size=2)

        # x = x.view(x.shape[0], -1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return x
