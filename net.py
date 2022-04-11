import torch
from torch import nn
import torch.nn.functional as F
from net_blocks.compression_block import CompressConv
from net_blocks.convolution_block import Convolution
from net_blocks.policy_head import PolicyHead
from net_blocks.residual_block import ResidualBlock
from net_blocks.value_head import ValueHead


class NET(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_feature: int,
        res_net_depth: int,
        value_head_size: int,
    ) -> None:
        super().__init__()
        layers = []
        super().__init__()
        self.n_feature = n_feature
        self.input_conv = Convolution(input_channels, n_feature)
        for _ in range(res_net_depth):
            layers.append(ResidualBlock(n_feature))
        self.residuals = nn.Sequential(*layers)
        self.conv_compress = CompressConv(n_feature)
        self.policy_head = PolicyHead(n_feature)
        self.value_head = ValueHead(n_feature, value_head_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """
        :param x: batch of images with size [batch, 1, w, h]
        :returns: predictions with size [batch, output_size]
        """
        x = self.input_conv(x)
        x = self.residuals(x)
        policy = self.policy_head(x)
        value = self.value_head(x)

        # x = self.conv_compress(x)
        # x = F.max_pool2d(x, kernel_size=2)
        # value = x

        # x = x.view(x.shape[0], -1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return policy, value
