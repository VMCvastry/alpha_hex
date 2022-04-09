from torch import nn


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
