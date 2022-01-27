import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class Aggregation(nn.Module):
    def __init__(self, channels):
        super(Aggregation, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3*channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//4, channels, bias=False),
            nn.Sigmoid()
        )
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.relu(self.conv(self.upsample_2(x)))

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = torch.mul(y, x)

        return out

