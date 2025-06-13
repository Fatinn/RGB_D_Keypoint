

import torch
import torch.nn as nn

class HeatmapHead(nn.Module):
    def __init__(self, in_channels, mid_channels=256, num_points=1, heatmap_size=(256, 256), dropout_p=0.3):
        super().__init__()
        self.num_points = num_points
        self.heatmap_size = heatmap_size

        self.upconv1 = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_p)

        self.upconv2 = nn.ConvTranspose2d(mid_channels, mid_channels // 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_p)

        self.heatmap_conv = nn.Conv2d(mid_channels // 2, num_points, kernel_size=1)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.upconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        heatmaps = self.heatmap_conv(x)
        return heatmaps
