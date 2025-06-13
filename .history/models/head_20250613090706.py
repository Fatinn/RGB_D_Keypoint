import torch
import torch.nn as nn
import torch.nn.functional as F

# 通道+空间注意力模块（简化 CBAM）
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # 通道注意力
        chn_att = self.avg_pool(x).view(b, c)
        chn_att = self.channel_fc(chn_att).view(b, c, 1, 1)
        x = x * chn_att

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x

# 主模块
class HeatmapHead(nn.Module):
    def __init__(self, channels, num_points=1, mid_channels=256):
        super().__init__()
        self.num_points = num_points
        _, _, c3, c4 = channels  # 只用 c3 和 c4

        self.reduce_c4 = nn.Sequential(
            nn.Conv2d(c4, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.reduce_c3 = nn.Sequential(
            nn.Conv2d(c3, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.up_c4_to_c3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.attn = AttentionBlock(in_channels=mid_channels * 2)

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, num_points, kernel_size=1)
        )

    def forward(self, features):
        _, _, c3, c4 = features

        x_c4 = self.reduce_c4(c4)
        x_c3 = self.reduce_c3(c3)

        x = self.up_c4_to_c3(x_c4)
        x = torch.cat([x, x_c3], dim=1)

        x = self.attn(x)
        x = self.fuse(x)

        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
        heatmap = self.heatmap_head(x)
        return heatmap
