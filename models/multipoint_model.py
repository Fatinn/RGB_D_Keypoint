# my_project/models/multipoint_model.py

import torch
import torch.nn as nn
from .head import HeatmapHead


class DFormerMultiPoint(nn.Module):
    def __init__(self, base_model, num_points=1, heatmap_size=(256, 256)):
        super().__init__()
        self.encoder = base_model.backbone
        self.num_points = num_points
        self.heatmap_size = heatmap_size

        self.heatmap_head = HeatmapHead(
            in_channels=112,  # 默认输出通道数为112，可根据实际模型修改
            num_points=num_points,
            heatmap_size=heatmap_size
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        feat = self.encoder(rgb, depth)
        if isinstance(feat, tuple):
            feat = feat[0]

        heatmaps = self.heatmap_head(feat)
        coords = self.get_keypoints_from_heatmaps(heatmaps)

        return {
            "heatmaps": heatmaps,
            "coords": coords
        }

    def get_keypoints_from_heatmaps(self, heatmaps: torch.Tensor):
        B, N, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, N, -1)

        max_vals, _ = torch.max(heatmaps_flat, dim=2, keepdim=True)
        exp_maps = torch.exp(256 * (heatmaps_flat - max_vals))
        softmax_maps = exp_maps / (exp_maps.sum(dim=2, keepdim=True) + 1e-10)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=heatmaps.device),
            torch.arange(W, device=heatmaps.device),
            indexing='ij'
        )
        grid_x = grid_x.reshape(1, 1, -1).float()
        grid_y = grid_y.reshape(1, 1, -1).float()

        x_coord = (softmax_maps * grid_x).sum(dim=2) / (W - 1)
        y_coord = (softmax_maps * grid_y).sum(dim=2) / (H - 1)

        coords = torch.stack([x_coord, y_coord], dim=2)
        return coords.view(B, -1)


def freeze_encoder(model: nn.Module):
    """冻结模型编码器部分"""
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module):
    """解冻模型编码器部分"""
    for param in model.encoder.parameters():
        param.requires_grad = True
