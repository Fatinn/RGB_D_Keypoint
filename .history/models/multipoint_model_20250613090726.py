import torch
import torch.nn as nn
from .head import HeatmapHead


class DFormerMultiPoint(nn.Module):
    def __init__(self, base_model, num_points=1, heatmap_size=(256, 256)):
        super().__init__()
        self.encoder = base_model.backbone
        self.num_points = num_points
        self.heatmap_size = heatmap_size

        # 使用 backbone channels，根据 DFormerv2_L 修改，其他型号按需调整
        self.heatmap_head = HeatmapHead(
            channels=[112, 224, 448, 640],
            num_points=num_points,
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        feats = self.encoder(rgb, depth)
        if isinstance(feats, tuple) or isinstance(feats, list):
            # 确保返回四级特征
            if len(feats) == 4:
                c1, c2, c3, c4 = feats
                # print("feats shape: ", c1.shape, c2.shape, c3.shape, c4.shape)
            else:
                # 如果返回结构不同，请调整此处解析逻辑
                raise ValueError("Backbone output feature count != 4, got {}".format(len(feats)))
        else:
            raise ValueError("Backbone output is not tuple or list")

        heatmaps = self.heatmap_head([c1, c2, c3, c4])
        coords = self.get_keypoints_from_heatmaps(heatmaps)

        return {
            "heatmaps": heatmaps,
            "coords": coords
        }

    def get_keypoints_from_heatmaps(self, heatmaps: torch.Tensor):
        B, N, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, N, -1)

        max_vals, _ = torch.max(heatmaps_flat, dim=2, keepdim=True)
        exp_maps = torch.exp(128 * (heatmaps_flat - max_vals))
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
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module):
    for param in model.encoder.parameters():
        param.requires_grad = True