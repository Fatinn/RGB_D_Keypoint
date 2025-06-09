

import torch
import torch.nn as nn

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, pred_heatmaps, target_heatmaps):
        return self.criterion(pred_heatmaps, target_heatmaps) / pred_heatmaps.size(0)


class CoordinateLoss(nn.Module):
    def __init__(self, image_size=(256, 256)):
        super().__init__()
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.image_size = torch.tensor(image_size, dtype=torch.float32)

    def forward(self, pred_coords, target_coords):
        if self.image_size.device != pred_coords.device:
            self.image_size = self.image_size.to(pred_coords.device)

        scale = torch.tensor([self.image_size[1], self.image_size[0]],
                             device=pred_coords.device, dtype=torch.float32)

        pred_pixels = pred_coords * scale
        target_pixels = target_coords * scale

        pixel_loss = self.criterion(pred_pixels, target_pixels)
        return pixel_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, coord_weight=0.1, image_size=(256, 256)):
        super().__init__()
        self.heatmap_loss = HeatmapLoss()
        self.coord_loss = CoordinateLoss(image_size=image_size)
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight

    def forward(self, pred, target):
        pred_heatmaps = pred['heatmaps']
        pred_coords = pred['coords']
        target_heatmaps = target['heatmaps']
        target_coords = target['coords']

        h_loss = self.heatmap_loss(pred_heatmaps, target_heatmaps)
        c_loss = self.coord_loss(pred_coords, target_coords)

        total_loss = self.heatmap_weight * h_loss + self.coord_weight * c_loss

        return total_loss, {
            'heatmap_loss': h_loss.item(),
            'coord_loss': c_loss.item()
        }
