# my_project/utils/metrics.py

import torch
import numpy as np

def calculate_pck(pred, target, threshold=0.1, image_size=None):
    """
    Percentage of Correct Keypoints (PCK)
    """
    pred = pred.view(-1, 2)
    target = target.view(-1, 2)

    if image_size is not None:
        w, h = image_size
        scale = torch.tensor([w, h], dtype=pred.dtype, device=pred.device)
        pred = pred / scale
        target = target / scale

    distances = torch.norm(pred - target, dim=1)
    pck = (distances < threshold).float().mean().item()
    return pck


def calculate_oks(pred, target, image_size=None, sigma=0.1, use_normalized_coords=True):
    """
    Object Keypoint Similarity (OKS)
    """
    pred = pred.view(-1, 2).detach().cpu().numpy()
    target = target.view(-1, 2).detach().cpu().numpy()

    if use_normalized_coords:
        squared_distances = np.sum((pred - target) ** 2, axis=1)
        scale = 1.0
    else:
        assert image_size is not None
        w, h = image_size
        squared_distances = np.sum((pred - target) ** 2, axis=1)
        scale = np.sqrt(w * h)

    oks = np.exp(-squared_distances / (2 * (sigma * scale) ** 2))
    return float(np.mean(oks))


def calculate_accuracy(pred, target, threshold=10.0, image_size=None):
    """
    计算欧氏距离小于 threshold 像素的点占比
    """
    pred = pred.view(-1, 2)
    target = target.view(-1, 2)

    if image_size is not None:
        w, h = image_size
        scale = torch.tensor([w, h], dtype=pred.dtype, device=pred.device)
        pred = pred * scale
        target = target * scale

    distances = torch.norm(pred - target, dim=1)
    acc = (distances < threshold).float().mean().item()
    return acc
