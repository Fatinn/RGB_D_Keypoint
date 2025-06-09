# my_project/utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(
    image,
    target_coords,
    pred_coords,
    heatmaps=None,
    save_path=None
):
    """
    显示图像 + GT点 + 预测点 + 可选热图

    参数:
    - image: 原始 RGB 图像 [H, W, 3]
    - target_coords: GT 点，归一化坐标 [num_points, 2]
    - pred_coords: 预测点，归一化坐标 [num_points, 2]
    - heatmaps: 可选，每个点的热图 [num_points, H, W]
    - save_path: 可选，若给出则保存到文件
    """
    h, w = image.shape[:2]
    target = target_coords.clone().cpu().numpy() * np.array([w, h])
    pred = pred_coords.clone().cpu().numpy() * np.array([w, h])

    plt.figure(figsize=(12, 8))

    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    # 预测与GT点
    plt.subplot(2, 2, 2)
    plt.imshow(image)
    for i, (x, y) in enumerate(target):
        plt.scatter(x, y, c='g', marker='o', s=60, label='GT' if i == 0 else None)
    for i, (x, y) in enumerate(pred):
        plt.scatter(x, y, c='r', marker='x', s=60, label='Pred' if i == 0 else None)
    plt.title("Predictions vs Ground Truth")
    plt.legend()

    # 热图（最多显示前2个）
    if heatmaps is not None:
        for i in range(min(2, heatmaps.shape[0])):
            plt.subplot(2, 2, 3 + i)
            plt.imshow(heatmaps[i].cpu().numpy(), cmap='hot')
            plt.title(f"Heatmap {i+1}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
