import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
from PIL import Image

def generate_gaussian_heatmap(center_x, center_y, height, width, sigma=3.0):
    """
    生成以(center_x, center_y)为中心的高斯热图
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    y = y[:, np.newaxis]
    heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    return heatmap

def enhance_depth_contrast(depth, gain=5.0):
    """
    增强深度图对比度，使深度变化更明显。
    """
    d_min, d_max = np.min(depth), np.max(depth)
    if d_max - d_min < 1e-3:
        return depth.astype(np.float32) / 255.0  # 几乎无变化，直接归一化

    depth_normalized = (depth - d_min) / (d_max - d_min)
    depth_enhanced = np.clip(depth_normalized * gain, 0, 1)
    return depth_enhanced.astype(np.float32)

def generate_target_heatmaps(points, height, width, sigma=3.0):
    """
    为多个点生成热图目标
    """
    heatmaps = []
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            heatmap = generate_gaussian_heatmap(x, y, height, width, sigma)
        else:
            heatmap = np.zeros((height, width))
        heatmaps.append(heatmap)
    return np.stack(heatmaps, axis=0)


class MultiPointDataset(Dataset):
    def __init__(self, csv_file, img_dir, depth_dir, config,
                 is_train=False):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.config = config
        self.img_size = tuple(config["image_size"])
        self.heatmap_size = tuple(config["heatmap_size"])
        self.sigma = config["gaussian_sigma"]
        self.max_points = config["num_points"]
        self.is_train = is_train

        df = pd.read_csv(csv_file)
        self.samples = {}
        self.original_sizes = {}

        for _, row in df.iterrows():
            filename = row["filename"]
            if filename not in self.samples:
                self.samples[filename] = []
                img_path = os.path.join(img_dir, filename)
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        self.original_sizes[filename] = img.size
                else:
                    self.original_sizes[filename] = self.img_size[::-1]
            self.samples[filename].append((float(row["x"]), float(row["y"])))

        self.filenames = [f for f in self.samples if len(self.samples[f]) > 0]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        orig_w, orig_h = self.original_sizes[filename]
        rgb_path = os.path.join(self.img_dir, filename)
        depth_path = os.path.join(self.depth_dir, os.path.splitext(filename)[0] + '.tiff')

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            return self.__getitem__((idx + 1) % len(self))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, self.img_size[::-1])
        depth = enhance_depth_contrast(depth, gain=5.0)  # 拉大深度差异

        rgb = cv2.resize(rgb, self.img_size[::-1])

        # 坐标缩放
        points = []
        for x, y in self.samples[filename]:
            x_scaled = x * (self.img_size[0] / orig_w)
            y_scaled = y * (self.img_size[1] / orig_h)
            points.append((x_scaled, y_scaled))
        points = points[:self.max_points]

        # 归一化图像和深度
        rgb_norm = rgb.astype(np.float32) / 255.0
        rgb_norm = (rgb_norm - np.array(self.config["norm_mean"])) / np.array(self.config["norm_std"])
        depth_norm = (depth - 0.5) / 0.25  # 新归一化方式，增强后为中心 0

        rgb_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).float()
        depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).float()

        heatmaps = np.zeros((self.max_points, *self.heatmap_size), dtype=np.float32)
        for i, (x, y) in enumerate(points):
            heat_x = int(x * self.heatmap_size[0] / self.img_size[0])
            heat_y = int(y * self.heatmap_size[1] / self.img_size[1])
            heat_x = max(0, min(heat_x, self.heatmap_size[0] - 1))
            heat_y = max(0, min(heat_y, self.heatmap_size[1] - 1))
            heatmaps[i] = generate_gaussian_heatmap(heat_x, heat_y, *self.heatmap_size, sigma=self.sigma)

        coord_points = np.zeros((self.max_points, 2), dtype=np.float32)
        for i, (x, y) in enumerate(points):
            coord_points[i, 0] = x / self.img_size[0]
            coord_points[i, 1] = y / self.img_size[1]
        coord_tensor = torch.from_numpy(coord_points.flatten()).float()

        valid_mask = torch.zeros(self.max_points, dtype=torch.float32)
        valid_mask[:len(points)] = 1.0

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "heatmap": torch.from_numpy(heatmaps).float(),
            "coords": coord_tensor,
            "valid_mask": valid_mask,
            "img_path": filename,
            "original_rgb": rgb,
            "original_size": (orig_w, orig_h)
        }

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        return torch.utils.data.default_collate(batch)
