# my_project/config/nyu_dformerv2_config.py

C = {
    "cfg_name": "DFormer.local_configs.NYUDepthv2.DFormerv2_L",
    "pretrained_path": "checkpoints/pretrained/DFormerv2_Large_NYU.pth",
    "norm_mean": [0.5, 0.5, 0.5],
    "norm_std": [0.5, 0.5, 0.5],
    "image_size": (256, 256),
    "heatmap_size": (256, 256),
    "gaussian_sigma": 3.0,
    "num_points": 1,
    "freeze_epochs": 70
}
