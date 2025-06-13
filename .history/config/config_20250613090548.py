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

     # --- 训练控制 ---
    "freeze_epochs": 0,
    "total_epochs": 50,
    "batch_size": 16,
    "num_workers": 8,
    "val_num_workers": 2,
    "print_freq": 1,

    # --- 学习率与优化器 ---
    "lr_encoder": 6e-5,
    "lr_head": 1e-4,
    "weight_decay": 1e-4,
    "scheduler_T_max": 20,
    "scheduler_eta_min": 1e-6,

    # --- 损失权重 ---
    "heatmap_weight": 1.0,
    "coord_weight": 0.1,

    # --- 数据路径 ---
    "train_csv": "dataset/train.csv",
    "val_csv":   "dataset/val_clear.csv",
    "rgb_dir":   "dataset/rgb",
    "depth_dir": "dataset/depth",

    # --- 可视化输出 ---
    "visualize_every_n_epoch": 5,
    "visual_output_dir": "outputs/visualizations",
    "checkpoint_dir": "outputs",

    # === 模型保存配置 ===
    "save_best_by": "pck",  # 可选："pck"、"loss"、"oks"
    "save_last": True,      # 是否保存最后一个 epoch 的模型
    "best_model_path": "outputs/best_model_v1_4.pth",
    "last_model_path": "outputs/last_model_v1_4.pth"
}
