# RGB-D Multi-Point Detection (v1.4.0)

📅 Updated: 2025-06-12

本项目基于 DFormer 主干网络，实现了多点 RGB+Depth 关键点检测任务，具备高斯热图监督、坐标解码、联合损失优化、PCK/OKS 评估及可视化功能，配置灵活。

<img src="outputs\pred_Image__Rgb_54_point1_aug1.png" style="zoom:33%;" />

---

## 📁 一、项目模块划分

```plaintext
代码主要模块：
├── config/           # 所有训练、模型、数据参数集中配置
├── data/             # 数据增强、热图生成、多点数据集定义
├── models/           # DFormer 主干 + 检测头 + 完整结构封装
├── losses/           # 热图损失 + 坐标损失 + 联合损失
├── utils/            # PCK、OKS、可视化工具
├── train.py          # 主训练入口，自动加载配置并训练
```

---

## 🧠 二、模型结构描述

```plaintext
[ RGB 图像 + 深度图（增强对比度）输入 ]
                ↓
    DFormerV2 编码器（融合 RGB + Depth，输出多尺度特征）
                ↓
    多尺度特征 [c1, c2, c3, c4]（分别为 1/4, 1/8, 1/16, 1/32 尺度）
                ↓
    HeatmapHead（多尺度融合模块）：
        - 1x1 Conv 降维（各层通道 → mid_channels）
        - ConvTranspose2d 上采样逐层融合
        - 注意力增强（加法 + 乘法）
        - 3×3 refine 卷积 + Dropout
                ↓
    最终输出热图：
        - 1x1 Conv → 输出维度为 (B, num_points, H/2, W/2)
        - 上采样（F.interpolate）恢复至原图分辨率 → (B, num_points, 256, 256)
                ↓
    Soft-argmax：
        - 对每个关键点热图进行 softmax 平滑归一化
        - 使用 weighted average 计算出 (x, y) 坐标
                ↓
    最终坐标输出：(B, num_points × 2)

```

支持联合输出：

* 热图 (监督用)
* 归一化坐标 (评估用)

---

## ⚙️ 三、训练配置（集中于 `config/config.py`）

| 配置项           | 示例值 / 描述                                          |
| ---------------- | ------------------------------------------------------ |
| 图像尺寸         | `image_size=(256, 256)`                                |
| 关键点数         | `num_points=1`                                         |
| 高斯核标准差 σ   | `gaussian_sigma=3.0`                                   |
| 模型结构         | `cfg_name='local_configs.NYUDepthv2.DFormerv2_L'`      |
| 预训练权重路径   | `'checkpoints/pretrained/DFormerv2_Large_NYU.pth'`     |
| 训练集CSV        | `'dataset/train.csv'`                                  |
| RGB/深度图目录   | `'dataset/rgb'`, `'dataset/depth'`                     |
| 联合损失权重     | `heatmap_weight=1.0`, `coord_weight=0.1`               |
| 批大小           | `batch_size=16`                                        |
| 总训练轮数       | `total_epochs=100`                                     |
| 冻结编码器轮数   | `freeze_epochs=70`                                     |
| 学习率配置       | `lr_encoder=1e-5`, `lr_head=5e-4`, `weight_decay=1e-4` |
| 学习率调度器     | `CosineAnnealingLR`, `T_max=30`                        |
| 模型保存策略     | `save_best_by='pck'`, `save_last=True`                 |
| 可视化周期与路径 | 每 `5` 轮 → `outputs/visualizations/`                  |
| 模型保存路径     | `best_model_path='outputs/best_model.pth'` 等          |

---

## 🧪 四、损失函数设计

| 类型   | 实现方式                       | 描述          | 权重  |
| ---- | -------------------------- | ----------- | --- |
| 热图损失 | `MSELoss(reduction='sum')` | 每个关键点的热图监督  | 1.0 |
| 坐标损失 | `SmoothL1Loss` + 反归一化      | 坐标预测在原图上的误差 | 0.1 |
| 联合损失 | `HeatmapLoss + CoordLoss`  | 最终总损失       | -   |

---

## 🧱 五、数据增强策略

| 操作       | 参数 / 范围            | 作用      |
| -------- | ------------------ | ------- |
| 水平翻转     | 50% 概率             | 增强左右对称  |
| 垂直翻转     | 50% 概率             | 增强上下对称  |
| 随机旋转     | ±15°，50% 概率        | 增强旋转不变性 |
| 亮度/对比度调整 | \[0.8, 1.2]，50% 概率 | 提高光照鲁棒性 |

---

## 📊 六、训练指标

| 指标名称  | 训练集 | 验证集 | 说明                          |
| --------- | ------ | ------ | ----------------------------- |
| 联合损失  | 7.82   | 10.095 | 越低越好                      |
| PCK\@0.05 | 0.9212 | 0.9166 | 距离真实点<5%图像宽度的点比例 |
| OKS       | 0.9256 | 0.9235 | 热图重合度相似性              |

> 注：val中的工件在训练中见过

---

## 🚀 七、使用方法

### 1. 安装依赖

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -r requirements.txt
```

> 如使用本地 DFormer（包含 `mmseg/`），需在 `train.py` 顶部加入：
>
> ```python
> sys.path.append("DFormer")
> ```

---

### 2. 训练模型

```bash
python train.py
```

将会：

* 自动加载配置
* 保存可视化结果于 `outputs/visualizations/`
* 按 `config["save_best_by"]` 策略保存模型

---

### 3. 可视化输出

* 热图、预测点、GT 点将可视化保存每 `n` 轮
* 图像位于 `outputs/visualizations/epoch_*.png`

---

## 🧬 八、伪代码（训练流程）

```python
加载配置 → 初始化模型 → 加载预训练权重
冻结编码器（前 N 轮）
for epoch in range(total_epochs):
    解冻编码器（可选）
    for batch in train_loader:
        → 模型前向 + 损失计算 + 反向传播
    for batch in val_loader:
        → 评估 PCK、OKS
        → 每 N 轮保存可视化
    若当前指标最优 → 保存模型
    若为最后一轮 → 保存最终模型（可选）
```

---

## 🧠 九、备注与扩展建议

* [ ] 支持多点标注（num\_points > 1）
* [ ] 添加 `test.py` 进行单张图像推理
* [ ] 支持 `yaml` 或 `argparse` 动态配置

---

## 📎 引用与参考

* [DFormer: Transformer-Based RGB-D Segmentation](https://github.com/zcablii/DFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)


