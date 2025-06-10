# RGB_D_Keypoint

# 📝 训练脚本版本更迭日志（Changelog）

# 📌 v1.3.1 - 2025-06-09

### 🔧 改进

* 将原型产品进行解耦，分离部件。

* ```
  RGB_D_Keypoint/
  ├── config/
  │   └── config.py
  ├── data/
  │   ├── augmentation.py
  │   └── dataset.py
  ├── models/
  │   ├── dformer_backbone.py
  │   ├── head.py
  │   └── multipoint_model.py
  ├── losses/
  │   └── combined_loss.py
  ├── utils/
  │   ├── metrics.py
  │   └── visualization.py
  ├── train.py
  ```
* 将训练配置全部移入config

---



## 📌 v1.3.0 - 2025-06-06

### ✨ 新增

* 引入 `evaluate` 函数，对验证集进行 OKS 和 PCK 指标评估。
* 加入验证阶段模型可视化结果保存（`output_dir/epoch_{:02d}_val_predictions.png`）。
* 新增 OKS 作为主评估指标，用于保存最佳模型。

### 🔧 改进

* `CombinedLoss` 支持动态调节 `lambda_coord` 权重。
* 模型保存从每轮保存更改为“只保存验证 OKS 最佳的模型”。

---

## 📌 v1.2.0 - 2025-06-05

### ✨ 新增

* 支持从指定路径加载预训练权重 `load_pretrained_weights()`。
* 添加冻结主干网络参数逻辑，实现逐步微调（前 `epoch=70` 只训练头部）。

### 🐛 修复

* 修复由于 `base_model` 输出维度不匹配 `DFormerMultiPoint` 导致的 `mat1 and mat2 shapes cannot be multiplied` 报错。

---

## 📌 v1.1.0 - 2025-06-04

### ✨ 新增

* 添加 `CombinedLoss`，综合热图 MSE 与坐标 SmoothL1 Loss。
* 模型结构首次引入 `DFormerMultiPoint`，用于多点热图输出与坐标回归。

### 🔧 改进

* `train_one_epoch` 支持返回 OKS 与 PCK。

---

## 📌 v1.0.0 - 2025-06-02

### 🚀 初始版本

* 支持 RGB 与深度图双模态输入。
* 构建基本训练流程：加载数据、构建模型、优化器、损失函数。
* 实现最基本的训练与评估框架。

---

