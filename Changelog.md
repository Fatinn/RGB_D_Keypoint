# 📝 训练脚本版本更迭日志（Changelog）

> 项目：多点检测模型训练脚本（基于 DFormer + 多模态 + 热图监督）
> 格式参考语义化版本控制 `v主.次.修`（Major.Minor.Patch）

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
* 添加冻结主干网络参数逻辑，实现逐步微调（前 `epoch=0` 只训练头部）。

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

如果你提供更多提交记录或更早的代码变更，我可以继续为你补充更完整的版本历史。是否需要我将这份 `Changelog` 写入 `CHANGELOG.md` 文件或转成 Markdown 格式？
