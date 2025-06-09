

import torch
from importlib import import_module
from DFormer.models.builder import EncoderDecoder as DFormerModel  # 根据你原有路径保持不变

def build_dformer(cfg_dict, use_bn=True):
    """
    构建 DFormer 主干网络
    参数:
        cfg_dict: 字典格式配置 (通常来自 config 文件)
        use_bn: 是否使用 BatchNorm2d
    返回:
        初始化的 DFormerModel 实例
    """
    config_module = import_module(cfg_dict["cfg_name"])
    config = getattr(config_module, "C")
    norm_layer = torch.nn.BatchNorm2d if use_bn else None

    model = DFormerModel(cfg=config, criterion=None, norm_layer=norm_layer)
    return model


def load_pretrained_weights(model, path):
    """
    加载预训练权重
    """
    weight = torch.load(path, map_location="cpu")
    if "model" in weight:
        weight = weight["model"]
    model.load_state_dict(weight, strict=False)
    return model
