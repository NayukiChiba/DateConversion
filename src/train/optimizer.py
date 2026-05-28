"""
优化器构建模块

提供 build_optimizer 函数,支持 Adam / SGD / AdamW 三种优化器。
"""

from typing import Literal

import torch
import torch.nn as nn

from config.defaults import TrainingParams


def build_optimizer(
    model: nn.Module,
    learning_rate: float = TrainingParams.LEARNING_RATE,
    optimizer_type: Literal["Adam", "SGD", "AdamW"] = TrainingParams.OPTIMIZER,
    weight_decay: float = TrainingParams.WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    """
    构建优化器

    Args:
        model: 待训练的模型
        learning_rate: 学习率
        optimizer_type: 优化器类型
        weight_decay: 权重衰减系数(L2 正则化)

    Returns:
        配置好的优化器实例
    """
    if optimizer_type == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"未知的优化器类型: '{optimizer_type}'")
