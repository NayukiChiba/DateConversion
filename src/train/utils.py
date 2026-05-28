"""
训练工具函数

提供 set_seed、get_device、count_parameters、format_time 等辅助函数。
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子,保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """获取可用计算设备(cuda / cpu)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    统计模型参数数量

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds: float) -> str:
    """将秒数格式化为 mm:ss 字符串"""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"
