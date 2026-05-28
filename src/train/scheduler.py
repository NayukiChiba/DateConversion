"""
学习率调度器构建模块

提供 build_scheduler 函数,支持 StepLR / CosineAnnealingLR / ReduceLROnPlateau。
"""

from typing import Literal

import torch

from config.defaults import TrainingParams


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Literal[
        "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"
    ] = TrainingParams.LR_SCHEDULER,
    step_size: int = TrainingParams.LR_STEP_SIZE,
    gamma: float = TrainingParams.LR_GAMMA,
    epochs: int = TrainingParams.EPOCHS,
    reduce_factor: float = TrainingParams.LR_REDUCE_FACTOR,
    reduce_patience: int = TrainingParams.LR_REDUCE_PATIENCE,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    构建学习率调度器

    Args:
        optimizer: 优化器实例
        scheduler_type: 调度器类型
        step_size: StepLR 的衰减周期(每 step_size 轮衰减一次)
        gamma: 衰减因子
        epochs: 总训练轮数(CosineAnnealingLR 需要)
        reduce_factor: ReduceLROnPlateau 的衰减因子
        reduce_patience: ReduceLROnPlateau 的容忍轮数

    Returns:
        配置好的调度器实例
    """
    if scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    elif scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=reduce_factor,
            patience=reduce_patience,
        )
    else:
        raise ValueError(f"未知的调度器类型: '{scheduler_type}'")


def is_plateau_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> bool:
    """判断调度器是否为 Plateau 类型(需要传入 val_loss)"""
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
