"""
Checkpoint 管理模块

提供 save_checkpoint / load_checkpoint 函数,
将模型权重、优化器状态、调度器状态、词表等封装为单个 .pth 文件。
"""

from pathlib import Path

import torch
import torch.nn as nn

from src.data.mapping import VocabMapping


def save_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_loss: float,
    vocab: VocabMapping,
    early_stopping_state: dict | None = None,
    attention_type: str | None = None,
) -> None:
    """
    保存完整训练状态到 checkpoint 文件

    Args:
        filepath: 保存路径
        model: 模型实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例
        epoch: 当前 epoch 编号(0-based)
        val_loss: 当前验证损失
        vocab: 词表映射,推理时需要重建
        early_stopping_state: 早停状态字典
        attention_type: 注意力机制类型,推理时需要重建相同结构
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "vocab": vocab.to_dict(),
    }
    if early_stopping_state is not None:
        checkpoint["early_stopping_state"] = early_stopping_state
    if attention_type is not None:
        checkpoint["attention_type"] = attention_type

    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str = "cpu",
) -> dict:
    """
    从 checkpoint 文件加载训练状态

    Args:
        filepath: checkpoint 文件路径
        model: 模型实例(状态加载目标)
        optimizer: 优化器实例(可选)
        scheduler: 调度器实例(可选)
        device: 计算设备

    Returns:
        完整的 checkpoint 字典
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
