"""
训练器模块

提供 Trainer 类,封装 Seq2Seq 模型的完整训练循环。
包含: 训练/验证 epoch、Teacher Forcing、梯度裁剪、学习率调度、
早停检查、checkpoint 自动保存、训练历史记录。
"""

import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.defaults import ModelParams, TrainingParams
from config.paths import BEST_MODEL_PATH, LAST_MODEL_PATH
from src.data.mapping import VocabMapping
from src.train.checkpoint import save_checkpoint
from src.train.early_stopping import EarlyStopping
from src.train.logger import Logger
from src.train.scheduler import is_plateau_scheduler


class Trainer:
    """
    Seq2Seq 训练器

    封装模型训练的完整流程。每个 epoch 执行:
    1. train_epoch()      — 训练模式
    2. validate_epoch()   — 验证模式
    3. scheduler.step()   — 学习率调度
    4. save_checkpoint()  — 保存最佳/最新模型
    5. early_stopping()   — 早停检查

    Args:
        model: Seq2Seq 模型实例
        train_loader: 训练集 DataLoader
        valid_loader: 验证集 DataLoader
        optimizer: 优化器实例
        scheduler: 学习率调度器实例
        criterion: 损失函数(CrossEntropyLoss)
        early_stopping: 早停实例
        vocab: 词表映射
        device: 计算设备
        epochs: 最大训练轮数
        grad_clip: 梯度裁剪阈值
        teacher_forcing_ratio: Teacher Forcing 概率
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        early_stopping: EarlyStopping,
        vocab: VocabMapping,
        device: torch.device,
        epochs: int = TrainingParams.EPOCHS,
        grad_clip: float = TrainingParams.GRAD_CLIP,
        teacher_forcing_ratio: float = ModelParams.TEACHER_FORCING_RATIO,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.early_stopping = early_stopping
        self.vocab = vocab
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.logger = Logger()
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.best_val_loss = float("inf")

    # ==================================================================
    # 训练 epoch
    # ==================================================================
    def train_epoch(self, epoch: int) -> float:
        """
        执行一个训练 epoch

        Returns:
            该 epoch 的平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        desc = f"[Train] Epoch {epoch + 1}/{self.epochs}"
        progress_bar = tqdm(self.train_loader, desc=desc, unit="batch")

        for encoder_input, decoder_input, target_output, encoder_mask in progress_bar:
            encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            target_output = target_output.to(self.device)
            encoder_mask = encoder_mask.to(self.device)

            batch_size = encoder_input.size(0)

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(
                encoder_input,
                decoder_input,
                encoder_mask,
                self.teacher_forcing_ratio,
            )

            # 损失计算: 展平 batch+时间步
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)
            loss = self.criterion(logits_flat, target_flat)

            # 反向传播 + 梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # 训练准确率(贪心解码,不影响梯度)
            with torch.no_grad():
                generated_ids, _ = self.model.generate(
                    encoder_input,
                    encoder_mask,
                    max_generation_length=target_output.size(1),
                )
                for i in range(batch_size):
                    tgt = target_output[i]
                    pred = generated_ids[i, : tgt.size(0)]
                    if torch.equal(pred, tgt):
                        total_correct += 1

            progress_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

        return total_loss / total_samples

    # ==================================================================
    # 验证 epoch
    # ==================================================================
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> tuple:
        """
        执行一个验证 epoch

        每个 batch 计算:
        1. 验证损失(Teacher Forcing ratio=1.0)
        2. 完全匹配准确率(贪心解码)

        Returns:
            (val_loss, val_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        desc = f"[Valid] Epoch {epoch + 1}/{self.epochs}"
        progress_bar = tqdm(self.valid_loader, desc=desc, unit="batch")

        for encoder_input, decoder_input, target_output, encoder_mask in progress_bar:
            encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            target_output = target_output.to(self.device)
            encoder_mask = encoder_mask.to(self.device)

            batch_size = encoder_input.size(0)

            # 验证损失(Teacher Forcing,稳定可复现)
            logits = self.model(
                encoder_input,
                decoder_input,
                encoder_mask,
                teacher_forcing_ratio=1.0,
            )
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)
            loss = self.criterion(logits_flat, target_flat)
            total_loss += loss.item() * batch_size

            # 完全匹配准确率(贪心解码)
            generated_ids, _ = self.model.generate(
                encoder_input,
                encoder_mask,
                max_generation_length=target_output.size(1),
            )

            for i in range(batch_size):
                target_seq = target_output[i]
                pred_seq = generated_ids[i, : target_seq.size(0)]
                if torch.equal(pred_seq, target_seq):
                    total_correct += 1

            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

        return total_loss / total_samples, total_correct / total_samples

    # ==================================================================
    # 完整训练流程
    # ==================================================================
    def train(self) -> Dict[str, List[float]]:
        """
        执行完整训练流程

        Returns:
            history: 训练历史字典
        """
        print(f"开始训练: epochs={self.epochs}, device={self.device}")
        print(f"Teacher Forcing ratio: {self.teacher_forcing_ratio}")

        self.logger.start()
        start_time = time.time()

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate_epoch(epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if is_plateau_scheduler(self.scheduler):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            print(
                f"Epoch {epoch + 1:3d}/{self.epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_accuracy:.4f} | "
                f"lr: {current_lr:.2e}"
            )

            self.logger.log_epoch(
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": current_lr,
                },
                epoch=epoch,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    BEST_MODEL_PATH,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_loss,
                    self.vocab,
                    self.early_stopping.state_dict(),
                    attention_type=self.model.attention_type
                    if hasattr(self.model, "attention_type")
                    else None,
                )
                print(f"  >> 最佳模型已保存 (val_loss={val_loss:.4f})")

            save_checkpoint(
                LAST_MODEL_PATH,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                val_loss,
                self.vocab,
                self.early_stopping.state_dict(),
                attention_type=self.model.attention_type
                if hasattr(self.model, "attention_type")
                else None,
            )

            if self.early_stopping(val_loss):
                print(
                    f"早停触发: 验证损失在 {self.early_stopping.patience} 轮内未显著改善"
                )
                break

        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(f"训练完成,总耗时: {minutes}m {seconds}s")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")

        self.logger.log_message(
            f"训练结束,epoch={epoch + 1}, "
            f"best_val_loss={self.best_val_loss:.4f}, "
            f"time={minutes}m{seconds}s"
        )
        self.logger.close()

        return self.history
