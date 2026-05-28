"""
训练日志模块

提供 Logger 类,同时记录:
- CSV 文件: 每个 epoch 一行,包含全部指标
- 文本日志: 训练过程中的事件和消息
- TensorBoard: 可选,需安装 tensorboard
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from config.paths import LOGS_DIR


class Logger:
    """
    训练日志记录器

    每个训练会话创建一个以时间戳命名的子目录,
    在其中保存 metrics.csv 和 train.log。

    使用方式:
        logger = Logger()
        logger.start()
        logger.log_epoch({"train_loss": 0.5, "val_loss": 0.4}, epoch=0)
        logger.log_message("训练完成")
        logger.close()
    """

    def __init__(self, log_dir: Path = LOGS_DIR):
        self.log_dir = log_dir
        self.session_dir: Path | None = None
        self.csv_path: Path | None = None
        self.log_path: Path | None = None
        self.csv_file = None
        self.csv_writer = None
        self.logger: logging.Logger | None = None
        self.field_names: list[str] | None = None

    def start(self) -> None:
        """初始化日志会话,创建输出目录和文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.session_dir / "metrics.csv"
        self.log_path = self.session_dir / "train.log"

        # 文本日志
        self.logger = logging.getLogger(f"train_{timestamp}")
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

        # CSV 文件(延迟初始化 writer,在第一个 log_epoch 时确定表头)
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.log_message(f"日志会话开始: {self.session_dir}")

    def log_epoch(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        记录一个 epoch 的指标

        Args:
            metrics: 指标字典,如 {"train_loss": 0.5, "val_loss": 0.4}
            epoch: epoch 编号(0-based)
        """
        if self.csv_writer is None:
            # 首次调用时确定字段列表并写入表头
            self.field_names = ["epoch"] + list(metrics.keys())
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.field_names)
            self.csv_writer.writeheader()

        row = {"epoch": epoch + 1, **metrics}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # 同时写入文本日志
        self.logger.info(
            f"Epoch {epoch + 1}: "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    def log_message(self, message: str) -> None:
        """记录一条文本消息"""
        if self.logger:
            self.logger.info(message)

    def close(self) -> None:
        """关闭日志会话,释放文件句柄"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
