"""
早停机制模块

监控验证损失,当连续 patience 轮内改善不足 min_delta 时触发早停,
避免过拟合并节省训练时间。
"""

from config.defaults import TrainingParams


class EarlyStopping:
    """
    早停机制

    跟踪最佳验证损失和未改善的连续轮次计数,
    当计数超过 patience 时触发早停。

    Args:
        patience: 容忍轮数
        min_delta: 最小改善阈值,低于此值视为未改善
    """

    def __init__(
        self,
        patience: int = TrainingParams.EARLY_STOP_PATIENCE,
        min_delta: float = TrainingParams.EARLY_STOP_MIN_DELTA,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应触发早停

        Args:
            val_loss: 当前轮的验证损失

        Returns:
            True 表示应停止训练
        """
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def state_dict(self) -> dict:
        """导出当前状态,用于 checkpoint 保存"""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "counter": self.counter,
        }

    def load_state_dict(self, state: dict) -> None:
        """从 checkpoint 恢复状态"""
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.best_score = state.get("best_score", float("inf"))
        self.counter = state.get("counter", 0)
