"""
可视化模块

基于 matplotlib 提供训练相关的可视化功能:
- 训练历史曲线(损失 + 准确率双 Y 轴)
- 预测样本展示
- 错误分析(按长度/格式对分类)
- 注意力热力图
"""

import os

# 修复 Windows 上 PyTorch 与 matplotlib 的 OpenMP 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 无图形界面环境使用非交互式后端
matplotlib.use("Agg")

from config.paths import FIGURES_DIR


def setup_chinese_font():
    """配置中文字体支持"""
    chinese_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> Path:
    """绘制训练历史曲线(双 Y 轴: 左=损失,右=准确率)"""
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "training_history.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, left_axis = plt.subplots(figsize=(10, 6))

    color_train = "#1f77b4"
    color_val = "#ff7f0e"
    left_axis.set_xlabel("Epoch", fontsize=12)
    left_axis.set_ylabel("Loss", fontsize=12, color="black")

    if "train_loss" in history and len(history["train_loss"]) > 0:
        left_axis.plot(
            epochs,
            history["train_loss"],
            color=color_train,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Train Loss",
        )
    if "val_loss" in history and len(history["val_loss"]) > 0:
        left_axis.plot(
            epochs,
            history["val_loss"],
            color=color_val,
            linewidth=1.5,
            marker="s",
            markersize=4,
            label="Val Loss",
        )

    left_axis.tick_params(axis="y")
    left_axis.grid(True, alpha=0.3)

    right_axis = left_axis.twinx()
    right_axis.set_ylabel("Accuracy", fontsize=12, color="green")

    if "val_accuracy" in history and len(history["val_accuracy"]) > 0:
        right_axis.plot(
            epochs,
            history["val_accuracy"],
            color="green",
            linewidth=1.5,
            marker="^",
            markersize=4,
            label="Val Accuracy",
        )
    right_axis.set_ylim(0, 1.05)
    right_axis.tick_params(axis="y", labelcolor="green")

    lines_left, labels_left = left_axis.get_legend_handles_labels()
    lines_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper right",
        fontsize=10,
    )

    plt.title("Training History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_prediction_samples(
    inputs: List[str],
    predictions: List[str],
    targets: List[str],
    num_samples: int = 20,
    save_path: Optional[Path] = None,
) -> Path:
    """绘制预测样本对比表格(错误行为红色高亮)"""
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "prediction_samples.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(inputs)
    if total > num_samples:
        indices = np.random.choice(total, num_samples, replace=False)
    else:
        indices = range(total)
        num_samples = total

    rows: List[List[str]] = []
    cell_colors: List[List[str]] = []

    for idx in indices:
        is_correct = predictions[idx] == targets[idx]
        rows.append([inputs[idx], targets[idx], predictions[idx]])
        if is_correct:
            cell_colors.append(["white", "white", "white"])
        else:
            cell_colors.append(["#ffcccc", "#ffcccc", "#ffcccc"])

    fig, ax = plt.subplots(figsize=(14, num_samples * 0.4 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Input", "Target", "Prediction"],
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    for col_idx in range(3):
        header_cell = table[0, col_idx]
        header_cell.set_facecolor("#4472C4")
        header_cell.set_text_props(color="white", fontweight="bold")

    plt.title("Prediction Samples (Red = Error)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_error_by_length(
    inputs: List[str],
    predictions: List[str],
    targets: List[str],
    save_path: Optional[Path] = None,
) -> Path:
    """按输入字符串长度分组绘制准确率柱状图"""
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "error_by_length.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    length_stats: Dict[int, Dict[str, int]] = {}
    for input_str, prediction, target in zip(inputs, predictions, targets):
        input_len = len(input_str)
        if input_len not in length_stats:
            length_stats[input_len] = {"correct": 0, "total": 0}
        length_stats[input_len]["total"] += 1
        if prediction == target:
            length_stats[input_len]["correct"] += 1

    sorted_lengths = sorted(length_stats.keys())
    accuracies = [
        length_stats[length]["correct"] / length_stats[length]["total"]
        for length in sorted_lengths
    ]
    totals = [length_stats[length]["total"] for length in sorted_lengths]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        [str(length) for length in sorted_lengths],
        accuracies,
        color="#4472C4",
        alpha=0.85,
    )
    ax.set_xlabel("Input Length (characters)", fontsize=12)
    ax.set_ylabel("Exact Match Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, total, accuracy in zip(bars, totals, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="gray",
        )

    plt.title("Exact Match Accuracy by Input Length", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_error_by_format_pair(
    inputs: List[str],
    predictions: List[str],
    targets: List[str],
    save_path: Optional[Path] = None,
) -> Path:
    """
    按格式转换对分组绘制准确率柱状图

    从输入/输出字符串的特征推断源格式和目标格式,
    按转换对分组统计准确率。
    """
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "error_by_format.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    format_stats: Dict[str, Dict[str, int]] = {}

    for input_str, prediction, target in zip(inputs, predictions, targets):
        # 从字符串特征推断格式标签
        src_label = _infer_format_label(input_str)
        tgt_label = _infer_format_label(target)
        pair_key = f"{src_label} -> {tgt_label}"

        if pair_key not in format_stats:
            format_stats[pair_key] = {"correct": 0, "total": 0}
        format_stats[pair_key]["total"] += 1
        if prediction == target:
            format_stats[pair_key]["correct"] += 1

    # 按准确率排序
    sorted_pairs = sorted(
        format_stats.keys(),
        key=lambda k: (
            format_stats[k]["correct"] / format_stats[k]["total"]
            if format_stats[k]["total"] > 0
            else 0
        ),
        reverse=True,
    )
    accuracies = [
        format_stats[pair]["correct"] / format_stats[pair]["total"]
        for pair in sorted_pairs
    ]
    totals = [format_stats[pair]["total"] for pair in sorted_pairs]

    fig, ax = plt.subplots(figsize=(12, max(5, len(sorted_pairs) * 0.5)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_pairs)))
    bars = ax.barh(sorted_pairs, accuracies, color=colors, alpha=0.85)
    ax.set_xlabel("Exact Match Accuracy", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    for bar, total, accuracy in zip(bars, totals, accuracies):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy:.2%} (n={total})",
            va="center",
            fontsize=9,
        )

    plt.title("Exact Match Accuracy by Format Pair", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_metrics_report(
    metrics: dict,
    save_path: Optional[Path] = None,
) -> Path:
    """绘制评估指标汇总图(水平柱状图)"""
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "metrics_report.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 3))

    metric_names = ["Exact Match", "Token Accuracy"]
    metric_values = [
        metrics.get("exact_match", 0.0) * 100,
        metrics.get("token_accuracy", 0.0) * 100,
    ]
    bar_colors = ["#4472C4", "#ED7D31"]

    bars = ax.barh(
        metric_names, metric_values, color=bar_colors, alpha=0.85, height=0.5
    )
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.title("Evaluation Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_attention_heatmap(
    input_tokens: List[str],
    output_tokens: List[str],
    attention_weights: np.ndarray,
    save_path: Optional[Path] = None,
) -> Path:
    """
    绘制注意力权重热力图

    Args:
        input_tokens: 输入字符列表(Encoder 输入)
        output_tokens: 输出字符列表(Decoder 生成,去 SOS/EOS)
        attention_weights: 注意力权重矩阵 [output_len, input_len]
        save_path: 图表保存路径

    Returns:
        保存的文件路径
    """
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "attention_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    output_len, input_len = attention_weights.shape
    fig, ax = plt.subplots(figsize=(max(8, input_len * 0.6), max(6, output_len * 0.6)))

    im = ax.imshow(attention_weights, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(input_len))
    ax.set_xticklabels(input_tokens, fontsize=10)
    ax.set_yticks(range(output_len))
    ax.set_yticklabels(output_tokens, fontsize=10)

    ax.set_xlabel("Encoder Input", fontsize=12)
    ax.set_ylabel("Decoder Output", fontsize=12)
    ax.set_title("Attention Weights Heatmap", fontsize=14, fontweight="bold")

    # 在每个单元格标注数值
    for i in range(output_len):
        for j in range(input_len):
            weight = attention_weights[i, j]
            if weight > 0.05:
                text_color = "white" if weight > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{weight:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _infer_format_label(date_str: str) -> str:
    """从日期字符串推断格式标签(用于可视化分组)"""
    if "-" in date_str and date_str[0].isdigit():
        return "ISO"
    elif "/" in date_str:
        return "Slash"
    elif "年" in date_str or "月" in date_str or "日" in date_str:
        return "Chinese"
    elif date_str[0].isupper() and len(date_str.split()[0]) > 3:
        return "EnglishFull"
    elif date_str[0].isupper():
        return "EnglishAbbr"
    return "Other"
