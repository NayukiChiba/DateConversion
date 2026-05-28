"""
评估指标模块

提供 Seq2Seq 日期转换任务的评估指标:
- Exact Match(完全匹配): 生成序列与目标序列逐 token 一致
- Token Accuracy(字符级准确率): 每个位置预测正确的字符占比
"""

from typing import Tuple

import torch


def compute_exact_match(
    generated_ids: torch.Tensor,
    target_output: torch.Tensor,
    eos_index: int = 3,
) -> Tuple[int, int]:
    """
    计算完全匹配准确率

    逐样本比较生成序列与目标序列,所有非 PAD 位置全部一致才算正确。

    Returns:
        (correct, total)
    """
    batch_size = target_output.size(0)
    correct = 0

    for i in range(batch_size):
        target_sequence = target_output[i]

        # 找到 EOS 位置截断有效长度
        eos_mask = target_sequence == eos_index
        if eos_mask.any():
            target_length = eos_mask.nonzero(as_tuple=True)[0][0].item() + 1
            target_sequence = target_sequence[:target_length]

        prediction_sequence = generated_ids[i, :target_length]

        if torch.equal(prediction_sequence, target_sequence):
            correct += 1

    return correct, batch_size


def compute_token_accuracy(
    generated_ids: torch.Tensor,
    target_output: torch.Tensor,
    pad_index: int = 0,
    eos_index: int = 3,
) -> Tuple[int, int]:
    """
    计算字符级准确率

    逐位置比较预测 token 与目标 token,忽略 PAD 位置。

    Returns:
        (correct_tokens, total_tokens)
    """
    batch_size = target_output.size(0)
    total_correct = 0
    total_tokens = 0

    for i in range(batch_size):
        target_sequence = target_output[i]
        prediction_sequence = generated_ids[i, : target_sequence.size(0)]

        for j in range(target_sequence.size(0)):
            if target_sequence[j].item() == pad_index:
                continue
            total_tokens += 1
            if prediction_sequence[j].item() == target_sequence[j].item():
                total_correct += 1

    return total_correct, total_tokens


def compute_metrics(
    generated_ids: torch.Tensor,
    target_output: torch.Tensor,
    pad_index: int = 0,
    eos_index: int = 3,
) -> dict:
    """
    计算全部评估指标

    Returns:
        {
            "exact_match": float,
            "exact_match_correct": int,
            "exact_match_total": int,
            "token_accuracy": float,
            "token_correct": int,
            "token_total": int,
        }
    """
    exact_match_correct, exact_match_total = compute_exact_match(
        generated_ids, target_output, eos_index
    )
    token_correct, token_total = compute_token_accuracy(
        generated_ids, target_output, pad_index, eos_index
    )

    return {
        "exact_match": exact_match_correct / exact_match_total
        if exact_match_total > 0
        else 0.0,
        "exact_match_correct": exact_match_correct,
        "exact_match_total": exact_match_total,
        "token_accuracy": token_correct / token_total if token_total > 0 else 0.0,
        "token_correct": token_correct,
        "token_total": token_total,
    }
