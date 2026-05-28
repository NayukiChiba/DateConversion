"""
推理模块

提供 Predictor 类,封装训练好的 Seq2Seq 模型对单条/批量/文件输入
进行日期格式转换推理的完整流程。
"""

import csv
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from config.defaults import DefaultParams, InferenceParams
from src.data.mapping import VocabMapping
from src.model import build_model


class Predictor:
    """
    Seq2Seq 日期转换推理器

    加载训练好的模型和词表,将输入日期字符串转换为目标格式。

    使用方式:
        predictor = Predictor.from_checkpoint("outputs/checkpoints/best_model.pth")
        result = predictor.predict("2024-01-15")  # -> "January 15, 2024"
        results = predictor.predict_batch(["2024-01-15", "2024-03-20"])
        predictor.predict_file("input.csv", "output.csv")
    """

    def __init__(
        self,
        model: nn.Module,
        vocab: VocabMapping,
        device: torch.device | str = DefaultParams.DEVICE,
    ):
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device) if isinstance(device, str) else device

        self.model.eval()
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device | str = DefaultParams.DEVICE,
    ) -> "Predictor":
        """
        从 checkpoint 文件构建 Predictor

        自动加载模型权重、重建词表。
        """
        device_obj = torch.device(device) if isinstance(device, str) else device

        checkpoint = torch.load(
            checkpoint_path, map_location=device_obj, weights_only=False
        )

        vocab = VocabMapping.from_dict(checkpoint["vocab"])

        model = build_model(
            vocab_size=vocab.vocabulary_size,
            pad_index=vocab.pad_index,
            sos_index=vocab.sos_index,
            eos_index=vocab.eos_index,
            attention_type=checkpoint.get("attention_type", "bahdanau"),
            device=device_obj,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, vocab, device_obj)

    # ==================================================================
    # 单条预测
    # ==================================================================
    def predict(self, date_string: str) -> str:
        """对单条日期字符串进行格式转换推理"""
        return self.predict_batch([date_string])[0]

    # ==================================================================
    # 批量预测
    # ==================================================================
    def predict_batch(self, date_strings: List[str]) -> List[str]:
        """对多条日期字符串进行批量推理"""
        if not date_strings:
            return []

        # Step 1: 编码所有日期字符串
        encoded_sequences = [
            self.vocab.encode(date_string, add_sos_eos=False)
            for date_string in date_strings
        ]
        max_length = max(len(sequence) for sequence in encoded_sequences)

        batch_size = len(encoded_sequences)
        encoder_input = torch.full(
            (batch_size, max_length),
            self.vocab.pad_index,
            dtype=torch.long,
            device=self.device,
        )
        encoder_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.bool,
            device=self.device,
        )

        for i, sequence in enumerate(encoded_sequences):
            seq_len = len(sequence)
            encoder_input[i, :seq_len] = torch.tensor(sequence, dtype=torch.long)
            encoder_mask[i, :seq_len] = True

        # Step 2: 模型生成
        with torch.no_grad():
            generated_ids, _ = self.model.generate(
                encoder_input,
                encoder_mask,
                max_generation_length=InferenceParams.MAX_GEN_LENGTH,
            )

        # Step 3: 解码为字符串
        results = []
        for i in range(batch_size):
            result_string = self.vocab.decode(
                generated_ids[i].tolist(),
                strip_special=True,
            )
            results.append(result_string)

        return results

    # ==================================================================
    # 文件预测
    # ==================================================================
    def predict_file(
        self,
        input_path: Path,
        output_path: Path,
        date_column: str = "input",
    ) -> Path:
        """
        对 CSV 文件中指定列的日期批量推理,结果写入新 CSV

        Args:
            input_path: 输入 CSV 文件路径
            output_path: 输出 CSV 文件路径
            date_column: 日期所在的列名,默认 "input"

        Returns:
            输出文件路径
        """
        with input_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_names = reader.fieldnames or []
            rows = list(reader)

        date_strings = [row[date_column].strip() for row in rows]
        predictions = self.predict_batch(date_strings)

        output_field_names = list(field_names) + ["prediction"]
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_field_names)
            writer.writeheader()
            for row, prediction in zip(rows, predictions):
                row["prediction"] = prediction
                writer.writerow(row)

        # 如果有真实答案列,计算准确率
        if "output" in field_names:
            targets = [row["output"].strip() for row in rows]
            correct = sum(1 for p, t in zip(predictions, targets) if p == t)
            total = len(targets)
            accuracy = correct / total if total > 0 else 0.0
            print(f"  文件推理完成: {input_path}")
            print(f"  样本数: {total}")
            print(f"  正确数: {correct}")
            print(f"  准确率: {accuracy * 100:.2f}%")

        return output_path
