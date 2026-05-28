"""
评估器模块

提供 Evaluator 类,封装测试集评估的完整流程:
1. 遍历测试集,用 model.generate() 贪心解码生成预测
2. 计算 Exact Match + Token Accuracy 指标
3. 生成可视化图表(训练曲线、预测样本、错误分析、注意力热力图)
4. 输出格式化评估报告
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.defaults import InferenceParams
from config.paths import FIGURES_DIR
from src.data.mapping import VocabMapping
from src.evaluate.metrics import compute_metrics
from src.evaluate.visualize import (
    plot_attention_heatmap,
    plot_error_by_format_pair,
    plot_error_by_length,
    plot_metrics_report,
    plot_prediction_samples,
    plot_training_history,
)


class Evaluator:
    """
    测试集评估器

    使用模型的自回归生成能力在测试集上评估,
    收集预测结果、计算指标、生成可视化报告(含注意力热力图)。

    使用方式:
        evaluator = Evaluator(model, test_loader, vocab, device)
        report = evaluator.evaluate(history={"train_loss": [...], ...})
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        vocab: VocabMapping,
        device: torch.device,
    ):
        self.model = model
        self.test_loader = test_loader
        self.vocab = vocab
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        history: Optional[Dict[str, List[float]]] = None,
        output_dir: Path = FIGURES_DIR,
    ) -> dict:
        """
        执行完整评估流程

        Args:
            history: 训练历史字典(可选),传入则绘制训练曲线
            output_dir: 图表输出目录

        Returns:
            评估报告字典
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        # Step 1: 收集所有预测结果
        all_inputs: List[str] = []
        all_predictions: List[str] = []
        all_targets: List[str] = []

        all_generated_ids: List[torch.Tensor] = []
        all_target_tensors: List[torch.Tensor] = []

        # 使用固定的最大生成长度,避免不同 batch 间序列长度不一致
        max_gen_len = InferenceParams.MAX_GEN_LENGTH

        desc = "[Evaluate] Testing"
        for encoder_input, decoder_input, target_output, encoder_mask in tqdm(
            self.test_loader, desc=desc, unit="batch"
        ):
            encoder_input = encoder_input.to(self.device)
            encoder_mask = encoder_mask.to(self.device)
            target_output = target_output.to(self.device)

            batch_size = encoder_input.size(0)

            generated_ids, _ = self.model.generate(
                encoder_input,
                encoder_mask,
                max_generation_length=max_gen_len,
            )

            # 将生成序列和目标序列统一补齐到 max_gen_len
            if generated_ids.size(1) < max_gen_len:
                pad_tensor = torch.full(
                    (batch_size, max_gen_len - generated_ids.size(1)),
                    self.vocab.pad_index,
                    dtype=torch.long,
                    device=self.device,
                )
                generated_ids = torch.cat([generated_ids, pad_tensor], dim=1)
            if target_output.size(1) < max_gen_len:
                pad_tensor = torch.full(
                    (batch_size, max_gen_len - target_output.size(1)),
                    self.vocab.pad_index,
                    dtype=torch.long,
                    device=self.device,
                )
                target_output = torch.cat([target_output, pad_tensor], dim=1)

            for i in range(batch_size):
                input_ids = encoder_input[i].tolist()
                input_string = self.vocab.decode(input_ids, strip_special=True)

                target_ids = target_output[i].tolist()
                target_string = self.vocab.decode(target_ids, strip_special=True)

                prediction_ids = generated_ids[i].tolist()
                prediction_string = self.vocab.decode(
                    prediction_ids, strip_special=True
                )

                all_inputs.append(input_string)
                all_targets.append(target_string)
                all_predictions.append(prediction_string)

            all_generated_ids.append(generated_ids.cpu())
            all_target_tensors.append(target_output.cpu())

        # Step 2: 计算评估指标
        generated_concatenated = torch.cat(all_generated_ids, dim=0)
        target_concatenated = torch.cat(all_target_tensors, dim=0)

        # 如果生成序列比目标序列短,补 PAD 使维度一致
        if generated_concatenated.size(1) < target_concatenated.size(1):
            padding = torch.full(
                (
                    generated_concatenated.size(0),
                    target_concatenated.size(1) - generated_concatenated.size(1),
                ),
                self.vocab.pad_index,
                dtype=torch.long,
            )
            generated_concatenated = torch.cat([generated_concatenated, padding], dim=1)

        metrics = compute_metrics(
            generated_concatenated,
            target_concatenated,
            pad_index=self.vocab.pad_index,
            eos_index=self.vocab.eos_index,
        )

        # Step 3: 收集错误样本
        error_samples = []
        for i in range(len(all_inputs)):
            if all_predictions[i] != all_targets[i]:
                error_samples.append(
                    {
                        "input": all_inputs[i],
                        "target": all_targets[i],
                        "prediction": all_predictions[i],
                    }
                )

        # Step 4: 生成可视化图表
        generated_plots: List[Path] = []

        if history is not None and len(history.get("train_loss", [])) > 0:
            plot_path = plot_training_history(
                history, save_path=output_dir / "training_history.png"
            )
            generated_plots.append(plot_path)

        plot_path = plot_metrics_report(
            metrics, save_path=output_dir / "metrics_report.png"
        )
        generated_plots.append(plot_path)

        plot_path = plot_prediction_samples(
            all_inputs,
            all_predictions,
            all_targets,
            save_path=output_dir / "prediction_samples.png",
        )
        generated_plots.append(plot_path)

        plot_path = plot_error_by_length(
            all_inputs,
            all_predictions,
            all_targets,
            save_path=output_dir / "error_by_length.png",
        )
        generated_plots.append(plot_path)

        plot_path = plot_error_by_format_pair(
            all_inputs,
            all_predictions,
            all_targets,
            save_path=output_dir / "error_by_format.png",
        )
        generated_plots.append(plot_path)

        # Step 5: 注意力热力图(选取少量样本)
        try:
            attention_samples = min(5, len(self.test_loader.dataset))
            self._generate_attention_plots(
                attention_samples, output_dir, generated_plots
            )
        except Exception as e:
            print(f"  注意: 注意力热力图生成失败 ({e})")

        # Step 6: 打印报告
        self._print_report(metrics, error_samples, generated_plots)

        return {
            "metrics": metrics,
            "total_samples": len(all_inputs),
            "error_samples": error_samples,
            "generated_plots": [str(p) for p in generated_plots],
        }

    def _generate_attention_plots(
        self,
        num_samples: int,
        output_dir: Path,
        generated_plots: List[Path],
    ) -> None:
        """
        生成注意力热力图

        从测试集中取少量样本,调用 model.generate(return_attention=True),
        绘制每个样本的注意力权重热力图。
        """
        # 检查模型是否支持返回注意力权重
        if (
            not hasattr(self.model, "attention_type")
            or self.model.attention_type == "none"
        ):
            return

        # 获取一个 batch
        test_iter = iter(self.test_loader)
        try:
            encoder_input, _, target_output, encoder_mask = next(test_iter)
        except StopIteration:
            return

        encoder_input = encoder_input[:num_samples].to(self.device)
        target_output = target_output[:num_samples]
        encoder_mask = encoder_mask[:num_samples].to(self.device)

        generated_ids, _, attention_batch = self.model.generate(
            encoder_input,
            encoder_mask,
            max_generation_length=target_output.size(1),
            return_attention=True,
        )

        for i in range(min(num_samples, encoder_input.size(0))):
            # 解码输入字符序列
            input_ids = encoder_input[i].tolist()
            input_tokens = list(self.vocab.decode(input_ids, strip_special=True))

            # 解码生成的输出字符序列
            output_ids = generated_ids[i].tolist()
            output_tokens = list(self.vocab.decode(output_ids, strip_special=True))

            # 注意力权重: [gen_len, enc_len] -> 截取有效部分
            seq_len = min(len(output_tokens), attention_batch.size(1))
            enc_len = len(input_tokens)
            attn = attention_batch[i, :seq_len, :enc_len].cpu().numpy()

            if len(output_tokens) == 0 or len(input_tokens) == 0:
                continue

            save_path = output_dir / f"attention_heatmap_{i + 1}.png"
            plot_attention_heatmap(
                input_tokens,
                output_tokens,
                attn,
                save_path=save_path,
            )
            generated_plots.append(save_path)

    def _print_report(
        self,
        metrics: dict,
        error_samples: List[dict],
        generated_plots: List[Path],
    ) -> None:
        """打印格式化的评估报告"""
        print()
        print("=" * 60)
        print("  评估报告")
        print("=" * 60)
        print(f"  测试样本数:      {metrics['exact_match_total']}")
        print(f"  完全正确数:      {metrics['exact_match_correct']}")
        print(f"  完全匹配准确率:  {metrics['exact_match'] * 100:.2f}%")
        print(f"  字符级准确率:    {metrics['token_accuracy'] * 100:.2f}%")
        print(f"  (有效 token 数:  {metrics['token_total']})")
        print()

        error_count = len(error_samples)
        if error_count > 0:
            print(f"  错误样本数: {error_count}")
            print("  " + "-" * 56)
            print(f"  {'输入':<20} {'真实':<20} {'预测':<20}")
            print("  " + "-" * 56)
            display_count = min(10, error_count)
            for sample in error_samples[:display_count]:
                print(
                    f"  {sample['input']:<20} "
                    f"{sample['target']:<20} "
                    f"{sample['prediction']:<20}"
                )
            if error_count > display_count:
                print(f"  ... 还有 {error_count - display_count} 条错误")
            print()

        print(f"  图表输出: {len(generated_plots)} 个")
        for plot_path in generated_plots:
            print(f"    - {plot_path}")
        print("=" * 60)
        print()
