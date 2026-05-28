"""
Seq2Seq 模型封装模块(含 Attention)

将 Encoder 和 Decoder 串联为完整的序列到序列模型。
提供训练前向(含 Teacher Forcing)和推理生成(贪心解码)两种接口。

与无 Attention 版本的核心区别:
- forward() 中保存 encoder_outputs 并传递给 decoder 每一步
- generate() 支持 return_attention 参数,用于可视化注意力权重
"""

import random
from typing import Tuple

import torch
import torch.nn as nn

from config.defaults import DefaultParams, ModelParams
from src.model.decoder import Decoder
from src.model.encoder import Encoder


class Seq2Seq(nn.Module):
    """
    Seq2Seq 模型(含 Attention)

    完整流程:
    训练时: Encoder 编码输入 -> Teacher Forcing 驱动 Decoder(+Attention) -> 输出 logits
    推理时: Encoder 编码输入 -> 贪心解码自回归生成(+Attention) -> 输出 token 序列

    Args:
        encoder: Encoder 实例
        decoder: Decoder 实例
        vocab_size: 词表大小
        pad_index: PAD 标记索引
        sos_index: SOS 起始标记索引
        eos_index: EOS 终止标记索引
        device: 计算设备
        attention_type: 注意力类型(用于判断是否传递 encoder_outputs)
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        vocab_size: int,
        pad_index: int = 0,
        sos_index: int = 2,
        eos_index: int = 3,
        device: torch.device | None = DefaultParams.DEVICE,
        attention_type: str = ModelParams.ATTENTION_TYPE,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.device = device
        self.attention_type = attention_type

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        teacher_forcing_ratio: float = ModelParams.TEACHER_FORCING_RATIO,
    ) -> torch.Tensor:
        """
        训练前向传播

        先通过 Encoder 编码输入序列得到 outputs 和 hidden,
        再用 Teacher Forcing 驱动 Decoder(+Attention) 生成每个时间步的 logits。

        Args:
            encoder_input: 编码器输入索引,[batch_size, encoder_length]
            decoder_input: 解码器输入索引(含 SOS 不含 EOS),[batch_size, decoder_length]
            encoder_mask: 编码器有效位置 mask,[batch_size, encoder_length]
            teacher_forcing_ratio: Teacher Forcing 概率

        Returns:
            logits: [batch_size, decoder_length, vocab_size]
        """
        decoder_length = decoder_input.size(1)

        # Step 1: Encoder 编码
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)

        # Step 2: 准备 Decoder 初始输入与隐藏状态
        current_input = decoder_input[:, 0].unsqueeze(1)
        hidden_state = encoder_hidden

        # Step 3: 逐时间步解码
        logits_list = []

        for step in range(decoder_length):
            # 单步解码
            step_output, hidden_state, _ = self.decoder(
                current_input, hidden_state, encoder_outputs, encoder_mask
            )
            logits_list.append(step_output)

            # 最后一步不需要准备下一输入
            if step == decoder_length - 1:
                break

            # Teacher Forcing 决策
            if random.random() < teacher_forcing_ratio:
                current_input = decoder_input[:, step + 1].unsqueeze(1)
            else:
                predicted_token = step_output.squeeze(1).argmax(dim=1)
                current_input = predicted_token.unsqueeze(1)

        # Step 4: 拼接所有时间步的 logits
        logits = torch.cat(logits_list, dim=1)

        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        max_generation_length: int = 25,
        return_attention: bool = False,
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        推理生成(贪心解码)

        Encoder 编码后,从 <SOS> 开始自回归生成,
        每步选取概率最高的 token,直到遇到 <EOS> 或达到最大生成长度。

        Args:
            encoder_input: 编码器输入索引,[batch_size, encoder_length]
            encoder_mask: 编码器有效位置 mask
            max_generation_length: 最大生成长度
            return_attention: 是否返回注意力权重(用于可视化)

        Returns:
            若 return_attention=False:
                (generated_ids, sequence_lengths)
            若 return_attention=True:
                (generated_ids, sequence_lengths, attention_weights)
                attention_weights: [batch_size, max_len, encoder_seq_len]
        """
        batch_size = encoder_input.size(0)

        # Step 1: Encoder 编码
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)

        # Step 2: 初始化生成状态
        current_input = torch.full(
            (batch_size, 1),
            self.sos_index,
            dtype=torch.long,
            device=self.device,
        )
        hidden_state = encoder_hidden

        generated_ids = torch.full(
            (batch_size, max_generation_length),
            self.pad_index,
            dtype=torch.long,
            device=self.device,
        )
        sequence_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 收集注意力权重(如果需要)
        attention_weights_list = [] if return_attention else None

        # Step 3: 自回归逐时间步生成
        for step in range(max_generation_length):
            step_output, hidden_state, attn_weights = self.decoder(
                current_input, hidden_state, encoder_outputs, encoder_mask
            )

            if return_attention and attn_weights is not None:
                attention_weights_list.append(attn_weights)

            predicted_token = step_output.squeeze(1).argmax(dim=1)

            # 只更新未完成样本的位置,已完成样本保持初始化时的 PAD
            generated_ids[~is_finished, step] = predicted_token[~is_finished]

            # 记录 <EOS> 位置
            just_finished = (predicted_token == self.eos_index) & ~is_finished
            sequence_lengths[just_finished] = step + 1
            is_finished = is_finished | just_finished

            if is_finished.all():
                break

            # 已完成样本喂 PAD,避免 EOS 进入 decoder(训练时从未见过)
            next_token = predicted_token.clone()
            next_token[is_finished] = self.pad_index
            current_input = next_token.unsqueeze(1)

        sequence_lengths[~is_finished] = max_generation_length

        if return_attention and attention_weights_list:
            # attention_weights: [batch, gen_len, enc_len]
            attention_batch = torch.cat(attention_weights_list, dim=1)
            return generated_ids, sequence_lengths, attention_batch

        return generated_ids, sequence_lengths
