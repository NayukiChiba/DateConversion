"""
Decoder 解码器模块(含 Attention)

从 Encoder 的上下文向量出发,自回归地逐字符生成目标序列。
每个解码步通过 Attention 动态关注 Encoder 输出的不同位置,
缓解长序列的信息瓶颈问题。

架构: Embedding -> Dropout -> RNN -> Attention -> Linear(-> vocab_size)
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn

from config.defaults import ModelParams
from src.model.attention import Attention


class Decoder(nn.Module):
    """
    Seq2Seq 解码器(含 Attention)

    以 Encoder 的 final hidden state 初始化 RNN 的隐藏状态,
    每步生成时通过 Attention 计算上下文向量,
    将 RNN 输出与上下文向量拼接后映射到词表空间。

    Args:
        vocab_size: 词表大小
        embedding_dim: 词嵌入向量维度
        hidden_dim: RNN 隐藏层维度
        num_layers: RNN 堆叠层数
        rnn_type: RNN 类型,"LSTM" / "RNN" / "GRU"
        dropout: Dropout 概率
        pad_index: PAD 标记索引
        attention_type: 注意力类型
        encoder_hidden_dim: Encoder 输出维度(用于 Attention 和输出层)
        encoder_bidirectional: Encoder 是否双向(影响 encoder_output_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = ModelParams.DECODER_EMBEDDING_DIM,
        hidden_dim: int = ModelParams.HIDDEN_DIM,
        num_layers: int = ModelParams.DECODER_NUM_LAYERS,
        rnn_type: Literal["LSTM", "RNN", "GRU"] = ModelParams.RNN_TYPE,
        dropout: float = ModelParams.DROPOUT,
        pad_index: int = 0,
        attention_type: str = ModelParams.ATTENTION_TYPE,
        encoder_hidden_dim: int | None = None,
        encoder_bidirectional: bool = ModelParams.BIDIRECTIONAL,
    ):
        super().__init__()

        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type

        # RNN 层: Decoder 始终单向
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_class = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=False,
            batch_first=True,
        )

        # ——— Attention 层 ———
        # Encoder 输出维度(考虑了双向)
        if encoder_hidden_dim is None:
            encoder_hidden_dim = hidden_dim
        self.encoder_output_dim = encoder_hidden_dim * (
            2 if encoder_bidirectional else 1
        )

        if attention_type != "none":
            self.attention = Attention(
                encoder_hidden_dim=self.encoder_output_dim,
                decoder_hidden_dim=hidden_dim,
                attention_type=attention_type,
            )
        else:
            self.attention = None

        # ——— 输出投影层 ———
        # 有 Attention: [rnn_output; context_vector] -> vocab
        # 无 Attention: rnn_output -> vocab
        if attention_type != "none":
            output_input_dim = hidden_dim + self.encoder_output_dim
        else:
            output_input_dim = hidden_dim
        self.output_layer = nn.Linear(output_input_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor | None = None,
        encoder_mask: torch.Tensor | None = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor | None,
    ]:
        """
        前向传播(单步或全序列)

        Args:
            input_ids: 解码器输入索引,形状 (batch_size, seq_len)
            hidden: 从 Encoder 传递来的初始隐藏状态
            encoder_outputs: Encoder 所有时间步的输出
                             [batch_size, encoder_seq_len, encoder_output_dim]
                             无 Attention 时为 None
            encoder_mask: Encoder 有效位置 mask

        Returns:
            logits: 预测分数,(batch_size, seq_len, vocab_size)
            hidden: 更新后的隐藏状态
            attention_weights: 注意力权重,无 Attention 时为 None
                               [batch_size, seq_len, encoder_seq_len]
        """
        # Step 1: Embedding + Dropout
        embedded = self.dropout(self.embedding(input_ids))

        # Step 2: RNN 前向
        rnn_outputs, hidden = self.rnn(embedded, hidden)

        # Step 3: Attention + 输出投影
        if self.attention is not None and encoder_outputs is not None:
            seq_len = rnn_outputs.size(1)

            logits_list = []
            attn_weights_list = []

            for t in range(seq_len):
                # 当前时间步的 RNN 输出: [batch, hidden_dim]
                rnn_output_t = rnn_outputs[:, t, :]

                # 计算注意力: 用当前 RNN 输出作为 query
                context, attn_weights = self.attention(
                    rnn_output_t, encoder_outputs, encoder_mask
                )

                # 拼接 RNN 输出和上下文向量
                combined = torch.cat([rnn_output_t, context], dim=-1)

                # 输出投影
                logits_t = self.output_layer(combined).unsqueeze(1)
                logits_list.append(logits_t)
                attn_weights_list.append(attn_weights.unsqueeze(1))

            logits = torch.cat(logits_list, dim=1)
            # attention_weights: [batch, seq_len, encoder_seq_len]
            attention_weights = torch.cat(attn_weights_list, dim=1)
        else:
            # 无 Attention: 直接投影
            logits = self.output_layer(rnn_outputs)
            attention_weights = None

        return logits, hidden, attention_weights
