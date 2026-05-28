"""
Encoder 编码器模块

将输入字符序列(如 "2024-01-15")编码为隐状态序列。
通过 RNN 逐字符处理,返回全部时间步的 outputs 和最后时刻的 hidden state。

支持 SimpleRNN / LSTM / GRU 三种 RNN 类型。
"""

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config.defaults import ModelParams


class Encoder(nn.Module):
    """
    Seq2Seq 编码器

    架构: Embedding -> Dropout -> RNN
    输出 RNN 全部时刻的 outputs 和最后时刻的 hidden state。

    Args:
        vocab_size: 词表大小
        embedding_dim: 词嵌入向量维度
        hidden_dim: RNN 隐藏层维度
        num_layers: RNN 堆叠层数
        rnn_type: RNN 类型,"LSTM" / "RNN" / "GRU"
        dropout: Dropout 概率
        pad_index: PAD 标记的索引
        bidirectional: 是否使用双向 RNN
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = ModelParams.ENCODER_EMBEDDING_DIM,
        hidden_dim: int = ModelParams.HIDDEN_DIM,
        num_layers: int = ModelParams.ENCODER_NUM_LAYERS,
        rnn_type: Literal["LSTM", "RNN", "GRU"] = ModelParams.RNN_TYPE,
        dropout: float = ModelParams.DROPOUT,
        pad_index: int = 0,
        bidirectional: bool = ModelParams.BIDIRECTIONAL,
    ):
        super().__init__()

        # Embedding 层: PAD 索引位置的嵌入向量为零,不参与训练更新
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )

        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # RNN 层: 根据 rnn_type 选择类型
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 编码器输入索引,形状 (batch_size, seq_len)
            mask: 有效位置掩码,形状 (batch_size, seq_len)
                  True=有效, False=PAD

        Returns:
            outputs: RNN 所有时刻的输出,
                     形状 (batch_size, seq_len, hidden_dim * num_directions)
            hidden: 最后时刻的隐藏状态
                   - LSTM: (h_n, c_n)
                   - RNN / GRU: h_n
        """
        # Step 1: Embedding + Dropout
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        # Step 2: RNN 前向
        if mask is not None:
            # 使用 pack_padded_sequence 跳过 PAD 位置加速计算
            seq_lengths = mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(
                embedded, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_outputs, hidden = self.rnn(packed)
            outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)

        return outputs, hidden
