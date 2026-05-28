"""
模型模块

提供模型注册表、build_model 工厂函数，
以及 Attention / Encoder / Decoder / Seq2Seq 的导出。
"""

from typing import Dict

import torch

from config.defaults import DefaultParams, ModelParams
from src.model.attention import Attention
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq

# RNN 类型注册表
MODEL_REGISTRY: Dict[str, str] = {
    "RNN": "RNN",
    "LSTM": "LSTM",
    "GRU": "GRU",
}

# 注意力类型注册表
ATTENTION_REGISTRY: Dict[str, str] = {
    "bahdanau": "Bahdanau Attention",
    "luong": "Luong Attention",
    "nadaraya_watson": "Nadaraya-Watson Attention",
    "none": "No Attention",
}


def build_model(
    vocab_size: int,
    pad_index: int = 0,
    sos_index: int = 2,
    eos_index: int = 3,
    # --- Encoder 参数 ---
    encoder_embedding_dim: int = ModelParams.ENCODER_EMBEDDING_DIM,
    encoder_num_layers: int = ModelParams.ENCODER_NUM_LAYERS,
    encoder_bidirectional: bool = ModelParams.BIDIRECTIONAL,
    # --- Decoder 参数 ---
    decoder_embedding_dim: int = ModelParams.DECODER_EMBEDDING_DIM,
    decoder_num_layers: int = ModelParams.DECODER_NUM_LAYERS,
    # --- 共享参数 ---
    hidden_dim: int = ModelParams.HIDDEN_DIM,
    rnn_type: str = ModelParams.RNN_TYPE,
    dropout: float = ModelParams.DROPOUT,
    # --- Attention 参数 ---
    attention_type: str = ModelParams.ATTENTION_TYPE,
    # --- 设备 ---
    device: torch.device | None = DefaultParams.DEVICE,
) -> Seq2Seq:
    """
    模型工厂函数

    根据传入的超参数构建 Encoder、Decoder 和 Seq2Seq 模型。

    Args:
        vocab_size: 词表大小
        pad_index: PAD 标记索引
        sos_index: SOS 起始标记索引
        eos_index: EOS 终止标记索引
        encoder_embedding_dim: Encoder 词嵌入维度
        encoder_num_layers: Encoder RNN 层数
        encoder_bidirectional: Encoder 是否双向
        decoder_embedding_dim: Decoder 词嵌入维度
        decoder_num_layers: Decoder RNN 层数
        hidden_dim: RNN 隐藏层维度
        rnn_type: RNN 类型
        dropout: Dropout 概率
        attention_type: 注意力类型
        device: 计算设备

    Returns:
        Seq2Seq 实例

    Raises:
        ValueError: 当 rnn_type 或 attention_type 不在注册表中时抛出
    """
    if rnn_type not in MODEL_REGISTRY:
        raise ValueError(
            f"未知的 RNN 类型: '{rnn_type}'，请从 {list(MODEL_REGISTRY.keys())} 中选择"
        )
    if attention_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"未知的注意力类型: '{attention_type}'，请从 {list(ATTENTION_REGISTRY.keys())} 中选择"
        )

    # 构建 Encoder
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=encoder_embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=encoder_num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        pad_index=pad_index,
        bidirectional=encoder_bidirectional,
    )

    # 构建 Decoder(含 Attention)
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=decoder_embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=decoder_num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        pad_index=pad_index,
        attention_type=attention_type,
        encoder_hidden_dim=hidden_dim,
        encoder_bidirectional=encoder_bidirectional,
    )

    # 封装为 Seq2Seq
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        pad_index=pad_index,
        sos_index=sos_index,
        eos_index=eos_index,
        device=device,
        attention_type=attention_type,
    )

    return model


__all__ = [
    "Attention",
    "Encoder",
    "Decoder",
    "Seq2Seq",
    "MODEL_REGISTRY",
    "ATTENTION_REGISTRY",
    "build_model",
]
