"""
注意力机制模块

实现三种注意力类型:

1. Bahdanau (加性注意力):
   score(s_i, h_j) = v^T * tanh(W_dec * s_i + W_enc * h_j)
   编码器和解码器维度可以不同。

2. Luong (乘性注意力, General):
   score(s_i, h_j) = s_i^T * W * h_j
   通过投影矩阵 W 对齐维度。

3. Nadaraya-Watson (非参数化核回归):
   score(s_i, h_j) = -||s_i - h_j||^2 / (2 * sigma^2)
   无任何可学习参数, sigma 为配置超参数。

所有类型最终都通过 softmax 归一化得到注意力权重,
再对 encoder_outputs 加权求和得到上下文向量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.defaults import ModelParams


class Attention(nn.Module):
    """
    注意力机制

    支持 Bahdanau / Luong / Nadaraya-Watson 三种类型。

    Args:
        encoder_hidden_dim: Encoder 输出维度
        decoder_hidden_dim: Decoder 隐藏状态维度
        attention_type: 注意力类型
        sigma: Nadaraya-Watson 的 Gaussian 核带宽参数
    """

    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_type: str = ModelParams.ATTENTION_TYPE,
        sigma: float = ModelParams.NW_SIGMA,
    ):
        super().__init__()

        self.attention_type = attention_type
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        if attention_type == "bahdanau":
            # 加性注意力: 将 decoder_hidden 和 encoder_outputs 映射到同一空间
            self.attn_dim = decoder_hidden_dim
            self.W_dec = nn.Linear(decoder_hidden_dim, self.attn_dim, bias=False)
            self.W_enc = nn.Linear(encoder_hidden_dim, self.attn_dim, bias=False)
            self.v = nn.Linear(self.attn_dim, 1, bias=False)

        elif attention_type == "luong":
            # 乘性注意力: decoder_hidden_dim 通过 W 投影与 encoder_hidden_dim 对齐
            self.W = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)

        elif attention_type == "nadaraya_watson":
            # 非参数化: 无任何可学习参数,sigma 为带宽超参数
            self.sigma = sigma

        else:
            raise ValueError(
                f"未知的注意力类型: '{attention_type}', "
                "可选: bahdanau / luong / nadaraya_watson / none"
            )

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力权重和上下文向量

        Args:
            decoder_hidden: 当前时间步 Decoder 的隐藏状态
                            [batch_size, decoder_hidden_dim]
            encoder_outputs: Encoder 所有时间步的输出
                             [batch_size, encoder_seq_len, encoder_hidden_dim]
            encoder_mask: 有效位置 mask
                          [batch_size, encoder_seq_len], True=有效

        Returns:
            context_vector: 加权求和后的上下文向量
                            [batch_size, encoder_hidden_dim]
            attention_weights: 每个编码器时间步的注意力权重
                               [batch_size, encoder_seq_len]
        """
        batch_size, encoder_seq_len, _ = encoder_outputs.shape

        # ——— 计算注意力分数 ———
        if self.attention_type == "bahdanau":
            scores = self._bahdanau_score(decoder_hidden, encoder_outputs)
        elif self.attention_type == "luong":
            scores = self._luong_score(decoder_hidden, encoder_outputs)
        elif self.attention_type == "nadaraya_watson":
            scores = self._nadaraya_watson_score(decoder_hidden, encoder_outputs)
        else:
            raise ValueError(f"未知的注意力类型: {self.attention_type}")

        # ——— 对 PAD 位置施加负无穷,使其在 softmax 后权重为 0 ———
        if encoder_mask is not None:
            # mask: True=有效, False=PAD -> scores 的 PAD 位置置为 -inf
            scores = scores.masked_fill(~encoder_mask, float("-inf"))

        # ——— softmax 归一化得到注意力权重 ———
        attention_weights = F.softmax(scores, dim=-1)

        # ——— 加权求和得到上下文向量 ———
        # attention_weights: [batch_size, encoder_seq_len]
        # encoder_outputs:   [batch_size, encoder_seq_len, encoder_hidden_dim]
        # -> context_vector:  [batch_size, encoder_hidden_dim]
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)

        return context_vector, attention_weights

    # ==================================================================
    # 三种注意力评分函数
    # ==================================================================

    def _bahdanau_score(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bahdanau 加性注意力评分

        score = v^T * tanh(W_dec * s + W_enc * h)

        返回 [batch_size, encoder_seq_len]
        """
        # dec_proj: [batch, decoder_hid_dim] -> [batch, 1, attn_dim]
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)

        # enc_proj: [batch, enc_seq_len, enc_hid_dim] -> [batch, enc_seq_len, attn_dim]
        enc_proj = self.W_enc(encoder_outputs)

        # energy: [batch, enc_seq_len, attn_dim]
        energy = torch.tanh(dec_proj + enc_proj)

        # scores: [batch, enc_seq_len, 1] -> [batch, enc_seq_len]
        scores = self.v(energy).squeeze(-1)

        return scores

    def _luong_score(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Luong General 乘性注意力评分

        score = s^T * W * h

        返回 [batch_size, encoder_seq_len]
        """
        # enc_proj: [batch, enc_seq_len, enc_hid_dim] -> [batch, enc_seq_len, dec_hid_dim]
        enc_proj = self.W(encoder_outputs)

        # scores = decoder_hidden @ enc_proj^T
        # [batch, dec_hid_dim] @ [batch, dec_hid_dim, enc_seq_len] -> [batch, enc_seq_len]
        scores = torch.bmm(
            decoder_hidden.unsqueeze(1),
            enc_proj.transpose(1, 2),
        ).squeeze(1)

        return scores

    def _nadaraya_watson_score(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Nadaraya-Watson 非参数化核回归评分

        K(u) = exp(-||u||^2 / (2 * sigma^2))
        score = -||s - h_j||^2 / (2 * sigma^2)

        使用 Gaussian 核度量 decoder_hidden 与每个 encoder_output 的相似度,
        无任何可学习参数。

        返回 [batch_size, encoder_seq_len]
        """
        # 计算平方欧氏距离: ||s - h_j||^2
        # decoder_hidden: [batch, dec_dim] -> [batch, 1, dec_dim]
        # encoder_outputs: [batch, enc_seq_len, enc_dim]
        # 注意: 需要 dec_dim == enc_dim
        dec_expanded = decoder_hidden.unsqueeze(1)  # [batch, 1, dec_dim]

        # squared_diff: [batch, enc_seq_len, dim]
        squared_diff = (dec_expanded - encoder_outputs) ** 2

        # dist_sq: [batch, enc_seq_len] — 各位置的平方欧氏距离
        dist_sq = squared_diff.sum(dim=-1)

        # Gaussian 核评分: -dist^2 / (2 * sigma^2)
        scores = -dist_sq / (2.0 * self.sigma**2)

        return scores
