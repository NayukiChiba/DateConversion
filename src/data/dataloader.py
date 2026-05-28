"""
DataLoader 模块

提供 DateDataset 类和 create_data_loaders 工厂函数,
封装 PyTorch Dataset/DataLoader 的创建逻辑。

序列构造规则:
- Encoder 输入: [2, 0, 2, 4, -, 0, 1, -, 1, 5]   ← 纯字符索引,无 SOS/EOS
- Decoder 输入: [SOS, J, a, n, u, a, r, y, ...]   ← 含 SOS,不含 EOS
- Target 输出: [J, a, n, u, a, r, y, ..., EOS]     ← 不含 SOS,含 EOS

通过 Shift 对齐: decoder_input[i] 的监督目标是 target_output[i],
即每个时间步预测下一个字符。交叉熵计算时 ignore_index=PAD_IDX。
"""

import csv
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config.defaults import DataParams
from config.paths import RAW_TEST_PATH, RAW_TRAIN_PATH, RAW_VAL_PATH
from src.data.mapping import VocabMapping


class DateDataset(Dataset):
    """
    日期格式转换 Dataset

    加载 CSV 数据,通过 VocabMapping 将字符串 tokenize 为索引序列。

    每条 CSV 行 "2024-01-15","January 15, 2024" 被处理为三组索引:
    - encoder_input:  [4, 4, 4, 4, 69, 4, 4, 69, 4, 4]   ← "2024-01-15" 的索引
    - decoder_input:  [2, 40, 14, 27, 24, 14, 27, 34, ...]  ← "<SOS>January..." 的索引
    - target_output:  [40, 14, 27, 24, 14, 27, 34, ..., 3]  ← "January...<EOS>" 的索引
    """

    def __init__(self, csv_path: Path, vocab: VocabMapping):
        """
        Args:
            csv_path: CSV 文件路径(含 input/output 列)
            vocab: VocabMapping 实例
        """
        self.vocab = vocab
        self.samples: List[Tuple[List[int], List[int], List[int]]] = []

        # 在构造阶段一次性加载并 tokenize 全部样本
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_expr = row["input"].strip()
                output_result = row["output"].strip()

                # Encoder 输入: 纯字符索引,不加 SOS/EOS
                encoder_input = vocab.encode(input_expr, add_sos_eos=False)

                # Decoder 输入: 加 SOS/EOS,再去掉末尾 EOS -> [SOS, ...]
                decoder_input = vocab.encode(output_result, add_sos_eos=True)[:-1]

                # Target 输出: 加 SOS/EOS,再去掉开头 SOS -> [..., EOS]
                target_output = vocab.encode(output_result, add_sos_eos=True)[1:]

                self.samples.append((encoder_input, decoder_input, target_output))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_input, decoder_input, target_output = self.samples[idx]
        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target_output, dtype=torch.long),
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批次整理函数

    对 encoder/decoder/target 三组序列分别做 padding 到批次内最大长度。
    PAD_IDX=0 作为填充值。

    Returns:
        (encoder_input, decoder_input, target_output, encoder_mask)
    """
    encoder_inputs, decoder_inputs, target_outputs = zip(*batch)

    max_encoder_len = max(s.size(0) for s in encoder_inputs)
    max_decoder_len = max(s.size(0) for s in decoder_inputs)
    max_target_len = max(s.size(0) for s in target_outputs)
    batch_size = len(batch)

    # 全量初始化为 PAD_IDX=0,再逐条填入实际序列
    padded_encoder = torch.zeros((batch_size, max_encoder_len), dtype=torch.long)
    padded_decoder = torch.zeros((batch_size, max_decoder_len), dtype=torch.long)
    padded_target = torch.zeros((batch_size, max_target_len), dtype=torch.long)
    encode_mask = torch.zeros((batch_size, max_encoder_len), dtype=torch.bool)

    for i, (enc, dec, tgt) in enumerate(
        zip(encoder_inputs, decoder_inputs, target_outputs)
    ):
        padded_encoder[i, : enc.size(0)] = enc
        padded_decoder[i, : dec.size(0)] = dec
        padded_target[i, : tgt.size(0)] = tgt
        encode_mask[i, : enc.size(0)] = True

    return padded_encoder, padded_decoder, padded_target, encode_mask


def create_data_loaders(
    vocab: VocabMapping,
    train_csv_path: Path = RAW_TRAIN_PATH,
    val_csv_path: Path = RAW_VAL_PATH,
    test_csv_path: Path = RAW_TEST_PATH,
    batch_size: int = DataParams.BATCH_SIZE,
    num_workers: int = DataParams.NUM_WORKERS,
    pin_memory: bool = DataParams.PIN_MEMORY,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练集、验证集和测试集的 DataLoader

    训练集设 shuffle=True 以打乱样本顺序;
    验证集和测试集设 shuffle=False,保证每次评估结果一致。

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = DateDataset(train_csv_path, vocab)
    val_dataset = DateDataset(val_csv_path, vocab)
    test_dataset = DateDataset(test_csv_path, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
