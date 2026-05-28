"""
默认超参数配置

- DefaultParams: 全局参数(随机种子、设备)
- DataParams: 数据生成与加载参数
- ModelParams: Seq2Seq 模型结构参数(含 Attention)
- TrainingParams: 训练相关参数
- InferenceParams: 推理相关参数
"""

from typing import Literal

import torch


class DefaultParams:
    """全局参数"""

    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DataParams:
    """数据生成与加载参数"""

    # 年份范围
    YEAR_MIN = 1900
    YEAR_MAX = 2100

    # 日期格式标识符
    DATE_FORMATS = ["ISO", "ENGLISH_FULL", "ENGLISH_ABBR", "SLASH", "CHINESE"]

    # 格式转换对: None 表示使用所有组合(共 5×4=20 对)
    FORMAT_PAIRS = None

    # 生成样本数量
    TRAIN_SIZE = 50000
    VAL_SIZE = 5000
    TEST_SIZE = 5000

    # DataLoader 参数
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    PIN_MEMORY = True
    SHUFFLE = True

    # 输入/输出序列最大长度, None 表示不限制
    MAX_INPUT_LENGTH = None
    MAX_OUTPUT_LENGTH = None

    # 词表最小频率
    MIN_FREQ = 1


class ModelParams:
    """Seq2Seq 模型结构参数(含 Attention)"""

    RNN_TYPE: Literal["LSTM", "RNN", "GRU"] = "LSTM"
    ATTENTION_TYPE: Literal["bahdanau", "luong", "nadaraya_watson", "none"] = "bahdanau"

    # 嵌入维度(日期词表约 73 字符,需要比算术任务更大的嵌入空间)
    ENCODER_EMBEDDING_DIM = 256
    DECODER_EMBEDDING_DIM = 256

    # 隐藏层维度(Encoder 与 Decoder 共享)
    HIDDEN_DIM = 256

    # RNN 层数
    ENCODER_NUM_LAYERS = 2
    DECODER_NUM_LAYERS = 2

    # Encoder 是否使用双向 RNN
    BIDIRECTIONAL = False

    # Teacher Forcing 概率
    TEACHER_FORCING_RATIO = 0.5

    # RNN Dropout
    DROPOUT = 0.3

    # Nadaraya-Watson 注意力的 Gaussian 核 sigma 参数
    NW_SIGMA = 1.0


class TrainingParams:
    """训练相关参数"""

    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 50
    GRAD_CLIP = 5.0

    OPTIMIZER: Literal["Adam", "SGD", "AdamW"] = "AdamW"
    WEIGHT_DECAY = 1e-4

    LR_SCHEDULER: Literal["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"] = (
        "ReduceLROnPlateau"
    )
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.5
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 3

    # 早停
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_MIN_DELTA = 1e-4

    LOG_INTERVAL = 50
    CHECKPOINT_INTERVAL = 5


class InferenceParams:
    """推理相关参数"""

    # 最长日期格式 "January 15, 2024" 约 18 字符,加 SOS/EOS 约 20,取 25 留余量
    MAX_GEN_LENGTH = 25
    TEMPERATURE = 0.8
    TOP_K = 3
    TOP_P = 0.9
