# DateConversion

使用有注意力的 Seq2Seq(Encoder-Decoder)模型进行日期格式转换.

## 项目简介

本项目构建一个基于 RNN 的序列到序列模型,**使用注意力机制**,让模型学习将一种日期格式(如 "2024-01-15")转换为另一种日期格式(如 "January 15, 2024").

- **输入**:源日期格式字符串(字符级 token 序列)
- **输出**:目标日期格式字符串(字符级 token 序列)
- **模型**:Encoder-Decoder + Attention(可选 SimpleRNN / LSTM / GRU)
- **任务**:多种日期格式之间的相互转换

## 项目架构

```
DateConversion/
├── config/                          # 配置模块
│   ├── __init__.py
│   ├── paths.py                     # 路径常量
│   └── defaults.py                  # 默认超参数(dataclass)
├── src/
│   ├── __init__.py
│   ├── cli/                         # CLI 模块
│   │   ├── __init__.py
│   │   ├── parser.py                # 参数解析器(argparse)
│   │   └── menu.py                  # 子命令路由分发
│   ├── data/                        # 数据模块
│   │   ├── __init__.py
│   │   ├── generate.py              # 数据集生成(随机日期+格式转换)
│   │   ├── mapping.py               # 词表映射(char <-> index)
│   │   └── dataloader.py            # DataLoader 构建
│   ├── model/                       # 模型模块
│   │   ├── __init__.py              # 注册表 + buildModel()
│   │   ├── encoder.py               # Encoder(RNN/LSTM/GRU)
│   │   ├── decoder.py               # Decoder(RNN/LSTM/GRU,含 Attention)
│   │   ├── attention.py             # Attention 机制(Bahdanau / Luong)
│   │   └── seq2seq.py               # Seq2Seq 封装
│   ├── train/                       # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py               # 训练主循环(含 teacher forcing)
│   │   ├── optimizer.py             # 优化器构建
│   │   ├── scheduler.py             # 学习率调度器
│   │   ├── earlyStopping.py         # 早停机制
│   │   ├── checkpoint.py            # Checkpoint 管理
│   │   ├── logger.py                # 训练日志
│   │   └── utils.py                 # 工具函数
│   ├── evaluate/                    # 评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py             # 评估器
│   │   ├── metrics.py               # 评估指标(准确率、逐位匹配率)
│   │   └── visualize.py             # 可视化(训练曲线、注意力热力图)
│   └── inference/                   # 推理模块
│       ├── __init__.py
│       └── predictor.py             # 推理器(贪心解码/束搜索)
├── datasets/                        # 数据集目录
│   ├── raw/                         # 原始生成数据(train/val/test CSV)
│   └── processed/                   # 预处理缓存(tokenized 序列)
├── outputs/                         # 输出目录
│   ├── checkpoints/                 # 模型权重
│   ├── logs/                        # 训练日志
│   └── figures/                     # 评估图表(含注意力可视化)
├── notebooks/                       # 探索性分析
├── tests/                           # 单元测试
├── main.py                          # CLI 主入口
└── pyproject.toml                   # 项目元数据
```

## 快速开始

### 环境要求

- Python >= 3.11
- PyTorch >= 2.0(CUDA 可选)
- 其余依赖见 `pyproject.toml`

### 安装

```bash
# 克隆仓库
git clone https://github.com/NayukiChiba/DateConversion.git
cd DateConversion

# 创建虚拟环境并安装依赖
uv sync
uv sync --group dev
```

### 数据准备

```bash
# 生成训练/验证/测试数据集
python main.py data generate --train-size 50000 --val-size 5000 --test-size 5000
```

### 训练

```bash
# 使用 LSTM 训练
python main.py train --model lstm --epochs 50 --batch-size 128 --lr 0.001

# 使用 GRU 训练
python main.py train --model gru --epochs 50 --batch-size 128 --lr 0.001

# 使用 SimpleRNN 训练
python main.py train --model rnn --epochs 50 --batch-size 128 --lr 0.001
```

### 评估

```bash
python main.py eval --checkpoint outputs/checkpoints/best.pt
```

### 推理

```bash
# 单条推理
python main.py predict --checkpoint outputs/checkpoints/best.pt --date "2024-01-15"

# 批量推理
python main.py predict --checkpoint outputs/checkpoints/best.pt --file dates.txt
```

## 技术要点

### 数据生成

- 随机生成日期,覆盖不同年份(含闰年)、月份、日期的合法组合
- 支持多种日期格式间的相互转换,如:
  - ISO 格式:`2024-01-15`
  - 英文完整格式:`January 15, 2024`
  - 英文缩写格式:`Jan 15, 2024`
  - 斜杠格式:`01/15/2024`
  - 中文格式:`2024年1月15日`
- 特殊 token:`<SOS>`(解码开始)、`<EOS>`(解码结束)、`<PAD>`(填充)、`<UNK>`(未知)

### 模型架构

- **Encoder**:将输入字符序列编码为隐状态序列
- **Attention**:在每个解码步计算与 Encoder 隐状态的对齐权重,生成上下文向量
- **Decoder**:结合上下文向量和上一时刻输出,自回归解码,Teacher Forcing 训练
- **有 Attention**:Decoder 可动态关注输入序列的不同位置,缓解长序列信息瓶颈

### 训练策略

- 损失函数:CrossEntropyLoss(忽略 `<PAD>`)
- Teacher Forcing 比例:可配置(默认 0.5 概率)
- 优化器:AdamW
- 学习率调度:ReduceLROnPlateau
- 早停:监控 val loss,patience 可配置
- 梯度裁剪

### 评估指标

- 完全匹配准确率(Exact Match):预测序列与真实序列完全一致的样本比例
- 逐位准确率(Character Accuracy):每个位置预测正确的字符占比
- 注意力可视化:绘制注意力权重热力图,观察模型关注位置

## 配置

修改 `config/defaults.py` 调整默认参数:

```python
class ModelParams:
    RNN_TYPE = "LSTM"         # RNN / LSTM / GRU
    EMBEDDING_DIM = 256       # 嵌入维度
    HIDDEN_DIM = 256          # 隐藏层维度
    NUM_LAYERS = 2            # RNN 层数
    DROPOUT = 0.5             # Dropout 比例
    ATTENTION_TYPE = "bahdanau"  # bahdanau / luong

class TrainingParams:
    LEARNING_RATE = 0.001
    EPOCHS = 50
    CLIP_GRAD = 5.0           # 梯度裁剪阈值
    TEACHER_FORCING_RATIO = 0.5
    OPTIMIZER = "AdamW"
    LR_SCHEDULER = "ReduceLROnPlateau"
```

## 开发

```bash
uv run ruff check .    # 代码检查
uv run ruff format .   # 代码格式化
uv run pytest          # 运行测试
```

## 注意事项

- 注意力机制允许 Decoder 在每个解码步查看 Encoder 所有隐状态,相比无注意力模型能更好地处理长序列
- 日期转换要求精确的字符级输出,单个字符错误即视为完全匹配失败
- 生成数据时会自动过滤非法日期(如 2 月 30 日),确保训练数据有效性
- 闰年规则完全遵循公历标准
