"""
CLI 参数解析模块

提供 build_parser 函数,构造 argparse 解析器,
支持 data generate / train / eval / predict 四个子命令。
"""

import argparse

from config.defaults import DataParams, InferenceParams, ModelParams, TrainingParams


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器

    四个子命令:
        data generate — 生成日期转换数据集
        train         — 训练模型
        eval          — 评估模型
        predict       — 推理预测

    Returns:
        配置好的 argparse.ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        prog="DateConversion",
        description="基于 Seq2Seq+Attention 的日期格式转换工具",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ==================================================================
    # data generate 子命令
    # ==================================================================
    data_parser = subparsers.add_parser("data", help="数据管理")
    data_subparsers = data_parser.add_subparsers(dest="data_command")

    gen_parser = data_subparsers.add_parser("generate", help="生成日期转换数据集")
    gen_parser.add_argument(
        "--train-size",
        type=int,
        default=DataParams.TRAIN_SIZE,
        help="训练集样本数 (默认: %(default)s)",
    )
    gen_parser.add_argument(
        "--val-size",
        type=int,
        default=DataParams.VAL_SIZE,
        help="验证集样本数 (默认: %(default)s)",
    )
    gen_parser.add_argument(
        "--test-size",
        type=int,
        default=DataParams.TEST_SIZE,
        help="测试集样本数 (默认: %(default)s)",
    )
    gen_parser.add_argument(
        "--year-min",
        type=int,
        default=DataParams.YEAR_MIN,
        help="年份范围最小值 (默认: %(default)s)",
    )
    gen_parser.add_argument(
        "--year-max",
        type=int,
        default=DataParams.YEAR_MAX,
        help="年份范围最大值 (默认: %(default)s)",
    )

    # ==================================================================
    # train 子命令
    # ==================================================================
    train_parser = subparsers.add_parser("train", help="训练模型")

    # 模型结构
    model_group = train_parser.add_argument_group("模型结构")
    model_group.add_argument(
        "--rnn-type",
        type=str,
        default=ModelParams.RNN_TYPE,
        choices=["LSTM", "RNN", "GRU"],
        help="RNN 类型 (默认: %(default)s)",
    )
    model_group.add_argument(
        "--attention-type",
        type=str,
        default=ModelParams.ATTENTION_TYPE,
        choices=["bahdanau", "luong", "nadaraya_watson", "none"],
        help="注意力机制类型 (默认: %(default)s)",
    )
    model_group.add_argument(
        "--hidden-dim",
        type=int,
        default=ModelParams.HIDDEN_DIM,
        help="RNN 隐藏层维度 (默认: %(default)s)",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=ModelParams.DROPOUT,
        help="Dropout 概率 (默认: %(default)s)",
    )
    model_group.add_argument(
        "--teacher-forcing-ratio",
        type=float,
        default=ModelParams.TEACHER_FORCING_RATIO,
        help="Teacher Forcing 概率 (默认: %(default)s)",
    )
    model_group.add_argument(
        "--bidirectional",
        action="store_true",
        default=ModelParams.BIDIRECTIONAL,
        help="Encoder 使用双向 RNN",
    )

    # 训练超参数
    train_group = train_parser.add_argument_group("训练超参数")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=TrainingParams.EPOCHS,
        help="训练轮数 (默认: %(default)s)",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=DataParams.BATCH_SIZE,
        help="批大小 (默认: %(default)s)",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=TrainingParams.LEARNING_RATE,
        help="学习率 (默认: %(default)s)",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default=TrainingParams.OPTIMIZER,
        choices=["Adam", "SGD", "AdamW"],
        help="优化器类型 (默认: %(default)s)",
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=TrainingParams.WEIGHT_DECAY,
        help="权重衰减系数 (默认: %(default)s)",
    )
    train_group.add_argument(
        "--grad-clip",
        type=float,
        default=TrainingParams.GRAD_CLIP,
        help="梯度裁剪阈值 (默认: %(default)s)",
    )

    # 学习率调度
    scheduler_group = train_parser.add_argument_group("学习率调度")
    scheduler_group.add_argument(
        "--lr-scheduler",
        type=str,
        default=TrainingParams.LR_SCHEDULER,
        choices=["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"],
        help="学习率调度器类型 (默认: %(default)s)",
    )
    scheduler_group.add_argument(
        "--lr-step-size",
        type=int,
        default=TrainingParams.LR_STEP_SIZE,
        help="StepLR 衰减周期 (默认: %(default)s)",
    )
    scheduler_group.add_argument(
        "--lr-gamma",
        type=float,
        default=TrainingParams.LR_GAMMA,
        help="StepLR 衰减因子 (默认: %(default)s)",
    )

    # 早停
    early_stop_group = train_parser.add_argument_group("早停")
    early_stop_group.add_argument(
        "--early-stop-patience",
        type=int,
        default=TrainingParams.EARLY_STOP_PATIENCE,
        help="早停容忍轮数 (默认: %(default)s)",
    )
    early_stop_group.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=TrainingParams.EARLY_STOP_MIN_DELTA,
        help="早停最小改善阈值 (默认: %(default)s)",
    )

    # 其他
    other_group = train_parser.add_argument_group("其他")
    other_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 恢复训练的路径",
    )
    other_group.add_argument(
        "--skip-generate",
        action="store_true",
        help="跳过数据生成(使用已有 CSV 文件)",
    )

    # ==================================================================
    # eval 子命令
    # ==================================================================
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 文件路径",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=DataParams.BATCH_SIZE,
        help="批大小 (默认: %(default)s)",
    )

    # ==================================================================
    # predict 子命令
    # ==================================================================
    predict_parser = subparsers.add_parser("predict", help="推理预测")
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 文件路径",
    )
    predict_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="要转换的日期,如 '2024-01-15'",
    )
    predict_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="批量预测的输入 CSV 文件路径",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="批量预测的输出 CSV 文件路径",
    )
    predict_parser.add_argument(
        "--max-generation-length",
        type=int,
        default=InferenceParams.MAX_GEN_LENGTH,
        help="最大生成长度 (默认: %(default)s)",
    )

    return parser
