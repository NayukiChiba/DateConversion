"""
DateConversion CLI 主入口

基于 Seq2Seq + Attention 的日期格式转换工具。

支持两种启动方式:

1. 交互式菜单(无参数):
       python main.py

2. 命令行子命令(带参数):
       python main.py data generate [OPTIONS]    — 生成数据集
       python main.py train   [OPTIONS]          — 训练模型
       python main.py eval    --checkpoint <path> — 评估模型
       python main.py predict --checkpoint <path> — 推理预测

CLI 用法示例:
    python main.py data generate --train-size 50000
    python main.py train --rnn-type LSTM --attention-type bahdanau --epochs 50
    python main.py train --resume outputs/checkpoints/last_model.pth
    python main.py eval --checkpoint outputs/checkpoints/best_model.pth
    python main.py predict --checkpoint outputs/checkpoints/best_model.pth --date "2024-01-15"
    python main.py predict --checkpoint outputs/checkpoints/best_model.pth --input test.csv --output result.csv
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli.menu import dispatch, show_menu
from src.cli.parser import build_parser


def main() -> None:
    """主入口函数"""

    # 无参数 -> 交互式菜单
    if len(sys.argv) == 1:
        show_menu()
        return

    # 有参数 -> 命令行解析
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch(args)


if __name__ == "__main__":
    main()
