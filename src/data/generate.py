"""
数据集生成模块

随机生成合法日期,并转换为多种格式,构造"源格式->目标格式"的转换对。

五种日期格式:
- ISO:          2024-01-15
- ENGLISH_FULL: January 15, 2024
- ENGLISH_ABBR: Jan 15, 2024
- SLASH:        01/15/2024
- CHINESE:      2024年1月15日

规则:
- 随机生成 (year, month, day) 合法元组
- 处理闰年及各月天数
- 随机选择不同的源格式和目标格式
- 保存为 CSV(input, output 两列)

使用方法:
    python -m src.data.generate
"""

import csv
import random
from pathlib import Path

from config.defaults import DataParams, DefaultParams
from config.paths import RAW_TEST_PATH, RAW_TRAIN_PATH, RAW_VAL_PATH

# 月份名称常量
MONTHS_EN_FULL = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_EN_ABBR = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
# 每月天数(非闰年,2 月为 28)
MONTH_DAYS_NO_LEAP = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap_year(year: int) -> bool:
    """
    判断是否为闰年

    公历闰年规则:
    - 能被 4 整除但不能被 100 整除
    - 能被 400 整除
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def get_days_in_month(year: int, month: int) -> int:
    """返回指定年月的天数"""
    if month == 2 and is_leap_year(year):
        return 29
    return MONTH_DAYS_NO_LEAP[month - 1]


def generate_one_date(
    year_min: int = DataParams.YEAR_MIN,
    year_max: int = DataParams.YEAR_MAX,
) -> tuple[int, int, int]:
    """
    随机生成一个合法日期

    Returns:
        (year, month, day) 元组
    """
    year = random.randint(year_min, year_max)
    month = random.randint(1, 12)
    day = random.randint(1, get_days_in_month(year, month))
    return year, month, day


def format_date(year: int, month: int, day: int, fmt: str) -> str:
    """
    按指定格式输出日期字符串

    Args:
        year: 年份
        month: 月份(1-12)
        day: 日期(1-31)
        fmt: 格式标识符,取值为 DATE_FORMATS 中的值

    Returns:
        格式化后的日期字符串
    """
    if fmt == "ISO":
        return f"{year:04d}-{month:02d}-{day:02d}"
    elif fmt == "ENGLISH_FULL":
        month_name = MONTHS_EN_FULL[month - 1]
        return f"{month_name} {day}, {year}"
    elif fmt == "ENGLISH_ABBR":
        month_abbr = MONTHS_EN_ABBR[month - 1]
        return f"{month_abbr} {day}, {year}"
    elif fmt == "SLASH":
        return f"{month:02d}/{day:02d}/{year}"
    elif fmt == "CHINESE":
        return f"{year}年{month}月{day}日"
    else:
        raise ValueError(f"未知的日期格式: {fmt}")


def generate_one_sample(
    year_min: int = DataParams.YEAR_MIN,
    year_max: int = DataParams.YEAR_MAX,
    formats: list | None = None,
) -> tuple[str, str]:
    """
    生成一个日期格式转换对

    随机选取不同的源格式和目标格式。
    输入前缀目标格式标记,消除歧义(同一源日期可能对应多种目标格式)。

    Returns:
        (input_seq, output_seq): 如 ("ENGLISH_FULL:2024-01-15", "January 15, 2024")
    """
    if formats is None:
        formats = DataParams.DATE_FORMATS

    year, month, day = generate_one_date(year_min, year_max)

    # 随机选取两个不同的格式
    source_fmt, target_fmt = random.sample(formats, 2)

    input_seq = f"{target_fmt}:{format_date(year, month, day, source_fmt)}"
    output_seq = format_date(year, month, day, target_fmt)
    return input_seq, output_seq


def generate_samples(
    num_samples: int,
    year_min: int = DataParams.YEAR_MIN,
    year_max: int = DataParams.YEAR_MAX,
    formats: list | None = None,
) -> list[tuple[str, str]]:
    """
    生成指定数量的日期格式转换对

    Args:
        num_samples: 样本数量
        year_min: 年份范围最小值
        year_max: 年份范围最大值
        formats: 日期格式列表,None 则使用全部五种格式

    Returns:
        [(input_seq, output_seq), ...]
    """
    samples = []
    for _ in range(num_samples):
        samples.append(generate_one_sample(year_min, year_max, formats))
    return samples


def save_to_csv(samples: list[tuple[str, str]], file_path: Path) -> None:
    """将样本保存为 CSV 文件"""
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["input", "output"])
        writer.writerows(samples)
    print(f"Saved {len(samples)} samples to {file_path}")


def generate_datasets(
    train_size: int = DataParams.TRAIN_SIZE,
    val_size: int = DataParams.VAL_SIZE,
    test_size: int = DataParams.TEST_SIZE,
    seed: int = DefaultParams.RANDOM_SEED,
) -> None:
    """
    生成训练集、验证集和测试集

    Args:
        train_size: 训练集样本数
        val_size: 验证集样本数
        test_size: 测试集样本数
        seed: 随机种子
    """
    random.seed(seed)

    train_samples = generate_samples(train_size)
    val_samples = generate_samples(val_size)
    test_samples = generate_samples(test_size)

    save_to_csv(train_samples, RAW_TRAIN_PATH)
    save_to_csv(val_samples, RAW_VAL_PATH)
    save_to_csv(test_samples, RAW_TEST_PATH)


if __name__ == "__main__":
    generate_datasets()
