# -*- coding: utf-8 -*-
"""问题二 v2 的集中配置。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "2023年C题"
QUESTION_DIR = PROJECT_ROOT / "问题二"
OUTPUT_DIR = QUESTION_DIR / "结果_v2"
FIGURE_DIR = QUESTION_DIR / "图表_v2"
CACHE_DIR = QUESTION_DIR / ".cache_v2"

CATEGORIES = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
FUTURE_DATES = pd.date_range("2023-07-01", "2023-07-07", freq="D")
DATA_END = pd.Timestamp("2023-06-30")

RANDOM_SEED = 20230827
FULL_SCENARIOS = 6000
QUICK_SCENARIOS = 600
BOOTSTRAP_REPS = 120
QUICK_BOOTSTRAP_REPS = 30

MAIN_MARKUP_BAND = (0.35, 0.65)
MARKUP_BANDS = {
    "百分之二十五至百分之七十五": (0.25, 0.75),
    "百分之三十至百分之七十": (0.30, 0.70),
    "百分之三十五至百分之六十五": (0.35, 0.65),
    "百分之四十至百分之六十": (0.40, 0.60),
    "百分之四十五至百分之五十五": (0.45, 0.55),
}
LOSS_FACTORS = (0.8, 1.0, 1.2)
BOOTSTRAP_BLOCK_LENGTHS = (7, 14)
WEIGHT_WINDOWS = (90, 180, 365)
WEIGHT_SHRINK_K = (10, 20, 40)

DEMAND_MODELS = [
    "同星期最近4次均值",
    "同星期最近8次中位数",
    "近7日均值",
    "近14日均值",
    "星期加月份对数回归",
    "星期加月份及趋势对数回归",
]

COST_METHODS = [
    "近7日均值",
    "近14日均值",
    "近7日中位数",
    "近14日中位数",
    "指数加权移动平均",
    "同星期最近4次统计",
    "同星期最近8次统计",
    "指数加权移动平均加周内收缩",
    "阻尼趋势",
    "指数平滑",
    "滞后稳健回归",
]

PRICE_RESPONSE_MODELS = [
    "半对数加成",
    "对数加成",
    "半对数售价偏离",
    "对数售价比",
]

def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def figure_path(name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR / name


def all_source_paths() -> list[Path]:
    return [DATA_DIR / f"附件{i}.xlsx" for i in range(1, 5)] + [DATA_DIR / "C题.pdf"]


def config_snapshot() -> dict:
    return {
        "版本": __version__,
        "随机种子": RANDOM_SEED,
        "正式情景数": FULL_SCENARIOS,
        "快速情景数": QUICK_SCENARIOS,
        "自助法次数": BOOTSTRAP_REPS,
        "主要价格带": MAIN_MARKUP_BAND,
        "价格带敏感性": MARKUP_BANDS,
        "损耗率情景": LOSS_FACTORS,
        "权重窗口": WEIGHT_WINDOWS,
        "权重收缩参数": WEIGHT_SHRINK_K,
        "需求模型": DEMAND_MODELS,
        "成本模型": COST_METHODS,
        "价格响应模型": PRICE_RESPONSE_MODELS,
    }
