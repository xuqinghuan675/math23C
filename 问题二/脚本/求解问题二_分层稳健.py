# -*- coding: utf-8 -*-
"""2023 C题问题二正式求解入口。

正式链路：
全量净销量预测基础需求 -> 正常销售识别价格响应 -> 动态七日批发价曲线 ->
弹性局部线性化形成逐日需求曲线 -> 同星期条件经营区间内定价 -> 损耗修正报童补货。

底层实现放在 `内部/`，避免正式目录同时出现多套“求解器”。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERNAL = HERE / "内部"
REPO_ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dynamic_cost = _load("q2_dynamic_cost", INTERNAL / "动态成本.py")
core = _load("q2_robust_core", INTERNAL / "分层稳健核心.py")
dynamic_pricing = _load("q2_dynamic_pricing", INTERNAL / "动态定价.py")

# 内部模块比原脚本多了一层目录，统一把数据/结果根目录重新指向仓库根目录。
core.base.ROOT = REPO_ROOT
core.base.DATA = REPO_ROOT / "2023年C题"
core.base.OUT = REPO_ROOT / "问题二" / "结果"
core.base.FIG = REPO_ROOT / "问题二" / "图表"
core.OUT = core.base.OUT
core.base.OUT.mkdir(parents=True, exist_ok=True)

# 成本层：水平 + 动态候选滚动回测，兼顾绝对水平和七日路径变化。
core.base.cost_forecast = dynamic_cost.cost_forecast
core.base.cost_backtest = dynamic_cost.cost_backtest

# 定价层：保留稳健弹性判别，将可靠弹性在历史经营点附近局部线性化，
# 使每天不同的基础需求真正移动利润曲线，而不是只做等比例缩放。
dynamic_pricing.bind(core, dynamic_cost)
core.optimize_hybrid = dynamic_pricing.optimize_hybrid


if __name__ == "__main__":
    core.main()
