# -*- coding: utf-8 -*-
"""2023 C题问题二正式求解入口。

正式链路：
全量净销量预测基础需求 -> 正常销售识别价格响应 -> 动态七日批发价曲线 ->
分层稳健定价 -> 损耗修正报童补货。

底层实现放在 `内部/`，避免正式目录同时出现多套“求解器”。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERNAL = HERE / "内部"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dynamic_cost = _load("q2_dynamic_cost", INTERNAL / "动态成本.py")
core = _load("q2_robust_core", INTERNAL / "分层稳健核心.py")

# 仅替换成本预测层；数据口径、需求模型、价格可靠性和报童模型保持正式方案原定义。
core.base.cost_forecast = dynamic_cost.cost_forecast
core.base.cost_backtest = dynamic_cost.cost_backtest


if __name__ == "__main__":
    core.main()
