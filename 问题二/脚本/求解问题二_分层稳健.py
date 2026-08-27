# -*- coding: utf-8 -*-
"""2023 C题问题二正式求解入口。

正式链路：
全量净销量预测基础需求 -> 正常销售识别价格响应 -> 七日批发价路径回测预测 ->
同星期中央经营带内逐日局部稳健定价 -> 损耗修正报童补货。

底层实现放在 `内部/`，仓库外部只保留这一个正式入口。
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

# 内部模块比原脚本多一层目录，统一重新指向仓库根目录。
core.base.ROOT = REPO_ROOT
core.base.DATA = REPO_ROOT / "2023年C题"
core.base.OUT = REPO_ROOT / "问题二" / "结果"
core.base.FIG = REPO_ROOT / "问题二" / "图表"
core.OUT = core.base.OUT
core.base.OUT.mkdir(parents=True, exist_ok=True)

# 成本层：水平与动态候选共同滚动回测，只有动态路径确有证据时才采用。
core.base.cost_forecast = dynamic_cost.cost_forecast
core.base.cost_backtest = dynamic_cost.cost_backtest

# 定价层：保留原稳健价格弹性；按星期构造中央常规经营带并逐日优化。
dynamic_pricing.bind(core, dynamic_cost)
core.optimize_hybrid = dynamic_pricing.optimize_hybrid

# 正式建模说明由仓库根文件维护，不再额外生成重复说明文件。
core.write_summary = lambda *args, **kwargs: None


if __name__ == "__main__":
    core.main()
