# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题二" / "结果_v2"

import sys

sys.path.insert(0, str(ROOT / "问题二" / "脚本"))

from v2.optimization import scenario_profit_quantiles


def test_discrete_grids_and_profit_intervals():
    final = pd.read_csv(OUT / "09_七天六品类最终策略.csv", encoding="utf-8-sig")
    assert np.allclose(final["稳健推荐售价"] * 100, np.round(final["稳健推荐售价"] * 100))
    assert np.allclose(final["建议补货量"] * 10, np.round(final["建议补货量"] * 10))
    assert (final["稳健推荐售价"] > final["预测批发价P50"] / (1.0 - final["附件四品类损耗率"])).all()
    assert (final["毛利P10"] <= final["毛利P50"]).all()
    assert (final["毛利P50"] <= final["毛利P90"]).all()


def test_neighbour_check_and_strategy_algebra():
    cells = pd.read_csv(OUT / "07_策略逐日结果.csv", encoding="utf-8-sig")
    assert set(cells["策略"]) == {"A传统基准", "B仅优化补货", "C期望利润最大", "D稳健经营"}
    assert len(cells) == 42 * 4
    decomp = pd.read_csv(OUT / "07_策略收益分解.csv", encoding="utf-8-sig")
    for cat, sub in decomp.groupby("品类"):
        values = sub.set_index("策略")["七天预计毛利"]
        improvements = sub.set_index("策略")["相对A改善"]
        assert abs(improvements["B仅优化补货"] - (values["B仅优化补货"] - values["A传统基准"])) < 1e-7
        assert abs(improvements["C期望利润最大"] - (values["C期望利润最大"] - values["A传统基准"])) < 1e-7
        assert abs(improvements["D稳健经营"] - (values["D稳健经营"] - values["A传统基准"])) < 1e-7


def test_joint_profit_quantiles_are_not_sums_of_daily_quantiles():
    day_profit = np.asarray(
        [
            [0.0, 10.0, 20.0, 30.0],
            [30.0, 20.0, 10.0, 0.0],
        ]
    )
    result = scenario_profit_quantiles(day_profit)
    assert result["P10"] == 30.0
    assert result["P50"] == 30.0
    assert result["P90"] == 30.0
    assert result["P10"] != float(np.quantile(day_profit[0], 0.10) + np.quantile(day_profit[1], 0.10))
