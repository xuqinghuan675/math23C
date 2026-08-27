# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题二" / "结果_v2"


def _check_temporal_table(name: str):
    table = pd.read_csv(OUT / name, encoding="utf-8-sig")
    train = pd.to_datetime(table["训练截止日"])
    test = pd.to_datetime(table["测试开始日"])
    assert (train < test).all()
    assert (table["泄漏检查"] == "通过").all()
    return table


def test_all_rolling_training_dates_precede_tests():
    demand = _check_temporal_table("03_需求回测明细.csv")
    price = _check_temporal_table("04_价格响应回测明细.csv")
    end_to_end = _check_temporal_table("05_端到端需求回测明细.csv")
    cost = _check_temporal_table("06_成本回测明细.csv")
    assert len(demand[demand["折组"] == "八折主要回测"]) >= 6 * 6 * 7
    assert len(price) > 0 and len(end_to_end) > 0 and len(cost) > 0


def test_pseudo_future_and_weight_boundary_are_explicit():
    demand = pd.read_csv(OUT / "03_需求回测明细.csv", encoding="utf-8-sig")
    pseudo = demand[demand["是否伪未来"] == "是"]
    assert set(pd.to_datetime(pseudo["测试开始日"]).dt.year.unique()) == {2021, 2022}
    index = pd.read_csv(OUT / "02_指数覆盖率.csv", encoding="utf-8-sig")
    assert "固定篮子覆盖率" in index.columns


def test_price_response_source_uses_lagged_structure_controls():
    source = (ROOT / "问题二" / "脚本" / "v2" / "price_response.py").read_text(encoding="utf-8")
    assert "STRUCTURE_CONTROL_COLUMNS" in source
    assert '"前一日销量集中度HHI"' in source
    assert '"前一日前三单品销量占比"' in source
    assert '"前一日当前权重与基准权重距离"' in source
