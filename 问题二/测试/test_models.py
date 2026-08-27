# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题二" / "结果_v2"
CATEGORIES = {"花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"}


def test_required_demand_candidates_are_reported():
    selection = pd.read_csv(OUT / "03_需求模型选择.csv", encoding="utf-8-sig")
    normal = selection[selection["口径"] == "正常销售"]
    assert set(normal["品类"]) == CATEGORIES
    summary = pd.read_csv(OUT / "03_需求回测汇总.csv", encoding="utf-8-sig")
    assert "百分之八十区间覆盖率" in summary.columns
    assert "百分之九十区间覆盖率" in summary.columns
    for cat in CATEGORIES:
        names = set(summary[(summary["口径"] == "正常销售") & (summary["品类"] == cat)]["需求模型"])
        assert len(names) == 6


def test_price_response_and_end_to_end_outputs_are_finite():
    relation = pd.read_csv(OUT / "04_价格关系可靠性.csv", encoding="utf-8-sig")
    assert set(relation["品类"]) == CATEGORIES
    assert relation["价格响应系数"].notna().all()
    assert relation["端到端不过度劣化"].notna().all()
    end = pd.read_csv(OUT / "05_端到端需求回测汇总.csv", encoding="utf-8-sig")
    assert len(end) == 12
    numeric = end.select_dtypes(include=["number"])
    assert np.isfinite(numeric.to_numpy()).all()


def test_cost_candidates_and_intervals():
    cost = pd.read_csv(OUT / "06_成本回测汇总.csv", encoding="utf-8-sig")
    assert set(cost["成本预测方法"]) >= {"近7日均值", "近14日均值", "指数加权移动平均", "指数平滑"}
    assert cost["池化加权绝对百分比误差"].notna().all()
    assert "百分之九十区间覆盖率" in cost.columns
    future = pd.read_csv(OUT / "06_未来成本路径.csv", encoding="utf-8-sig")
    assert len(future) == 42
    assert (future["成本P10"] <= future["成本P50"]).all()
    assert (future["成本P50"] <= future["成本P90"]).all()
