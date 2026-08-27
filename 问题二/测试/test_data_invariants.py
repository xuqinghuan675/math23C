# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题二" / "结果_v2"


def test_attachment_counts_and_core_audit():
    audit = json.loads((OUT / "01_数据审计.json").read_text(encoding="utf-8"))
    assert audit["附件一行数"] == 251
    assert audit["附件二行数"] == 878503
    assert audit["附件三行数"] == 55982
    assert audit["附件二退货记录数"] == 461
    assert audit["附件二完整重复行数"] == 0
    assert audit["附件二忽略扫码时间候选重复额外行数"] > 0
    assert audit["日期—单品批发价匹配率"] == 1.0
    assert audit["销售量符号不一致数"] == 0
    assert audit["无流水日期数"] == 10
    assert audit["附件四零损耗率单品数"] == 22


def test_audit_has_conservation_and_gap_evidence():
    audit_table = pd.read_csv(OUT / "01_数据审计.csv", encoding="utf-8-sig")
    connection = pd.read_csv(OUT / "01_附件连接审计.csv", encoding="utf-8-sig")
    assert len(audit_table) == 69
    assert (connection["是否通过"].fillna("是") == "是").all()
    conservation = connection[connection["检查类别"].fillna("") == "各品类守恒"]
    assert len(conservation) == 36
    assert (conservation["是否通过"] == "是").all()


def test_panel_row_counts_and_no_future_data():
    full = pd.read_csv(OUT / "02_全量净需求面板.csv", encoding="utf-8-sig")
    normal = pd.read_csv(OUT / "02_正常销售面板.csv", encoding="utf-8-sig")
    item = pd.read_csv(OUT / "02_单品日期正常销售面板.csv", encoding="utf-8-sig")
    assert len(full) == 6474
    assert len(normal) == 6473
    assert len(item) > 40000
    assert pd.to_datetime(full["销售日期"]).max() <= pd.Timestamp("2023-06-30")
    assert set(full["品类"].unique()) == {"花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"}


def test_structure_controls_are_previous_calendar_day_values():
    structure = pd.read_csv(OUT / "02_商品结构指标.csv", encoding="utf-8-sig")
    dates = pd.to_datetime(structure["销售日期"])
    assert {"前一日销量集中度HHI", "前一日前三单品销量占比", "前一日当前权重与基准权重距离"}.issubset(structure.columns)
    for _, sub in structure.assign(_日期=dates).sort_values(["品类", "_日期"]).groupby("品类"):
        sub = sub.reset_index(drop=True)
        yesterday = sub["_日期"].diff().dt.days.eq(1)
        for current, lagged in [
            ("销量集中度HHI", "前一日销量集中度HHI"),
            ("前三单品销量占比", "前一日前三单品销量占比"),
            ("当前权重与基准权重距离", "前一日当前权重与基准权重距离"),
        ]:
            expected = sub[current].shift(1)
            valid = yesterday & expected.notna() & sub[lagged].notna()
            assert np.allclose(sub.loc[valid, lagged].to_numpy(float), expected.loc[valid].to_numpy(float))
