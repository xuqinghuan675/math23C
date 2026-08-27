# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

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
    assert len(audit_table) >= 20
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
