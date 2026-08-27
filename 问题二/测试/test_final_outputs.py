# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题二" / "结果_v2"
CATEGORIES = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]

sys.path.insert(0, str(ROOT / "问题二" / "脚本"))

from v2.reporting import code_tree_hash


def test_exactly_42_final_rows_and_complete_date_category_grid():
    final = pd.read_csv(OUT / "09_七天六品类最终策略.csv", encoding="utf-8-sig")
    assert len(final) == 42
    assert sorted(final["日期"].unique()) == [f"2023-07-{day:02d}" for day in range(1, 8)]
    assert final.groupby("日期")["品类"].nunique().eq(6).all()
    assert final.groupby("品类")["日期"].nunique().eq(7).all()
    assert final.notna().all().all()


def test_final_profit_matches_daily_strategy_and_manifest_is_present():
    final = pd.read_csv(OUT / "09_七天六品类最终策略.csv", encoding="utf-8-sig")
    daily = pd.read_csv(OUT / "07_策略逐日结果.csv", encoding="utf-8-sig")
    robust = daily[daily["策略"] == "D稳健经营"]
    merged = final.merge(robust, on=["日期", "品类"], suffixes=("_最终", "_逐日"))
    assert (abs(merged["预计毛利_最终"] - merged["预计毛利_逐日"]) < 1e-7).all()
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["数据使用截止日"] == "2023-06-30"


def test_unreliable_categories_do_not_have_price_optimum_or_boundary_claim():
    final = pd.read_csv(OUT / "09_七天六品类最终策略.csv", encoding="utf-8-sig")
    unreliable = final[final["价格关系是否可靠"] == "否"]
    assert (unreliable["数学期望利润最大售价"] == "不适用").all()
    assert (unreliable["是否边界解"] == "不适用").all()
    assert (unreliable["边界方向"] == "不适用").all()


def test_manifest_records_the_current_source_tree():
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["程序版本"] == "2.1.0"
    assert manifest["代码树哈希"] == code_tree_hash()
