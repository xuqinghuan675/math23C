# -*- coding: utf-8 -*-
"""销量加权指标、固定篮子指标和商品结构指标。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CATEGORIES, OUTPUT_DIR, WEIGHT_WINDOWS


def _normalise(values: pd.Series) -> pd.Series:
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        return pd.Series(0.0, index=values.index)
    return values.astype(float) / total


def compute_weight_snapshot(
    item_daily: pd.DataFrame,
    category: str,
    target_date: pd.Timestamp,
    window_days: int = 180,
    shrink_k: int = 40,
) -> dict[str, Any]:
    """构造严格截止目标日前的固定篮子权重。"""
    target_date = pd.Timestamp(target_date).normalize()
    category_items = item_daily[item_daily["品类"] == category].copy()
    history = category_items[category_items["销售日期"] < target_date]
    global_weights = _normalise(history.groupby("单品编码")["正常销售量"].sum())
    recent = history[history["销售日期"] >= target_date - pd.Timedelta(days=window_days)]
    recent_weights = _normalise(recent.groupby("单品编码")["正常销售量"].sum())
    item_codes = global_weights.index.union(recent_weights.index)
    if len(item_codes) == 0:
        return {
            "品类": category,
            "目标日期": target_date,
            "权重": pd.Series(dtype=float),
            "历史样本数": 0,
            "权重覆盖": 0.0,
        }
    global_weights = global_weights.reindex(item_codes, fill_value=0.0)
    recent_weights = recent_weights.reindex(item_codes, fill_value=0.0)
    sample_count = int(recent["单品编码"].nunique())
    shrink = sample_count / (sample_count + max(1, int(shrink_k)))
    weights = shrink * recent_weights + (1.0 - shrink) * global_weights
    weights = _normalise(weights)
    return {
        "品类": category,
        "目标日期": target_date,
        "权重": weights,
        "历史样本数": sample_count,
        "权重覆盖": float(weights.sum()),
        "收缩系数": float(shrink),
    }


def _index_for_date(
    item_daily: pd.DataFrame,
    category: str,
    date: pd.Timestamp,
    window_days: int = 180,
    shrink_k: int = 40,
) -> dict[str, Any]:
    snapshot = compute_weight_snapshot(item_daily, category, date, window_days, shrink_k)
    weights = snapshot["权重"]
    current = item_daily[(item_daily["品类"] == category) & (item_daily["销售日期"] == date)].copy()
    if current.empty or weights.empty:
        fixed_price = np.nan
        fixed_cost = np.nan
        coverage = 0.0
        distance = np.nan
    else:
        current = current.set_index("单品编码")
        available_price = current["正常销量加权售价"].notna()
        available_cost = current["当日批发价"].notna()
        price_weights = weights.reindex(current.index, fill_value=0.0)
        price_weights = price_weights.where(available_price, 0.0)
        cost_weights = weights.reindex(current.index, fill_value=0.0)
        cost_weights = cost_weights.where(available_cost, 0.0)
        price_coverage = float(price_weights.sum())
        cost_coverage = float(cost_weights.sum())
        if price_coverage > 0:
            fixed_price = float((price_weights * current["正常销量加权售价"]).sum() / price_coverage)
        else:
            fixed_price = np.nan
        if cost_coverage > 0:
            fixed_cost = float((cost_weights * current["当日批发价"]).sum() / cost_coverage)
        else:
            fixed_cost = np.nan
        coverage = min(price_coverage, cost_coverage)
        current_weights = _normalise(current["正常销售量"])
        aligned_current = current_weights.reindex(weights.index, fill_value=0.0)
        distance = float(0.5 * np.abs(aligned_current - weights).sum())
    return {
        "销售日期": pd.Timestamp(date),
        "品类": category,
        "固定篮子价格指数": fixed_price,
        "固定篮子成本指数": fixed_cost,
        "固定篮子加成率": fixed_price / fixed_cost - 1.0 if np.isfinite(fixed_price) and np.isfinite(fixed_cost) and fixed_cost > 0 else np.nan,
        "固定篮子覆盖率": coverage,
        "权重收缩系数": snapshot.get("收缩系数", np.nan),
        "当前权重与基准权重距离": distance,
        "权重历史样本数": snapshot.get("历史样本数", 0),
        "权重窗口天数": window_days,
    }


def build_indices(
    full_panel: pd.DataFrame,
    normal_panel: pd.DataFrame,
    item_daily: pd.DataFrame,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造品类日面板、固定篮子指数、覆盖率和结构指标。"""
    rows: list[dict[str, Any]] = []
    # 先把每个品类做成“日期×单品”矩阵，避免对每个日期重复筛选和分组。
    for cat in CATEGORIES:
        item_sub = item_daily[item_daily["品类"] == cat].copy()
        dates = pd.DatetimeIndex(full_panel.loc[full_panel["品类"] == cat, "销售日期"].drop_duplicates().sort_values())
        if item_sub.empty:
            continue
        item_sub["销售日期"] = pd.to_datetime(item_sub["销售日期"]).dt.normalize()
        items = pd.Index(sorted(item_sub["单品编码"].astype(str).unique()))
        start = min(item_sub["销售日期"].min(), dates.min())
        end = max(item_sub["销售日期"].max(), dates.max())
        calendar = pd.date_range(start, end, freq="D")
        matrix = (
            item_sub.pivot_table(index="销售日期", columns="单品编码", values="正常销售量", aggfunc="sum", fill_value=0.0)
            .reindex(index=calendar, columns=items, fill_value=0.0)
            .fillna(0.0)
        )
        current_map = {date: sub.set_index("单品编码") for date, sub in item_sub.groupby("销售日期", observed=True)}
        matrix_values = matrix.to_numpy(float)
        date_values = matrix.index.to_numpy(dtype="datetime64[ns]")
        for date in dates:
            date = pd.Timestamp(date).normalize()
            before_mask = date_values < np.datetime64(date)
            recent_mask = before_mask & (date_values >= np.datetime64(date - pd.Timedelta(days=180)))
            global_qty = matrix_values[before_mask].sum(axis=0) if before_mask.any() else np.zeros(len(items))
            recent_qty = matrix_values[recent_mask].sum(axis=0) if recent_mask.any() else np.zeros(len(items))
            global_weights = _normalise(pd.Series(global_qty, index=items))
            recent_weights = _normalise(pd.Series(recent_qty, index=items))
            sample_count = int((recent_qty > 0).sum())
            shrink = sample_count / (sample_count + 40) if sample_count else 0.0
            weights = _normalise(shrink * recent_weights + (1.0 - shrink) * global_weights)
            current = current_map.get(date)
            if current is None or weights.sum() <= 0:
                fixed_price = np.nan
                fixed_cost = np.nan
                coverage = 0.0
                distance = np.nan
            else:
                current = current.reindex(items)
                current_price = pd.to_numeric(current["正常销量加权售价"], errors="coerce")
                current_cost = pd.to_numeric(current["当日批发价"], errors="coerce")
                price_weights = weights.where(current_price.notna(), 0.0)
                cost_weights = weights.where(current_cost.notna(), 0.0)
                price_coverage = float(price_weights.sum())
                cost_coverage = float(cost_weights.sum())
                fixed_price = float((price_weights * current_price.fillna(0.0)).sum() / price_coverage) if price_coverage > 0 else np.nan
                fixed_cost = float((cost_weights * current_cost.fillna(0.0)).sum() / cost_coverage) if cost_coverage > 0 else np.nan
                coverage = min(price_coverage, cost_coverage)
                current_weights = _normalise(pd.to_numeric(current["正常销售量"], errors="coerce").fillna(0.0))
                distance = float(0.5 * np.abs(current_weights - weights).sum())
            rows.append(
                {
                    "销售日期": date,
                    "品类": cat,
                    "固定篮子价格指数": fixed_price,
                    "固定篮子成本指数": fixed_cost,
                    "固定篮子加成率": fixed_price / fixed_cost - 1.0 if np.isfinite(fixed_price) and np.isfinite(fixed_cost) and fixed_cost > 0 else np.nan,
                    "固定篮子覆盖率": coverage,
                    "权重收缩系数": shrink,
                    "当前权重与基准权重距离": distance,
                    "权重历史样本数": sample_count,
                    "权重窗口天数": 180,
                }
            )
    index_df = pd.DataFrame(rows)

    structure = (
        item_daily.groupby(["销售日期", "品类"], as_index=False, observed=True)
        .agg(当日单品数=("单品编码", "nunique"), 当日正常销量=("正常销售量", "sum"))
    )
    structure_rows: list[dict[str, Any]] = []
    for (date, cat), sub in item_daily.groupby(["销售日期", "品类"], observed=True):
        shares = _normalise(sub.set_index("单品编码")["正常销售量"])
        sorted_shares = shares.sort_values(ascending=False)
        structure_rows.append(
            {
                "销售日期": date,
                "品类": cat,
                "当日单品数": int(len(sub)),
                "销量集中度HHI": float((shares**2).sum()),
                "前三单品销量占比": float(sorted_shares.head(3).sum()),
            }
        )
    structure_df = pd.DataFrame(structure_rows)
    panel = full_panel.merge(index_df, on=["销售日期", "品类"], how="left", validate="one_to_one")
    panel = panel.merge(structure_df, on=["销售日期", "品类"], how="left", validate="one_to_one")
    panel["销量加权加成率"] = panel["销量加权售价"] / panel["销量加权进价"] - 1.0
    panel["当前权重与基准权重距离"] = panel["当前权重与基准权重距离"].astype(float)

    coverage_df = panel[
        [
            "销售日期", "品类", "固定篮子覆盖率", "权重收缩系数", "权重历史样本数", "权重窗口天数",
        ]
    ].copy()
    index_export = panel[
        [
            "销售日期", "品类", "销量加权售价", "销量加权进价", "销量加权加成率",
            "固定篮子价格指数", "固定篮子成本指数", "固定篮子加成率", "固定篮子覆盖率",
            "权重窗口天数", "权重收缩系数", "权重历史样本数",
        ]
    ].copy()
    structure_export = panel[
        [
            "销售日期", "品类", "当日单品数", "单品数", "销量集中度HHI", "前三单品销量占比",
            "固定篮子覆盖率", "当前权重与基准权重距离",
        ]
    ].copy()
    if write_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_csv(OUTPUT_DIR / "02_品类日面板.csv", index=False, encoding="utf-8-sig")
        index_export.to_csv(OUTPUT_DIR / "02_固定篮子价格成本指数.csv", index=False, encoding="utf-8-sig")
        coverage_df.to_csv(OUTPUT_DIR / "02_指数覆盖率.csv", index=False, encoding="utf-8-sig")
        structure_export.to_csv(OUTPUT_DIR / "02_商品结构指标.csv", index=False, encoding="utf-8-sig")
    return panel, index_export, coverage_df, structure_export
