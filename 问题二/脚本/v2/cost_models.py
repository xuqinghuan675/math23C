# -*- coding: utf-8 -*-
"""批发成本候选模型、滚动回测和未来成本路径。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CATEGORIES, COST_METHODS, FUTURE_DATES, OUTPUT_DIR
from .demand_models import make_folds


def _series(train: pd.DataFrame, target_col: str) -> pd.Series:
    result = train.set_index("销售日期")[target_col].sort_index().dropna().astype(float)
    result = result[result > 0]
    return result


def forecast_cost(train: pd.DataFrame, dates: pd.DatetimeIndex, method: str, target_col: str) -> pd.Series:
    series = _series(train, target_col)
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if series.empty:
        raise ValueError("成本训练序列为空")
    if method == "近7日均值":
        value = series.tail(7).mean()
        return pd.Series(float(value), index=dates)
    if method == "近14日均值":
        value = series.tail(14).mean()
        return pd.Series(float(value), index=dates)
    if method == "近7日中位数":
        value = series.tail(7).median()
        return pd.Series(float(value), index=dates)
    if method == "近14日中位数":
        value = series.tail(14).median()
        return pd.Series(float(value), index=dates)
    if method == "指数加权移动平均":
        value = series.ewm(alpha=0.35, adjust=False).mean().iloc[-1]
        return pd.Series(float(value), index=dates)
    if method in {"同星期最近4次统计", "同星期最近8次统计"}:
        count = 4 if method == "同星期最近4次统计" else 8
        values = []
        for date in dates:
            same = series[series.index.weekday == date.weekday()].tail(count)
            values.append(float(same.median()) if len(same) else float(series.tail(7).median()))
        return pd.Series(values, index=dates)
    if method == "指数加权移动平均加周内收缩":
        ewma = float(series.ewm(alpha=0.35, adjust=False).mean().iloc[-1])
        values = []
        for date in dates:
            same = series[series.index.weekday == date.weekday()].tail(8)
            weekday_value = float(same.median()) if len(same) else ewma
            values.append(0.65 * ewma + 0.35 * weekday_value)
        return pd.Series(values, index=dates)
    if method == "阻尼趋势":
        recent = series.tail(min(60, len(series)))
        x = np.arange(len(recent), dtype=float)
        if len(recent) >= 5:
            slope, intercept = np.polyfit(x, recent.to_numpy(float), 1)
            slope = float(np.clip(slope, -0.05 * recent.mean(), 0.05 * recent.mean()))
        else:
            intercept, slope = float(recent.mean()), 0.0
        horizon = np.arange(1, len(dates) + 1, dtype=float)
        damp = 0.75
        base = float(recent.iloc[-1])
        values = base + slope * (1.0 - damp**horizon) / max(1e-9, 1.0 - damp)
        low, high = np.quantile(recent, [0.05, 0.95])
        return pd.Series(np.clip(values, low, high), index=dates)
    if method == "指数平滑":
        # 用固定平滑系数的递推平滑，避免在每个滚动折上进行数值优化；
        # 该候选仍与指数平滑定义一致，且完全由训练期数据得到。
        values = series.to_numpy(float)
        level = float(values[0])
        for value in values[1:]:
            level = 0.30 * float(value) + 0.70 * level
        value = level
        low, high = np.quantile(series.tail(min(90, len(series))), [0.01, 0.99])
        return pd.Series(float(np.clip(value, low, high)), index=dates)
    if method == "滞后稳健回归":
        recent = series.tail(min(90, len(series)))
        lag = recent.shift(7)
        valid = pd.concat([recent.rename("当前"), lag.rename("滞后")], axis=1).dropna()
        if len(valid) >= 10:
            difference = float((valid["当前"] - valid["滞后"]).median())
            value = float(recent.iloc[-1] + 0.25 * difference)
        else:
            value = float(recent.median())
        low, high = np.quantile(recent, [0.01, 0.99])
        return pd.Series(float(np.clip(value, low, high)), index=dates)
    raise ValueError(f"未知成本预测方法: {method}")


def _path_error(actual: pd.Series, predicted: pd.Series) -> float:
    joined = pd.concat([actual.rename("实际"), predicted.rename("预测")], axis=1).dropna()
    if len(joined) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(joined["实际"]) - np.diff(joined["预测"]))))


def run_cost_backtests(panel: pd.DataFrame, last_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    folds = make_folds(last_date)
    targets = [("销量加权成本", "销量加权进价"), ("固定篮子成本", "固定篮子成本指数")]
    for cat in CATEGORIES:
        frame = panel[panel["品类"] == cat].sort_values("销售日期").copy()
        for target_name, target_col in targets:
            for method in COST_METHODS:
                for fold in folds:
                    cutoff = pd.Timestamp(fold["截止日"])
                    train = frame[frame["销售日期"] <= cutoff]
                    test = frame[(frame["销售日期"] > cutoff) & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))]
                    train = train.dropna(subset=[target_col])
                    test = test.dropna(subset=[target_col])
                    if len(train) < 60 or len(test) < 2:
                        continue
                    pred = forecast_cost(train, pd.DatetimeIndex(test["销售日期"]), method, target_col)
                    actual = test.set_index("销售日期")[target_col].astype(float)
                    joined = pd.concat([actual.rename("实际"), pred.rename("预测")], axis=1).dropna()
                    error = joined["实际"] - joined["预测"]
                    previous = float(train[target_col].iloc[-1])
                    actual_direction = np.sign(joined["实际"].to_numpy() - previous)
                    pred_direction = np.sign(joined["预测"].to_numpy() - previous)
                    recent_values = train[target_col].tail(min(60, len(train))).to_numpy(float)
                    spread = float(np.std(recent_values, ddof=1)) if len(recent_values) > 1 else 0.0
                    low_band = joined["预测"] - 1.645 * spread
                    high_band = joined["预测"] + 1.645 * spread
                    detail_rows.append(
                        {
                            "品类": cat,
                            "成本口径": target_name,
                            "成本预测方法": method,
                            "折组": fold["折组"],
                            "训练截止日": cutoff.date().isoformat(),
                            "测试开始日": pd.Timestamp(test["销售日期"].min()).date().isoformat(),
                            "测试结束日": pd.Timestamp(test["销售日期"].max()).date().isoformat(),
                            "训练样本数": int(len(train)),
                            "测试样本数": int(len(joined)),
                            "使用最大日期": pd.Timestamp(train["销售日期"].max()).date().isoformat(),
                            "泄漏检查": "通过" if pd.Timestamp(train["销售日期"].max()) < pd.Timestamp(test["销售日期"].min()) else "失败",
                            "加权绝对百分比误差": float(np.abs(error).sum() / max(np.abs(joined["实际"]).sum(), 1e-9)),
                            "平均绝对误差": float(np.abs(error).mean()),
                            "路径变化误差": _path_error(actual, pred),
                            "方向准确率": float(np.mean(actual_direction == pred_direction)),
                            "百分之九十区间覆盖率": float(np.mean((joined["实际"] >= low_band) & (joined["实际"] <= high_band))),
                            "百分之九十区间平均宽度": float(np.mean(high_band - low_band)),
                        }
                    )
    detail_df = pd.DataFrame(detail_rows)
    for (cat, target_name, method), sub in detail_df.groupby(["品类", "成本口径", "成本预测方法"], sort=False):
        summary_rows.append(
            {
                "品类": cat,
                "成本口径": target_name,
                "成本预测方法": method,
                "池化加权绝对百分比误差": float(np.average(sub["加权绝对百分比误差"], weights=sub["测试样本数"])),
                "折均加权绝对百分比误差": float(sub["加权绝对百分比误差"].mean()),
                "平均绝对误差": float(sub["平均绝对误差"].mean()),
                "路径变化误差": float(sub["路径变化误差"].mean()),
                "方向准确率": float(sub["方向准确率"].mean()),
                "百分之九十区间覆盖率": float(sub["百分之九十区间覆盖率"].mean()),
                "百分之九十区间平均宽度": float(sub["百分之九十区间平均宽度"].mean()),
                "回测折数": int(len(sub)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    selected: dict[str, dict[str, str]] = {}
    if not summary_df.empty:
        for cat in CATEGORIES:
            sub = summary_df[summary_df["品类"] == cat].sort_values(["池化加权绝对百分比误差", "平均绝对误差"])
            if sub.empty:
                continue
            best = sub.iloc[0]
            selected[cat] = {"成本口径": str(best["成本口径"]), "成本预测方法": str(best["成本预测方法"])}
        summary_df["是否入选"] = [
            "是" if selected.get(row["品类"], {}).get("成本口径") == row["成本口径"] and selected.get(row["品类"], {}).get("成本预测方法") == row["成本预测方法"] else "否"
            for _, row in summary_df.iterrows()
        ]
    detail_df.to_csv(OUTPUT_DIR / "06_成本回测明细.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "06_成本回测汇总.csv", index=False, encoding="utf-8-sig")
    return detail_df, summary_df, selected


def future_cost_points(panel: pd.DataFrame, selected: dict[str, dict[str, str]], last_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        choice = selected[cat]
        target_col = "销量加权进价" if choice["成本口径"] == "销量加权成本" else "固定篮子成本指数"
        train = panel[(panel["品类"] == cat) & (panel["销售日期"] <= last_date)].dropna(subset=[target_col]).sort_values("销售日期")
        prediction = forecast_cost(train, FUTURE_DATES, choice["成本预测方法"], target_col)
        for date in FUTURE_DATES:
            rows.append(
                {
                    "日期": date.date().isoformat(),
                    "品类": cat,
                    "成本口径": choice["成本口径"],
                    "成本预测方法": choice["成本预测方法"],
                    "预测批发价": float(max(0.01, prediction.loc[date])),
                }
            )
    result = pd.DataFrame(rows)
    return result
