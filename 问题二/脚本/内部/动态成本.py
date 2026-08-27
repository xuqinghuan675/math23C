# -*- coding: utf-8 -*-
"""问题二动态批发价曲线预测。

目标不是为了“让七天价格看起来不同”而制造波动，而是在保持价格水平预测误差可控的前提下，
额外评价未来七日路径变化的拟合能力。模型选择使用滚动七日伪未来回测：

1. 主指标：批发价水平 WAPE；
2. 路径指标：逐日价格变化误差 / 实际价格水平；
3. 仅在水平 WAPE 距最优不超过 5%（且至少容忍 0.3 个百分点）的候选中，
   再按“水平 WAPE + 0.5×路径变化误差”选择。

动态候选采用“水平预测 + 最近一周同星期形状的收缩修正”，避免直接照搬上一周噪声。
因此 rho 越小越接近稳健水平预测，rho 越大越强调近期周内形状。
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "问题二" / "结果"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
RHO_VALUES = (0.15, 0.30, 0.45, 0.60)
LEVEL_METHODS = ("近7日均值", "近14日均值", "指数加权移动平均")
SHAPE_BASES = ("近7日均值", "指数加权移动平均", "近14日均值")
DYNAMIC_METHODS = tuple(
    f"{base_name}+近周形状rho={rho:.2f}"
    for base_name in SHAPE_BASES
    for rho in RHO_VALUES
) + ("近21日阻尼趋势",)
COST_METHODS = LEVEL_METHODS + DYNAMIC_METHODS


def _write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def _validation_cutoffs(frame: pd.DataFrame) -> list[pd.Timestamp]:
    end = pd.Timestamp(frame["销售日期"].max())
    cutoffs = []
    for offset in [14 + 35 * i for i in range(8)]:
        cutoff = end - pd.Timedelta(days=int(offset))
        train_n = int((frame["销售日期"] <= cutoff).sum())
        test_n = int(
            (
                (frame["销售日期"] > cutoff)
                & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
            ).sum()
        )
        if train_n >= 300 and test_n >= 4:
            cutoffs.append(cutoff)
    return sorted(cutoffs)


def _series(train: pd.DataFrame) -> pd.Series:
    series = (
        train[["销售日期", "日平均进价"]]
        .dropna()
        .sort_values("销售日期")
        .drop_duplicates("销售日期", keep="last")
        .set_index("销售日期")["日平均进价"]
        .astype(float)
    )
    if len(series) == 0:
        raise ValueError("进价序列为空")
    return series


def _level_value(series: pd.Series, method: str) -> float:
    if method == "近7日均值":
        return float(series.tail(7).mean())
    if method == "近14日均值":
        return float(series.tail(14).mean())
    if method == "指数加权移动平均":
        return float(series.ewm(alpha=0.4, adjust=False).mean().iloc[-1])
    raise ValueError(f"未知水平预测方法: {method}")


def _clip_recent(series: pd.Series, prediction: np.ndarray) -> np.ndarray:
    recent = series.tail(90)
    low = float(recent.quantile(0.03))
    high = float(recent.quantile(0.97))
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or low >= high:
        low = max(1e-6, float(recent.min()))
        high = float(recent.max())
    # 给未来留少量外推空间，但不允许短期模型把进价推到历史极端之外。
    span = high - low
    return np.clip(prediction, max(1e-6, low - 0.15 * span), high + 0.15 * span)


def _weekly_shape_forecast(
    series: pd.Series,
    future_dates: pd.DatetimeIndex,
    base_method: str,
    rho: float,
) -> pd.Series:
    level = _level_value(series, base_method)
    recent_mean = float(series.tail(7).mean())
    if not np.isfinite(recent_mean) or recent_mean <= 0:
        recent_mean = level

    values = []
    for date in pd.DatetimeIndex(future_dates):
        same_weekday = series[series.index.weekday == date.weekday()]
        if len(same_weekday):
            reference = float(same_weekday.iloc[-1])
        else:
            reference = float(series.iloc[-1])
        ratio = reference / recent_mean if recent_mean > 0 else 1.0
        # 日均批发价本身包含商品结构噪声，因此只允许近期周形状做有限修正。
        ratio = float(np.clip(ratio, 0.75, 1.25))
        values.append(level * (1.0 + rho * (ratio - 1.0)))

    prediction = _clip_recent(series, np.asarray(values, dtype=float))
    return pd.Series(prediction, index=pd.DatetimeIndex(future_dates))


def _damped_trend_forecast(
    series: pd.Series,
    future_dates: pd.DatetimeIndex,
) -> pd.Series:
    recent = series.tail(21)
    if len(recent) < 7:
        return pd.Series(_level_value(series, "指数加权移动平均"), index=future_dates)

    end = recent.index.max()
    x = (recent.index - end).days.astype(float)
    y = recent.to_numpy(float)
    design = np.column_stack([np.ones(len(recent)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    level = _level_value(series, "指数加权移动平均")
    # 短期日斜率不超过当前水平的 2.5%，防止噪声外推爆炸。
    slope = float(np.clip(slope, -0.025 * level, 0.025 * level))
    phi = 0.82
    values = []
    for h, _date in enumerate(pd.DatetimeIndex(future_dates), start=1):
        cumulative = sum(phi**k for k in range(h))
        values.append(level + slope * cumulative)
    prediction = _clip_recent(series, np.asarray(values, dtype=float))
    return pd.Series(prediction, index=pd.DatetimeIndex(future_dates))


def cost_forecast(
    train: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    method: str,
) -> pd.Series:
    """输出未来七日批发价路径；接口与原求解器保持一致。"""
    series = _series(train)
    future_dates = pd.DatetimeIndex(future_dates)

    if method in LEVEL_METHODS:
        value = _level_value(series, method)
        return pd.Series(value, index=future_dates)

    match = re.fullmatch(r"(.+)\+近周形状rho=(0\.15|0\.30|0\.45|0\.60)", method)
    if match:
        base_method = match.group(1)
        rho = float(match.group(2))
        return _weekly_shape_forecast(series, future_dates, base_method, rho)

    if method == "近21日阻尼趋势":
        return _damped_trend_forecast(series, future_dates)

    raise ValueError(f"未知动态进价预测方法: {method}")


def _fold_metrics(train: pd.DataFrame, test: pd.DataFrame, method: str) -> dict:
    prediction = cost_forecast(train, pd.DatetimeIndex(test["销售日期"]), method)
    actual = test.set_index("销售日期")["日平均进价"].astype(float).sort_index()
    joined = pd.concat(
        [actual.rename("实际进价"), prediction.rename("预测进价")], axis=1
    ).dropna()
    if len(joined) == 0:
        raise ValueError("动态进价回测没有可比较日期")

    error = joined["实际进价"] - joined["预测进价"]
    last_train = float(_series(train).iloc[-1])
    actual_path = np.r_[last_train, joined["实际进价"].to_numpy(float)]
    pred_path = np.r_[last_train, joined["预测进价"].to_numpy(float)]
    actual_delta = np.diff(actual_path)
    pred_delta = np.diff(pred_path)
    delta_error = actual_delta - pred_delta

    nonzero = np.abs(actual_delta) > 1e-8
    if nonzero.any():
        direction_accuracy = float(
            np.mean(np.sign(actual_delta[nonzero]) == np.sign(pred_delta[nonzero]))
        )
    else:
        direction_accuracy = np.nan

    actual_std = float(joined["实际进价"].std(ddof=0))
    pred_std = float(joined["预测进价"].std(ddof=0))
    volatility_ratio = pred_std / actual_std if actual_std > 1e-10 else np.nan

    return {
        "验证天数": int(len(joined)),
        "绝对误差和": float(np.abs(error).sum()),
        "实际价格和": float(np.abs(joined["实际进价"]).sum()),
        "平均绝对误差": float(np.abs(error).mean()),
        "路径变化绝对误差和": float(np.abs(delta_error).sum()),
        "路径归一化分母": float(np.abs(joined["实际进价"]).sum()),
        "方向命中率": direction_accuracy,
        "预测波动比": volatility_ratio,
    }


def cost_backtest(
    panel_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """滚动七日同时评价价格水平与未来路径，逐品类选择批发价曲线模型。"""
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    selected_methods: dict[str, str] = {}

    for cat in CATEGORIES:
        frame = panel_all[panel_all["品类"] == cat].sort_values("销售日期").copy()
        cutoffs = _validation_cutoffs(frame)
        if not cutoffs:
            raise ValueError(f"{cat} 没有足够样本进行动态进价回测")

        for method in COST_METHODS:
            for cutoff in cutoffs:
                train = frame[frame["销售日期"] <= cutoff]
                test = frame[
                    (frame["销售日期"] > cutoff)
                    & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
                ]
                metrics = _fold_metrics(train, test, method)
                detail_rows.append(
                    {
                        "品类": cat,
                        "方法": method,
                        "验证截止日": cutoff.date().isoformat(),
                        **metrics,
                    }
                )

        detail_cat = pd.DataFrame([row for row in detail_rows if row["品类"] == cat])
        for method, sub in detail_cat.groupby("方法", sort=False):
            abs_sum = float(sub["绝对误差和"].sum())
            actual_sum = float(sub["实际价格和"].sum())
            path_abs = float(sub["路径变化绝对误差和"].sum())
            path_den = float(sub["路径归一化分母"].sum())
            wape = abs_sum / actual_sum
            path_wape = path_abs / path_den
            direction = float(sub["方向命中率"].dropna().mean()) if sub["方向命中率"].notna().any() else np.nan
            volatility = float(sub["预测波动比"].dropna().median()) if sub["预测波动比"].notna().any() else np.nan
            summary_rows.append(
                {
                    "品类": cat,
                    "方法": method,
                    "回测WAPE": wape,
                    "回测MAE": float(sub["平均绝对误差"].mean()),
                    "路径变化误差": path_wape,
                    "方向命中率": direction,
                    "预测波动比中位数": volatility,
                    "回测次数": int(len(sub)),
                    "是否动态": "否" if method in LEVEL_METHODS else "是",
                }
            )

        cat_summary = pd.DataFrame([row for row in summary_rows if row["品类"] == cat])
        best_wape = float(cat_summary["回测WAPE"].min())
        tolerance = max(0.003, 0.05 * best_wape)
        admissible = cat_summary[
            cat_summary["回测WAPE"] <= best_wape + tolerance + 1e-12
        ].copy()
        admissible["综合路径评分"] = (
            admissible["回测WAPE"] + 0.5 * admissible["路径变化误差"]
        )
        admissible = admissible.sort_values(
            ["综合路径评分", "回测WAPE", "路径变化误差", "方法"]
        )
        selected_methods[cat] = str(admissible.iloc[0]["方法"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df["距最优WAPE"] = summary_df.groupby("品类")["回测WAPE"].transform(
        lambda s: s - s.min()
    )
    summary_df["是否入选"] = [
        "是" if row["方法"] == selected_methods[row["品类"]] else "否"
        for _, row in summary_df.iterrows()
    ]
    summary_df["综合路径评分"] = summary_df["回测WAPE"] + 0.5 * summary_df["路径变化误差"]
    summary_df = summary_df.sort_values(
        ["品类", "是否入选", "回测WAPE"], ascending=[True, False, True]
    ).reset_index(drop=True)
    _write_csv(summary_df, "成本预测回测.csv")

    detail_df = pd.DataFrame(detail_rows)
    return detail_df, summary_df, selected_methods
