# -*- coding: utf-8 -*-
"""需求预测候选模型、严格滚动回测和模型选择。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import CATEGORIES, DEMAND_MODELS, OUTPUT_DIR


@dataclass
class DemandFit:
    category: str
    model_name: str
    target_col: str
    origin: pd.Timestamp
    history: pd.DataFrame
    coefficients: np.ndarray | None
    design_names: list[str]
    residual_log: np.ndarray
    residual_quantiles: tuple[float, float, float, float, float]
    smear_factor: float


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """用正规方程快速求解；只有矩阵病态时才退回奇异值分解。"""
    gram = x.T @ x
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(x, y, rcond=None)[0]


def make_folds(last_date: pd.Timestamp) -> list[dict[str, Any]]:
    """给出八个主要折和十四个近期滚动折。"""
    main_dates = pd.to_datetime(
        [
            "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
            "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
        ]
    )
    result: list[dict[str, Any]] = []
    used: set[pd.Timestamp] = set()
    for date in main_dates:
        date = pd.Timestamp(date)
        if date >= last_date - pd.Timedelta(days=7):
            continue
        result.append({"截止日": date, "折组": "八折主要回测", "是否伪未来": date.month == 6 and date.day == 30})
        used.add(date)
    for weeks in range(2, 16):
        date = pd.Timestamp(last_date) - pd.Timedelta(days=7 * weeks)
        date = date.normalize()
        if date in used or date < pd.Timestamp("2021-01-01"):
            continue
        result.append({"截止日": date, "折组": "近期滚动回测", "是否伪未来": False})
    result.sort(key=lambda row: row["截止日"])
    return result


def _design_frame(dates: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp, include_trend: bool) -> tuple[np.ndarray, list[str]]:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    frame = pd.DataFrame(index=np.arange(len(dates)))
    frame["常数项"] = 1.0
    names = ["常数项"]
    weekday = dates.weekday + 1
    month = dates.month
    for value in range(2, 8):
        name = f"星期{value}"
        frame[name] = (weekday == value).astype(float)
        names.append(name)
    for value in range(2, 13):
        name = f"月份{value}"
        frame[name] = (month == value).astype(float)
        names.append(name)
    if include_trend:
        frame["时间趋势"] = (dates - pd.Timestamp(origin)).days.astype(float) / 365.25
        names.append("时间趋势")
    return frame[names].to_numpy(float), names


def _as_history(frame: pd.DataFrame, category: str, target_col: str) -> pd.DataFrame:
    history = frame[frame["品类"] == category][["销售日期", target_col]].copy()
    history["销售日期"] = pd.to_datetime(history["销售日期"]).dt.normalize()
    history[target_col] = pd.to_numeric(history[target_col], errors="coerce")
    history = history.dropna().sort_values("销售日期")
    history = history[history[target_col] > 0].reset_index(drop=True)
    return history


def _simple_center(history: pd.DataFrame, model_name: str, date: pd.Timestamp, target_col: str) -> float:
    if history.empty:
        return 0.01
    date = pd.Timestamp(date).normalize()
    if model_name == "同星期最近4次均值":
        values = history[history["销售日期"].dt.weekday == date.weekday()][target_col].tail(4)
        value = values.mean() if len(values) else history[target_col].tail(7).mean()
    elif model_name == "同星期最近8次中位数":
        values = history[history["销售日期"].dt.weekday == date.weekday()][target_col].tail(8)
        value = values.median() if len(values) else history[target_col].tail(14).median()
    elif model_name == "近7日均值":
        value = history[target_col].tail(7).mean()
    elif model_name == "近14日均值":
        value = history[target_col].tail(14).mean()
    else:
        value = history[target_col].tail(7).mean()
    return float(max(0.01, value))


def fit_demand_model(frame: pd.DataFrame, category: str, model_name: str, target_col: str) -> DemandFit:
    history = _as_history(frame, category, target_col)
    if history.empty:
        raise ValueError(f"{category} 的需求训练样本为空")
    origin = pd.Timestamp(history["销售日期"].min())
    include_trend = model_name == "星期加月份及趋势对数回归"
    is_regression = model_name in {"星期加月份对数回归", "星期加月份及趋势对数回归"}
    coefficients: np.ndarray | None = None
    design_names: list[str] = []
    if is_regression:
        x, design_names = _design_frame(history["销售日期"], origin, include_trend)
        y = np.log(np.maximum(history[target_col].to_numpy(float), 1e-6))
        coefficients = _ols(x, y)
        fitted = np.exp(x @ coefficients)
        residual_log = y - np.log(np.maximum(fitted, 1e-6))
        smear_factor = float(np.mean(np.exp(residual_log)))
    else:
        if model_name == "同星期最近4次均值":
            grouped = history.assign(_星期=history["销售日期"].dt.weekday).groupby("_星期")[target_col].mean()
            centers = history["销售日期"].dt.weekday.map(grouped).fillna(history[target_col].mean()).to_numpy(float)
        elif model_name == "同星期最近8次中位数":
            grouped = history.assign(_星期=history["销售日期"].dt.weekday).groupby("_星期")[target_col].median()
            centers = history["销售日期"].dt.weekday.map(grouped).fillna(history[target_col].median()).to_numpy(float)
        else:
            center = history[target_col].tail(7 if model_name == "近7日均值" else 14).mean()
            centers = np.full(len(history), float(center))
        centers = np.maximum(np.asarray(centers, dtype=float), 0.01)
        residual_log = np.log(np.maximum(history[target_col].to_numpy(float), 1e-6)) - np.log(centers)
        smear_factor = float(np.mean(np.exp(residual_log)))
    if len(residual_log) < 3:
        residual_log = np.array([0.0, 0.0, 0.0], dtype=float)
    quantiles = tuple(float(x) for x in np.quantile(residual_log, [0.05, 0.10, 0.50, 0.90, 0.95]))
    return DemandFit(
        category=category,
        model_name=model_name,
        target_col=target_col,
        origin=origin,
        history=history,
        coefficients=coefficients,
        design_names=design_names,
        residual_log=np.asarray(residual_log, dtype=float),
        residual_quantiles=quantiles,
        smear_factor=smear_factor,
    )


def predict_point(model: DemandFit, dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    is_regression = model.coefficients is not None
    if is_regression:
        x, names = _design_frame(dates, model.origin, model.model_name == "星期加月份及趋势对数回归")
        if names != model.design_names:
            raise ValueError("需求模型训练与预测的设计矩阵不一致")
        values = np.exp(x @ model.coefficients) * model.smear_factor
    else:
        values = np.array([_simple_center(model.history, model.model_name, date, model.target_col) for date in dates])
    return np.maximum(np.asarray(values, dtype=float), 0.01)


def predict_distribution(model: DemandFit, dates: pd.Series | pd.DatetimeIndex) -> dict[str, np.ndarray]:
    point = predict_point(model, dates)
    q05, q10, q50, q90, q95 = model.residual_quantiles
    return {
        "P05": np.maximum(0.0, point * np.exp(q05)),
        "P10": np.maximum(0.0, point * np.exp(q10)),
        "P50": np.maximum(0.0, point * np.exp(q50)),
        "P90": np.maximum(0.0, point * np.exp(q90)),
        "P95": np.maximum(0.0, point * np.exp(q95)),
        "中心": point,
    }


def _quantile_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = actual - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def _mase(actual: np.ndarray, prediction: np.ndarray, train_values: np.ndarray) -> float:
    if len(train_values) < 8:
        return float(np.mean(np.abs(actual - prediction)))
    scale = np.mean(np.abs(train_values[7:] - train_values[:-7]))
    return float(np.mean(np.abs(actual - prediction)) / scale) if scale > 0 else float(np.mean(np.abs(actual - prediction)))


def _metrics(actual: np.ndarray, prediction: dict[str, np.ndarray], train_values: np.ndarray) -> dict[str, float]:
    p50 = prediction["P50"]
    error = actual - p50
    return {
        "加权绝对百分比误差": float(np.abs(error).sum() / max(np.abs(actual).sum(), 1e-9)),
        "平均绝对误差": float(np.mean(np.abs(error))),
        "均方根误差": float(np.sqrt(np.mean(error**2))),
        "平均绝对尺度误差": _mase(actual, p50, train_values),
        "百分之十分位损失": _quantile_loss(actual, prediction["P10"], 0.10),
        "百分之五十分位损失": _quantile_loss(actual, prediction["P50"], 0.50),
        "百分之九十分位损失": _quantile_loss(actual, prediction["P90"], 0.90),
        "百分之八十区间覆盖率": float(np.mean((actual >= prediction["P10"]) & (actual <= prediction["P90"]))),
        "百分之八十区间平均宽度": float(np.mean(prediction["P90"] - prediction["P10"])),
        "百分之九十区间覆盖率": float(np.mean((actual >= prediction["P05"]) & (actual <= prediction["P95"]))),
        "百分之九十区间平均宽度": float(np.mean(prediction["P95"] - prediction["P05"])),
    }


def run_demand_backtests(
    normal_panel: pd.DataFrame,
    full_panel: pd.DataFrame,
    last_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, DemandFit], dict[str, DemandFit], dict[str, str], dict[str, str]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected_normal: dict[str, str] = {}
    selected_full: dict[str, str] = {}
    normal_fits: dict[str, DemandFit] = {}
    full_fits: dict[str, DemandFit] = {}
    folds = make_folds(last_date)

    for regime_name, panel, target_col, selected_store, fit_store in [
        ("正常销售", normal_panel, "正常销售量", selected_normal, normal_fits),
        ("全量净需求", full_panel, "净销售量", selected_full, full_fits),
    ]:
        for cat in CATEGORIES:
            frame = panel[panel["品类"] == cat].sort_values("销售日期").copy()
            model_fold_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in DEMAND_MODELS}
            for fold in folds:
                cutoff = pd.Timestamp(fold["截止日"])
                train = frame[frame["销售日期"] <= cutoff]
                test = frame[(frame["销售日期"] > cutoff) & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))]
                if len(train) < 120 or len(test) < 2:
                    continue
                for model_name in DEMAND_MODELS:
                    model = fit_demand_model(train, cat, model_name, target_col)
                    distribution = predict_distribution(model, test["销售日期"])
                    actual = test[target_col].to_numpy(float)
                    metrics = _metrics(actual, distribution, train[target_col].to_numpy(float))
                    row = {
                        "口径": regime_name,
                        "品类": cat,
                        "需求模型": model_name,
                        "折组": fold["折组"],
                        "训练截止日": cutoff.date().isoformat(),
                        "测试开始日": pd.Timestamp(test["销售日期"].min()).date().isoformat(),
                        "测试结束日": pd.Timestamp(test["销售日期"].max()).date().isoformat(),
                        "训练样本数": int(len(train)),
                        "测试样本数": int(len(test)),
                        "使用最大日期": pd.Timestamp(train["销售日期"].max()).date().isoformat(),
                        "是否伪未来": "是" if fold["是否伪未来"] else "否",
                        "泄漏检查": "通过" if pd.Timestamp(train["销售日期"].max()) < pd.Timestamp(test["销售日期"].min()) else "失败",
                    }
                    row.update(metrics)
                    detail_rows.append(row)
                    model_fold_rows[model_name].append({"row": row, "actual": actual, "pred": distribution["P50"]})
            for model_name, rows in model_fold_rows.items():
                if not rows:
                    continue
                actual_all = np.concatenate([x["actual"] for x in rows])
                pred_all = np.concatenate([x["pred"] for x in rows])
                fold_wape = np.array([x["row"]["加权绝对百分比误差"] for x in rows], dtype=float)
                summary_rows.append(
                    {
                        "口径": regime_name,
                        "品类": cat,
                        "需求模型": model_name,
                        "池化加权绝对百分比误差": float(np.abs(actual_all - pred_all).sum() / max(np.abs(actual_all).sum(), 1e-9)),
                        "折均加权绝对百分比误差": float(fold_wape.mean()),
                        "折中位加权绝对百分比误差": float(np.median(fold_wape)),
                        "折间标准差": float(fold_wape.std(ddof=1)) if len(fold_wape) > 1 else 0.0,
                        "折间标准误": float(fold_wape.std(ddof=1) / np.sqrt(len(fold_wape))) if len(fold_wape) > 1 else 0.0,
                        "平均绝对误差": float(np.mean([x["row"]["平均绝对误差"] for x in rows])),
                        "均方根误差": float(np.mean([x["row"]["均方根误差"] for x in rows])),
                        "平均绝对尺度误差": float(np.mean([x["row"]["平均绝对尺度误差"] for x in rows])),
                        "百分之十分位损失": float(np.mean([x["row"]["百分之十分位损失"] for x in rows])),
                        "百分之五十分位损失": float(np.mean([x["row"]["百分之五十分位损失"] for x in rows])),
                        "百分之九十分位损失": float(np.mean([x["row"]["百分之九十分位损失"] for x in rows])),
                        "百分之八十区间覆盖率": float(np.mean([x["row"]["百分之八十区间覆盖率"] for x in rows])),
                        "百分之八十区间平均宽度": float(np.mean([x["row"]["百分之八十区间平均宽度"] for x in rows])),
                        "百分之九十区间覆盖率": float(np.mean([x["row"]["百分之九十区间覆盖率"] for x in rows])),
                        "百分之九十区间平均宽度": float(np.mean([x["row"]["百分之九十区间平均宽度"] for x in rows])),
                        "回测折数": int(len(rows)),
                    }
                )
            summary_cat = [
                row for row in summary_rows if row["口径"] == regime_name and row["品类"] == cat
            ]
            summary_cat.sort(key=lambda x: (x["池化加权绝对百分比误差"], x["折均加权绝对百分比误差"]))
            best = summary_cat[0]
            best_wape = best["池化加权加权绝对百分比误差"] if "池化加权加权绝对百分比误差" in best else best["池化加权绝对百分比误差"]
            simple_candidates = [row for row in summary_cat if row["需求模型"] in DEMAND_MODELS[:4]]
            simple_best = min(simple_candidates, key=lambda x: x["池化加权绝对百分比误差"]) if simple_candidates else best
            one_se = best["折间标准误"]
            eligible = [row for row in summary_cat if row["池化加权绝对百分比误差"] <= best_wape + one_se]
            eligible.sort(key=lambda x: (DEMAND_MODELS.index(x["需求模型"]), x["池化加权绝对百分比误差"]))
            chosen = eligible[0] if eligible else best
            simple_improvement = (simple_best["池化加权绝对百分比误差"] - best_wape) / max(simple_best["池化加权绝对百分比误差"], 1e-9)
            if chosen["需求模型"] not in DEMAND_MODELS[:4] and simple_improvement < 0.03:
                chosen = simple_best
                reason = "复杂模型改善不足3%，采用简单模型"
            elif chosen["需求模型"] != best["需求模型"]:
                reason = "采用一标准误范围内更简单模型"
            else:
                reason = "池化回测误差最小且通过无泄漏检查"
            selected_store[cat] = chosen["需求模型"]
            fit_store[cat] = fit_demand_model(frame, cat, chosen["需求模型"], target_col)
            selection_rows.append(
                {
                    "口径": regime_name,
                    "品类": cat,
                    "入选需求模型": chosen["需求模型"],
                    "入选池化加权绝对百分比误差": chosen["池化加权绝对百分比误差"],
                    "最优池化加权绝对百分比误差": best_wape,
                    "简单模型最佳误差": simple_best["池化加权绝对百分比误差"],
                    "简单模型相对改善": simple_improvement,
                    "一标准误范围": one_se,
                    "回测折数": chosen["回测折数"],
                    "选择理由": reason,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    selection_df = pd.DataFrame(selection_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(OUTPUT_DIR / "03_需求回测明细.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "03_需求回测汇总.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(OUTPUT_DIR / "03_需求模型选择.csv", index=False, encoding="utf-8-sig")
    detail_df[detail_df["是否伪未来"] == "是"].to_csv(OUTPUT_DIR / "03_伪未来七月回测.csv", index=False, encoding="utf-8-sig")
    return detail_df, summary_df, selection_df, normal_fits, full_fits, selected_normal, selected_full
