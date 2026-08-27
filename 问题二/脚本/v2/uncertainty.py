# -*- coding: utf-8 -*-
"""需求、成本、系数、损耗和折扣回收的联合情景。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CATEGORIES, FUTURE_DATES, LOSS_FACTORS
from .demand_models import DemandFit, predict_point
from .price_response import PriceFit


def moving_block_sample(matrix: np.ndarray, scenario_count: int, horizon: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """按连续日期区块抽样，返回情景、日期、品类三维数组。"""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) == 0:
        return np.zeros((scenario_count, horizon, matrix.shape[1]), dtype=float)
    starts = rng.integers(0, len(matrix), size=(scenario_count, int(np.ceil(horizon / block_length))))
    result = np.empty((scenario_count, horizon, matrix.shape[1]), dtype=float)
    for scenario in range(scenario_count):
        values: list[np.ndarray] = []
        for start in starts[scenario]:
            indices = (int(start) + np.arange(block_length)) % len(matrix)
            values.append(matrix[indices])
        result[scenario] = np.concatenate(values, axis=0)[:horizon]
    return result


def _demand_residual_matrix(fits: dict[str, DemandFit]) -> np.ndarray:
    dates: set[pd.Timestamp] = set()
    for fit in fits.values():
        dates.update(pd.to_datetime(fit.history["销售日期"]).dt.normalize())
    index = pd.DatetimeIndex(sorted(dates))
    matrix = np.full((len(index), len(CATEGORIES)), np.nan, dtype=float)
    for j, cat in enumerate(CATEGORIES):
        fit = fits[cat]
        residual = pd.Series(fit.residual_log - np.median(fit.residual_log), index=pd.to_datetime(fit.history["销售日期"]).dt.normalize())
        aligned = residual.reindex(index)
        matrix[:, j] = aligned.interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    return np.clip(matrix, -2.5, 2.5)


def _one_step_cost_residuals(panel: pd.DataFrame, selected_cost: dict[str, dict[str, str]], last_date: pd.Timestamp) -> np.ndarray:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["销售日期"]).dt.normalize().unique()))
    matrix = np.full((len(dates), len(CATEGORIES)), np.nan, dtype=float)
    for j, cat in enumerate(CATEGORIES):
        choice = selected_cost[cat]
        target_col = "销量加权进价" if choice["成本口径"] == "销量加权成本" else "固定篮子成本指数"
        frame = panel[panel["品类"] == cat].sort_values("销售日期").dropna(subset=[target_col])
        values = frame.set_index("销售日期")[target_col].astype(float)
        residuals: dict[pd.Timestamp, float] = {}
        for date, actual in values.items():
            date = pd.Timestamp(date).normalize()
            before = frame[frame["销售日期"] < date]
            if len(before) < 14:
                continue
            previous = values[values.index < date]
            if previous.empty:
                continue
            method = choice["成本预测方法"]
            from .cost_models import forecast_cost

            predicted = float(forecast_cost(before, pd.DatetimeIndex([date]), method, target_col).iloc[0])
            residuals[date] = float(actual - predicted)
        series = pd.Series(residuals, dtype=float).reindex(dates)
        matrix[:, j] = series.interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    return np.clip(matrix, -5.0, 5.0)


def _discount_demand(panel: pd.DataFrame) -> np.ndarray:
    values = []
    for cat in CATEGORIES:
        sub = panel[(panel["品类"] == cat) & (panel["折扣销售量"] > 0)]
        if len(sub):
            values.append(float(sub["折扣销售量"].median()))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=float)


def generate_scenario_bundle(
    normal_fits: dict[str, DemandFit],
    price_fits: dict[str, PriceFit],
    panel: pd.DataFrame,
    selected_cost: dict[str, dict[str, str]],
    future_cost_points: pd.DataFrame,
    category_loss: dict[str, float],
    reliability: dict[str, bool],
    scenario_count: int,
    seed: int,
    last_date: pd.Timestamp,
    calibration: dict[str, float] | None = None,
    block_length: int = 7,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    calibration = calibration or {cat: 1.0 for cat in CATEGORIES}
    demand_matrix = _demand_residual_matrix(normal_fits)
    demand_shocks = moving_block_sample(demand_matrix, scenario_count, len(FUTURE_DATES), block_length, rng)
    cost_matrix = _one_step_cost_residuals(panel, selected_cost, last_date)
    cost_shocks = moving_block_sample(cost_matrix, scenario_count, len(FUTURE_DATES), block_length, rng)
    demand_reference = np.zeros((scenario_count, len(FUTURE_DATES), len(CATEGORIES)), dtype=float)
    cost_reference = np.zeros_like(demand_reference)
    for j, cat in enumerate(CATEGORIES):
        fit = normal_fits[cat]
        point = predict_point(fit, FUTURE_DATES) * float(calibration.get(cat, 1.0))
        demand_reference[:, :, j] = point[None, :] * np.exp(demand_shocks[:, :, j])
        point_cost = future_cost_points[future_cost_points["品类"] == cat].sort_values("日期")["预测批发价"].to_numpy(float)
        cost_reference[:, :, j] = np.maximum(0.01, point_cost[None, :] + cost_shocks[:, :, j])

    beta_draws: dict[str, np.ndarray] = {}
    for cat in CATEGORIES:
        fit = price_fits[cat]
        if not reliability.get(cat, False):
            beta_draws[cat] = np.zeros(scenario_count, dtype=float)
            continue
        if len(fit.bootstrap7) >= 10:
            beta_draws[cat] = rng.choice(fit.bootstrap7, size=scenario_count, replace=True)
        else:
            beta_draws[cat] = rng.normal(fit.coefficient, max(fit.se7, 1e-3), size=scenario_count)
        beta_draws[cat] = np.minimum(beta_draws[cat], -1e-6)

    loss_paths = np.zeros((scenario_count, len(FUTURE_DATES), len(CATEGORIES)), dtype=float)
    loss_levels = np.asarray(LOSS_FACTORS, dtype=float)
    for j, cat in enumerate(CATEGORIES):
        base = float(category_loss[cat])
        factors = rng.choice(loss_levels, size=(scenario_count, len(FUTURE_DATES)), p=[0.2, 0.6, 0.2])
        loss_paths[:, :, j] = np.clip(base * factors, 0.0, 0.99)

    discount_base = _discount_demand(panel)
    discount_noise = np.exp(rng.normal(0.0, 0.25, size=(scenario_count, len(FUTURE_DATES), len(CATEGORIES))))
    discount_demand = discount_base[None, None, :] * discount_noise
    historical_ratios = panel.get("折扣价比中位数", pd.Series(dtype=float)).dropna().to_numpy(float)
    historical_ratio = float(np.median(historical_ratios)) if len(historical_ratios) else 0.7
    discount_recovery_choices = np.asarray([0.0, historical_ratio, 1.0], dtype=float)
    recovery_scenarios = rng.choice(discount_recovery_choices, size=scenario_count, p=[0.65, 0.30, 0.05])
    return {
        "需求基准情景": demand_reference,
        "成本情景": cost_reference,
        "价格系数抽样": beta_draws,
        "损耗情景": loss_paths,
        "折扣潜在需求": discount_demand,
        "折扣回收比例情景": recovery_scenarios,
        "历史折扣价比中位数": historical_ratio,
        "需求扰动区块长度": block_length,
        "情景数": scenario_count,
        "未来日期": [str(x.date()) for x in FUTURE_DATES],
    }
