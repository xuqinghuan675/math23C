# -*- coding: utf-8 -*-
"""透明的价格—补货离散搜索、策略收益分解和敏感性分析。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CATEGORIES,
    FUTURE_DATES,
    MAIN_MARKUP_BAND,
    MARKUP_BANDS,
    OUTPUT_DIR,
)
from .demand_models import DemandFit, predict_point
from .price_response import PriceFit, reference_markup
from .uncertainty import generate_scenario_bundle


def _round_tenth(value: float) -> float:
    return float(max(0.0, round(float(value) * 10.0) / 10.0))


def _price_grid(cost: float, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    low_cent = int(math.ceil(float(cost) * (1.0 + low) * 100.0 - 1e-9))
    high_cent = int(math.floor(float(cost) * (1.0 + high) * 100.0 + 1e-9))
    if high_cent < low_cent:
        high_cent = low_cent
    return np.arange(max(1, low_cent), max(1, high_cent) + 1, dtype=float) / 100.0


def _price_variable(fit: PriceFit, date: pd.Timestamp, price: float, costs: np.ndarray) -> np.ndarray:
    costs = np.maximum(np.asarray(costs, dtype=float), 1e-6)
    ref = reference_markup(fit, date)
    markup = float(price) / costs - 1.0
    if fit.model_name == "半对数加成":
        value = markup - ref
    elif fit.model_name == "对数加成":
        value = np.log1p(np.maximum(markup, -0.95)) - math.log1p(max(ref, -0.95))
    elif fit.model_name == "半对数售价偏离":
        value = float(price) - costs * (1.0 + ref)
    elif fit.model_name == "对数售价比":
        value = np.log(np.maximum(float(price) / np.maximum(costs * (1.0 + ref), 1e-8), 1e-8))
    else:
        value = np.zeros_like(costs)
    return np.asarray(value, dtype=float)


def _profit_arrays(
    price: float,
    order: float,
    demand: np.ndarray,
    costs: np.ndarray,
    losses: np.ndarray,
    discount_demand: np.ndarray,
    recovery_rate: float | np.ndarray = 0.0,
    discount_ratio: float = 0.0,
) -> dict[str, np.ndarray]:
    sellable = max(0.0, float(order)) * np.maximum(0.0, 1.0 - losses)
    normal_sales = np.minimum(np.maximum(demand, 0.0), sellable)
    remaining = np.maximum(sellable - normal_sales, 0.0)
    recovery_rate_array = np.asarray(recovery_rate, dtype=float)
    discount_sales = np.minimum(np.maximum(discount_demand, 0.0) * np.maximum(recovery_rate_array, 0.0), remaining)
    profit = float(price) * normal_sales + float(price) * float(discount_ratio) * discount_sales - costs * float(order)
    return {
        "利润": profit,
        "正常销售量": normal_sales,
        "折扣销售量": discount_sales,
        "剩余量": np.maximum(remaining - discount_sales, 0.0),
        "可售量": sellable,
    }


def _summarise(price: float, order: float, arrays: dict[str, np.ndarray], demand: np.ndarray, cost: np.ndarray, lower: float, upper: float, support: str) -> dict[str, Any]:
    profit = arrays["利润"]
    sold = arrays["正常销售量"] + arrays["折扣销售量"]
    return {
        "售价": float(price),
        "补货量": _round_tenth(order),
        "加成率": float(price / max(float(np.mean(cost)), 1e-9) - 1.0),
        "平均需求": float(np.mean(demand)),
        "平均正常销售量": float(np.mean(arrays["正常销售量"])),
        "平均折扣销售量": float(np.mean(arrays["折扣销售量"])),
        "平均剩余量": float(np.mean(arrays["剩余量"])),
        "平均利润": float(np.mean(profit)),
        "利润P10": float(np.quantile(profit, 0.10)),
        "利润P50": float(np.quantile(profit, 0.50)),
        "利润P90": float(np.quantile(profit, 0.90)),
        "亏损概率": float(np.mean(profit < 0)),
        "缺货概率": float(np.mean(demand > arrays["可售量"])),
        "剩余概率": float(np.mean(arrays["剩余量"] > 1e-9)),
        "销售总量": float(np.mean(sold)),
        "下界加成率": float(lower),
        "上界加成率": float(upper),
        "历史支持等级": support,
    }


def _order_candidates(center: float, coarse: bool = True) -> np.ndarray:
    center = _round_tenth(center)
    if coarse:
        offsets = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    else:
        offsets = np.arange(-2.0, 2.0001, 0.1).tolist()
    values = {_round_tenth(center + offset) for offset in offsets if center + offset >= 0.1}
    values.add(max(0.1, center))
    return np.asarray(sorted(values), dtype=float)


def _best_order(
    price: float,
    base_demand: np.ndarray,
    beta: np.ndarray,
    fit: PriceFit,
    date: pd.Timestamp,
    costs: np.ndarray,
    losses: np.ndarray,
    discount_demand: np.ndarray,
    discount_ratio: float,
    recovery_rate: float | np.ndarray = 0.0,
    initial: float | None = None,
    refine: bool = False,
) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
    effect = _price_variable(fit, date, price, costs)
    demand = np.maximum(0.0, base_demand * np.exp(np.clip(beta * effect, -10.0, 10.0)))
    effective_loss = np.maximum(1e-6, 1.0 - float(np.mean(losses)))
    effective_cost = float(np.mean(costs)) / effective_loss
    fractile = float(np.clip(1.0 - effective_cost / max(float(price), 1e-6), 0.01, 0.99))
    initial = float(initial) if initial is not None else float(np.quantile(demand / np.maximum(1e-6, 1.0 - losses), fractile))
    initial = max(0.1, _round_tenth(initial))
    orders = _order_candidates(initial, coarse=not refine)
    q_matrix = orders[:, None]
    sellable = q_matrix * np.maximum(0.0, 1.0 - losses[None, :])
    normal_sales = np.minimum(demand[None, :], sellable)
    remaining = np.maximum(sellable - normal_sales, 0.0)
    recovery = np.asarray(recovery_rate, dtype=float)
    discount_sales = np.minimum(discount_demand[None, :] * np.maximum(recovery, 0.0), remaining)
    profit = float(price) * normal_sales + float(price) * float(discount_ratio) * discount_sales - costs[None, :] * q_matrix
    means = profit.mean(axis=1)
    best_idx = int(np.argmax(means))
    best_order = float(orders[best_idx])
    arrays = _profit_arrays(price, best_order, demand, costs, losses, discount_demand, recovery_rate, discount_ratio)
    return best_order, arrays, demand


def _refine_order(
    price: float,
    center: float,
    base_demand: np.ndarray,
    beta: np.ndarray,
    fit: PriceFit,
    date: pd.Timestamp,
    costs: np.ndarray,
    losses: np.ndarray,
    discount_demand: np.ndarray,
    discount_ratio: float,
    recovery_rate: float | np.ndarray = 0.0,
) -> tuple[float, dict[str, np.ndarray], np.ndarray, float]:
    order, _, _ = _best_order(price, base_demand, beta, fit, date, costs, losses, discount_demand, discount_ratio, recovery_rate, center, refine=True)
    orders = _order_candidates(order, coarse=False)
    effect = _price_variable(fit, date, price, costs)
    demand = np.maximum(0.0, base_demand * np.exp(np.clip(beta * effect, -10.0, 10.0)))
    means = []
    arrays_by_order: list[dict[str, np.ndarray]] = []
    for current in orders:
        arrays = _profit_arrays(price, current, demand, costs, losses, discount_demand, recovery_rate, discount_ratio)
        arrays_by_order.append(arrays)
        means.append(float(arrays["利润"].mean()))
    idx = int(np.argmax(means))
    selected = float(orders[idx])
    # 关键邻点核验：最终网格解不劣于其存在的相邻 0.1 千克方案。
    neighbour_means = []
    for neighbor in [selected - 0.1, selected + 0.1]:
        if neighbor >= 0.1:
            arrays = _profit_arrays(price, neighbor, demand, costs, losses, discount_demand, recovery_rate, discount_ratio)
            neighbour_means.append(float(arrays["利润"].mean()))
    margin = float(means[idx] - max(neighbour_means)) if neighbour_means else np.nan
    return selected, arrays_by_order[idx], demand, margin


def _historical_support(panel: pd.DataFrame, category: str) -> dict[str, float]:
    sub = panel[(panel["品类"] == category) & (panel["正常销售量"] > 0)].copy()
    prices = pd.to_numeric(sub["正常销售售价"], errors="coerce").dropna().to_numpy(float)
    costs = pd.to_numeric(sub["正常销售进价"], errors="coerce").dropna().to_numpy(float)
    markup = prices / np.maximum(costs, 1e-8) - 1.0
    markup = markup[np.isfinite(markup)]
    if len(markup) == 0:
        markup = np.asarray([0.5])
    if len(prices) == 0:
        prices = np.asarray([1.0])
    if len(costs) == 0:
        costs = np.asarray([1.0])
    return {
        "加成P01": float(np.quantile(markup, 0.01)),
        "加成P05": float(np.quantile(markup, 0.05)),
        "加成P25": float(np.quantile(markup, 0.25)),
        "加成P50": float(np.quantile(markup, 0.50)),
        "加成P75": float(np.quantile(markup, 0.75)),
        "加成P95": float(np.quantile(markup, 0.95)),
        "加成P99": float(np.quantile(markup, 0.99)),
        "售价P01": float(np.quantile(prices, 0.01)),
        "售价P05": float(np.quantile(prices, 0.05)),
        "售价P25": float(np.quantile(prices, 0.25)),
        "售价P50": float(np.quantile(prices, 0.50)),
        "售价P75": float(np.quantile(prices, 0.75)),
        "售价P95": float(np.quantile(prices, 0.95)),
        "售价P99": float(np.quantile(prices, 0.99)),
        "成本P10": float(np.quantile(costs, 0.10)),
        "成本P90": float(np.quantile(costs, 0.90)),
        "最近历史售价": float(prices[-1]),
        "历史售价四分位距": float(np.quantile(prices, 0.75) - np.quantile(prices, 0.25)),
    }


def check_price_support(price: float, cost: float, date: pd.Timestamp, band: tuple[float, float], support: dict[str, float]) -> tuple[str, bool, float, float, float]:
    markup = float(price / max(cost, 1e-8) - 1.0)
    markup_position = (markup - support["加成P01"]) / max(support["加成P99"] - support["加成P01"], 1e-9)
    price_position = (price - support["售价P01"]) / max(support["售价P99"] - support["售价P01"], 1e-9)
    recent_distance = abs(price - support["最近历史售价"]) / max(support["历史售价四分位距"], 1e-6)
    within_core = support["加成P05"] <= markup <= support["加成P95"] and support["售价P05"] <= price <= support["售价P95"] and support["成本P10"] * 0.8 <= cost <= support["成本P90"] * 1.2
    within_outer = support["加成P01"] <= markup <= support["加成P99"] and support["售价P01"] <= price <= support["售价P99"]
    if within_core:
        level = "强"
    elif within_outer:
        level = "中"
    else:
        level = "弱或外推"
    return level, bool(within_outer), float(markup_position), float(price_position), float(recent_distance)


def _cell(
    category: str,
    day_index: int,
    date: pd.Timestamp,
    future_cost: float,
    loss_rate: float,
    demand_fit: DemandFit,
    price_fit: PriceFit,
    reliable: bool,
    band: tuple[float, float],
    support: dict[str, float],
    bundle: dict[str, Any],
    calibration: float,
    collect_curve: bool = True,
) -> dict[str, Any]:
    base = bundle["需求基准情景"][:, day_index, CATEGORIES.index(category)]
    costs = bundle["成本情景"][:, day_index, CATEGORIES.index(category)]
    losses = bundle["损耗情景"][:, day_index, CATEGORIES.index(category)]
    beta = bundle["价格系数抽样"][category]
    discount_demand = bundle["折扣潜在需求"][:, day_index, CATEGORIES.index(category)]
    discount_ratio = float(bundle.get("历史折扣价比中位数", 0.0))
    candidates: list[dict[str, Any]] = []
    for price in _price_grid(future_cost, band):
        support_level, supported, markup_pos, price_pos, recent_distance = check_price_support(price, future_cost, date, band, support)
        order, arrays, demand = _best_order(price, base, beta if reliable else np.zeros_like(beta), price_fit, date, costs, losses, discount_demand, discount_ratio, 0.0, None, False)
        summary = _summarise(price, order, arrays, demand, costs, band[0], band[1], support_level)
        summary.update({"日期": date.date().isoformat(), "品类": category, "是否在历史外圈": "否" if supported else "是", "加成分位位置": markup_pos, "绝对售价分位位置": price_pos, "最近历史距离": recent_distance})
        candidates.append(summary)
    if not candidates:
        raise ValueError(f"{category} {date.date()} 没有价格候选")
    candidate_df = pd.DataFrame(candidates)
    math_idx = int(candidate_df["平均利润"].to_numpy(float).argmax())
    math_summary = candidates[math_idx]
    near_best = candidate_df[candidate_df["平均利润"] >= float(candidate_df["平均利润"].max()) * 0.99].copy()
    near_best = near_best.sort_values(["利润P10", "最近历史距离", "售价"], ascending=[False, True, True])
    robust_summary = near_best.iloc[0].to_dict()
    ref_markup = reference_markup(price_fit, date)
    reference_price = float(np.round(future_cost * (1.0 + ref_markup), 2))
    reference_price = float(np.clip(reference_price, candidates[0]["售价"], candidates[-1]["售价"]))
    reference_order, reference_arrays, reference_demand = _best_order(reference_price, base, np.zeros_like(beta) if not reliable else beta, price_fit, date, costs, losses, discount_demand, discount_ratio, 0.0, None, False)
    reference_summary = _summarise(reference_price, reference_order, reference_arrays, reference_demand, costs, band[0], band[1], check_price_support(reference_price, future_cost, date, band, support)[0])
    # 对数学价、稳健价和参考价分别进行 0.1 千克细化，并保留邻点差值。
    refined: dict[str, tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, float]] = {}
    for name, summary in [("数学", math_summary), ("稳健", robust_summary), ("参考", reference_summary)]:
        price = float(summary["售价"])
        refined_order, refined_arrays, refined_demand, neighbour_margin = _refine_order(
            price, float(summary["补货量"]), base, beta if reliable else np.zeros_like(beta), price_fit, date, costs, losses, discount_demand, discount_ratio, 0.0
        )
        refined_summary = _summarise(price, refined_order, refined_arrays, refined_demand, costs, band[0], band[1], summary["历史支持等级"])
        refined[name] = (refined_summary, refined_arrays, refined_demand, neighbour_margin)
    math_final, math_arrays, math_demand, math_neighbour = refined["数学"]
    robust_final, robust_arrays, robust_demand, robust_neighbour = refined["稳健"]
    reference_final, reference_arrays, reference_demand, reference_neighbour = refined["参考"]
    final_name = "稳健" if reliable else "参考"
    final_summary, final_arrays, final_demand, final_neighbour = refined[final_name]
    # 四套策略使用同一联合情景，确保收益差额有代数意义。
    mean_base_order = _round_tenth(float(np.mean(base) / max(1e-6, 1.0 - loss_rate)))
    base_arrays = _profit_arrays(reference_final["售价"], mean_base_order, reference_demand, costs, losses, discount_demand, 0.0, 0.0)
    only_replenishment_order, only_replenishment_arrays, _ = _best_order(reference_final["售价"], base, beta if reliable else np.zeros_like(beta), price_fit, date, costs, losses, discount_demand, discount_ratio, 0.0, reference_final["补货量"], True)
    only_replenishment_arrays = _profit_arrays(reference_final["售价"], only_replenishment_order, reference_demand, costs, losses, discount_demand, 0.0, 0.0)
    if reliable:
        c_arrays = math_arrays
        c_summary = math_final
    else:
        c_arrays = only_replenishment_arrays
        c_summary = _summarise(reference_final["售价"], only_replenishment_order, c_arrays, reference_demand, costs, band[0], band[1], reference_final["历史支持等级"])
    d_arrays = final_arrays
    d_summary = final_summary
    strategies = {
        "A传统基准": {"售价": reference_final["售价"], "补货量": mean_base_order, "利润数组": base_arrays, "汇总": _summarise(reference_final["售价"], mean_base_order, base_arrays, reference_demand, costs, band[0], band[1], reference_final["历史支持等级"])},
        "B仅优化补货": {"售价": reference_final["售价"], "补货量": only_replenishment_order, "利润数组": only_replenishment_arrays, "汇总": _summarise(reference_final["售价"], only_replenishment_order, only_replenishment_arrays, reference_demand, costs, band[0], band[1], reference_final["历史支持等级"])},
        "C期望利润最大": {"售价": c_summary["售价"], "补货量": c_summary["补货量"], "利润数组": c_arrays, "汇总": c_summary},
        "D稳健经营": {"售价": d_summary["售价"], "补货量": d_summary["补货量"], "利润数组": d_arrays, "汇总": d_summary},
    }
    math_markup = float(math_final["加成率"])
    boundary = abs(math_markup - band[1]) <= 0.015 or abs(math_markup - band[0]) <= 0.015
    direction = "上界" if abs(math_markup - band[1]) <= 0.015 else "下界" if abs(math_markup - band[0]) <= 0.015 else "无"
    return {
        "日期": date.date().isoformat(),
        "品类": category,
        "未来成本点值": float(future_cost),
        "损耗率": float(loss_rate),
        "参考加成率": float(ref_markup),
        "参考正常售价": float(reference_final["售价"]),
        "数学最优": math_final,
        "稳健推荐": robust_final,
        "最终": final_summary,
        "最终数组": final_arrays,
        "最终需求": final_demand,
        "数学需求": math_demand,
        "可靠": bool(reliable),
        "候选": candidates if collect_curve else [],
        "策略": strategies,
        "是否边界": "是" if boundary else "否",
        "边界方向": direction,
        "数学补货邻点差": math_neighbour,
        "稳健补货邻点差": robust_neighbour,
        "参考补货邻点差": reference_neighbour,
        "支持": support,
        "校准系数": float(calibration),
    }


def _future_cost_exports(bundle: dict[str, Any], future_cost_points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    for j, cat in enumerate(CATEGORIES):
        point = future_cost_points[future_cost_points["品类"] == cat].sort_values("日期")["预测批发价"].to_numpy(float)
        samples = bundle["成本情景"][:, :, j]
        for i, date in enumerate(FUTURE_DATES):
            values = samples[:, i]
            path_rows.append({"日期": date.date().isoformat(), "品类": cat, "成本口径": future_cost_points[future_cost_points["品类"] == cat].iloc[0]["成本口径"], "成本预测方法": future_cost_points[future_cost_points["品类"] == cat].iloc[0]["成本预测方法"], "预测批发价": float(point[i]), "成本P10": float(np.quantile(values, 0.10)), "成本P50": float(np.quantile(values, 0.50)), "成本P90": float(np.quantile(values, 0.90))})
            quantile_rows.append({"日期": date.date().isoformat(), "品类": cat, "P10": float(np.quantile(values, 0.10)), "P25": float(np.quantile(values, 0.25)), "P50": float(np.quantile(values, 0.50)), "P75": float(np.quantile(values, 0.75)), "P90": float(np.quantile(values, 0.90))})
    return pd.DataFrame(path_rows), pd.DataFrame(quantile_rows)


def run_optimization(
    enriched_panel: pd.DataFrame,
    normal_fits: dict[str, DemandFit],
    price_info: dict[str, Any],
    selected_cost: dict[str, dict[str, str]],
    future_cost_points: pd.DataFrame,
    category_loss: dict[str, float],
    calibration: dict[str, float],
    scenario_count: int,
    seed: int,
    last_date: pd.Timestamp,
    quick: bool = False,
) -> dict[str, Any]:
    relation = price_info["relation"].set_index("品类")
    price_fits: dict[str, PriceFit] = price_info["selected"]
    reliability = {cat: str(relation.loc[cat, "价格关系是否可靠"]) == "是" for cat in CATEGORIES}
    bundle = generate_scenario_bundle(
        normal_fits,
        price_fits,
        enriched_panel,
        selected_cost,
        future_cost_points,
        category_loss,
        reliability,
        scenario_count,
        seed,
        last_date,
        calibration,
        block_length=7,
    )
    support_map = {cat: _historical_support(enriched_panel, cat) for cat in CATEGORIES}
    cell_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        cost_sub = future_cost_points[future_cost_points["品类"] == cat].sort_values("日期")
        for day_index, date in enumerate(FUTURE_DATES):
            future_cost = float(cost_sub.iloc[day_index]["预测批发价"])
            result = _cell(cat, day_index, date, future_cost, category_loss[cat], normal_fits[cat], price_fits[cat], reliability[cat], MAIN_MARKUP_BAND, support_map[cat], bundle, calibration.get(cat, 1.0), collect_curve=True)
            cell_rows.append(result)
            for candidate in result["候选"]:
                curve_rows.append(candidate)
            for strategy_name, strategy in result["策略"].items():
                strategy_rows.append({"日期": result["日期"], "品类": cat, "策略": strategy_name, "售价": strategy["售价"], "补货量": strategy["补货量"], "预计毛利": float(strategy["汇总"]["平均利润"]), "毛利P10": float(strategy["汇总"]["利润P10"]), "毛利P50": float(strategy["汇总"]["利润P50"]), "毛利P90": float(strategy["汇总"]["利润P90"])})
            final_summary = result["最终"]
            cost_values = bundle["成本情景"][:, day_index, CATEGORIES.index(cat)]
            demand_values = result["最终需求"]
            final_rows.append(
                {
                    "日期": result["日期"],
                    "品类": cat,
                    "需求模型": normal_fits[cat].model_name,
                    "价格响应模型": price_fits[cat].model_name,
                    "价格关系是否可靠": "是" if reliability[cat] else "否",
                    "价格响应系数": price_fits[cat].coefficient,
                    "参考点弹性": price_fits[cat].coefficient * (1.0 + price_fits[cat].reference_fallback),
                    "预测批发价P10": float(np.quantile(cost_values, 0.10)),
                    "预测批发价P50": float(np.quantile(cost_values, 0.50)),
                    "预测批发价P90": float(np.quantile(cost_values, 0.90)),
                    "附件四品类损耗率": category_loss[cat],
                    "条件参考加成率": result["参考加成率"],
                    "价格经营带下限": MAIN_MARKUP_BAND[0],
                    "价格经营带上限": MAIN_MARKUP_BAND[1],
                    "参考正常售价": result["参考正常售价"],
                    "数学期望利润最大售价": result["数学最优"]["售价"],
                    "稳健推荐售价": result["稳健推荐"]["售价"] if reliability[cat] else result["参考正常售价"],
                    "建议成本加成率": final_summary["加成率"],
                    "正常需求P10": float(np.quantile(demand_values, 0.10)),
                    "正常需求P50": float(np.quantile(demand_values, 0.50)),
                    "正常需求P90": float(np.quantile(demand_values, 0.90)),
                    "建议补货量": final_summary["补货量"],
                    "补货量敏感区间": f"{_round_tenth(np.quantile(demand_values / np.maximum(1e-6, 1.0 - bundle['损耗情景'][:, day_index, CATEGORIES.index(cat)]), 0.10)):.1f}-{_round_tenth(np.quantile(demand_values / np.maximum(1e-6, 1.0 - bundle['损耗情景'][:, day_index, CATEGORIES.index(cat)]), 0.90)):.1f}千克",
                    "预计正常销售量": final_summary["平均正常销售量"],
                    "预计折扣销售量": final_summary["平均折扣销售量"],
                    "预计剩余量": final_summary["平均剩余量"],
                    "缺货概率": final_summary["缺货概率"],
                    "剩余概率": final_summary["剩余概率"],
                    "预计毛利": final_summary["平均利润"],
                    "毛利P10": final_summary["利润P10"],
                    "毛利P50": final_summary["利润P50"],
                    "毛利P90": final_summary["利润P90"],
                    "是否边界解": result["是否边界"],
                    "边界方向": result["边界方向"],
                    "历史支持等级": final_summary["历史支持等级"],
                    "定价依据": "可靠价格关系下在历史支持价格带内先最大化期望利润，再在不超过最优利润1%的候选中选取利润下界较稳健的价格。" if reliability[cat] else "价格关系不可识别，未采用精细弹性最优价；采用截止日前条件中位加成作为经营价格。",
                    "补货依据": "联合需求、成本和损耗情景下的报童初值，并在0.1千克网格邻域离散复核。",
                    "主要风险提示": "无库存、缺货和剩余记录，历史销量可能低估潜在需求；价格关系是条件关联。" + ("数学最优价触及经营带边界，需结合经营试验复核。" if result["是否边界"] == "是" else ""),
                }
            )
    final_df = pd.DataFrame(final_rows).sort_values(["日期", "品类"]).reset_index(drop=True)
    curve_df = pd.DataFrame(curve_rows).sort_values(["品类", "日期", "售价"]).reset_index(drop=True)
    strategy_df = pd.DataFrame(strategy_rows).sort_values(["日期", "品类", "策略"]).reset_index(drop=True)
    path_df, quantile_df = _future_cost_exports(bundle, future_cost_points)
    strategy_summary = strategy_df.groupby(["品类", "策略"], as_index=False).agg(七天预计毛利=("预计毛利", "sum"), 毛利P10=("毛利P10", "sum"), 毛利P50=("毛利P50", "sum"), 毛利P90=("毛利P90", "sum"), 七天补货量=("补货量", "sum"))
    decomp_rows: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        sub = strategy_summary[strategy_summary["品类"] == cat].set_index("策略")
        a = float(sub.loc["A传统基准", "七天预计毛利"])
        b = float(sub.loc["B仅优化补货", "七天预计毛利"])
        c = float(sub.loc["C期望利润最大", "七天预计毛利"])
        d = float(sub.loc["D稳健经营", "七天预计毛利"])
        decomp_rows.extend(
            [
                {"品类": cat, "策略": "A传统基准", "七天预计毛利": a, "相对A改善": 0.0, "说明": "条件中位加成，平均需求除以可售比例"},
                {"品类": cat, "策略": "B仅优化补货", "七天预计毛利": b, "相对A改善": b - a, "说明": "保持参考价格，仅改变补货量"},
                {"品类": cat, "策略": "C期望利润最大", "七天预计毛利": c, "相对A改善": c - a, "说明": "可靠品类优化价格和补货；不可靠品类不精细调价"},
                {"品类": cat, "策略": "D稳健经营", "七天预计毛利": d, "相对A改善": d - a, "说明": "利润下界或1%平台内的稳健价格和补货"},
            ]
        )
    decomp_df = pd.DataFrame(decomp_rows)
    total_a = float(strategy_summary.loc[strategy_summary["策略"] == "A传统基准", "七天预计毛利"].sum())
    total_b = float(strategy_summary.loc[strategy_summary["策略"] == "B仅优化补货", "七天预计毛利"].sum())
    total_c = float(strategy_summary.loc[strategy_summary["策略"] == "C期望利润最大", "七天预计毛利"].sum())
    total_d = float(strategy_summary.loc[strategy_summary["策略"] == "D稳健经营", "七天预计毛利"].sum())
    strategy_total = pd.DataFrame(
        [
            {"策略": "A传统基准", "七天预计毛利": total_a, "相对A改善": 0.0},
            {"策略": "B仅优化补货", "七天预计毛利": total_b, "相对A改善": total_b - total_a},
            {"策略": "C期望利润最大", "七天预计毛利": total_c, "相对A改善": total_c - total_a},
            {"策略": "D稳健经营", "七天预计毛利": total_d, "相对A改善": total_d - total_a},
        ]
    )
    final_df.to_csv(OUTPUT_DIR / "09_七天六品类最终策略.csv", index=False, encoding="utf-8-sig")
    curve_df.to_csv(OUTPUT_DIR / "07_价格利润曲线.csv", index=False, encoding="utf-8-sig")
    strategy_df.to_csv(OUTPUT_DIR / "07_策略逐日结果.csv", index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(OUTPUT_DIR / "07_策略对比汇总.csv", index=False, encoding="utf-8-sig")
    decomp_df.to_csv(OUTPUT_DIR / "07_策略收益分解.csv", index=False, encoding="utf-8-sig")
    strategy_total.to_csv(OUTPUT_DIR / "07_策略收益总表.csv", index=False, encoding="utf-8-sig")
    path_df.to_csv(OUTPUT_DIR / "06_未来成本路径.csv", index=False, encoding="utf-8-sig")
    quantile_df.to_csv(OUTPUT_DIR / "06_未来成本情景分位数.csv", index=False, encoding="utf-8-sig")
    return {
        "bundle": bundle,
        "cells": cell_rows,
        "final": final_df,
        "curve": curve_df,
        "strategy": strategy_df,
        "strategy_summary": strategy_summary,
        "decomposition": decomp_df,
        "strategy_total": strategy_total,
        "support": support_map,
        "cost_path": path_df,
        "cost_quantiles": quantile_df,
        "reliability": reliability,
    }


def run_sensitivities(
    base_result: dict[str, Any],
    enriched_panel: pd.DataFrame,
    normal_fits: dict[str, DemandFit],
    price_info: dict[str, Any],
    selected_cost: dict[str, dict[str, str]],
    future_cost_points: pd.DataFrame,
    category_loss: dict[str, float],
    calibration: dict[str, float],
    seed: int,
    last_date: pd.Timestamp,
    quick: bool = False,
) -> dict[str, pd.DataFrame]:
    bundle = base_result["bundle"]
    sensitivity_count = min(800 if not quick else 250, int(bundle["情景数"]))
    small_bundle = dict(bundle)
    small_bundle["情景数"] = sensitivity_count
    for key in ["需求基准情景", "成本情景", "损耗情景", "折扣潜在需求"]:
        small_bundle[key] = bundle[key][:sensitivity_count]
    small_bundle["价格系数抽样"] = {
        cat: values[:sensitivity_count] for cat, values in bundle["价格系数抽样"].items()
    }
    final = base_result["final"]
    price_fits: dict[str, PriceFit] = price_info["selected"]
    relation = price_info["relation"].set_index("品类")
    reliable = base_result["reliability"]
    support_map = base_result["support"]
    band_rows: list[dict[str, Any]] = []
    # 经营带敏感性使用同一组情景，避免把情景变化误认为经营带变化。
    for band_name, band in MARKUP_BANDS.items():
        for cat in CATEGORIES:
            profits = []
            touch = 0
            cost_sub = future_cost_points[future_cost_points["品类"] == cat].sort_values("日期")
            for i, date in enumerate(FUTURE_DATES):
                result = _cell(cat, i, date, float(cost_sub.iloc[i]["预测批发价"]), category_loss[cat], normal_fits[cat], price_fits[cat], reliable[cat], band, support_map[cat], small_bundle, calibration.get(cat, 1.0), collect_curve=False)
                profits.append(float(result["数学最优"]["平均利润"]))
                touch += int(result["是否边界"] == "是")
            band_rows.append({"经营带": band_name, "品类": cat, "下限": band[0], "上限": band[1], "七天数学搜索毛利": float(np.sum(profits)), "边界解天数": touch})
    band_df = pd.DataFrame(band_rows)

    loss_rows: list[dict[str, Any]] = []
    for factor in [0.8, 1.0, 1.2]:
        for cat in CATEGORIES:
            sub = final[final["品类"] == cat]
            losses = np.clip(category_loss[cat] * factor, 0.0, 0.99)
            values = []
            for i, row in sub.reset_index(drop=True).iterrows():
                base = small_bundle["需求基准情景"][:, i, CATEGORIES.index(cat)]
                costs = small_bundle["成本情景"][:, i, CATEGORIES.index(cat)]
                beta = small_bundle["价格系数抽样"][cat] if reliable[cat] else np.zeros(small_bundle["情景数"])
                demand = base * np.exp(np.clip(beta * _price_variable(price_fits[cat], pd.Timestamp(row["日期"]), float(row["稳健推荐售价"]), costs), -10, 10))
                arrays = _profit_arrays(float(row["稳健推荐售价"]), float(row["建议补货量"]), demand, costs, np.full_like(costs, losses), np.zeros_like(costs), 0.0, 0.0)
                values.append(float(np.mean(arrays["利润"])))
            loss_rows.append({"损耗情景": f"{factor:.1f}倍附件四损耗率", "损耗倍数": factor, "品类": cat, "七天预计毛利": float(np.sum(values)), "采用损耗率": losses})
    loss_df = pd.DataFrame(loss_rows)

    cost_rows: list[dict[str, Any]] = []
    for level, quantile in [("成本P10", 0.10), ("成本P50", 0.50), ("成本P90", 0.90)]:
        for cat in CATEGORIES:
            sub = final[final["品类"] == cat].sort_values("日期").reset_index(drop=True)
            values = []
            for i, row in sub.iterrows():
                cost_s = small_bundle["成本情景"][:, i, CATEGORIES.index(cat)]
                fixed_cost = float(np.quantile(cost_s, quantile))
                demand = small_bundle["需求基准情景"][:, i, CATEGORIES.index(cat)]
                beta = small_bundle["价格系数抽样"][cat] if reliable[cat] else np.zeros(small_bundle["情景数"])
                price = float(row["稳健推荐售价"])
                demand = demand * np.exp(np.clip(beta * _price_variable(price_fits[cat], pd.Timestamp(row["日期"]), price, np.full_like(cost_s, fixed_cost)), -10, 10))
                arrays = _profit_arrays(price, float(row["建议补货量"]), demand, np.full_like(cost_s, fixed_cost), small_bundle["损耗情景"][:, i, CATEGORIES.index(cat)], np.zeros_like(cost_s), 0.0, 0.0)
                values.append(float(np.mean(arrays["利润"])))
            cost_rows.append({"成本情景": level, "品类": cat, "七天预计毛利": float(np.sum(values))})
    cost_df = pd.DataFrame(cost_rows)

    beta_rows: list[dict[str, Any]] = []
    for level in ["点估计", "95%区间下限", "95%区间上限", "自助法抽样"]:
        for cat in CATEGORIES:
            fit = price_fits[cat]
            if level == "点估计":
                beta_values = np.full(sensitivity_count, fit.coefficient)
            elif level == "95%区间下限":
                beta_values = np.full(sensitivity_count, fit.lower95)
            elif level == "95%区间上限":
                beta_values = np.full(sensitivity_count, fit.upper95)
            else:
                beta_values = fit.bootstrap7 if len(fit.bootstrap7) else np.full(bundle["情景数"], fit.coefficient)
                beta_values = np.resize(beta_values, sensitivity_count)
            values = []
            sub = final[final["品类"] == cat].sort_values("日期").reset_index(drop=True)
            for i, row in sub.iterrows():
                costs = small_bundle["成本情景"][:, i, CATEGORIES.index(cat)]
                demand = small_bundle["需求基准情景"][:, i, CATEGORIES.index(cat)]
                price = float(row["稳健推荐售价"])
                demand = demand * np.exp(np.clip(beta_values * _price_variable(fit, pd.Timestamp(row["日期"]), price, costs), -10, 10))
                arrays = _profit_arrays(price, float(row["建议补货量"]), demand, costs, small_bundle["损耗情景"][:, i, CATEGORIES.index(cat)], np.zeros_like(costs), 0.0, 0.0)
                values.append(float(np.mean(arrays["利润"])))
            beta_rows.append({"价格系数情景": level, "品类": cat, "采用系数中心": float(np.mean(beta_values)), "七天预计毛利": float(np.sum(values)), "是否纳入优化": "是" if reliable[cat] and level != "95%区间上限" else "否"})
    beta_df = pd.DataFrame(beta_rows)

    recovery_rows: list[dict[str, Any]] = []
    historical_ratio = float(bundle.get("历史折扣价比中位数", 0.0))
    for label, recovery, ratio in [("零残值", 0.0, 0.0), ("历史中位折扣比值有限回收", 1.0, historical_ratio), ("乐观回收上界", 1.0, 1.0)]:
        for cat in CATEGORIES:
            values = []
            sub = final[final["品类"] == cat].sort_values("日期").reset_index(drop=True)
            for i, row in sub.iterrows():
                costs = small_bundle["成本情景"][:, i, CATEGORIES.index(cat)]
                demand = small_bundle["需求基准情景"][:, i, CATEGORIES.index(cat)]
                beta = small_bundle["价格系数抽样"][cat] if reliable[cat] else np.zeros(small_bundle["情景数"])
                price = float(row["稳健推荐售价"])
                demand = demand * np.exp(np.clip(beta * _price_variable(price_fits[cat], pd.Timestamp(row["日期"]), price, costs), -10, 10))
                discount_demand = small_bundle["折扣潜在需求"][:, i, CATEGORIES.index(cat)]
                arrays = _profit_arrays(price, float(row["建议补货量"]), demand, costs, small_bundle["损耗情景"][:, i, CATEGORIES.index(cat)], discount_demand, recovery, ratio)
                values.append(float(np.mean(arrays["利润"])))
            recovery_rows.append({"折扣残值情景": label, "折扣价相对正常价比值": ratio, "品类": cat, "七天预计毛利": float(np.sum(values)), "主方案是否采用": "是" if label == "零残值" else "否"})
    recovery_df = pd.DataFrame(recovery_rows)

    model_rows: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        relation_row = relation.loc[cat]
        model_rows.extend(
            [
                {"品类": cat, "需求口径": "正常销量", "价格口径": relation_row["主价格口径"], "价格响应模型": relation_row["主价格响应模型"], "价格响应系数": relation_row["价格响应系数"], "可靠性": relation_row["价格关系是否可靠"], "说明": "主方案口径"},
                {"品类": cat, "需求口径": "净销量", "价格口径": relation_row["主价格口径"], "价格响应模型": relation_row["主价格响应模型"], "价格响应系数": price_info["selected_full"][cat].coefficient, "可靠性": "仅作总量敏感性", "说明": "全量净需求敏感性，不直接替代正常价格响应"},
                {"品类": cat, "需求口径": "正常销量", "价格口径": "销量加权", "价格响应模型": relation_row["主价格响应模型"], "价格响应系数": price_info["source_fits"][(cat, "销量加权", relation_row["主价格响应模型"])].coefficient, "可靠性": "敏感性", "说明": "销量加权价格口径"},
            ]
        )
    model_df = pd.DataFrame(model_rows)
    band_df.to_csv(OUTPUT_DIR / "08_经营带敏感性.csv", index=False, encoding="utf-8-sig")
    loss_df.to_csv(OUTPUT_DIR / "08_损耗敏感性.csv", index=False, encoding="utf-8-sig")
    cost_df.to_csv(OUTPUT_DIR / "08_成本敏感性.csv", index=False, encoding="utf-8-sig")
    beta_df.to_csv(OUTPUT_DIR / "08_价格系数敏感性.csv", index=False, encoding="utf-8-sig")
    recovery_df.to_csv(OUTPUT_DIR / "08_折扣残值敏感性.csv", index=False, encoding="utf-8-sig")
    model_df.to_csv(OUTPUT_DIR / "08_模型口径敏感性.csv", index=False, encoding="utf-8-sig")
    return {"band": band_df, "loss": loss_df, "cost": cost_df, "beta": beta_df, "recovery": recovery_df, "model": model_df}
