# -*- coding: utf-8 -*-
"""问题二逐日动态定价层。

原分层稳健方案使用 isoelastic 需求：D_t(p)=A_t*(p/p0)^beta。
在成本和 beta 固定时，A_t 只缩放利润曲线，因此不同日期很容易得到相同最优加成率。

本层保留已经通过稳健检验的价格弹性，但在历史正常经营点附近做一阶局部线性化：

    dD/dP = beta * Q_ref / P_ref
    D_t(P) ≈ D_t^0 + (dD/dP) * (P - P_ref,t)

这里 Q_ref、P_ref 使用正常销售历史中位日销量和中位售价，斜率固定在历史经营尺度；
而 D_t^0 是每天不同的基础需求预测，因此不同日期的利润曲线不再只是等比例缩放。

同时，价格可行域不再用一个全局 IQR 复制七天，而使用最近一年“同星期正常销售加成率”
的条件 IQR，并向全局 IQR 收缩。这样星期差异只负责界定当天常规经营范围，不负责制造收益。
价格关系不可靠的品类不做利润型价格优化，而采用同星期条件中位加成。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

CORE = None
DYNAMIC_COST = None


def bind(core_module, dynamic_cost_module) -> None:
    global CORE, DYNAMIC_COST
    CORE = core_module
    DYNAMIC_COST = dynamic_cost_module


def _weekday_markup_band(
    panel_normal: pd.DataFrame,
    cat: str,
    date: pd.Timestamp,
    global_info: dict,
) -> tuple[float, float, float, int]:
    frame = panel_normal[panel_normal["品类"] == cat].sort_values("销售日期").copy()
    end = pd.Timestamp(frame["销售日期"].max())
    recent = frame[frame["销售日期"] >= end - pd.Timedelta(days=364)].copy()
    same = recent[recent["销售日期"].dt.weekday == pd.Timestamp(date).weekday()]

    global_q25 = float(global_info["百分之二十五分位"])
    global_med = float(global_info["中位数"])
    global_q75 = float(global_info["百分之七十五分位"])

    if len(same) < 8:
        return global_q25, global_med, global_q75, int(len(same))

    local_q25 = float(same["加成率"].quantile(0.25))
    local_med = float(same["加成率"].median())
    local_q75 = float(same["加成率"].quantile(0.75))

    # 条件样本约 50 个/星期；加入 20 个等效全局样本做收缩，避免某个星期被偶然噪声支配。
    weight = float(len(same) / (len(same) + 20.0))
    low = weight * local_q25 + (1.0 - weight) * global_q25
    center = weight * local_med + (1.0 - weight) * global_med
    high = weight * local_q75 + (1.0 - weight) * global_q75

    low = max(0.0, float(low))
    high = max(low + 0.01, float(high))
    center = float(np.clip(center, low, high))
    return low, center, high, int(len(same))


def _local_linear_slopes(
    panel_normal: pd.DataFrame,
    cat: str,
    beta_point: float,
    beta_low: float,
) -> tuple[float, float, float, float]:
    frame = panel_normal[panel_normal["品类"] == cat]
    q_ref = float(frame["日销售量"].median())
    p_ref = float(frame["日平均售价"].median())
    if q_ref <= 0 or p_ref <= 0:
        raise ValueError(f"{cat} 正常销售历史经营点非法")
    slope_point = float(beta_point * q_ref / p_ref)
    slope_conservative = float(beta_low * q_ref / p_ref)
    return slope_point, slope_conservative, q_ref, p_ref


def _evaluate_linear_candidate(
    base_samples: np.ndarray,
    price: float,
    reference_price: float,
    slope: float,
    future_cost: float,
    loss_rate: float,
) -> dict | None:
    effective_cost = future_cost / (1.0 - loss_rate)
    if price <= effective_cost or price <= 0 or reference_price <= 0:
        return None

    demand = np.maximum(0.05, base_samples + slope * (price - reference_price))
    fractile = float(np.clip(1.0 - effective_cost / price, 0.001, 0.999))
    target_sellable = float(np.quantile(demand, fractile))
    order = max(0.1, round(target_sellable / (1.0 - loss_rate), 1))
    sellable = order * (1.0 - loss_rate)
    sales = np.minimum(demand, sellable)
    return {
        "售价": float(price),
        "加成率": float(price / future_cost - 1.0),
        "预测需求量": float(demand.mean()),
        "补货量": float(order),
        "预计满足量": float(sales.mean()),
        "预计利润": float(np.mean(price * sales - future_cost * order)),
        "临界分位数": fractile,
    }


def optimize_hybrid(
    panel_all: pd.DataFrame,
    panel_normal: pd.DataFrame,
    category_loss: dict,
    markup_info: dict,
    selected_cost_methods: dict,
    base_specs: dict,
    normal_specs: dict,
    reliable: dict,
    first_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if CORE is None or DYNAMIC_COST is None:
        raise RuntimeError("动态定价模块尚未 bind")

    rng = np.random.default_rng(CORE.RANDOM_SEED + 701)
    rows = []
    diagnostics = []

    for cat in CORE.CATEGORIES:
        cost_frame = panel_all[panel_all["品类"] == cat].sort_values("销售日期")
        costs = DYNAMIC_COST.cost_forecast(
            cost_frame, CORE.FUTURE_DATES, selected_cost_methods[cat]
        )
        price_spec = normal_specs[cat]
        beta_point = float(price_spec["价格系数"])
        beta_low = float(price_spec["稳健区间下限"])
        info = markup_info[cat]["正常销售"]
        slope_point, slope_conservative, q_ref, p_ref = _local_linear_slopes(
            panel_normal, cat, beta_point, beta_low
        )

        for date in CORE.FUTURE_DATES:
            future_cost = float(costs.loc[date])
            low, center, high, weekday_n = _weekday_markup_band(
                panel_normal, cat, date, info
            )
            reference_price = float(np.round(future_cost * (1.0 + center), 2))

            residual_draws = rng.choice(
                np.asarray(base_specs[cat]["残差"], dtype=float),
                size=CORE.SAMPLE_COUNT,
                replace=True,
            )
            base_samples = CORE.base_demand_samples(
                base_specs[cat], date, residual_draws, first_date
            )

            if reliable[cat] and slope_point < 0:
                low_grid = math.floor(low * 100.0) / 100.0
                high_grid = math.ceil(high * 100.0) / 100.0
                markups = np.arange(low_grid, high_grid + 0.0001, 0.01)
                candidates = []
                for markup in markups:
                    price = float(np.round(future_cost * (1.0 + markup), 2))
                    slope_used = slope_conservative if price >= reference_price else slope_point
                    item = _evaluate_linear_candidate(
                        base_samples,
                        price,
                        reference_price,
                        slope_used,
                        future_cost,
                        category_loss[cat],
                    )
                    if item is not None:
                        candidates.append(item)
                if not candidates:
                    raise ValueError(f"{cat} {date.date()} 没有有效逐日价格候选")
                best = max(candidates, key=lambda x: x["预计利润"])
                local_upper = float(np.round(future_cost * (1.0 + high_grid), 2))
                boundary = abs(best["售价"] - local_upper) <= 0.011
                pricing_basis = "价格关系稳定：逐日需求曲线+同星期条件IQR内局部稳健优化"
            else:
                best = _evaluate_linear_candidate(
                    base_samples,
                    reference_price,
                    reference_price,
                    0.0,
                    future_cost,
                    category_loss[cat],
                )
                if best is None:
                    raise ValueError(f"{cat} {date.date()} 条件参考价低于有效成本")
                boundary = False
                pricing_basis = "价格关系证据不足：采用同星期条件中位成本加成"

            base_mean = float(base_samples.mean())
            mean_order = max(0.1, round(base_mean / (1.0 - category_loss[cat]), 1))
            baseline_sellable = mean_order * (1.0 - category_loss[cat])
            baseline_sales = np.minimum(base_samples, baseline_sellable)
            baseline_profit = float(
                np.mean(reference_price * baseline_sales - future_cost * mean_order)
            )

            rows.append(
                {
                    "日期": date.date().isoformat(),
                    "品类": cat,
                    "预测批发价": future_cost,
                    "损耗率（百分数）": category_loss[cat] * 100.0,
                    "价格关系可靠": "是" if reliable[cat] else "否",
                    "正常销售价格弹性": beta_point,
                    "局部线性价格斜率": slope_point,
                    "历史经营参考日销量": q_ref,
                    "历史经营参考售价": p_ref,
                    "同星期参考样本数": weekday_n,
                    "日条件加成下限（百分数）": low * 100.0,
                    "日条件参考加成（百分数）": center * 100.0,
                    "日条件加成上限（百分数）": high * 100.0,
                    "参考中位售价": reference_price,
                    "建议成本加成率（百分数）": best["加成率"] * 100.0,
                    "建议售价": best["售价"],
                    "预测需求量": best["预测需求量"],
                    "建议补货量": best["补货量"],
                    "预计满足量": best["预计满足量"],
                    "临界分位数": best["临界分位数"],
                    "预计利润": best["预计利润"],
                    "是否触及局部上界": "是" if boundary else "否",
                    "定价依据": pricing_basis,
                }
            )
            diagnostics.append(
                {
                    "日期": date.date().isoformat(),
                    "品类": cat,
                    "参考中位售价": reference_price,
                    "参考平均需求补货利润": baseline_profit,
                    "最终建议利润": best["预计利润"],
                    "相对基准改善": best["预计利润"] - baseline_profit,
                    "基础需求均值": base_mean,
                    "价格关系可靠": "是" if reliable[cat] else "否",
                }
            )

    result = pd.DataFrame(rows).sort_values(["日期", "品类"]).reset_index(drop=True)
    diag = pd.DataFrame(diagnostics).sort_values(["日期", "品类"]).reset_index(drop=True)

    export = result.copy()
    round2 = [
        "预测批发价",
        "损耗率（百分数）",
        "局部线性价格斜率",
        "历史经营参考日销量",
        "历史经营参考售价",
        "日条件加成下限（百分数）",
        "日条件参考加成（百分数）",
        "日条件加成上限（百分数）",
        "参考中位售价",
        "建议成本加成率（百分数）",
        "建议售价",
        "预测需求量",
        "建议补货量",
        "预计满足量",
        "预计利润",
    ]
    for col in round2:
        export[col] = export[col].round(2)
    export["正常销售价格弹性"] = export["正常销售价格弹性"].round(4)
    export["临界分位数"] = export["临界分位数"].round(4)
    CORE.base.write_csv(export, "七天六品类最终策略_分层稳健.csv")

    diag_export = diag.copy()
    for col in ["参考中位售价", "参考平均需求补货利润", "最终建议利润", "相对基准改善", "基础需求均值"]:
        diag_export[col] = diag_export[col].round(2)
    CORE.base.write_csv(diag_export, "策略分解_分层稳健.csv")
    return result, diag
