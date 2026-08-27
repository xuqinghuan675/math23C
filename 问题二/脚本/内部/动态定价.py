# -*- coding: utf-8 -*-
"""问题二逐日局部稳健定价层。

核心原则：
1. 价格响应仍使用已通过正常销售稳健检验的对数价格弹性，不引入人为绝对斜率；
2. 每天的参考加成率来自最近一年“同星期正常销售加成率”的条件中位数，并向全局中位数收缩；
3. 可靠品类只在同星期正常销售 35%~65% 分位的中央经营带内做局部利润优化；
4. 不可靠品类不做伪精确优化，直接采用同星期条件中位加成；
5. 成本预测允许动态模型与水平模型滚动回测竞争，因此日售价变化来自有数据依据的成本路径和星期经营规律。

这样既避免七天机械复制同一价格，也避免把条件 IQR 上沿当成高利润捷径。
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


def _weekday_local_band(
    panel_normal: pd.DataFrame,
    cat: str,
    date: pd.Timestamp,
) -> tuple[float, float, float, int]:
    frame = panel_normal[panel_normal["品类"] == cat].sort_values("销售日期").copy()
    end = pd.Timestamp(frame["销售日期"].max())
    recent = frame[frame["销售日期"] >= end - pd.Timedelta(days=364)].copy()
    same = recent[recent["销售日期"].dt.weekday == pd.Timestamp(date).weekday()]

    global_low = float(frame["加成率"].quantile(0.35))
    global_center = float(frame["加成率"].median())
    global_high = float(frame["加成率"].quantile(0.65))

    if len(same) < 8:
        return global_low, global_center, global_high, int(len(same))

    local_low = float(same["加成率"].quantile(0.35))
    local_center = float(same["加成率"].median())
    local_high = float(same["加成率"].quantile(0.65))

    # 20 个等效全局样本做收缩，避免某一星期的短期偶然波动主导定价。
    weight = float(len(same) / (len(same) + 20.0))
    low = weight * local_low + (1.0 - weight) * global_low
    center = weight * local_center + (1.0 - weight) * global_center
    high = weight * local_high + (1.0 - weight) * global_high

    low = max(0.0, float(low))
    high = max(low + 0.005, float(high))
    center = float(np.clip(center, low, high))
    return low, center, high, int(len(same))


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

        for date in CORE.FUTURE_DATES:
            future_cost = float(costs.loc[date])
            low, center, high, weekday_n = _weekday_local_band(panel_normal, cat, date)
            reference_price = float(np.round(future_cost * (1.0 + center), 2))

            residual_draws = rng.choice(
                np.asarray(base_specs[cat]["残差"], dtype=float),
                size=CORE.SAMPLE_COUNT,
                replace=True,
            )
            base_samples = CORE.base_demand_samples(
                base_specs[cat], date, residual_draws, first_date
            )

            if reliable[cat] and beta_point < 0:
                low_grid = math.floor(low * 100.0) / 100.0
                high_grid = math.ceil(high * 100.0) / 100.0
                markups = np.arange(low_grid, high_grid + 0.0001, 0.005)
                candidates = []
                for markup in markups:
                    price = float(np.round(future_cost * (1.0 + markup), 2))
                    beta_used = beta_low if price >= reference_price else beta_point
                    item = CORE.evaluate_candidate(
                        base_samples,
                        price,
                        reference_price,
                        beta_used,
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
                pricing_basis = "价格关系稳定：同星期中央经营带内逐日局部稳健优化"
            else:
                best = CORE.evaluate_candidate(
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
                    "同星期参考样本数": weekday_n,
                    "日局部加成下限（百分数）": low * 100.0,
                    "日条件参考加成（百分数）": center * 100.0,
                    "日局部加成上限（百分数）": high * 100.0,
                    "参考售价": reference_price,
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
                    "参考售价": reference_price,
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
        "预测批发价", "损耗率（百分数）", "日局部加成下限（百分数）",
        "日条件参考加成（百分数）", "日局部加成上限（百分数）", "参考售价",
        "建议成本加成率（百分数）", "建议售价", "预测需求量", "建议补货量",
        "预计满足量", "预计利润",
    ]
    for col in round2:
        export[col] = export[col].round(2)
    export["正常销售价格弹性"] = export["正常销售价格弹性"].round(4)
    export["临界分位数"] = export["临界分位数"].round(4)
    CORE.base.write_csv(export, "七天六品类最终策略_分层稳健.csv")

    diag_export = diag.copy()
    for col in ["参考售价", "参考平均需求补货利润", "最终建议利润", "相对基准改善", "基础需求均值"]:
        diag_export[col] = diag_export[col].round(2)
    CORE.base.write_csv(diag_export, "策略分解_分层稳健.csv")
    return result, diag
