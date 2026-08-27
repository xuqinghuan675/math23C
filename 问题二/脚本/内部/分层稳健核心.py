# -*- coding: utf-8 -*-
"""2023 C题问题二：分层稳健定价 + 报童补货。

本脚本复用 `求解问题二.py` 的数据读取、审计、成本预测和正常销售价格回归，
但把两个不同任务明确拆开：

1. 基础需求：用全量有效净销量，仅根据星期、月份和可选趋势预测；
2. 价格响应：只用正常销售数据估计价格弹性；
3. 对价格关系可靠的品类，在正常销售历史 IQR 内做局部利润优化；
4. 对价格关系不可靠的品类，采用正常销售历史中位加成；
5. 所有品类均用同一随机需求分布下的报童分位数确定补货量。

这样避免“折扣价污染正常价格关系”，也避免把所有品类都退化为固定中位定价。
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "求解问题二.py"
_spec = importlib.util.spec_from_file_location("q2_base", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("无法载入求解问题二.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

OUT = base.OUT
FUTURE_DATES = base.FUTURE_DATES
CATEGORIES = base.CATEGORIES
RANDOM_SEED = base.RANDOM_SEED + 202
SAMPLE_COUNT = 8000


def _time_design(frame: pd.DataFrame, include_trend: bool) -> tuple[np.ndarray, list[str]]:
    """基础需求设计矩阵：故意不放价格，只刻画时间基准需求。"""
    x = pd.DataFrame(index=frame.index)
    names: list[str] = []
    for weekday in range(2, 8):
        name = f"星期{weekday}"
        x[name] = (frame["星期"] == weekday).astype(float)
        names.append(name)
    for month in range(2, 13):
        name = f"月份{month}"
        x[name] = (frame["月份"] == month).astype(float)
        names.append(name)
    if include_trend:
        x["时间趋势"] = frame["时间趋势"].astype(float)
        names.append("时间趋势")
    x.insert(0, "常数项", 1.0)
    names.insert(0, "常数项")
    return x[names].to_numpy(float), names


def fit_base_demand(frame: pd.DataFrame, include_trend: bool) -> dict:
    x, names = _time_design(frame, include_trend)
    y = np.log(frame["日销售量"].astype(float).to_numpy())
    params = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ params
    return {
        "含时间趋势": bool(include_trend),
        "系数": params,
        "设计列": names,
        "残差": residuals,
        "平滑还原因子": float(np.mean(np.exp(residuals))),
        "样本数": int(len(frame)),
    }


def predict_base_mean(spec: dict, frame: pd.DataFrame) -> np.ndarray:
    x, names = _time_design(frame, bool(spec["含时间趋势"]))
    if names != spec["设计列"]:
        raise ValueError("基础需求训练与预测矩阵不一致")
    eta = x @ np.asarray(spec["系数"], dtype=float)
    return np.exp(eta) * float(spec["平滑还原因子"])


def base_demand_backtest(panel_all: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """滚动七日回测选择是否保留趋势；评价对象是全量有效净销量。"""
    rows = []
    selected: dict[str, dict] = {}
    for cat in CATEGORIES:
        frame = panel_all[panel_all["品类"] == cat].sort_values("销售日期").copy()
        model_scores = {}
        for include_trend in (False, True):
            model_name = "时间基准需求（星期和月份）" if not include_trend else "时间基准需求（星期、月份和趋势）"
            total_abs = 0.0
            total_actual = 0.0
            fold_rows = []
            for cutoff in base.validation_cutoffs(frame):
                train = frame[frame["销售日期"] <= cutoff]
                test = frame[
                    (frame["销售日期"] > cutoff)
                    & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
                ]
                spec = fit_base_demand(train, include_trend)
                pred = predict_base_mean(spec, test)
                actual = test["日销售量"].to_numpy(float)
                err = actual - pred
                total_abs += float(np.abs(err).sum())
                total_actual += float(actual.sum())
                fold_rows.append(
                    {
                        "品类": cat,
                        "模型": model_name,
                        "验证截止日": cutoff.date().isoformat(),
                        "WAPE": float(np.abs(err).sum() / actual.sum()),
                        "MAE": float(np.abs(err).mean()),
                        "RMSE": float(np.sqrt(np.mean(err**2))),
                    }
                )
            rows.extend(fold_rows)
            model_scores[include_trend] = total_abs / total_actual

        plain = float(model_scores[False])
        trend = float(model_scores[True])
        chosen_trend = bool(trend < 0.95 * plain)
        selected[cat] = fit_base_demand(frame, chosen_trend)

    detail = pd.DataFrame(rows)
    base.write_csv(detail, "基础需求模型回测_分层稳健.csv")
    summary = (
        detail.groupby(["品类", "模型"], as_index=False)
        .agg(WAPE=("WAPE", "mean"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"))
    )
    base.write_csv(summary, "基础需求模型回测汇总_分层稳健.csv")
    return summary, selected


def future_time_frame(date: pd.Timestamp, first_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "销售日期": [pd.Timestamp(date)],
            "日销售量": [1.0],
            "星期": [pd.Timestamp(date).weekday() + 1],
            "月份": [pd.Timestamp(date).month],
            "时间趋势": [(pd.Timestamp(date) - pd.Timestamp(first_date)).days / 365.25],
        }
    )


def base_demand_samples(
    spec: dict,
    date: pd.Timestamp,
    residual_draws: np.ndarray,
    first_date: pd.Timestamp,
) -> np.ndarray:
    frame = future_time_frame(date, first_date)
    x, names = _time_design(frame, bool(spec["含时间趋势"]))
    if names != spec["设计列"]:
        raise ValueError("未来基础需求矩阵不一致")
    eta = float((x @ np.asarray(spec["系数"], dtype=float)).ravel()[0])
    return np.exp(eta + residual_draws)


def evaluate_candidate(
    base_samples: np.ndarray,
    price: float,
    reference_price: float,
    beta: float,
    future_cost: float,
    loss_rate: float,
) -> dict | None:
    effective_cost = future_cost / (1.0 - loss_rate)
    if price <= effective_cost or price <= 0 or reference_price <= 0:
        return None

    ratio = max(price / reference_price, 1e-8)
    demand = base_samples * (ratio ** beta)
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


def reliability_table(
    panel_normal: pd.DataFrame,
    normal_detail: pd.DataFrame,
    normal_selected: dict,
    normal_specs: dict,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    reliable: dict[str, bool] = {}
    for cat in CATEGORIES:
        selected_name = base.demand_model_name(bool(normal_selected[cat]))
        folds = normal_detail[
            (normal_detail["品类"] == cat) & (normal_detail["模型"] == selected_name)
        ]
        spec = normal_specs[cat]
        beta = float(spec["价格系数"])
        p = float(spec["稳健概率值"])
        negative_folds = int((folds["价格系数"] < 0).sum())
        fold_count = int(len(folds))
        ok = bool(beta < 0 and p < 0.05 and negative_folds >= max(1, fold_count - 1))
        reliable[cat] = ok
        rows.append(
            {
                "品类": cat,
                "正常销售价格弹性": beta,
                "稳健概率值": p,
                "负向回测折数": negative_folds,
                "回测折数": fold_count,
                "价格关系可靠": "是" if ok else "否",
                "处理": "正常销售IQR内局部利润优化" if ok else "正常销售历史中位加成",
            }
        )
    df = pd.DataFrame(rows)
    base.write_csv(df, "价格关系可靠性_分层稳健.csv")
    return df, reliable


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
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    diagnostics = []

    for cat in CATEGORIES:
        cost_frame = panel_all[panel_all["品类"] == cat].sort_values("销售日期")
        costs = base.cost_forecast(
            cost_frame, FUTURE_DATES, selected_cost_methods[cat]
        )
        price_spec = normal_specs[cat]
        beta_point = float(price_spec["价格系数"])
        beta_low = float(price_spec["稳健区间下限"])
        info = markup_info[cat]["正常销售"]
        q25 = float(info["百分之二十五分位"])
        median = float(info["中位数"])
        q75 = float(info["百分之七十五分位"])

        for date in FUTURE_DATES:
            future_cost = float(costs.loc[date])
            reference_price = float(np.round(future_cost * (1.0 + median), 2))
            residual_draws = rng.choice(
                np.asarray(base_specs[cat]["残差"], dtype=float),
                size=SAMPLE_COUNT,
                replace=True,
            )
            base_samples = base_demand_samples(
                base_specs[cat], date, residual_draws, first_date
            )

            if reliable[cat]:
                low = math.floor(q25 * 100.0) / 100.0
                high = math.ceil(q75 * 100.0) / 100.0
                markups = np.arange(low, high + 0.0001, 0.01)
                candidates = []
                for markup in markups:
                    price = float(np.round(future_cost * (1.0 + markup), 2))
                    beta_used = beta_low if price >= reference_price else beta_point
                    item = evaluate_candidate(
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
                    raise ValueError(f"{cat} {date.date()} 没有有效价格候选")
                best = max(candidates, key=lambda x: x["预计利润"])
                pricing_basis = "价格关系稳定：正常销售IQR内局部稳健利润优化"
                local_upper = float(np.round(future_cost * (1.0 + high), 2))
                boundary = abs(best["售价"] - local_upper) <= 0.011
            else:
                item = evaluate_candidate(
                    base_samples,
                    reference_price,
                    reference_price,
                    0.0,
                    future_cost,
                    category_loss[cat],
                )
                if item is None:
                    raise ValueError(f"{cat} {date.date()} 历史中位价低于有效成本")
                best = item
                pricing_basis = "价格关系证据不足：采用正常销售历史中位加成"
                boundary = False

            base_mean = float(base_samples.mean())
            mean_order = max(
                0.1, round(base_mean / (1.0 - category_loss[cat]), 1)
            )
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

    round2 = [
        "预测批发价",
        "损耗率（百分数）",
        "参考中位售价",
        "建议成本加成率（百分数）",
        "建议售价",
        "预测需求量",
        "建议补货量",
        "预计满足量",
        "预计利润",
    ]
    export = result.copy()
    for col in round2:
        export[col] = export[col].round(2)
    export["正常销售价格弹性"] = export["正常销售价格弹性"].round(4)
    export["临界分位数"] = export["临界分位数"].round(4)
    base.write_csv(export, "七天六品类最终策略_分层稳健.csv")

    diag_export = diag.copy()
    for col in ["参考中位售价", "参考平均需求补货利润", "最终建议利润", "相对基准改善", "基础需求均值"]:
        diag_export[col] = diag_export[col].round(2)
    base.write_csv(diag_export, "策略分解_分层稳健.csv")
    return result, diag


def write_summary(
    result: pd.DataFrame,
    reliability: pd.DataFrame,
    base_summary: pd.DataFrame,
    selected_cost_methods: dict,
) -> None:
    total_profit = float(result["预计利润"].sum())
    reliable_cats = reliability.loc[
        reliability["价格关系可靠"] == "是", "品类"
    ].tolist()
    conservative_cats = reliability.loc[
        reliability["价格关系可靠"] == "否", "品类"
    ].tolist()
    upper_hits = int((result["是否触及局部上界"] == "是").sum())
    lines = [
        "# 问题二分层稳健方案说明",
        "",
        "## 主结论",
        "",
        "本版把“预测这一天总共会卖多少”和“正常售价变化会怎样影响销量”拆成两个模型：",
        "",
        "- 基础需求：使用全量有效净销量，只控制星期、月份和必要趋势；",
        "- 价格响应：只使用正常销售记录估计价格弹性；",
        "- 价格关系可靠的品类：仅在正常销售历史 IQR（25%~75%）内做局部稳健利润优化；",
        "- 价格关系不可靠的品类：采用正常销售历史中位加成；",
        "- 所有品类：使用损耗修正后的报童分位数确定补货量。",
        "",
        f"模型预计七天总利润：**{total_profit:.2f} 元**。",
        f"局部定价优化品类：{'、'.join(reliable_cats) if reliable_cats else '无'}。",
        f"保守中位定价品类：{'、'.join(conservative_cats) if conservative_cats else '无'}。",
        f"42 条策略中触及局部 IQR 上界：**{upper_hits} 条**；该标记表示约束内最优，不宣称无约束全局最优。",
        "",
        "## 为什么比上一版更适合论文",
        "",
        "上一版用正常销售数据同时承担需求预测和价格响应，剔除折扣后部分品类预测误差反而上升；本版用全量真实净销量预测基础需求，仅让正常销售数据识别价格响应，从而同时保留真实需求规模和干净的正常价格关系。",
        "",
        "对于价格关系可靠的品类，不再使用 5%~95% 甚至 1%~99% 的宽边界追逐高价，而限制在历史最常见的中间 50% 经营区间，并对提价使用更保守的价格弹性下界。对于关系不可靠的品类，不伪造精确最优价。",
        "",
        "## 进价模型",
        "",
    ]
    for cat in CATEGORIES:
        lines.append(f"- {cat}：{selected_cost_methods[cat]}")
    lines += [
        "",
        "## 论文表述边界",
        "",
        "价格系数是控制星期、月份后的条件关联，不是严格因果。历史销量仍可能受缺货影响；题目没有给出库存、预算、货架容量和包装约束，因此按品类独立决策。所谓“最优”均指在给定历史常规经营区间和模型假设下的局部稳健方案。",
        "",
        "完整 42 条结果见 `结果/七天六品类最终策略_分层稳健.csv`。",
    ]
    (base.ROOT / "问题二" / "最终建模说明_分层稳健.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    (
        merged,
        panel_all,
        panel_normal,
        _audit,
        category_loss,
        _item_loss,
        first_date,
    ) = base.read_source_data()

    base.discount_audit(merged)
    markups = base.markup_summary(panel_all, panel_normal)

    normal_detail, _normal_summary, normal_selected, normal_specs = base.demand_backtest(
        panel_normal, "正常销售"
    )
    reliability, reliable = reliability_table(
        panel_normal, normal_detail, normal_selected, normal_specs
    )

    base_summary, base_specs = base_demand_backtest(panel_all)
    _cost_detail, _cost_summary, selected_cost_methods = base.cost_backtest(panel_all)

    result, _diag = optimize_hybrid(
        panel_all,
        panel_normal,
        category_loss,
        markups,
        selected_cost_methods,
        base_specs,
        normal_specs,
        reliable,
        first_date,
    )
    write_summary(result, reliability, base_summary, selected_cost_methods)

    print("分层稳健问题二已完成")
    print(f"7天42条预计利润合计: {result['预计利润'].sum():.2f} 元")
    print("价格可靠性:")
    print(reliability[["品类", "价格关系可靠", "正常销售价格弹性", "稳健概率值"]].to_string(index=False))


if __name__ == "__main__":
    main()
