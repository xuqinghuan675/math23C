# -*- coding: utf-8 -*-
"""2023 C题问题二：分层稳健定价 + 报童补货。

正式职责拆分：
1. 基础需求：用全量有效净销量，仅根据星期、月份和可选趋势预测；
2. 价格响应：正常销售数据上比较半对数与对数-对数模型，但定价主模型采用半对数；
3. 半对数模型只有在滚动回测不过度劣化、价格系数显著为负且方向稳定时，才进入利润型定价；
4. 对价格关系不可靠的品类，采用同星期条件中位加成；
5. 所有品类均用随机需求分布下的损耗修正报童分位数确定补货量。

选择半对数作为定价主响应不是为了制造内部最优点，而是因为
log(D)=a+bP 的价格弹性 bP 会随价格变化；当 b<0 时，价格升高会逐渐增强需求收缩，
避免常弹性模型在 |elasticity|<1 时产生无界提价倾向。对数-对数模型继续保留为
滚动预测基准，用于检查半对数形式是否明显损失样本外拟合能力。
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


# ---------------------------
# 价格响应模型：半对数主模型 + 对数-对数基准
# ---------------------------

def _price_design(
    frame: pd.DataFrame,
    include_trend: bool,
    response_model: str,
) -> tuple[np.ndarray, list[str]]:
    x = pd.DataFrame(index=frame.index)
    if response_model == "半对数":
        x["价格项"] = frame["日平均售价"].astype(float)
    elif response_model == "对数-对数":
        x["价格项"] = np.log(frame["日平均售价"].astype(float))
    else:
        raise ValueError(f"未知价格响应模型: {response_model}")
    names = ["价格项"]
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


def _fit_price_response(
    frame: pd.DataFrame,
    include_trend: bool,
    response_model: str,
) -> dict:
    """OLS 点估计 + 7 阶 Newey-West/HAC 协方差。"""
    x, names = _price_design(frame, include_trend, response_model)
    y = np.log(frame["日销售量"].astype(float).to_numpy())
    params = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ params

    xtx_inv = np.linalg.pinv(x.T @ x)
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    lag_count = min(7, max(0, len(frame) - 1))
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    covariance = xtx_inv @ meat @ xtx_inv
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    price_idx = names.index("价格项")
    beta = float(params[price_idx])
    se = float(standard_errors[price_idx])
    z = beta / se if se > 0 else 0.0
    p_value = float(math.erfc(abs(z) / math.sqrt(2.0)))

    return {
        "价格响应模型": response_model,
        "含时间趋势": bool(include_trend),
        "价格系数": beta,
        "稳健标准误": se,
        "稳健概率值": p_value,
        "稳健区间下限": beta - 1.96 * se,
        "稳健区间上限": beta + 1.96 * se,
        "系数": params,
        "设计列": names,
        "残差": residuals,
        "平滑还原因子": float(np.mean(np.exp(residuals))),
        "样本数": int(len(frame)),
    }


def _predict_price_response(spec: dict, frame: pd.DataFrame) -> np.ndarray:
    x, names = _price_design(
        frame,
        bool(spec["含时间趋势"]),
        str(spec["价格响应模型"]),
    )
    if names != spec["设计列"]:
        raise ValueError("价格响应训练与预测矩阵不一致")
    eta = x @ np.asarray(spec["系数"], dtype=float)
    return np.exp(eta) * float(spec["平滑还原因子"])


def price_response_backtest(
    panel_normal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict]:
    """比较半对数/对数-对数及是否带趋势，并决定定价响应是否可靠。

    规则：
    - 对数-对数是预测基准；半对数是唯一允许进入定价优化的函数形式；
    - 每种函数内部仍由滚动 WAPE 选择是否保留趋势（趋势至少改善 5% 才保留）；
    - 半对数最终规格必须 beta<0、HAC p<0.05，且至少 7/8 回测折价格系数为负；
    - 同时半对数滚动 WAPE 不得比对数-对数基准恶化超过 max(0.01, 10%)；
      若明显劣化，则不强行用它做定价，回退到条件中位加成。
    """
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    specs: dict[str, dict] = {}
    reliability_rows: list[dict] = []
    reliable: dict[str, bool] = {}

    for cat in CATEGORIES:
        frame = panel_normal[panel_normal["品类"] == cat].sort_values("销售日期").copy()
        cutoffs = base.validation_cutoffs(frame)
        if not cutoffs:
            raise ValueError(f"{cat} 没有足够样本进行价格响应滚动回测")

        chosen_by_form: dict[str, tuple[bool, float, list[float]]] = {}
        full_by_form: dict[str, dict] = {}

        for response_model in ("半对数", "对数-对数"):
            scores: dict[bool, float] = {}
            fold_betas: dict[bool, list[float]] = {}
            for include_trend in (False, True):
                total_abs = 0.0
                total_actual = 0.0
                betas: list[float] = []
                for cutoff in cutoffs:
                    train = frame[frame["销售日期"] <= cutoff]
                    test = frame[
                        (frame["销售日期"] > cutoff)
                        & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
                    ]
                    spec = _fit_price_response(train, include_trend, response_model)
                    pred = _predict_price_response(spec, test)
                    actual = test["日销售量"].to_numpy(float)
                    error = actual - pred
                    total_abs += float(np.abs(error).sum())
                    total_actual += float(actual.sum())
                    betas.append(float(spec["价格系数"]))
                    detail_rows.append(
                        {
                            "品类": cat,
                            "价格响应模型": response_model,
                            "含时间趋势": "是" if include_trend else "否",
                            "验证截止日": cutoff.date().isoformat(),
                            "WAPE": float(np.abs(error).sum() / actual.sum()),
                            "MAE": float(np.abs(error).mean()),
                            "RMSE": float(np.sqrt(np.mean(error**2))),
                            "价格系数": float(spec["价格系数"]),
                        }
                    )
                scores[include_trend] = total_abs / total_actual
                fold_betas[include_trend] = betas

            plain = float(scores[False])
            trend = float(scores[True])
            chosen_trend = bool(trend < 0.95 * plain)
            chosen_wape = float(scores[chosen_trend])
            chosen_betas = fold_betas[chosen_trend]
            full_spec = _fit_price_response(frame, chosen_trend, response_model)
            chosen_by_form[response_model] = (chosen_trend, chosen_wape, chosen_betas)
            full_by_form[response_model] = full_spec

            summary_rows.append(
                {
                    "品类": cat,
                    "价格响应模型": response_model,
                    "最终含趋势": "是" if chosen_trend else "否",
                    "滚动WAPE": chosen_wape,
                    "全样本价格系数": float(full_spec["价格系数"]),
                    "全样本稳健p值": float(full_spec["稳健概率值"]),
                    "负向回测折数": int(sum(beta < 0 for beta in chosen_betas)),
                    "回测折数": int(len(chosen_betas)),
                }
            )

        semi_trend, semi_wape, semi_betas = chosen_by_form["半对数"]
        loglog_trend, loglog_wape, _ = chosen_by_form["对数-对数"]
        semi_spec = full_by_form["半对数"]
        semi_negative = int(sum(beta < 0 for beta in semi_betas))
        fold_count = int(len(semi_betas))
        tolerance = max(0.01, 0.10 * loglog_wape)
        prediction_ok = bool(semi_wape <= loglog_wape + tolerance)
        statistical_ok = bool(
            float(semi_spec["价格系数"]) < 0
            and float(semi_spec["稳健概率值"]) < 0.05
            and semi_negative >= max(1, fold_count - 1)
        )
        ok = bool(prediction_ok and statistical_ok)
        reliable[cat] = ok
        specs[cat] = semi_spec

        # 为解释保留参考价处的点弹性；半对数弹性随价格变化，不再把 beta 本身称为弹性。
        reference_price = float(frame["日平均售价"].median())
        reference_elasticity = float(semi_spec["价格系数"] * reference_price)
        reliability_rows.append(
            {
                "品类": cat,
                "价格响应模型": "半对数",
                "半对数价格系数": float(semi_spec["价格系数"]),
                "参考中位售价": reference_price,
                "参考价处价格弹性": reference_elasticity,
                "稳健概率值": float(semi_spec["稳健概率值"]),
                "负向回测折数": semi_negative,
                "回测折数": fold_count,
                "半对数滚动WAPE": semi_wape,
                "对数-对数基准WAPE": loglog_wape,
                "相对基准WAPE差": semi_wape - loglog_wape,
                "预测不过度劣化": "是" if prediction_ok else "否",
                "价格关系可靠": "是" if ok else "否",
                "处理": "半对数局部利润优化" if ok else "同星期条件中位加成",
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    reliability_df = pd.DataFrame(reliability_rows)
    base.write_csv(detail, "价格响应模型回测明细_分层稳健.csv")
    base.write_csv(summary, "价格响应模型回测汇总_分层稳健.csv")
    base.write_csv(reliability_df, "价格关系可靠性_分层稳健.csv")
    return detail, summary, specs, reliability_df, reliable


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
    """占位实现；正式入口会由 `动态定价.py` 覆盖。"""
    raise RuntimeError("正式入口未绑定动态定价模块")


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
        "本版把‘预测这一天总共会卖多少’与‘改变正常售价后需求怎样变化’拆成两个模型：",
        "",
        "- 基础需求：使用全量有效净销量，只控制星期、月份和必要趋势；",
        "- 价格响应：正常销售数据上以对数-对数作为预测基准，半对数作为定价主函数；",
        "- 半对数只有在滚动预测不过度劣化、价格系数显著为负且跨折稳定时才进入利润优化；",
        "- 价格关系可靠的品类：在同星期中央经营带内做局部利润优化；",
        "- 价格关系不可靠的品类：采用同星期条件中位加成；",
        "- 所有品类：使用损耗修正后的报童分位数确定补货量。",
        "",
        f"模型预计七天总利润：**{total_profit:.2f} 元**。",
        f"局部定价优化品类：{'、'.join(reliable_cats) if reliable_cats else '无'}。",
        f"保守中位定价品类：{'、'.join(conservative_cats) if conservative_cats else '无'}。",
        f"42 条策略中触及局部经营带上界：**{upper_hits} 条**。",
        "",
        "## 为什么使用半对数定价响应",
        "",
        "对数-对数常弹性模型在绝对弹性小于 1 时，会让利润函数在较宽价格区间持续偏向高价，容易把最优解机械推到人为边界。半对数模型 log(D)=a+bP+controls 在 b<0 时的点弹性为 bP，会随价格上升而增强需求收缩，因此更适合有限价格区间内的经营定价。为避免为了得到内部峰值而牺牲拟合，本版仍把对数-对数作为滚动预测基准；半对数若样本外误差明显更差，则直接判定该品类不适合做利润型价格优化。",
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
        "价格系数是控制星期、月份后的条件关联，不是严格因果；半对数相对于对数-对数的作用首先是改善定价函数的结构合理性，并不把观察性回归包装成因果效应。历史销量仍可能受缺货影响；题目没有给出库存、预算、货架容量和包装约束，因此按品类独立决策。所谓‘最优’均指在给定历史常规经营区间和模型假设下的局部稳健方案。",
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

    _price_detail, _price_summary, normal_specs, reliability, reliable = price_response_backtest(
        panel_normal
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
    print(
        reliability[
            ["品类", "价格关系可靠", "半对数价格系数", "参考价处价格弹性", "稳健概率值"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
