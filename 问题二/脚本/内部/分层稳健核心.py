# -*- coding: utf-8 -*-
"""2023 C题问题二：分层稳健定价 + 报童补货。

正式职责拆分：
1. 基础需求：用全量有效净销量，仅根据星期、月份和可选趋势预测；
2. 价格响应：正常销售数据上比较三种函数形式：
   - 半对数售价：log(D)=...+bP；
   - 半对数加成：log(D)=...+bm，其中 m=P/C-1；
   - 对数-对数售价：log(D)=...+b log(P)，仅作滚动预测基准；
3. 两种可用于定价的半对数模型按滚动样本外 WAPE 竞争，且必须通过 HAC 显著性、
   跨折方向稳定和“相对全体最佳模型不过度劣化”三重门槛；
4. 对价格关系不可靠的品类，采用同星期条件中位加成；
5. 所有品类均用随机需求分布下的损耗修正报童分位数确定补货量。

加入“半对数加成”是因为题目明确给出商超采用成本加成定价。该模型使用零售商真正可控的
加成率解释需求变化，可减少批发成本波动直接推高售价对价格系数的污染；最终是否采用仍由
滚动回测决定，而不是为了得到内部最优点人工选择。
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
# 价格响应：半对数售价 / 半对数加成 / 对数-对数售价基准
# ---------------------------
PRICE_RESPONSE_MODELS = ("半对数售价", "半对数加成", "对数-对数售价")
PRICING_MODELS = ("半对数售价", "半对数加成")


def _response_design(
    frame: pd.DataFrame,
    include_trend: bool,
    response_model: str,
) -> tuple[np.ndarray, list[str]]:
    x = pd.DataFrame(index=frame.index)
    if response_model == "半对数售价":
        x["价格项"] = frame["日平均售价"].astype(float)
    elif response_model == "半对数加成":
        x["价格项"] = frame["加成率"].astype(float)
    elif response_model == "对数-对数售价":
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
    x, names = _response_design(frame, include_trend, response_model)
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
    x, names = _response_design(
        frame,
        bool(spec["含时间趋势"]),
        str(spec["价格响应模型"]),
    )
    if names != spec["设计列"]:
        raise ValueError("价格响应训练与预测矩阵不一致")
    eta = x @ np.asarray(spec["系数"], dtype=float)
    return np.exp(eta) * float(spec["平滑还原因子"])


def _reference_elasticity(spec: dict, frame: pd.DataFrame) -> float:
    model = str(spec["价格响应模型"])
    beta = float(spec["价格系数"])
    if model == "半对数售价":
        reference_price = float(frame["日平均售价"].median())
        return float(beta * reference_price)
    if model == "半对数加成":
        reference_markup = float(frame["加成率"].median())
        # m=P/C-1，成本固定时 dm/dlog(P)=P/C=1+m。
        return float(beta * (1.0 + reference_markup))
    if model == "对数-对数售价":
        return beta
    raise ValueError(f"未知价格响应模型: {model}")


def price_response_backtest(
    panel_normal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict]:
    """滚动比较价格响应函数，并只让可验证的半对数模型进入定价。

    每种函数先在“有/无时间趋势”中选择：趋势只有让滚动 WAPE 至少改善 5% 才保留。
    然后在两种半对数定价模型之间选择，候选必须同时满足：
    - 全样本价格系数 < 0；
    - HAC p < 0.05；
    - 至少 7/8 个滚动折价格系数为负；
    - 该模型 WAPE 不高于三种函数总体最佳 WAPE + max(0.01, 10%×best_WAPE)。

    若两个半对数候选都通过，选择 WAPE 更低者；都不通过则不做利润型价格搜索。
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

        chosen: dict[str, dict] = {}
        for response_model in PRICE_RESPONSE_MODELS:
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
            chosen[response_model] = {
                "含趋势": chosen_trend,
                "WAPE": chosen_wape,
                "回测价格系数": chosen_betas,
                "规格": full_spec,
            }
            summary_rows.append(
                {
                    "品类": cat,
                    "价格响应模型": response_model,
                    "最终含趋势": "是" if chosen_trend else "否",
                    "滚动WAPE": chosen_wape,
                    "全样本价格系数": float(full_spec["价格系数"]),
                    "参考价处价格弹性": _reference_elasticity(full_spec, frame),
                    "全样本稳健p值": float(full_spec["稳健概率值"]),
                    "负向回测折数": int(sum(beta < 0 for beta in chosen_betas)),
                    "回测折数": int(len(chosen_betas)),
                }
            )

        best_wape = min(float(info["WAPE"]) for info in chosen.values())
        tolerance = max(0.01, 0.10 * best_wape)
        admissible: list[tuple[str, float]] = []
        gate_info: dict[str, dict] = {}
        for response_model in PRICING_MODELS:
            info = chosen[response_model]
            spec = info["规格"]
            betas = info["回测价格系数"]
            fold_count = len(betas)
            negative_folds = int(sum(beta < 0 for beta in betas))
            statistical_ok = bool(
                float(spec["价格系数"]) < 0
                and float(spec["稳健概率值"]) < 0.05
                and negative_folds >= max(1, fold_count - 1)
            )
            prediction_ok = bool(float(info["WAPE"]) <= best_wape + tolerance + 1e-12)
            gate_info[response_model] = {
                "statistical_ok": statistical_ok,
                "prediction_ok": prediction_ok,
                "negative_folds": negative_folds,
                "fold_count": fold_count,
            }
            if statistical_ok and prediction_ok:
                admissible.append((response_model, float(info["WAPE"])))

        if admissible:
            selected_model = min(admissible, key=lambda item: (item[1], item[0]))[0]
            ok = True
        else:
            # 失败时仍保留预测较好的半对数规格供结果表解释，但不让其参与利润优化。
            selected_model = min(
                PRICING_MODELS,
                key=lambda model: (float(chosen[model]["WAPE"]), model),
            )
            ok = False

        selected_spec = dict(chosen[selected_model]["规格"])
        selected_elasticity = _reference_elasticity(selected_spec, frame)
        selected_spec["参考价处价格弹性"] = selected_elasticity
        specs[cat] = selected_spec
        reliable[cat] = ok

        selected_gate = gate_info[selected_model]
        price_wape = float(chosen["半对数售价"]["WAPE"])
        markup_wape = float(chosen["半对数加成"]["WAPE"])
        loglog_wape = float(chosen["对数-对数售价"]["WAPE"])
        reference_price = float(frame["日平均售价"].median())
        reference_markup = float(frame["加成率"].median())
        reliability_rows.append(
            {
                "品类": cat,
                "入选价格响应模型": selected_model,
                "价格响应系数": float(selected_spec["价格系数"]),
                "参考中位售价": reference_price,
                "参考中位加成率": reference_markup,
                "参考价处价格弹性": selected_elasticity,
                "稳健概率值": float(selected_spec["稳健概率值"]),
                "负向回测折数": int(selected_gate["negative_folds"]),
                "回测折数": int(selected_gate["fold_count"]),
                "半对数售价滚动WAPE": price_wape,
                "半对数加成滚动WAPE": markup_wape,
                "对数-对数基准WAPE": loglog_wape,
                "三模型最佳WAPE": best_wape,
                "入选模型预测不过度劣化": "是" if selected_gate["prediction_ok"] else "否",
                "入选模型统计稳定": "是" if selected_gate["statistical_ok"] else "否",
                "价格关系可靠": "是" if ok else "否",
                "处理": f"{selected_model}局部稳健利润优化" if ok else "同星期条件中位加成",
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
        "本版把‘预测这一天总共会卖多少’与‘零售商改变正常售价/加成后需求怎样变化’拆成两个模型。价格响应层同时比较半对数售价、半对数加成和对数-对数售价，其中只有两种半对数模型可以进入利润优化，并且必须通过滚动预测与稳健性门槛。",
        "",
        f"模型预计七天总利润：**{total_profit:.2f} 元**。",
        f"局部定价优化品类：{'、'.join(reliable_cats) if reliable_cats else '无'}。",
        f"保守中位定价品类：{'、'.join(conservative_cats) if conservative_cats else '无'}。",
        f"42 条策略中触及局部经营带上界：**{upper_hits} 条**。",
        "",
        "## 价格响应为何这样设计",
        "",
        "对数-对数常弹性模型在绝对弹性小于 1 时容易持续偏好更高价格，因此只作为样本外预测基准。半对数售价具有随价格上升而增强的点弹性；半对数加成则直接对应题目给出的成本加成定价机制，能把批发成本推动的被动涨价与零售商主动提高加成更好地区分。最终模型由滚动回测和显著稳定性共同决定。",
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
        "价格响应系数仍属于控制日历因素后的条件关联，不宣称严格因果；历史销量可能受缺货影响。题目未给出库存、预算、货架容量和包装约束，因此按品类独立决策。所谓‘最优’仅指历史常规经营区间和模型假设下的局部稳健方案。",
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
            ["品类", "入选价格响应模型", "价格关系可靠", "参考价处价格弹性", "稳健概率值"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
