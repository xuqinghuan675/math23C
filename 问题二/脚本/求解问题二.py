# -*- coding: utf-8 -*-
"""2023 年高教社杯全国大学生数学建模竞赛 C 题问题二。

本脚本只处理按品类的第二题，使用仓库内四个原始附件重新计算：
1. 保留合法销售、折扣和退货记录，退货按负销量净额处理；
2. 用销售量加权计算品类日售价和日进价；
3. 只比较近七日均值、指数加权移动平均、同星期历史中位数三种成本预测；
4. 用不含未来未知折扣输入的对数线性需求模型；
5. 在历史加成率区间内用网格搜索和易腐品报童模型联立定价、补货。

运行方式：在仓库根目录执行
    python 问题二/脚本/求解问题二.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "2023年C题"
OUT = ROOT / "问题二" / "结果"
FIG = ROOT / "问题二" / "图表"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
FUTURE_DATES = pd.date_range("2023-07-01", "2023-07-07", freq="D")
COST_METHODS = ["近七日均值", "指数加权移动平均", "同星期近八次中位数"]
MODEL_SPECS = {"不含趋势": False, "含时间趋势": True}
RANDOM_SEED = 20230826
SAMPLE_COUNT = 4000


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """读取并统一四个附件，返回交易明细、日面板、审计信息、损耗率和原始加成率。"""
    raw1 = pd.read_excel(DATA / "附件1.xlsx", dtype=str)
    raw2 = pd.read_excel(DATA / "附件2.xlsx")
    raw3 = pd.read_excel(DATA / "附件3.xlsx", dtype=str)
    raw4_category = pd.read_excel(DATA / "附件4.xlsx", sheet_name=0)
    raw4_item = pd.read_excel(DATA / "附件4.xlsx", sheet_name=1)

    item = raw1.iloc[:, [0, 3]].copy()
    item.columns = ["单品编码", "品类"]
    item["单品编码"] = item["单品编码"].astype(str).str.strip()
    item["品类"] = item["品类"].astype(str).str.strip()

    sales = raw2.iloc[:, :7].copy()
    sales.columns = ["销售日期", "扫码时间", "单品编码", "销量", "售价", "销售类型", "是否打折"]
    sales["单品编码"] = sales["单品编码"].astype(str).str.strip()
    sales["销售日期"] = pd.to_datetime(sales["销售日期"], errors="coerce")
    sales["销量"] = pd.to_numeric(sales["销量"], errors="coerce")
    sales["售价"] = pd.to_numeric(sales["售价"], errors="coerce")

    wholesale = raw3.iloc[:, [0, 1, 2]].copy()
    wholesale.columns = ["销售日期", "单品编码", "进价"]
    wholesale["单品编码"] = wholesale["单品编码"].astype(str).str.strip()
    wholesale["销售日期"] = pd.to_datetime(wholesale["销售日期"], errors="coerce")
    wholesale["进价"] = pd.to_numeric(wholesale["进价"], errors="coerce")

    loss_category = raw4_category.iloc[:, [1, 2]].copy()
    loss_category.columns = ["品类", "损耗率百分数"]
    loss_category["品类"] = loss_category["品类"].astype(str).str.strip()
    loss_category["损耗率百分数"] = pd.to_numeric(loss_category["损耗率百分数"], errors="coerce")
    category_loss = dict(
        zip(loss_category["品类"], loss_category["损耗率百分数"] / 100.0)
    )

    loss_item = raw4_item.iloc[:, [0, 2]].copy()
    loss_item.columns = ["单品编码", "损耗率百分数"]
    loss_item["单品编码"] = loss_item["单品编码"].astype(str).str.strip()
    loss_item["损耗率百分数"] = pd.to_numeric(loss_item["损耗率百分数"], errors="coerce")
    item_loss = dict(zip(loss_item["单品编码"], loss_item["损耗率百分数"] / 100.0))

    audit = {
        "附件二原始流水数": int(len(sales)),
        "退货记录数": int((sales["销售类型"] == "退货").sum()),
        "折扣记录数": int((sales["是否打折"] == "是").sum()),
        "完整重复行数": int(sales.duplicated().sum()),
        "附件一单品数": int(item["单品编码"].nunique()),
        "附件三日期单品组合数": int(wholesale[["销售日期", "单品编码"]].drop_duplicates().shape[0]),
        "销售日期起点": str(sales["销售日期"].min().date()),
        "销售日期终点": str(sales["销售日期"].max().date()),
        "品类数": int(item["品类"].nunique()),
    }

    merged = sales.merge(item, on="单品编码", how="left", validate="many_to_one")
    merged = merged.merge(
        wholesale, on=["销售日期", "单品编码"], how="left", validate="many_to_one"
    )
    bad = (
        merged["销售日期"].isna()
        | merged["单品编码"].isna()
        | merged["品类"].isna()
        | merged["销量"].isna()
        | merged["售价"].isna()
        | merged["进价"].isna()
        | (merged["售价"] <= 0)
        | (merged["进价"] <= 0)
        | (merged["销量"] == 0)
    )
    audit["无法进入分析的记录数"] = int(bad.sum())
    audit["批发价未匹配数"] = int(merged["进价"].isna().sum())
    audit["分类未匹配数"] = int(merged["品类"].isna().sum())
    if bad.any():
        raise ValueError("附件存在无法进入分析的记录，请先检查数据审计结果")

    wrong_return_sign = ((merged["销售类型"] == "退货") & (merged["销量"] >= 0)) | (
        (merged["销售类型"] == "销售") & (merged["销量"] < 0)
    )
    audit["业务类型与销量符号不一致数"] = int(wrong_return_sign.sum())
    if wrong_return_sign.any():
        raise ValueError("退货标记与销量符号不一致")

    merged["销售额"] = merged["销量"] * merged["售价"]
    merged["成本额"] = merged["销量"] * merged["进价"]
    merged["正销量"] = merged["销量"].clip(lower=0)
    merged["折扣正销量"] = np.where(merged["是否打折"] == "是", merged["正销量"], 0.0)

    panel = (
        merged.groupby(["销售日期", "品类"], as_index=False)
        .agg(
            日销售量=("销量", "sum"),
            日销售额=("销售额", "sum"),
            日成本额=("成本额", "sum"),
            正销售量=("正销量", "sum"),
            折扣销售量=("折扣正销量", "sum"),
            交易条数=("单品编码", "size"),
            单品数=("单品编码", "nunique"),
        )
    )
    panel = panel[panel["日销售量"] > 0].copy()
    panel["日平均售价"] = panel["日销售额"] / panel["日销售量"]
    panel["日平均进价"] = panel["日成本额"] / panel["日销售量"]
    panel["加成率"] = panel["日平均售价"] / panel["日平均进价"] - 1.0
    panel["折扣销量占比"] = panel["折扣销售量"] / panel["正销售量"].replace(0, np.nan)
    panel["星期"] = panel["销售日期"].dt.weekday + 1
    panel["月份"] = panel["销售日期"].dt.month
    first_date = panel["销售日期"].min()
    panel["时间趋势"] = (panel["销售日期"] - first_date).dt.days / 365.25
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["日平均售价", "日平均进价", "加成率"]
    )
    audit["面板行数"] = int(len(panel))
    audit["净销售量千克"] = float(panel["日销售量"].sum())
    audit["面板品类数"] = int(panel["品类"].nunique())
    audit["面板日期数"] = int(panel["销售日期"].nunique())
    audit["各品类面板行数"] = {
        str(k): int(v) for k, v in panel["品类"].value_counts().to_dict().items()
    }
    if set(CATEGORIES) != set(category_loss):
        raise ValueError("附件四的品类损耗率未覆盖六个题目品类")
    if set(CATEGORIES) != set(panel["品类"].unique()):
        raise ValueError("销售面板未覆盖六个题目品类")

    markup_summary = {}
    for cat in CATEGORIES:
        values = panel.loc[panel["品类"] == cat, "加成率"]
        markup_summary[cat] = {
            "百分之五分位": float(values.quantile(0.05)),
            "中位数": float(values.quantile(0.50)),
            "百分之九十五分位": float(values.quantile(0.95)),
            "百分之一分位": float(values.quantile(0.01)),
            "百分之九十九分位": float(values.quantile(0.99)),
        }

    # 用近期正销售量对单品损耗率做核对，但正式按品类直接使用附件四第一张表。
    recent_start = panel["销售日期"].max() - pd.Timedelta(days=29)
    recent = merged[(merged["销售日期"] >= recent_start) & (merged["销量"] > 0)].copy()
    recent["单品损耗率"] = recent["单品编码"].map(item_loss)
    recent = recent.dropna(subset=["单品损耗率"])
    recent_loss = {}
    for cat, sub in recent.groupby("品类"):
        recent_loss[cat] = float(np.average(sub["单品损耗率"], weights=sub["销量"]))

    loss_check = []
    for cat in CATEGORIES:
        loss_check.append(
            {
                "品类": cat,
                "附件四品类损耗率": category_loss[cat],
                "近期单品销售量加权损耗率": recent_loss.get(cat, np.nan),
                "正式采用损耗率": category_loss[cat],
                "采用理由": "题目按品类补货，直接采用附件四品类汇总值",
            }
        )
    pd.DataFrame(loss_check).to_csv(OUT / "损耗率核对.csv", index=False, encoding="utf-8-sig")
    markup_rows = []
    for cat in CATEGORIES:
        markup_rows.append({"品类": cat, **markup_summary[cat]})
    pd.DataFrame(markup_rows).to_csv(OUT / "历史加成率分布.csv", index=False, encoding="utf-8-sig")
    discount_summary = (
        panel.groupby("品类", as_index=False)
        .agg(
            平均折扣销量占比=("折扣销量占比", "mean"),
            中位数折扣销量占比=("折扣销量占比", "median"),
            有折扣记录的天数=("折扣销量占比", lambda x: int((x > 0).sum())),
            样本天数=("折扣销量占比", "size"),
        )
    )
    discount_summary.to_csv(OUT / "折扣辅助分析.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(OUT / "品类日面板.csv", index=False, encoding="utf-8-sig")
    (OUT / "数据审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return merged, panel, audit, category_loss, markup_summary


def design_matrix(frame: pd.DataFrame, include_trend: bool) -> pd.DataFrame:
    """构造需求模型矩阵，星期一和一月作为基准。"""
    x = pd.DataFrame(index=frame.index)
    x["对数售价"] = np.log(frame["日平均售价"].astype(float))
    for weekday in range(2, 8):
        x[f"星期{weekday}"] = (frame["星期"] == weekday).astype(float)
    for month in range(2, 13):
        x[f"月份{month}"] = (frame["月份"] == month).astype(float)
    if include_trend:
        x["时间趋势"] = frame["时间趋势"].astype(float)
    return sm.add_constant(x, has_constant="add")


def fit_demand(frame: pd.DataFrame, include_trend: bool) -> dict:
    y = np.log(frame["日销售量"].astype(float))
    x = design_matrix(frame, include_trend)
    model = sm.OLS(y, x).fit()
    try:
        robust = model.get_robustcov_results(cov_type="HAC", maxlags=7, use_correction=True)
    except TypeError:
        robust = model.get_robustcov_results(cov_type="HAC", maxlags=7)
    names = list(x.columns)
    price_idx = names.index("对数售价")
    params = np.asarray(model.params, dtype=float)
    robust_se = np.asarray(robust.bse, dtype=float)
    robust_p = np.asarray(robust.pvalues, dtype=float)
    ci = np.asarray(robust.conf_int(), dtype=float)
    residuals = np.asarray(model.resid, dtype=float)
    smearing = float(np.mean(np.exp(residuals)))
    return {
        "模型": "含时间趋势" if include_trend else "不含趋势",
        "含时间趋势": include_trend,
        "模型对象": model,
        "系数": params,
        "设计列": names,
        "价格弹性": float(params[price_idx]),
        "稳健标准误": float(robust_se[price_idx]),
        "稳健概率值": float(robust_p[price_idx]),
        "稳健区间下限": float(ci[price_idx, 0]),
        "稳健区间上限": float(ci[price_idx, 1]),
        "模型决定系数": float(model.rsquared),
        "调整决定系数": float(model.rsquared_adj),
        "残差": residuals,
        "平滑还原因子": smearing,
    }


def prediction_from_model(spec: dict, frame: pd.DataFrame) -> np.ndarray:
    x = design_matrix(frame, spec["含时间趋势"])
    eta = np.asarray(x, dtype=float) @ spec["系数"]
    return np.exp(eta) * spec["平滑还原因子"]


def validation_cutoffs(frame: pd.DataFrame) -> list[pd.Timestamp]:
    end = frame["销售日期"].max()
    offsets = [14 + 35 * i for i in range(8)]
    result = []
    for days in offsets:
        cutoff = end - pd.Timedelta(days=days)
        train_n = int((frame["销售日期"] <= cutoff).sum())
        test_n = int(
            (
                (frame["销售日期"] > cutoff)
                & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
            ).sum()
        )
        if train_n >= 300 and test_n >= 4:
            result.append(cutoff)
    return sorted(result)


def demand_backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    selected = {}
    relation_rows = []
    for cat in CATEGORIES:
        frame = panel[panel["品类"] == cat].sort_values("销售日期").copy()
        for model_name, include_trend in MODEL_SPECS.items():
            metrics = []
            for cutoff in validation_cutoffs(frame):
                train = frame[frame["销售日期"] <= cutoff]
                test = frame[
                    (frame["销售日期"] > cutoff)
                    & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
                ]
                spec = fit_demand(train, include_trend)
                pred = prediction_from_model(spec, test)
                actual = test["日销售量"].to_numpy(float)
                metrics.append(
                    {
                        "品类": cat,
                        "模型": model_name,
                        "验证截止日": cutoff.date().isoformat(),
                        "验证天数": int(len(test)),
                        "绝对百分比误差加权": float(
                            np.sum(np.abs(actual - pred)) / np.sum(actual)
                        ),
                        "平均绝对误差": float(np.mean(np.abs(actual - pred))),
                        "均方根误差": float(np.sqrt(np.mean((actual - pred) ** 2))),
                    }
                )
            rows.extend(metrics)
        sub = pd.DataFrame([r for r in rows if r["品类"] == cat])
        summary = sub.groupby("模型", as_index=False).agg(
            回测加权误差=("绝对百分比误差加权", "mean"),
            回测平均绝对误差=("平均绝对误差", "mean"),
            回测均方根误差=("均方根误差", "mean"),
            回测次数=("验证截止日", "count"),
        )
        summary = summary.sort_values(["回测加权误差", "回测平均绝对误差"])
        selected_name = str(summary.iloc[0]["模型"])
        selected[cat] = MODEL_SPECS[selected_name]
        full_spec = fit_demand(frame, selected[cat])
        log_quantity = np.log(frame["日销售量"].astype(float))
        markup = frame["加成率"].astype(float)
        relation_rows.append(
            {
                "品类": cat,
                "最终模型": selected_name,
                "价格弹性": full_spec["价格弹性"],
                "稳健标准误": full_spec["稳健标准误"],
                "稳健概率值": full_spec["稳健概率值"],
                "稳健区间下限": full_spec["稳健区间下限"],
                "稳健区间上限": full_spec["稳健区间上限"],
                "模型决定系数": full_spec["模型决定系数"],
                "调整决定系数": full_spec["调整决定系数"],
                "样本数": int(len(frame)),
                "平滑还原因子": full_spec["平滑还原因子"],
                "回测加权误差": float(summary.iloc[0]["回测加权误差"]),
                "加成率百分之五分位": float(markup.quantile(0.05)),
                "加成率中位数": float(markup.quantile(0.50)),
                "加成率百分之九十五分位": float(markup.quantile(0.95)),
                "加成率与销量相关系数": float(markup.corr(frame["日销售量"])),
                "加成率与对数销量相关系数": float(markup.corr(log_quantity)),
                "解释边界": "价格与销量的条件关联，不单独作因果证明",
            }
        )

    backtest_df = pd.DataFrame(rows)
    relation_df = pd.DataFrame(relation_rows)
    backtest_df.to_csv(OUT / "需求模型回测.csv", index=False, encoding="utf-8-sig")
    relation_df.to_csv(OUT / "销售量与成本加成关系.csv", index=False, encoding="utf-8-sig")
    return backtest_df, relation_df, selected


def cost_forecast(train: pd.DataFrame, future_dates: pd.DatetimeIndex, method: str) -> pd.Series:
    series = train.set_index("销售日期")["日平均进价"].sort_index().dropna()
    if len(series) == 0:
        raise ValueError("成本序列为空")
    if method == "近七日均值":
        value = float(series.tail(7).mean())
        return pd.Series(value, index=future_dates, dtype=float)
    if method == "指数加权移动平均":
        value = float(series.ewm(alpha=0.4, adjust=False).mean().iloc[-1])
        return pd.Series(value, index=future_dates, dtype=float)
    if method == "同星期近八次中位数":
        values = []
        for date in future_dates:
            same_weekday = series[series.index.weekday == date.weekday()].tail(8)
            if len(same_weekday) == 0:
                value = float(series.tail(7).mean())
            else:
                value = float(same_weekday.median())
            values.append(value)
        return pd.Series(values, index=future_dates, dtype=float)
    raise ValueError(f"未知成本预测方法: {method}")


def cost_backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    summary_rows = []
    selected = {}
    for cat in CATEGORIES:
        frame = panel[panel["品类"] == cat].sort_values("销售日期").copy()
        for cutoff in validation_cutoffs(frame):
            train = frame[frame["销售日期"] <= cutoff]
            test = frame[
                (frame["销售日期"] > cutoff)
                & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
            ]
            for method in COST_METHODS:
                pred = cost_forecast(train, pd.DatetimeIndex(test["销售日期"]), method)
                actual = test.set_index("销售日期")["日平均进价"]
                joined = pd.concat([actual.rename("实际"), pred.rename("预测")], axis=1).dropna()
                error = joined["实际"] - joined["预测"]
                actual_values = joined["实际"].to_numpy(float)
                rows.append(
                    {
                        "品类": cat,
                        "方法": method,
                        "验证截止日": cutoff.date().isoformat(),
                        "验证天数": int(len(joined)),
                        "绝对百分比误差加权": float(
                            np.sum(np.abs(error)) / np.sum(np.abs(actual_values))
                        ),
                        "平均绝对误差": float(np.mean(np.abs(error))),
                    }
                )
        sub = pd.DataFrame([r for r in rows if r["品类"] == cat])
        summary = sub.groupby("方法", as_index=False).agg(
            回测加权误差=("绝对百分比误差加权", "mean"),
            回测平均绝对误差=("平均绝对误差", "mean"),
            回测次数=("验证截止日", "count"),
        )
        summary = summary.sort_values(["回测加权误差", "回测平均绝对误差"])
        selected[cat] = str(summary.iloc[0]["方法"])
        for _, row in summary.iterrows():
            summary_rows.append(
                {
                    "品类": cat,
                    "方法": row["方法"],
                    "回测加权误差": row["回测加权误差"],
                    "回测平均绝对误差": row["回测平均绝对误差"],
                    "回测次数": int(row["回测次数"]),
                    "是否入选": "是" if row["方法"] == selected[cat] else "否",
                }
            )
    detail = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    detail.to_csv(OUT / "成本预测回测明细.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "成本预测回测.csv", index=False, encoding="utf-8-sig")
    return detail, selected, summary_df


def future_design(date: pd.Timestamp, price: float, first_date: pd.Timestamp) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "销售日期": [date],
            "日平均售价": [price],
            "星期": [date.weekday() + 1],
            "月份": [date.month],
            "时间趋势": [(date - first_date).days / 365.25],
        }
    )
    return frame


def evaluate_price(
    spec: dict,
    date: pd.Timestamp,
    price: float,
    future_cost: float,
    loss_rate: float,
    residual_draws: np.ndarray,
    first_date: pd.Timestamp,
) -> dict | None:
    effective_cost = future_cost / (1.0 - loss_rate)
    if price <= effective_cost or price <= 0:
        return None
    cr = float(np.clip(1.0 - effective_cost / price, 0.0, 0.999))
    frame = future_design(date, price, first_date)
    x = design_matrix(frame, spec["含时间趋势"])
    eta = float((np.asarray(x, dtype=float) @ spec["系数"]).ravel()[0])
    demand_samples = np.exp(eta + residual_draws)
    sellable = float(np.quantile(demand_samples, cr))
    replenishment = sellable / (1.0 - loss_rate)
    expected_profit = float(
        np.mean(price * np.minimum(demand_samples, sellable) - future_cost * replenishment)
    )
    return {
        "售价": float(price),
        "预测需求量": float(np.mean(demand_samples)),
        "准备销售量": sellable,
        "补货量": replenishment,
        "预期利润": expected_profit,
        "临界分位数": cr,
        "预计满足量": float(np.mean(np.minimum(demand_samples, sellable))),
    }


def price_grid(lower: float, upper: float, future_cost: float) -> np.ndarray:
    markups = np.arange(lower, upper + 0.0001, 0.01)
    prices = np.round(future_cost * (1.0 + markups), 2)
    prices = prices[(prices > 0) & np.isfinite(prices)]
    return np.unique(prices)


def optimize_day(
    cat: str,
    date: pd.Timestamp,
    future_cost: float,
    loss_rate: float,
    spec: dict,
    markup_bounds: tuple[float, float],
    markup_median: float,
    rng: np.random.Generator,
    first_date: pd.Timestamp,
) -> dict:
    pool = spec["残差"]
    # 价格候选之间共用同一组残差抽样，降低网格比较的随机噪声。
    residual_draws = rng.choice(pool, size=SAMPLE_COUNT, replace=True)
    lower, upper = markup_bounds
    candidates = []
    for price in price_grid(lower, upper, future_cost):
        result = evaluate_price(
            spec, date, float(price), future_cost, loss_rate, residual_draws, first_date
        )
        if result is not None:
            result["加成率"] = float(price / future_cost - 1.0)
            candidates.append(result)
    if not candidates:
        raise ValueError(f"{cat} {date.date()} 没有满足正毛利条件的价格候选")
    best = max(candidates, key=lambda x: x["预期利润"])

    baseline_markup = float(np.clip(markup_median, lower, upper))
    baseline_price = float(np.round(future_cost * (1.0 + baseline_markup), 2))
    baseline = evaluate_price(
        spec,
        date,
        baseline_price,
        future_cost,
        loss_rate,
        residual_draws,
        first_date,
    )
    if baseline is None:
        baseline = {
            "售价": baseline_price,
            "预测需求量": np.nan,
            "准备销售量": np.nan,
            "补货量": np.nan,
            "预期利润": np.nan,
            "临界分位数": np.nan,
            "预计满足量": np.nan,
        }
    # 口径一致的基础方案：固定历史中位数加成，按平均预测需求补货。
    base_frame = future_design(date, baseline_price, first_date)
    base_x = design_matrix(base_frame, spec["含时间趋势"])
    base_eta = float((np.asarray(base_x, dtype=float) @ spec["系数"]).ravel()[0])
    base_samples = np.exp(base_eta + residual_draws)
    base_mean = float(np.mean(base_samples))
    base_q = base_mean / (1.0 - loss_rate)
    base_profit = float(
        np.mean(baseline_price * np.minimum(base_samples, base_mean) - future_cost * base_q)
    )

    best_row = {
        "日期": date.date().isoformat(),
        "品类": cat,
        "预测进价": future_cost,
        "损耗率": loss_rate,
        "最优加成率": best["加成率"],
        "最优售价": best["售价"],
        "预测需求量": best["预测需求量"],
        "最优补货量": best["补货量"],
        "预计满足量": best["预计满足量"],
        "临界分位数": best["临界分位数"],
        "预计利润": best["预期利润"],
        "边界标记": "触及上限" if best["加成率"] >= upper - 0.005 else "未触边界",
        "基础方案售价": baseline_price,
        "基础方案补货量": base_q,
        "基础方案预计利润": base_profit,
        "候选价格数": int(len(candidates)),
    }
    return {"主方案": best_row, "候选": candidates}


def optimize_all(
    panel: pd.DataFrame,
    category_loss: dict,
    markup_summary: dict,
    selected_cost_method: dict,
    selected_model: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first_date = panel["销售日期"].min()
    final_rows = []
    baseline_rows = []
    curve_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for cat in CATEGORIES:
        frame = panel[panel["品类"] == cat].sort_values("销售日期").copy()
        spec = fit_demand(frame, selected_model[cat])
        costs = cost_forecast(frame, FUTURE_DATES, selected_cost_method[cat])
        bounds = (
            markup_summary[cat]["百分之五分位"],
            markup_summary[cat]["百分之九十五分位"],
        )
        for date in FUTURE_DATES:
            result = optimize_day(
                cat,
                date,
                float(costs.loc[date]),
                category_loss[cat],
                spec,
                bounds,
                markup_summary[cat]["中位数"],
                rng,
                first_date,
            )
            final_rows.append(result["主方案"])
            best_row = result["主方案"]
            baseline_rows.append(
                {
                    "日期": best_row["日期"],
                    "品类": cat,
                    "基础方案售价": best_row["基础方案售价"],
                    "基础方案补货量": best_row["基础方案补货量"],
                    "基础方案预计利润": best_row["基础方案预计利润"],
                    "主方案售价": best_row["最优售价"],
                    "主方案补货量": best_row["最优补货量"],
                    "主方案预计利润": best_row["预计利润"],
                }
            )
            if date == FUTURE_DATES[0]:
                for item in result["候选"]:
                    curve_rows.append(
                        {
                            "日期": date.date().isoformat(),
                            "品类": cat,
                            "候选售价": item["售价"],
                            "预测需求量": item["预测需求量"],
                            "预期利润": item["预期利润"],
                            "候选加成率": item["加成率"],
                        }
                    )

    final_df = pd.DataFrame(final_rows).sort_values(["日期", "品类"]).reset_index(drop=True)
    baseline_df = pd.DataFrame(baseline_rows).sort_values(["日期", "品类"]).reset_index(drop=True)
    curve_df = pd.DataFrame(curve_rows)
    final_export = final_df.copy()
    final_export["损耗率（百分数）"] = (final_export["损耗率"] * 100).round(2)
    final_export["最优加成率（百分数）"] = (final_export["最优加成率"] * 100).round(2)
    for column in ["预测进价", "最优售价", "预测需求量", "最优补货量", "预计满足量", "预计利润", "基础方案售价", "基础方案补货量", "基础方案预计利润"]:
        final_export[column] = final_export[column].round(2)
    final_export["临界分位数"] = final_export["临界分位数"].round(4)
    final_export = final_export[
        [
            "日期",
            "品类",
            "预测进价",
            "损耗率（百分数）",
            "最优加成率（百分数）",
            "最优售价",
            "预测需求量",
            "最优补货量",
            "预计满足量",
            "临界分位数",
            "预计利润",
            "边界标记",
        ]
    ]
    baseline_export = baseline_df.copy()
    for column in baseline_export.columns[2:]:
        baseline_export[column] = baseline_export[column].round(2)
    curve_export = curve_df.copy()
    for column in ["候选售价", "预测需求量", "预期利润"]:
        curve_export[column] = curve_export[column].round(2)
    curve_export["候选加成率"] = curve_export["候选加成率"].round(4)
    final_export.to_csv(OUT / "七天六品类最优方案.csv", index=False, encoding="utf-8-sig")
    baseline_export.to_csv(OUT / "基础方案与主方案比较.csv", index=False, encoding="utf-8-sig")
    curve_export.to_csv(OUT / "价格销量利润曲线_7月1日.csv", index=False, encoding="utf-8-sig")
    return final_df, baseline_df, curve_df


def boundary_sensitivity(
    panel: pd.DataFrame,
    category_loss: dict,
    markup_summary: dict,
    selected_cost_method: dict,
    selected_model: dict,
) -> pd.DataFrame:
    """比较百分之五至百分之九十五与百分之一至百分之九十九两个价格边界。"""
    first_date = panel["销售日期"].min()
    scenario_defs = {
        "保守边界（百分之五至百分之九十五）": ("百分之五分位", "百分之九十五分位"),
        "扩展边界（百分之一至百分之九十九）": ("百分之一分位", "百分之九十九分位"),
    }
    rows = []
    for scenario_id, (low_key, high_key) in scenario_defs.items():
        rng = np.random.default_rng(RANDOM_SEED + (1 if "扩展" in scenario_id else 0))
        for cat in CATEGORIES:
            frame = panel[panel["品类"] == cat].sort_values("销售日期").copy()
            spec = fit_demand(frame, selected_model[cat])
            costs = cost_forecast(frame, FUTURE_DATES, selected_cost_method[cat])
            bounds = (markup_summary[cat][low_key], markup_summary[cat][high_key])
            for date in FUTURE_DATES:
                result = optimize_day(
                    cat,
                    date,
                    float(costs.loc[date]),
                    category_loss[cat],
                    spec,
                    bounds,
                    markup_summary[cat]["中位数"],
                    rng,
                    first_date,
                )["主方案"]
                rows.append(
                    {
                        "边界情景": scenario_id,
                        "日期": result["日期"],
                        "品类": cat,
                        "最优售价": result["最优售价"],
                        "最优加成率": result["最优加成率"],
                        "最优补货量": result["最优补货量"],
                        "预计利润": result["预计利润"],
                        "边界标记": result["边界标记"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "价格边界敏感性.csv", index=False, encoding="utf-8-sig")
    return df


def make_curve_figure(curve_df: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, cat in zip(axes, CATEGORIES):
        sub = curve_df[curve_df["品类"] == cat].sort_values("候选售价")
        if sub.empty:
            continue
        ax2 = ax.twinx()
        line1 = ax.plot(
            sub["候选售价"],
            sub["预测需求量"],
            color="#2F6B8A",
            label="预测需求量",
        )
        line2 = ax2.plot(
            sub["候选售价"],
            sub["预期利润"],
            color="#C45A3C",
            label="预期利润",
        )
        best = sub.loc[sub["预期利润"].idxmax()]
        ax.axvline(best["候选售价"], color="#555555", linestyle="--", linewidth=0.8)
        ax.set_title(cat)
        ax.set_xlabel("售价（元/千克）")
        ax.set_ylabel("预测需求量（千克）", color="#2F6B8A")
        ax2.set_ylabel("预期利润（元）", color="#C45A3C")
        ax.grid(alpha=0.18)
        lines = line1 + line2
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
    fig.suptitle("2023年7月1日各品类候选售价、预测需求与预期利润")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "价格销量利润曲线_7月1日.png", dpi=180)
    plt.close(fig)


def write_report(
    audit: dict,
    category_loss: dict,
    selected_cost_method: dict,
    selected_model: dict,
    final_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    main_profit = float(final_df["预计利润"].sum())
    base_profit = float(baseline_df["基础方案预计利润"].sum())
    upper_count = int((final_df["边界标记"] == "触及上限").sum())
    sens_summary = (
        sensitivity_df.groupby("边界情景", as_index=False)
        .agg(
            七天总利润=("预计利润", "sum"),
            触及上限天数=("边界标记", lambda x: int((x == "触及上限").sum())),
        )
    )
    lines = [
        "# 2023年C题问题二求解结果",
        "",
        "## 结论",
        "",
        f"已按六个品类、2023年7月1日至7日生成42条定价与补货方案。主方案七天预计利润为 **{main_profit:.2f}元**；固定历史中位数加成的基础方案预计利润为 **{base_profit:.2f}元**。主方案中有 **{upper_count}条**触及保守价格上限，因此价格上限敏感性必须和正文结果一起报告。",
        "总利润使用未四舍五入的中间值汇总，逐行决策表仅显示两位小数，逐行相加出现0.01元左右的差异属于显示舍入。",
        "",
        "完整方案见 七天六品类最优方案.csv，论文可直接按日期和品类整理成结果表。",
        "",
        "## 计算口径",
        "",
        "- 合法销售、折扣和退货流水均保留；退货按负销量和负销售额计入日净额，没有把合法退货当作脏数据删除。",
        "- 品类日售价和日进价均按销量加权；成本加成率按日平均售价除以日平均进价再减一。",
        "- 损耗率直接使用附件四中与六个品类对应的汇总值；单品损耗率只用于核对，不在品类方案中重复重加权。",
        "- 主需求模型使用实际成交售价、星期、月份和可选时间趋势；没有把未来未知的折扣比例作为输入。折扣只在数据审计和辅助解释中保留。",
        "- 成本预测在近七日均值、指数加权移动平均、同星期近八次中位数之间做滚动七日回测，逐品类选回测误差较小者。",
        "- 需求不确定性在对数销量尺度上对模型残差重抽样，并使用反变换平滑因子；没有把对数尺度残差直接加到千克销量上。",
        "- 报童模型的有效单位成本为预测进价除以可销售比例，临界分位数为一减有效单位成本除以售价；没有可观残值假设。",
        "",
        "## 数据审计摘要",
        "",
        f"附件二共有{audit['附件二原始流水数']}条流水，其中退货{audit['退货记录数']}条、折扣{audit['折扣记录数']}条；完整重复行数为{audit['完整重复行数']}。分类和批发价匹配后，可形成{audit['面板行数']}个日期—品类观测，净销售量为{audit['净销售量千克']:.3f}千克。",
        "",
        "## 选择结果",
        "",
        "| 品类 | 成本预测方法 | 需求模型 | 损耗率 |",
        "|---|---|---|---:|",
    ]
    for cat in CATEGORIES:
        lines.append(
            f"| {cat} | {selected_cost_method[cat]} | {'含时间趋势' if selected_model[cat] else '不含趋势'} | {category_loss[cat] * 100:.2f}% |"
        )
    lines += [
        "",
        "## 价格边界敏感性",
        "",
        "| 边界情景 | 七天总利润（元） | 触及上限天数 |",
        "|---|---:|---:|",
    ]
    for _, row in sens_summary.iterrows():
        lines.append(
            f"| {row['边界情景']} | {row['七天总利润']:.2f} | {int(row['触及上限天数'])} |"
        )
    lines += [
        "",
        "扩展边界下利润继续随价格上限变化，说明历史分位边界是经营约束而不是数学上的无界最优证明；论文中应把它表述为“在给定历史经营区间内的最优方案”。",
        "",
        "## 生成文件",
        "",
        "- 七天六品类最优方案.csv：42条主方案。",
        "- 基础方案与主方案比较.csv：基础方案和报童网格主方案对照。",
        "- 销售量与成本加成关系.csv：六品类价格弹性及回测结果。",
        "- 成本预测回测.csv、需求模型回测.csv：滚动七日回测。",
        "- 价格边界敏感性.csv：价格上限敏感性。",
        "- 价格销量利润曲线_7月1日.csv及图：论文可用的价格—需求—利润关系。",
        "- 历史加成率分布.csv、折扣辅助分析.csv、损耗率核对.csv：口径审计与辅助分析。",
        "",
        "## 已知边界",
        "",
        "附件没有提供库存、缺货和剩余商品的逐日记录，因此历史销量被当作需求的可观测代理；价格弹性只能解释条件关联。题目没有给出货架容量、预算或箱规，故按品类分别优化，补货量保留到小数。",
        "",
        "## 复现",
        "",
        "在仓库根目录执行：",
        "",
        "    python 问题二/脚本/求解问题二.py",
    ]
    (OUT / "问题二正式结果说明.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    _, panel, audit, category_loss, markup_summary = read_inputs()
    _, _, selected_model = demand_backtest(panel)
    _, selected_cost_method, _ = cost_backtest(panel)
    final_df, baseline_df, curve_df = optimize_all(
        panel, category_loss, markup_summary, selected_cost_method, selected_model
    )
    sensitivity_df = boundary_sensitivity(
        panel, category_loss, markup_summary, selected_cost_method, selected_model
    )
    make_curve_figure(curve_df)
    write_report(
        audit,
        category_loss,
        selected_cost_method,
        selected_model,
        final_df,
        baseline_df,
        sensitivity_df,
    )
    print("问题二计算完成")
    print(f"面板观测数: {len(panel)}")
    print(f"七天方案数: {len(final_df)}")
    print(f"主方案七天预计利润: {final_df['预计利润'].sum():.2f} 元")
    print(f"触及保守价格上限: {(final_df['边界标记'] == '触及上限').sum()} 条")
    print("输出目录:", OUT)


if __name__ == "__main__":
    main()
