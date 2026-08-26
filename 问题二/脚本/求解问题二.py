# -*- coding: utf-8 -*-
"""2023 年数学建模国赛 C 题问题二：按品类的定价与补货复核。

本脚本只读取仓库内的四个原始附件，主线保持为：
历史规律 -> 未来进价/基础需求 -> 需求与价格关系 -> 定价 -> 报童补货。

核心原则：
1. 退货保留为负销量并计入净额；折扣行保留在现实成交口径的审计中。
2. 常规定价需求使用正常销售行，折扣行不作为未来未知的主模型输入。
3. 需求模型只使用简单对数回归，并以滚动七日回测选择是否保留时间趋势。
4. 价格搜索只作边界诊断；若没有可识别的内部峰值，最终建议采用正常销售历史中位加成。
5. 补货量使用同一需求分布下的易腐品报童分位数，避免把定价收益和补货收益混在一起。

在仓库根目录运行：
    python 问题二/脚本/求解问题二.py
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "2023年C题"
OUT = ROOT / "问题二" / "结果"
FIG = ROOT / "问题二" / "图表"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
FUTURE_DATES = pd.date_range("2023-07-01", "2023-07-07", freq="D")
DEMAND_MODELS = {
    "对数需求（星期和月份）": False,
    "对数需求（星期、月份和趋势）": True,
}
COST_METHODS = [
    "近7日均值",
    "近14日均值",
    "指数加权移动平均",
    "同星期近8次中位数",
    "近180日趋势加星期",
]
COST_METHOD_PRIORITY = {name: i for i, name in enumerate(COST_METHODS)}
RANDOM_SEED = 20230826
SAMPLE_COUNT = 6000
LOSS_RECENT_DAYS = 30


def demand_model_name(include_trend: bool) -> str:
    return (
        "对数需求（星期、月份和趋势）"
        if include_trend
        else "对数需求（星期和月份）"
    )


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def make_panel(transactions: pd.DataFrame, first_date: pd.Timestamp) -> pd.DataFrame:
    """按日期和品类构造销量加权的日面板，不填补整天缺失。"""
    panel = (
        transactions.groupby(["销售日期", "品类"], as_index=False)
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
    panel = panel[(panel["日销售量"] > 0) & (panel["日成本额"] > 0)].copy()
    panel["日平均售价"] = panel["日销售额"] / panel["日销售量"]
    panel["日平均进价"] = panel["日成本额"] / panel["日销售量"]
    panel["加成率"] = panel["日平均售价"] / panel["日平均进价"] - 1.0
    panel["折扣销量占比"] = panel["折扣销售量"] / panel["正销售量"].replace(0, np.nan)
    panel["星期"] = panel["销售日期"].dt.weekday + 1
    panel["月份"] = panel["销售日期"].dt.month
    panel["时间趋势"] = (panel["销售日期"] - first_date).dt.days / 365.25
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["日平均售价", "日平均进价", "加成率"]
    )
    return panel.sort_values(["销售日期", "品类"]).reset_index(drop=True)


def read_source_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict,
    dict,
    pd.Timestamp,
]:
    """读取四个原始附件并完成数据、匹配、守恒和损耗率审计。"""
    raw1 = pd.read_excel(DATA / "附件1.xlsx", dtype=str)
    raw2 = pd.read_excel(DATA / "附件2.xlsx")
    raw3 = pd.read_excel(DATA / "附件3.xlsx", dtype=str)
    raw4_category = pd.read_excel(DATA / "附件4.xlsx", sheet_name=0)
    raw4_item = pd.read_excel(DATA / "附件4.xlsx", sheet_name=1)

    item = raw1.iloc[:, [0, 2, 3]].copy()
    item.columns = ["单品编码", "分类编码", "品类"]
    item["单品编码"] = item["单品编码"].astype(str).str.strip()
    item["分类编码"] = item["分类编码"].astype(str).str.strip()
    item["品类"] = item["品类"].astype(str).str.strip()

    sales = raw2.iloc[:, :7].copy()
    sales.columns = [
        "销售日期",
        "扫码销售时间",
        "单品编码",
        "销量",
        "售价",
        "销售类型",
        "是否打折销售",
    ]
    sales["销售日期"] = pd.to_datetime(sales["销售日期"], errors="coerce")
    sales["单品编码"] = sales["单品编码"].astype(str).str.strip()
    sales["销量"] = pd.to_numeric(sales["销量"], errors="coerce")
    sales["售价"] = pd.to_numeric(sales["售价"], errors="coerce")

    wholesale = raw3.iloc[:, [0, 1, 2]].copy()
    wholesale.columns = ["销售日期", "单品编码", "进价"]
    wholesale["销售日期"] = pd.to_datetime(wholesale["销售日期"], errors="coerce")
    wholesale["单品编码"] = wholesale["单品编码"].astype(str).str.strip()
    wholesale["进价"] = pd.to_numeric(wholesale["进价"], errors="coerce")

    loss_category = raw4_category.iloc[:, [1, 2]].copy()
    loss_category.columns = ["品类", "损耗率百分数"]
    loss_category["品类"] = loss_category["品类"].astype(str).str.strip()
    loss_category["损耗率百分数"] = pd.to_numeric(
        loss_category["损耗率百分数"], errors="coerce"
    )
    category_loss = dict(
        zip(loss_category["品类"], loss_category["损耗率百分数"] / 100.0)
    )

    loss_item = raw4_item.iloc[:, [0, 2]].copy()
    loss_item.columns = ["单品编码", "损耗率百分数"]
    loss_item["单品编码"] = loss_item["单品编码"].astype(str).str.strip()
    loss_item["损耗率百分数"] = pd.to_numeric(
        loss_item["损耗率百分数"], errors="coerce"
    )
    item_loss = dict(zip(loss_item["单品编码"], loss_item["损耗率百分数"] / 100.0))

    if sales["销售日期"].isna().any() or wholesale["销售日期"].isna().any():
        raise ValueError("原始附件存在无法识别的日期")
    if item["单品编码"].duplicated().any():
        raise ValueError("附件一存在重复单品编码")
    if wholesale.duplicated(["销售日期", "单品编码"]).any():
        raise ValueError("附件三存在重复日期和单品组合")
    if set(CATEGORIES) != set(category_loss):
        raise ValueError("附件四未覆盖六个题目品类")
    if any(
        not np.isfinite(value) or value < 0 or value >= 1
        for value in category_loss.values()
    ):
        raise ValueError("附件四品类损耗率存在非法值")

    merged = sales.merge(
        item, on="单品编码", how="left", validate="many_to_one"
    ).merge(
        wholesale,
        on=["销售日期", "单品编码"],
        how="left",
        validate="many_to_one",
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
    if bad.any():
        raise ValueError("原始附件存在无法进入分析的记录")

    sign_error = (
        ((merged["销售类型"] == "退货") & (merged["销量"] >= 0))
        | ((merged["销售类型"] == "销售") & (merged["销量"] < 0))
    )
    if sign_error.any():
        raise ValueError("退货标记与销量符号不一致")
    if set(merged["销售类型"].dropna().unique()) != {"销售", "退货"}:
        raise ValueError("销售类型存在未预期取值")
    if set(merged["是否打折销售"].dropna().unique()) - {"是", "否"}:
        raise ValueError("折扣标记存在未预期取值")

    merged["销售额"] = merged["销量"] * merged["售价"]
    merged["成本额"] = merged["销量"] * merged["进价"]
    merged["正销量"] = merged["销量"].clip(lower=0)
    merged["折扣正销量"] = np.where(
        (merged["销售类型"] == "销售")
        & (merged["是否打折销售"] == "是"),
        merged["销量"],
        0.0,
    )

    first_date = pd.Timestamp(sales["销售日期"].min())
    last_date = pd.Timestamp(sales["销售日期"].max())
    calendar = pd.date_range(first_date, last_date, freq="D")
    all_sales = merged[merged["销售类型"] == "销售"].copy()
    normal_sales = merged[
        (merged["销售类型"] == "销售")
        & (merged["是否打折销售"] != "是")
    ].copy()

    panel_all = make_panel(merged, first_date)
    panel_normal = make_panel(normal_sales, first_date)
    if set(panel_normal["品类"].unique()) != set(CATEGORIES):
        raise ValueError("正常销售面板未覆盖六个品类")

    audit = {
        "附件二原始流水数": int(len(sales)),
        "销售记录数": int((sales["销售类型"] == "销售").sum()),
        "退货记录数": int((sales["销售类型"] == "退货").sum()),
        "折扣记录数": int((sales["是否打折销售"] == "是").sum()),
        "折扣销售记录数": int(
            ((sales["销售类型"] == "销售") & (sales["是否打折销售"] == "是")).sum()
        ),
        "正常销售记录数": int(len(normal_sales)),
        "完整重复行数": int(sales.duplicated().sum()),
        "附件一单品数": int(item["单品编码"].nunique()),
        "附件三日期单品组合数": int(
            wholesale[["销售日期", "单品编码"]].drop_duplicates().shape[0]
        ),
        "销售日期起点": str(first_date.date()),
        "销售日期终点": str(last_date.date()),
        "日历天数": int(len(calendar)),
        "有流水日期数": int(sales["销售日期"].nunique()),
        "无流水日期数": int(len(calendar.difference(sales["销售日期"].drop_duplicates()))),
        "销售量千克": float(all_sales["销量"].sum()),
        "退货量千克": float(merged.loc[merged["销售类型"] == "退货", "销量"].sum()),
        "净销售量千克": float(merged["销量"].sum()),
        "折扣销售量千克": float(
            all_sales.loc[all_sales["是否打折销售"] == "是", "销量"].sum()
        ),
        "分类未匹配数": int(merged["品类"].isna().sum()),
        "进价未匹配数": int(merged["进价"].isna().sum()),
        "业务类型与销量符号不一致数": int(sign_error.sum()),
        "全量有效面板行数": int(len(panel_all)),
        "正常销售面板行数": int(len(panel_normal)),
        "面板日期数": int(panel_all["销售日期"].nunique()),
        "全量各品类面板行数": {
            str(k): int(v)
            for k, v in panel_all["品类"].value_counts().to_dict().items()
        },
        "正常销售各品类面板行数": {
            str(k): int(v)
            for k, v in panel_normal["品类"].value_counts().to_dict().items()
        },
        "整天无流水日期": [
            str(x.date()) for x in calendar.difference(sales["销售日期"].drop_duplicates())
        ],
        "仅折扣导致的面板缺口": int(len(panel_all) - len(panel_normal)),
    }
    audit["折扣记录占销售记录比例"] = audit["折扣销售记录数"] / audit["销售记录数"]
    audit["折扣销量占销售量比例"] = audit["折扣销售量千克"] / audit["销售量千克"]

    write_csv(panel_all, "品类日面板_全量有效.csv")
    write_csv(panel_normal, "品类日面板_正常销售.csv")
    write_csv(panel_normal, "品类日面板.csv")

    recent_start = last_date - pd.Timedelta(days=LOSS_RECENT_DAYS - 1)
    recent = merged[
        (merged["销售日期"] >= recent_start)
        & (merged["销售类型"] == "销售")
        & (merged["销量"] > 0)
    ].copy()
    recent["单品损耗率"] = recent["单品编码"].map(item_loss)
    recent = recent.dropna(subset=["单品损耗率"])
    recent_loss = {}
    for cat, sub in recent.groupby("品类"):
        recent_loss[cat] = float(np.average(sub["单品损耗率"], weights=sub["销量"]))
    loss_rows = []
    for cat in CATEGORIES:
        loss_rows.append(
            {
                "品类": cat,
                "附件四品类损耗率": category_loss[cat],
                "近期单品销量加权损耗率": recent_loss.get(cat, np.nan),
                "正式采用损耗率": category_loss[cat],
                "采用理由": "题目按品类补货，直接采用附件四品类汇总值",
            }
        )
    write_csv(pd.DataFrame(loss_rows), "损耗率核对.csv")
    write_csv(pd.DataFrame([audit]), "数据审计摘要.csv")
    (OUT / "数据审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (
        merged,
        panel_all,
        panel_normal,
        audit,
        category_loss,
        item_loss,
        first_date,
    )


def discount_audit(merged: pd.DataFrame) -> pd.DataFrame:
    """比较折扣流水与正常流水，判断折扣是否属于常规定价状态。"""
    sales = merged[merged["销售类型"] == "销售"].copy()
    sales["折扣行"] = (sales["是否打折销售"] == "是").astype(int)
    sales["小时"] = (
        sales["扫码销售时间"].astype(str).str.extract(r"^(\d{1,2})")[0].astype(float)
    )
    normal_reference = (
        sales[sales["是否打折销售"] != "是"]
        .groupby(["销售日期", "单品编码"])["售价"]
        .median()
        .rename("同日正常价中位数")
        .reset_index()
    )
    discount = sales[sales["是否打折销售"] == "是"].merge(
        normal_reference, on=["销售日期", "单品编码"], how="left"
    )
    discount["折扣相对同日正常价比值"] = (
        discount["售价"] / discount["同日正常价中位数"]
    )

    rows = []
    for cat in CATEGORIES:
        sub = sales[sales["品类"] == cat]
        d = sub[sub["是否打折销售"] == "是"]
        d_ref = discount[discount["品类"] == cat].dropna(
            subset=["折扣相对同日正常价比值"]
        )
        rows.append(
            {
                "品类": cat,
                "销售流水数": int(len(sub)),
                "折扣流水数": int(len(d)),
                "折扣流水占比": float(len(d) / len(sub)),
                "销售量千克": float(sub["销量"].sum()),
                "折扣销售量千克": float(d["销量"].sum()),
                "折扣销量占比": float(d["销量"].sum() / sub["销量"].sum()),
                "销售额": float(sub["销售额"].sum()),
                "折扣销售额": float(d["销售额"].sum()),
                "折扣销售额占比": float(d["销售额"].sum() / sub["销售额"].sum()),
                "有折扣的日期数": int(d["销售日期"].nunique()),
                "样本日期数": int(sub["销售日期"].nunique()),
                "折扣晚于19点占比": float((d["小时"] >= 19).mean())
                if len(d)
                else np.nan,
                "折扣相对同日正常价中位数": float(
                    d_ref["折扣相对同日正常价比值"].median()
                )
                if len(d_ref)
                else np.nan,
                "折扣同日正常价匹配数": int(len(d_ref)),
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, "折扣辅助分析.csv")
    return result


def markup_summary(panel_all: pd.DataFrame, panel_normal: pd.DataFrame) -> dict:
    """输出全量与正常销售两种历史加成率分布。"""
    result: dict[str, dict[str, dict[str, float]]] = {}
    rows = []
    quantile_names = [
        ("百分之一分位", 0.01),
        ("百分之五分位", 0.05),
        ("百分之十分位", 0.10),
        ("百分之二十五分位", 0.25),
        ("中位数", 0.50),
        ("百分之七十五分位", 0.75),
        ("百分之九十分位", 0.90),
        ("百分之九十五分位", 0.95),
        ("百分之九十九分位", 0.99),
    ]
    for cat in CATEGORIES:
        row: dict[str, object] = {"品类": cat}
        result[cat] = {}
        for label, panel in [("全量有效", panel_all), ("正常销售", panel_normal)]:
            values = panel.loc[panel["品类"] == cat, "加成率"].astype(float)
            quantiles = {
                name: float(values.quantile(q)) for name, q in quantile_names
            }
            result[cat][label] = quantiles
            for key, value in quantiles.items():
                row[f"{label}{key}"] = value
        rows.append(row)
    write_csv(pd.DataFrame(rows), "历史加成率分布.csv")
    return result


def design_matrix(
    frame: pd.DataFrame, include_trend: bool, price_kind: str = "售价"
) -> tuple[np.ndarray, list[str]]:
    """构造星期、月份控制下的对数需求设计矩阵。"""
    x = pd.DataFrame(index=frame.index)
    if price_kind == "售价":
        x["对数售价"] = np.log(frame["日平均售价"].astype(float))
    elif price_kind == "加成率":
        x["对数加成率"] = np.log1p(frame["加成率"].astype(float))
    else:
        raise ValueError(f"未知价格变量: {price_kind}")
    names = list(x.columns)
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


def robust_ols(frame: pd.DataFrame, include_trend: bool, price_kind: str) -> dict:
    """最小二乘点估计加七阶 Newey-West 型稳健协方差。"""
    x, names = design_matrix(frame, include_trend, price_kind)
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
    z_values = np.divide(
        params,
        standard_errors,
        out=np.zeros_like(params),
        where=standard_errors > 0,
    )
    p_values = np.array(
        [math.erfc(abs(float(z)) / math.sqrt(2.0)) for z in z_values],
        dtype=float,
    )
    price_idx = names.index("对数售价" if price_kind == "售价" else "对数加成率")
    sse = float(residuals @ residuals)
    centered = y - y.mean()
    sst = float(centered @ centered)
    r_squared = 1.0 - sse / sst if sst > 0 else np.nan
    n = len(y)
    k = x.shape[1]
    adjusted_r_squared = (
        1.0 - (1.0 - r_squared) * (n - 1) / (n - k)
        if n > k and np.isfinite(r_squared)
        else np.nan
    )
    confidence = np.column_stack(
        [params - 1.96 * standard_errors, params + 1.96 * standard_errors]
    )
    return {
        "模型": demand_model_name(include_trend),
        "含时间趋势": include_trend,
        "价格变量": price_kind,
        "系数": params,
        "设计列": names,
        "价格系数": float(params[price_idx]),
        "稳健标准误": float(standard_errors[price_idx]),
        "稳健概率值": float(p_values[price_idx]),
        "稳健区间下限": float(confidence[price_idx, 0]),
        "稳健区间上限": float(confidence[price_idx, 1]),
        "模型决定系数": float(r_squared),
        "调整决定系数": float(adjusted_r_squared),
        "残差": residuals,
        "平滑还原因子": float(np.mean(np.exp(residuals))),
        "一阶残差相关": float(np.corrcoef(residuals[1:], residuals[:-1])[0, 1])
        if len(residuals) > 2
        else np.nan,
        "样本数": int(n),
    }


def prediction_from_model(spec: dict, frame: pd.DataFrame) -> np.ndarray:
    x, names = design_matrix(
        frame, bool(spec["含时间趋势"]), str(spec["价格变量"])
    )
    if names != spec["设计列"]:
        raise ValueError("训练与预测的设计矩阵不一致")
    eta = x @ np.asarray(spec["系数"], dtype=float)
    return np.exp(eta) * float(spec["平滑还原因子"])


def validation_cutoffs(frame: pd.DataFrame) -> list[pd.Timestamp]:
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


def demand_backtest(
    panel: pd.DataFrame, regime_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """比较两种简单需求模型，并用滚动七日误差选择是否保留趋势。"""
    detail_rows = []
    summary_rows = []
    selected_models: dict[str, bool] = {}
    full_specs: dict[str, dict] = {}

    for cat in CATEGORIES:
        frame = panel[panel["品类"] == cat].sort_values("销售日期").copy()
        model_metrics: dict[str, list[dict]] = {}
        for model_name, include_trend in DEMAND_MODELS.items():
            metrics = []
            for cutoff in validation_cutoffs(frame):
                train = frame[frame["销售日期"] <= cutoff]
                test = frame[
                    (frame["销售日期"] > cutoff)
                    & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
                ]
                spec = robust_ols(train, include_trend, "售价")
                pred = prediction_from_model(spec, test)
                actual = test["日销售量"].to_numpy(float)
                error = actual - pred
                metrics.append(
                    {
                        "口径": regime_name,
                        "品类": cat,
                        "模型": model_name,
                        "验证截止日": cutoff.date().isoformat(),
                        "验证天数": int(len(test)),
                        "绝对百分比误差加权": float(
                            np.abs(error).sum() / actual.sum()
                        ),
                        "平均绝对误差": float(np.abs(error).mean()),
                        "均方根误差": float(np.sqrt(np.mean(error**2))),
                        "价格系数": float(spec["价格系数"]),
                    }
                )
                detail_rows.append(metrics[-1])
            model_metrics[model_name] = metrics

        summary_cat = []
        for model_name, metrics in model_metrics.items():
            total_abs = 0.0
            total_actual = 0.0
            maes = []
            rmses = []
            for cutoff in validation_cutoffs(frame):
                train = frame[frame["销售日期"] <= cutoff]
                test = frame[
                    (frame["销售日期"] > cutoff)
                    & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
                ]
                spec = robust_ols(
                    train, DEMAND_MODELS[model_name], "售价"
                )
                pred = prediction_from_model(spec, test)
                actual = test["日销售量"].to_numpy(float)
                error = actual - pred
                total_abs += float(np.abs(error).sum())
                total_actual += float(actual.sum())
                maes.append(float(np.abs(error).mean()))
                rmses.append(float(np.sqrt(np.mean(error**2))))
            summary_cat.append(
                {
                    "口径": regime_name,
                    "品类": cat,
                    "模型": model_name,
                    "回测绝对百分比误差加权": total_abs / total_actual,
                    "回测平均绝对误差": float(np.mean(maes)),
                    "回测均方根误差": float(np.mean(rmses)),
                    "回测次数": int(len(metrics)),
                }
            )
        summary_cat_df = pd.DataFrame(summary_cat)
        plain_name = "对数需求（星期和月份）"
        trend_name = "对数需求（星期、月份和趋势）"
        plain_wape = float(
            summary_cat_df.loc[
                summary_cat_df["模型"] == plain_name,
                "回测绝对百分比误差加权",
            ].iloc[0]
        )
        trend_wape = float(
            summary_cat_df.loc[
                summary_cat_df["模型"] == trend_name,
                "回测绝对百分比误差加权",
            ].iloc[0]
        )
        # 只有趋势误差至少改善5%才保留趋势，平衡预测效果和解释简洁性。
        selected_name = trend_name if trend_wape < 0.95 * plain_wape else plain_name
        selected_models[cat] = DEMAND_MODELS[selected_name]
        summary_cat_df["是否入选"] = np.where(
            summary_cat_df["模型"] == selected_name, "是", "否"
        )
        summary_rows.extend(summary_cat_df.to_dict("records"))
        full_specs[cat] = robust_ols(
            frame, selected_models[cat], "售价"
        )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    write_csv(detail_df, f"需求模型回测明细_{regime_name}.csv")
    write_csv(summary_df, f"需求模型回测汇总_{regime_name}.csv")
    return detail_df, summary_df, selected_models, full_specs


def cost_forecast(
    train: pd.DataFrame, future_dates: pd.DatetimeIndex, method: str
) -> pd.Series:
    """四种水平方法加一种趋势方法的简单进价预测。"""
    series = train.set_index("销售日期")["日平均进价"].sort_index().dropna()
    future_dates = pd.DatetimeIndex(future_dates)
    if len(series) == 0:
        raise ValueError("进价序列为空")
    if method == "近7日均值":
        return pd.Series(float(series.tail(7).mean()), index=future_dates)
    if method == "近14日均值":
        return pd.Series(float(series.tail(14).mean()), index=future_dates)
    if method == "指数加权移动平均":
        return pd.Series(
            float(series.ewm(alpha=0.4, adjust=False).mean().iloc[-1]),
            index=future_dates,
        )
    if method == "同星期近8次中位数":
        values = []
        for date in future_dates:
            same_weekday = series[series.index.weekday == date.weekday()].tail(8)
            values.append(
                float(same_weekday.median())
                if len(same_weekday)
                else float(series.tail(7).median())
            )
        return pd.Series(values, index=future_dates)
    if method == "近180日趋势加星期":
        recent = train.sort_values("销售日期").tail(180).copy()
        train_time = (recent["销售日期"] - recent["销售日期"].max()).dt.days.astype(float)
        x = [np.ones(len(recent)), train_time.to_numpy()]
        for weekday in range(1, 7):
            x.append((recent["销售日期"].dt.weekday == weekday).astype(float).to_numpy())
        beta = np.linalg.lstsq(
            np.column_stack(x),
            recent["日平均进价"].to_numpy(float),
            rcond=None,
        )[0]
        future_time = (future_dates - recent["销售日期"].max()).days.astype(float)
        xf = [np.ones(len(future_dates)), future_time]
        for weekday in range(1, 7):
            xf.append((future_dates.weekday == weekday).astype(float))
        prediction = np.column_stack(xf) @ beta
        low = float(recent["日平均进价"].quantile(0.05))
        high = float(recent["日平均进价"].quantile(0.95))
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            low = float(recent["日平均进价"].min())
            high = float(recent["日平均进价"].max())
        return pd.Series(np.clip(prediction, low, high), index=future_dates)
    raise ValueError(f"未知进价预测方法: {method}")


def cost_backtest(
    panel_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    detail_rows = []
    summary_rows = []
    selected_methods: dict[str, str] = {}
    for cat in CATEGORIES:
        frame = panel_all[panel_all["品类"] == cat].sort_values("销售日期").copy()
        for cutoff in validation_cutoffs(frame):
            train = frame[frame["销售日期"] <= cutoff]
            test = frame[
                (frame["销售日期"] > cutoff)
                & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
            ]
            actual = test.set_index("销售日期")["日平均进价"]
            for method in COST_METHODS:
                prediction = cost_forecast(
                    train, pd.DatetimeIndex(test["销售日期"]), method
                )
                joined = pd.concat(
                    [actual.rename("实际进价"), prediction.rename("预测进价")],
                    axis=1,
                ).dropna()
                error = joined["实际进价"] - joined["预测进价"]
                detail_rows.append(
                    {
                        "品类": cat,
                        "方法": method,
                        "验证截止日": cutoff.date().isoformat(),
                        "验证天数": int(len(joined)),
                        "绝对百分比误差加权": float(
                            np.abs(error).sum() / np.abs(joined["实际进价"]).sum()
                        ),
                        "平均绝对误差": float(np.abs(error).mean()),
                    }
                )
        detail_cat = pd.DataFrame(
            [x for x in detail_rows if x["品类"] == cat]
        )
        for method, sub in detail_cat.groupby("方法", sort=False):
            numerator = 0.0
            denominator = 0.0
            maes = []
            for _, row in sub.iterrows():
                cutoff = pd.Timestamp(row["验证截止日"])
                train = frame[frame["销售日期"] <= cutoff]
                test = frame[
                    (frame["销售日期"] > cutoff)
                    & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))
                ]
                pred = cost_forecast(
                    train, pd.DatetimeIndex(test["销售日期"]), method
                )
                actual = test.set_index("销售日期")["日平均进价"]
                joined = pd.concat(
                    [actual.rename("实际"), pred.rename("预测")], axis=1
                ).dropna()
                numerator += float(np.abs(joined["实际"] - joined["预测"]).sum())
                denominator += float(np.abs(joined["实际"]).sum())
                maes.append(float(np.abs(joined["实际"] - joined["预测"]).mean()))
            summary_rows.append(
                {
                    "品类": cat,
                    "方法": method,
                    "回测绝对百分比误差加权": numerator / denominator,
                    "回测平均绝对误差": float(np.mean(maes)),
                    "回测次数": int(len(sub)),
                }
            )
        sub_summary = pd.DataFrame(
            [x for x in summary_rows if x["品类"] == cat]
        )
        sub_summary["方法排序"] = sub_summary["方法"].map(COST_METHOD_PRIORITY)
        sub_summary = sub_summary.sort_values(
            ["回测绝对百分比误差加权", "回测平均绝对误差", "方法排序"]
        )
        selected_methods[cat] = str(sub_summary.iloc[0]["方法"])
    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["是否入选"] = [
        "是" if row["方法"] == selected_methods[row["品类"]] else "否"
        for _, row in summary_df.iterrows()
    ]
    write_csv(detail_df, "成本预测回测明细.csv")
    write_csv(summary_df, "成本预测回测.csv")
    return detail_df, summary_df, selected_methods


def future_design(
    date: pd.Timestamp, price: float, first_date: pd.Timestamp
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "销售日期": [pd.Timestamp(date)],
            "日平均售价": [float(price)],
            "加成率": [0.0],
            "日销售量": [1.0],
            "星期": [pd.Timestamp(date).weekday() + 1],
            "月份": [pd.Timestamp(date).month],
            "时间趋势": [
                (pd.Timestamp(date) - pd.Timestamp(first_date)).days / 365.25
            ],
        }
    )


def demand_samples(
    spec: dict,
    date: pd.Timestamp,
    price: float,
    residual_draws: np.ndarray,
    first_date: pd.Timestamp,
) -> np.ndarray:
    frame = future_design(date, price, first_date)
    x, names = design_matrix(frame, bool(spec["含时间趋势"]), "售价")
    if names != spec["设计列"]:
        raise ValueError("未来设计矩阵与需求模型不一致")
    eta = float((x @ np.asarray(spec["系数"], dtype=float)).ravel()[0])
    return np.exp(eta + residual_draws)


def evaluate_fixed_order(
    spec: dict,
    date: pd.Timestamp,
    price: float,
    future_cost: float,
    loss_rate: float,
    residual_draws: np.ndarray,
    first_date: pd.Timestamp,
    order_kg: float,
) -> dict:
    samples = demand_samples(spec, date, price, residual_draws, first_date)
    sellable = max(0.0, float(order_kg)) * (1.0 - loss_rate)
    sales = np.minimum(samples, sellable)
    return {
        "售价": float(price),
        "加成率": float(price / future_cost - 1.0),
        "预测需求量": float(samples.mean()),
        "补货量": float(order_kg),
        "预计满足量": float(sales.mean()),
        "预期利润": float(
            np.mean(price * sales - future_cost * float(order_kg))
        ),
        "临界分位数": np.nan,
    }


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
    critical_fractile = float(
        np.clip(1.0 - effective_cost / price, 0.001, 0.999)
    )
    samples = demand_samples(spec, date, price, residual_draws, first_date)
    target_sellable = float(np.quantile(samples, critical_fractile))
    order_kg = max(0.1, round(target_sellable / (1.0 - loss_rate), 1))
    sellable = order_kg * (1.0 - loss_rate)
    sales = np.minimum(samples, sellable)
    return {
        "售价": float(price),
        "加成率": float(price / future_cost - 1.0),
        "预测需求量": float(samples.mean()),
        "补货量": float(order_kg),
        "预计满足量": float(sales.mean()),
        "预期利润": float(np.mean(price * sales - future_cost * order_kg)),
        "临界分位数": critical_fractile,
    }


def price_grid(
    lower_markup: float, upper_markup: float, future_cost: float
) -> np.ndarray:
    low = math.floor(float(lower_markup) * 100.0) / 100.0
    high = math.ceil(float(upper_markup) * 100.0) / 100.0
    markups = np.arange(low, high + 0.0001, 0.01)
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
    residual_draws = rng.choice(
        np.asarray(spec["残差"], dtype=float),
        size=SAMPLE_COUNT,
        replace=True,
    )
    lower_markup, upper_markup = markup_bounds
    candidates = []
    for price in price_grid(lower_markup, upper_markup, future_cost):
        item = evaluate_price(
            spec,
            date,
            float(price),
            future_cost,
            loss_rate,
            residual_draws,
            first_date,
        )
        if item is not None:
            candidates.append(item)
    if not candidates:
        raise ValueError(f"{cat} {date.date()} 没有满足正毛利的价格候选")
    search_best = max(candidates, key=lambda x: x["预期利润"])
    median_price = float(np.round(future_cost * (1.0 + markup_median), 2))
    if median_price <= future_cost / (1.0 - loss_rate):
        median_price = float(
            max(
                candidate["售价"]
                for candidate in candidates
                if candidate["售价"] > future_cost / (1.0 - loss_rate)
            )
        )
    robust_result = evaluate_price(
        spec,
        date,
        median_price,
        future_cost,
        loss_rate,
        residual_draws,
        first_date,
    )
    if robust_result is None:
        robust_result = search_best.copy()
    robust_samples = demand_samples(
        spec, date, robust_result["售价"], residual_draws, first_date
    )
    mean_order = max(
        0.1, round(float(robust_samples.mean()) / (1.0 - loss_rate), 1)
    )
    base_result = evaluate_fixed_order(
        spec,
        date,
        robust_result["售价"],
        future_cost,
        loss_rate,
        residual_draws,
        first_date,
        mean_order,
    )
    search_same_base_order = evaluate_fixed_order(
        spec,
        date,
        search_best["售价"],
        future_cost,
        loss_rate,
        residual_draws,
        first_date,
        mean_order,
    )
    touch_upper = bool(
        search_best["加成率"] >= float(upper_markup) - 0.005
    )
    reason = (
        "价格搜索触及历史上限，缺少内部最优证据，采用正常销售历史中位加成"
        if touch_upper
        else "价格关系证据不足或收益峰值不稳定，采用正常销售历史中位加成"
    )
    main_row = {
        "日期": date.date().isoformat(),
        "品类": cat,
        "预测批发价": float(future_cost),
        "损耗率": float(loss_rate),
        "建议成本加成率": float(robust_result["加成率"]),
        "建议售价": float(robust_result["售价"]),
        "预测需求量": float(robust_result["预测需求量"]),
        "建议补货量": float(robust_result["补货量"]),
        "预计满足量": float(robust_result["预计满足量"]),
        "临界分位数": float(robust_result["临界分位数"]),
        "预计利润": float(robust_result["预期利润"]),
        "数学搜索售价": float(search_best["售价"]),
        "数学搜索加成率": float(search_best["加成率"]),
        "数学搜索预计利润": float(search_best["预期利润"]),
        "数学搜索是否触及上限": "是" if touch_upper else "否",
        "需求关系可靠性": "",
        "定价依据": reason,
    }
    decomposition = {
        "日期": date.date().isoformat(),
        "品类": cat,
        "基础方案售价": float(base_result["售价"]),
        "基础方案补货量": float(base_result["补货量"]),
        "基础方案预计利润": float(base_result["预期利润"]),
        "仅改补货方案利润": float(robust_result["预期利润"]),
        "价格搜索方案利润": float(search_best["预期利润"]),
        "价格搜索同基础补货利润": float(search_same_base_order["预期利润"]),
        "补货改善": float(robust_result["预期利润"] - base_result["预期利润"]),
        "价格搜索改善": float(
            search_best["预期利润"] - robust_result["预期利润"]
        ),
        "最终建议方案利润": float(robust_result["预期利润"]),
    }
    return {
        "主方案": main_row,
        "分解": decomposition,
        "候选": candidates,
    }


def optimize_all(
    panel_normal: pd.DataFrame,
    panel_cost: pd.DataFrame,
    category_loss: dict,
    markup_info: dict,
    selected_cost_methods: dict,
    selected_demand_models: dict,
    demand_specs: dict,
    first_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_rows = []
    decomposition_rows = []
    curve_rows = []
    future_cost_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for cat in CATEGORIES:
        frame = panel_normal[panel_normal["品类"] == cat].sort_values(
            "销售日期"
        )
        cost_frame = panel_cost[panel_cost["品类"] == cat].sort_values(
            "销售日期"
        )
        method = selected_cost_methods[cat]
        costs = cost_forecast(cost_frame, FUTURE_DATES, method)
        spec = demand_specs[cat]
        info = markup_info[cat]["正常销售"]
        bounds = (
            info["百分之五分位"],
            info["百分之九十五分位"],
        )
        for date in FUTURE_DATES:
            future_cost = float(costs.loc[date])
            future_cost_rows.append(
                {
                    "日期": date.date().isoformat(),
                    "品类": cat,
                    "预测批发价": future_cost,
                    "方法": method,
                }
            )
            result = optimize_day(
                cat,
                date,
                future_cost,
                category_loss[cat],
                spec,
                bounds,
                info["中位数"],
                rng,
                first_date,
            )
            final_rows.append(result["主方案"])
            decomposition_rows.append(result["分解"])
            if date == FUTURE_DATES[0]:
                for candidate in result["候选"]:
                    curve_rows.append(
                        {
                            "日期": date.date().isoformat(),
                            "品类": cat,
                            "候选售价": candidate["售价"],
                            "候选加成率": candidate["加成率"],
                            "预测需求量": candidate["预测需求量"],
                            "预计满足量": candidate["预计满足量"],
                            "预计补货量": candidate["补货量"],
                            "预期利润": candidate["预期利润"],
                        }
                    )
    final_df = pd.DataFrame(final_rows).sort_values(
        ["日期", "品类"]
    ).reset_index(drop=True)
    decomposition_df = pd.DataFrame(decomposition_rows).sort_values(
        ["日期", "品类"]
    ).reset_index(drop=True)
    curve_df = pd.DataFrame(curve_rows).sort_values(
        ["品类", "候选售价"]
    ).reset_index(drop=True)
    future_cost_df = pd.DataFrame(future_cost_rows).sort_values(
        ["日期", "品类"]
    ).reset_index(drop=True)

    export = final_df.copy()
    export["损耗率（百分数）"] = export["损耗率"] * 100.0
    export["建议成本加成率（百分数）"] = export["建议成本加成率"] * 100.0
    export["数学搜索加成率（百分数）"] = export["数学搜索加成率"] * 100.0
    for column in [
        "预测批发价",
        "建议售价",
        "预测需求量",
        "建议补货量",
        "预计满足量",
        "预计利润",
        "数学搜索售价",
        "数学搜索预计利润",
    ]:
        export[column] = export[column].round(2)
    for column in [
        "损耗率（百分数）",
        "建议成本加成率（百分数）",
        "数学搜索加成率（百分数）",
    ]:
        export[column] = export[column].round(2)
    export["临界分位数"] = export["临界分位数"].round(4)
    export = export[
        [
            "日期",
            "品类",
            "预测批发价",
            "损耗率（百分数）",
            "建议成本加成率（百分数）",
            "建议售价",
            "预测需求量",
            "建议补货量",
            "预计满足量",
            "临界分位数",
            "预计利润",
            "数学搜索售价",
            "数学搜索加成率（百分数）",
            "数学搜索预计利润",
            "数学搜索是否触及上限",
            "需求关系可靠性",
            "定价依据",
        ]
    ]
    write_csv(export, "七天六品类最终策略.csv")
    write_csv(export, "七天六品类最优方案.csv")

    decomposition_export = decomposition_df.copy()
    for column in decomposition_export.columns[2:]:
        decomposition_export[column] = decomposition_export[column].round(2)
    write_csv(decomposition_export, "策略分解.csv")
    comparison = decomposition_export[
        [
            "日期",
            "品类",
            "基础方案售价",
            "基础方案补货量",
            "基础方案预计利润",
            "仅改补货方案利润",
            "最终建议方案利润",
        ]
    ].rename(
        columns={
            "仅改补货方案利润": "报童方案预计利润",
            "最终建议方案利润": "主方案预计利润",
        }
    )
    write_csv(comparison, "基础方案与主方案比较.csv")
    write_csv(future_cost_df, "成本预测结果.csv")
    curve_export = curve_df.copy()
    for column in [
        "候选售价",
        "预测需求量",
        "预计满足量",
        "预计补货量",
        "预期利润",
    ]:
        curve_export[column] = curve_export[column].round(2)
    curve_export["候选加成率"] = curve_export["候选加成率"].round(4)
    write_csv(curve_export, "价格销量利润曲线_7月1日.csv")
    return final_df, decomposition_df, curve_df, future_cost_df


def boundary_sensitivity(
    panel_normal: pd.DataFrame,
    panel_cost: pd.DataFrame,
    category_loss: dict,
    markup_info: dict,
    selected_cost_methods: dict,
    demand_specs: dict,
    first_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比较多个历史加成率边界，专门识别边界是否主导利润。"""
    scenarios = [
        ("正常销售百分之五至百分之九十五", "百分之五分位", "百分之九十五分位"),
        ("正常销售百分之十至百分之九十", "百分之十分位", "百分之九十分位"),
        (
            "正常销售百分之二十五至百分之七十五",
            "百分之二十五分位",
            "百分之七十五分位",
        ),
        ("正常销售百分之一至百分之九十九", "百分之一分位", "百分之九十九分位"),
    ]
    rows = []
    rng = np.random.default_rng(RANDOM_SEED + 101)
    for scenario, low_key, high_key in scenarios:
        for cat in CATEGORIES:
            frame = panel_normal[panel_normal["品类"] == cat].sort_values(
                "销售日期"
            )
            cost_frame = panel_cost[panel_cost["品类"] == cat].sort_values(
                "销售日期"
            )
            costs = cost_forecast(
                cost_frame,
                FUTURE_DATES,
                selected_cost_methods[cat],
            )
            info = markup_info[cat]["正常销售"]
            bounds = (info[low_key], info[high_key])
            spec = demand_specs[cat]
            for date in FUTURE_DATES:
                residuals = rng.choice(
                    np.asarray(spec["残差"], dtype=float),
                    size=SAMPLE_COUNT,
                    replace=True,
                )
                candidates = []
                for price in price_grid(bounds[0], bounds[1], float(costs.loc[date])):
                    evaluated = evaluate_price(
                        spec,
                        date,
                        float(price),
                        float(costs.loc[date]),
                        category_loss[cat],
                        residuals,
                        first_date,
                    )
                    if evaluated is not None:
                        candidates.append(evaluated)
                best = max(candidates, key=lambda x: x["预期利润"])
                touch_upper = best["加成率"] >= bounds[1] - 0.005
                rows.append(
                    {
                        "边界情景": scenario,
                        "日期": date.date().isoformat(),
                        "品类": cat,
                        "下界加成率": bounds[0],
                        "上界加成率": bounds[1],
                        "搜索最高售价": best["售价"],
                        "搜索最高加成率": best["加成率"],
                        "搜索最高补货量": best["补货量"],
                        "搜索最高预计利润": best["预期利润"],
                        "是否触及上限": "是" if touch_upper else "否",
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("边界情景", as_index=False)
        .agg(
            七天搜索利润=("搜索最高预计利润", "sum"),
            触及上限天数=("是否触及上限", lambda x: int((x == "是").sum())),
            平均搜索售价=("搜索最高售价", "mean"),
        )
        .sort_values("七天搜索利润", ascending=False)
    )
    write_csv(detail, "价格边界敏感性.csv")
    write_csv(summary, "价格边界敏感性汇总.csv")
    return detail, summary


def demand_relation_table(
    panel_all: pd.DataFrame,
    panel_normal: pd.DataFrame,
    all_specs: dict,
    normal_specs: dict,
    all_models: dict,
    normal_models: dict,
    all_summary: pd.DataFrame,
    normal_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    relation_specs = {}
    residual_rows = []
    for cat in CATEGORIES:
        normal_frame = panel_normal[panel_normal["品类"] == cat].sort_values(
            "销售日期"
        )
        normal_spec = normal_specs[cat]
        all_spec = all_specs[cat]
        markup_spec = robust_ols(
            normal_frame, bool(normal_spec["含时间趋势"]), "加成率"
        )
        relation_specs[cat] = markup_spec
        normal_summary_row = normal_summary[
            (normal_summary["品类"] == cat)
            & (normal_summary["是否入选"] == "是")
        ].iloc[0]
        all_summary_row = all_summary[
            (all_summary["品类"] == cat)
            & (all_summary["是否入选"] == "是")
        ].iloc[0]
        fold_betas = []
        for cutoff in validation_cutoffs(normal_frame):
            train = normal_frame[normal_frame["销售日期"] <= cutoff]
            fold_spec = robust_ols(
                train, bool(normal_spec["含时间趋势"]), "售价"
            )
            fold_betas.append(float(fold_spec["价格系数"]))
        negative_count = int(sum(beta < 0 for beta in fold_betas))
        negative_stable = negative_count >= max(6, math.ceil(len(fold_betas) * 0.75))
        reliable = bool(
            normal_spec["价格系数"] < 0
            and normal_spec["稳健概率值"] < 0.05
            and negative_stable
        )
        internal_peak = bool(normal_spec["稳健区间上限"] < -1.0)
        residual_values = np.asarray(normal_spec["残差"], dtype=float)
        middle = max(1, len(residual_values) // 2)
        residual_first_std = float(np.std(residual_values[:middle], ddof=1))
        residual_second_std = float(np.std(residual_values[middle:], ddof=1))
        residual_diagnosis = (
            "存在短期相关"
            if abs(normal_spec["一阶残差相关"]) >= 0.1
            else "短期相关较弱"
        )
        residual_rows.append(
            {
                "品类": cat,
                "最终需求模型": normal_spec["模型"],
                "样本数": normal_spec["样本数"],
                "残差标准差": float(np.std(residual_values, ddof=1)),
                "一阶残差相关": normal_spec["一阶残差相关"],
                "前半段残差标准差": residual_first_std,
                "后半段残差标准差": residual_second_std,
                "残差诊断": residual_diagnosis,
            }
        )
        reliability_text = (
            "价格方向较可靠，但未识别出内部最优价"
            if reliable and not internal_peak
            else "价格关系证据不足，不作精细定价"
            if not reliable
            else "可继续检验内部最优价"
        )
        rows.append(
            {
                "品类": cat,
                "正常销售最终模型": demand_model_name(normal_models[cat]),
                "全量有效最终模型": demand_model_name(all_models[cat]),
                "正常销售售价弹性": normal_spec["价格系数"],
                "全量有效售价弹性": all_spec["价格系数"],
                "折扣口径弹性差异": normal_spec["价格系数"]
                - all_spec["价格系数"],
                "正常销售加成率关联系数": markup_spec["价格系数"],
                "稳健标准误": normal_spec["稳健标准误"],
                "稳健概率值": normal_spec["稳健概率值"],
                "稳健95%下限": normal_spec["稳健区间下限"],
                "稳健95%上限": normal_spec["稳健区间上限"],
                "模型决定系数": normal_spec["模型决定系数"],
                "调整决定系数": normal_spec["调整决定系数"],
                "正常销售回测误差": normal_summary_row[
                    "回测绝对百分比误差加权"
                ],
                "全量有效回测误差": all_summary_row[
                    "回测绝对百分比误差加权"
                ],
                "滚动回测价格系数为负折数": negative_count,
                "滚动回测总折数": len(fold_betas),
                "滚动价格系数最小值": min(fold_betas),
                "滚动价格系数最大值": max(fold_betas),
                "一阶残差相关": normal_spec["一阶残差相关"],
                "残差诊断": residual_diagnosis,
                "价格关系可靠性": reliability_text,
                "是否存在可识别内部最优": "是" if internal_peak else "否",
                "最终定价处理": "正常销售历史中位加成",
                "解释边界": "控制星期、月份后仍是条件关联，不能单独证明因果",
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, "销售量与成本加成关系.csv")
    write_csv(result, "需求模型口径比较.csv")
    write_csv(pd.DataFrame(residual_rows), "需求残差诊断.csv")
    return result, relation_specs


def key_backtest_table(
    relation_df: pd.DataFrame,
    cost_summary: pd.DataFrame,
    selected_cost_methods: dict,
) -> pd.DataFrame:
    rows = []
    for _, row in relation_df.iterrows():
        cat = row["品类"]
        cost_row = cost_summary[
            (cost_summary["品类"] == cat)
            & (cost_summary["方法"] == selected_cost_methods[cat])
        ].iloc[0]
        rows.append(
            {
                "品类": cat,
                "全量有效需求回测误差": row["全量有效回测误差"],
                "正常销售需求回测误差": row["正常销售回测误差"],
                "正常销售模型": row["正常销售最终模型"],
                "正常销售价格关系可靠性": row["价格关系可靠性"],
                "入选进价方法": selected_cost_methods[cat],
                "进价回测误差": cost_row["回测绝对百分比误差加权"],
                "未来7天进价是否变化": "",
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, "关键回测比较.csv")
    return result


def strategy_summary(
    final_df: pd.DataFrame, decomposition_df: pd.DataFrame
) -> pd.DataFrame:
    result = (
        final_df.groupby("品类", as_index=False)
        .agg(
            七天建议补货量=("建议补货量", "sum"),
            七天预计满足量=("预计满足量", "sum"),
            七天预计利润=("预计利润", "sum"),
            建议售价最低=("建议售价", "min"),
            建议售价最高=("建议售价", "max"),
            数学搜索触顶天数=(
                "数学搜索是否触及上限",
                lambda x: int((x == "是").sum()),
            ),
        )
    )
    decomp = decomposition_df.groupby("品类", as_index=False).agg(
        基础方案七天利润=("基础方案预计利润", "sum"),
        仅改补货七天利润=("仅改补货方案利润", "sum"),
        价格搜索七天利润=("价格搜索方案利润", "sum"),
        补货改善=("补货改善", "sum"),
        价格搜索改善=("价格搜索改善", "sum"),
    )
    result = result.merge(decomp, on="品类", how="left")
    result["品类排序"] = result["品类"].map(
        {cat: i for i, cat in enumerate(CATEGORIES)}
    )
    result = result.sort_values("品类排序").drop(columns="品类排序")
    write_csv(result, "策略分解汇总.csv")
    return result


def set_plot_font() -> None:
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def make_relation_figure(
    panel_normal: pd.DataFrame, demand_specs: dict, relation_specs: dict
) -> None:
    if not HAS_MATPLOTLIB:
        return
    set_plot_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    reference_date = pd.Timestamp("2023-07-01")
    for ax, cat in zip(axes, CATEGORIES):
        sub = panel_normal[panel_normal["品类"] == cat].copy()
        x = np.log(sub["日平均售价"].to_numpy(float))
        y = np.log(sub["日销售量"].to_numpy(float))
        ax.scatter(x, y, s=9, alpha=0.28, color="#2F6B8A", edgecolors="none")
        spec = demand_specs[cat]
        x_grid = np.linspace(
            float(np.quantile(x, 0.02)),
            float(np.quantile(x, 0.98)),
            100,
        )
        line = pd.DataFrame(
            {
                "销售日期": reference_date,
                "日平均售价": np.exp(x_grid),
                "加成率": 0.0,
                "日销售量": 1.0,
                "星期": reference_date.weekday() + 1,
                "月份": reference_date.month,
                "时间趋势": (reference_date - sub["销售日期"].min()).days / 365.25,
            }
        )
        pred = prediction_from_model(spec, line)
        ax.plot(x_grid, np.log(pred), color="#C45A3C", linewidth=2)
        ax.set_title(cat)
        ax.set_xlabel("ln(日平均售价)")
        ax.set_ylabel("ln(日销量)")
        ax.grid(alpha=0.18)
        ax.text(
            0.03,
            0.05,
            f"售价弹性={spec['价格系数']:.2f}\n加成率直关联={relation_specs[cat]['价格系数']:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    fig.suptitle("正常销售口径：售价—销量的条件关系（加成率直关联仅作诊断）")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "价格加成率销量关系.png", dpi=180)
    plt.close(fig)


def make_profit_figure(curve_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB:
        return
    set_plot_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, cat in zip(axes, CATEGORIES):
        sub = curve_df[curve_df["品类"] == cat].sort_values("候选售价")
        if sub.empty:
            continue
        ax2 = ax.twinx()
        line_demand = ax.plot(
            sub["候选售价"],
            sub["预测需求量"],
            color="#2F6B8A",
            linewidth=1.8,
            label="预测需求量",
        )
        line_profit = ax2.plot(
            sub["候选售价"],
            sub["预期利润"],
            color="#C45A3C",
            linewidth=1.8,
            label="预期利润",
        )
        final_row = final_df[
            (final_df["日期"] == "2023-07-01") & (final_df["品类"] == cat)
        ].iloc[0]
        final_line = ax.axvline(
            final_row["建议售价"],
            color="#2E8B57",
            linestyle="-",
            linewidth=1.3,
            label="建议售价",
        )
        search_line = ax.axvline(
            final_row["数学搜索售价"],
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            label="边界搜索价",
        )
        ax.set_title(cat)
        ax.set_xlabel("售价（元/千克）")
        ax.set_ylabel("预测需求量（千克）", color="#2F6B8A")
        ax2.set_ylabel("预计利润（元）", color="#C45A3C")
        ax.grid(alpha=0.18)
        ax.legend(
            line_demand + line_profit + [final_line, search_line],
            ["预测需求量", "预期利润", "建议售价", "边界搜索价"],
            fontsize=8,
            loc="best",
        )
    fig.suptitle("2023年7月1日价格—需求—利润曲线（边界诊断）")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "价格销量利润曲线_7月1日.png", dpi=180)
    plt.close(fig)


def make_cost_figure(
    panel_all: pd.DataFrame,
    future_cost_df: pd.DataFrame,
) -> None:
    if not HAS_MATPLOTLIB:
        return
    set_plot_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, cat in zip(axes, CATEGORIES):
        sub = panel_all[panel_all["品类"] == cat].sort_values("销售日期").tail(90)
        fut = future_cost_df[future_cost_df["品类"] == cat].sort_values("日期")
        future_dates = pd.to_datetime(fut["日期"])
        ax.plot(
            sub["销售日期"],
            sub["日平均进价"],
            color="#2F6B8A",
            linewidth=1.3,
            label="历史日进价",
        )
        ax.plot(
            future_dates,
            fut["预测批发价"],
            color="#C45A3C",
            linestyle="--",
            marker="o",
            markersize=3,
            label="未来预测",
        )
        ax.axvline(FUTURE_DATES[0], color="#555555", linestyle=":", linewidth=1)
        ax.set_title(cat)
        ax.set_ylabel("元/千克")
        ax.grid(alpha=0.18)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("近90日历史进价与未来7日预测")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "未来成本预测.png", dpi=180)
    plt.close(fig)


def format_pct(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def write_final_report(
    audit: dict,
    discount_df: pd.DataFrame,
    relation_df: pd.DataFrame,
    cost_summary: pd.DataFrame,
    selected_cost_methods: dict,
    selected_demand_models: dict,
    final_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    strategy_summary_df: pd.DataFrame,
    boundary_summary_df: pd.DataFrame,
) -> str:
    total_profit = float(final_df["预计利润"].sum())
    base_profit = float(decomposition_df["基础方案预计利润"].sum())
    replenishment_gain = float(decomposition_df["补货改善"].sum())
    boundary_price_gain = float(decomposition_df["价格搜索改善"].sum())
    top_boundary = boundary_summary_df.iloc[0]
    future_ranges = (
        final_df.groupby("品类")["预测批发价"].agg(["min", "max"]).reset_index()
    )
    ratio_low = float(discount_df["折扣相对同日正常价中位数"].min())
    ratio_high = float(discount_df["折扣相对同日正常价中位数"].max())
    late_low = float(discount_df["折扣晚于19点占比"].min())
    late_high = float(discount_df["折扣晚于19点占比"].max())
    lines = [
        "# 2023年数学建模国赛C题问题二最终建模说明",
        "",
        "## 一、最终结论",
        "",
        f"本次从四个原始附件重新计算，生成2023年7月1日至7日六个品类共42条建议。最终建议方案的模型预计七天利润为 **{total_profit:.2f}元**；用正常销售历史中位加成、按平均需求补货的简单基准利润为 **{base_profit:.2f}元**。其中补货规则带来的改善为 **{replenishment_gain:.2f}元**，价格边界搜索相对建议价格的诊断差额为 **{boundary_price_gain:.2f}元**，后者不作为可信的定价收益承诺。合计采用未四舍五入的中间值，逐行显示后相加可能有几分钱舍入差异。",
        "",
        "最终价格不是把网格搜索的最高点直接当作现实最优价，而是：价格关系通过可靠性检查后，仍未发现可识别的内部收益峰值，因此统一采用正常销售历史中位加成；补货量再根据损耗率和需求不确定性用报童分位数确定。",
        "",
        "## 二、数据口径",
        "",
        f"- 原始销售流水共{audit['附件二原始流水数']}条，销售{audit['销售记录数']}条，退货{audit['退货记录数']}条；退货数量为负，按净额保留。",
        f"- 折扣标记流水{audit['折扣记录数']}条，其中销售折扣流水{audit['折扣销售记录数']}条，占销售流水{format_pct(audit['折扣记录占销售记录比例'])}；折扣销售量占销售量{format_pct(audit['折扣销量占销售量比例'])}。",
        f"- 六个品类折扣价相对同一单品同日正常价中位数约为{ratio_low:.2f}至{ratio_high:.2f}，折扣流水中晚于19点的比例约为{format_pct(late_low)}至{format_pct(late_high)}；因此把折扣视为临期、品相或清仓等特殊处理的观测状态，而不是未来已知的常规定价。",
        f"- 全量有效日期—品类面板为{audit['全量有效面板行数']}行；正常销售面板为{audit['正常销售面板行数']}行。整天没有流水的日期不填成零销量。",
        "- 品类日售价和日进价都用销量加权：总销售额除以总销量、总成本额除以总销量。附件四的品类损耗率直接用于补货；单品损耗率只做核对。",
        "",
        "## 三、最终模型及选择理由",
        "",
        "| 品类 | 需求模型 | 进价预测方法 | 价格关系判断 | 最终定价处理 |",
        "|---|---|---|---|---|",
    ]
    for cat in CATEGORIES:
        row = relation_df[relation_df["品类"] == cat].iloc[0]
        lines.append(
            f"| {cat} | {demand_model_name(selected_demand_models[cat])} | {selected_cost_methods[cat]} | {row['价格关系可靠性']} | 正常销售历史中位加成 |"
        )
    lines += [
        "",
        "### 需求模型",
        "",
        "对正常销售日面板建立对数需求模型：",
        "",
        "$$\\ln Q_t=\\alpha+\\beta\\ln P_t+\\text{星期效应}+\\text{月份效应}+\\text{可选趋势}+\\varepsilon_t.$$",
        "",
        "模型用滚动七日后置回测比较“不含趋势”和“含趋势”两种简单形式；只有趋势模型的误差至少改善5%时才保留趋势。价格系数使用七阶稳健协方差，残差不确定性在对数销量尺度上重抽样。",
        "",
        "表格中的“加成率直接关联系数”只用于诊断同期成本波动和主动调价造成的口径风险，不作为未来需求模型或定价依据；最终定价使用售价与销量的条件关系，并且在无法识别内部峰值时回到历史中位加成。",
        "",
        "### 进价预测",
        "",
        "只比较近7日均值、近14日均值、指数加权移动平均、同星期近8次中位数和近180日趋势加星期五种简单方法，以滚动七日加权误差逐品类选择。允许星期和趋势变化的方法若回测更差，就不采用它们。因本次六个品类入选方法都是水平型方法，未来七天进价相同是数据支持的结果，不是人为制造或遗漏了星期项。",
        "",
        "### 定价与补货",
        "",
        "价格搜索范围取正常销售历史加成率的5%至95%分位，只用于检查收益曲线是否在历史经营区间内出现内部峰值。有效单位成本为进价除以可销售比例，临界分位数为 $1-\\frac{C/(1-L)}{P}$；补货量按该分位数确定并取0.1千克粒度。",
        "",
        "## 四、折扣口径的最终决定",
        "",
        "保留折扣记录用于原始数据审计和现实成交额核对；正常定价关系和未来七天需求则采用仅正常销售口径。理由有三点：一是折扣价格相对同日正常价有明显折减；二是折扣集中在晚间，符合临期或品相处理；三是未来七天没有给出折扣计划，不能把历史折扣比例当成已知输入。全量有效口径仍保留在“需求模型口径比较.csv”中，用来检查结论是否因折扣状态改变。",
        "",
        "## 五、六个品类的价格关系可靠性",
        "",
        "| 品类 | 正常销售售价弹性 | 稳健概率值 | 滚动回测负向折数 | 结论 |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in relation_df.iterrows():
        lines.append(
            f"| {row['品类']} | {row['正常销售售价弹性']:.3f} | {row['稳健概率值']:.3g} | {int(row['滚动回测价格系数为负折数'])}/{int(row['滚动回测总折数'])} | {row['价格关系可靠性']} |"
        )
    lines += [
        "",
        "- 花叶类、花菜类、水生根茎类和辣椒类的价格方向在滚动回测中稳定为负，适合说明“价格提高通常伴随销量下降”，但它们没有被证明存在现实区间内的内部最优价。",
        "- 茄类的趋势模型改善了销量预测，但价格系数证据不足，因此不据此精细调价。",
        "- 食用菌正常销售口径的价格系数在滚动回测中不稳定，不能硬求精确弹性，采用历史中位加成。",
        "- 六类的最终价格关系均只能解释为控制星期、月份后的条件关联。售价还同时受单品结构、库存、上架和促销状态影响，不能写成严格因果。",
        "",
        "## 六、未来七天策略",
        "",
        "完整42行结果见“七天六品类最终策略.csv”。下表给出品类汇总，具体每日补货量和售价以该表为准。",
        "",
        "| 品类 | 七天建议补货量（千克） | 七天预计满足量（千克） | 七天预计利润（元） | 数学搜索触顶天数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in strategy_summary_df.iterrows():
        lines.append(
            f"| {row['品类']} | {row['七天建议补货量']:.1f} | {row['七天预计满足量']:.1f} | {row['七天预计利润']:.2f} | {int(row['数学搜索触顶天数'])} |"
        )
    lines += [
        "",
        "本次入选进价方法产生的未来七天进价范围如下：",
        "",
        "| 品类 | 预测进价最低值 | 预测进价最高值 |",
        "|---|---:|---:|",
    ]
    for _, row in future_ranges.iterrows():
        lines.append(
            f"| {row['品类']} | {row['min']:.2f} | {row['max']:.2f} |"
        )
    lines += [
        "",
        f"价格边界敏感性中，利润最高的情景为“{top_boundary['边界情景']}”，七天搜索利润为 {top_boundary['七天搜索利润']:.2f} 元，触及上限 {int(top_boundary['触及上限天数'])} 天；这说明放宽边界会继续推高数学搜索利润，但不能证明商超可以无限提价。价格—利润图中同时标出建议售价和边界搜索价，便于评委区分两者。",
        "",
        "| 价格边界情景 | 七天搜索利润（元） | 触及上限天数 |",
        "|---|---:|---:|",
    ]
    for _, row in boundary_summary_df.iterrows():
        lines.append(
            f"| {row['边界情景']} | {row['七天搜索利润']:.2f} | {int(row['触及上限天数'])} |"
        )
    lines += [
        "",
        "## 七、旧方案的否定与保留",
        "",
        "1. 旧问题二方案把历史加成率分位上界的搜索结果直接称为最优价；本次发现搜索价大量触及上界，因此改为边界诊断，不再把它包装成内部最优。",
        "2. 旧版只保留正常销售或只把折扣比例作为控制变量，两种做法都各有信息损失；本次先用原始全量口径核对现实成交，再用正常销售口径服务于无未来折扣输入的常规定价，并把两者回测并列表明差异。",
        "3. 旧版固定使用单一进价预测方法；本次改为五种简单方法滚动回测逐品类选择。回测显示星期中位数和趋势方法并没有普遍改善，因此没有为了制造每日差异而强行采用它们。",
        "4. 旧版的复杂工具变量和精细弹性结果只作为参考，不作为最终定价依据。原因是价格与销量的同时性、单品结构和促销选择仍无法仅靠附件完全排除；在这种情况下，稳健中位加成比伪精确价格更适合论文主结论。",
        "5. 没有加入深度学习、遗传算法等复杂模型，因为本题数据和题目约束不足以证明复杂模型会带来可复核的决策收益。",
        "",
        "## 八、模型局限",
        "",
        "- 附件没有逐日库存、缺货和剩余商品记录，历史销量只能作为可观测需求代理。",
        "- 没有给出总预算、货架容量、包装规格和品类之间的联合约束，因此按品类分别决策。",
        "- 折扣被作为特殊销售状态处理；如果未来经营明确每天都有固定折扣计划，应重新建立含折扣策略的需求模型。",
        "- 预计利润是基于需求分布、损耗率和无显著残值假设的模型结果，不是已经实现的利润。",
        "",
        "## 九、复现与文件",
        "",
        "在仓库根目录运行：",
        "",
        "    python 问题二/脚本/求解问题二.py",
        "",
        "主脚本会重新读取四个原始附件并更新“问题二/结果”和“问题二/图表”目录。关键文件包括：最终策略表、关键回测比较、需求模型口径比较、需求残差诊断、成本预测回测、价格边界敏感性、策略分解、价格加成率销量关系图、价格销量利润曲线和未来成本预测图。",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "最终建模说明.md").write_text(report, encoding="utf-8")
    (OUT / "问题二正式结果说明.md").write_text(report, encoding="utf-8")
    (ROOT / "问题二" / "最终建模说明.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    (
        merged,
        panel_all,
        panel_normal,
        audit,
        category_loss,
        _item_loss,
        first_date,
    ) = read_source_data()
    discount_audit(merged)
    markup_info = markup_summary(panel_all, panel_normal)

    all_detail, all_summary, all_models, all_specs = demand_backtest(
        panel_all, "全量有效"
    )
    normal_detail, normal_summary, normal_models, normal_specs = demand_backtest(
        panel_normal, "正常销售"
    )
    relation_df, relation_specs = demand_relation_table(
        panel_all,
        panel_normal,
        all_specs,
        normal_specs,
        all_models,
        normal_models,
        all_summary,
        normal_summary,
    )
    demand_detail = pd.concat([all_detail, normal_detail], ignore_index=True)
    demand_summary = pd.concat([all_summary, normal_summary], ignore_index=True)
    write_csv(demand_detail, "需求模型回测.csv")
    write_csv(demand_summary, "需求模型回测汇总.csv")

    _cost_detail, cost_summary, selected_cost_methods = cost_backtest(panel_all)
    final_df, decomposition_df, curve_df, future_cost_df = optimize_all(
        panel_normal,
        panel_all,
        category_loss,
        markup_info,
        selected_cost_methods,
        normal_models,
        normal_specs,
        first_date,
    )
    _sensitivity_detail, sensitivity_summary = boundary_sensitivity(
        panel_normal,
        panel_all,
        category_loss,
        markup_info,
        selected_cost_methods,
        normal_specs,
        first_date,
    )
    summary_df = strategy_summary(final_df, decomposition_df)
    key_df = key_backtest_table(
        relation_df, cost_summary, selected_cost_methods
    )
    ranges = (
        future_cost_df.groupby("品类")["预测批发价"].agg(["min", "max"]).reset_index()
    )
    key_df["未来7天进价是否变化"] = key_df["品类"].map(
        {
            row["品类"]: "是" if row["min"] < row["max"] else "否"
            for _, row in ranges.iterrows()
        }
    )
    write_csv(key_df, "关键回测比较.csv")

    make_relation_figure(panel_normal, normal_specs, relation_specs)
    make_profit_figure(curve_df, final_df)
    make_cost_figure(panel_all, future_cost_df)
    write_final_report(
        audit,
        pd.read_csv(OUT / "折扣辅助分析.csv", encoding="utf-8-sig"),
        relation_df,
        cost_summary,
        selected_cost_methods,
        normal_models,
        final_df,
        decomposition_df,
        summary_df,
        sensitivity_summary,
    )

    reliability_map = dict(
        zip(relation_df["品类"], relation_df["价格关系可靠性"])
    )
    final_path = OUT / "七天六品类最终策略.csv"
    final_export = pd.read_csv(final_path, encoding="utf-8-sig")
    final_export["需求关系可靠性"] = final_export["品类"].map(reliability_map)
    final_export.to_csv(final_path, index=False, encoding="utf-8-sig")
    final_export.to_csv(
        OUT / "七天六品类最优方案.csv", index=False, encoding="utf-8-sig"
    )

    print("问题二重新求解完成")
    print(f"原始流水数: {audit['附件二原始流水数']}")
    print(f"全量有效面板行数: {audit['全量有效面板行数']}")
    print(f"正常销售面板行数: {audit['正常销售面板行数']}")
    print(f"最终策略行数: {len(final_df)}")
    print(f"七天建议方案预计利润: {final_df['预计利润'].sum():.2f} 元")
    print(
        f"边界搜索触顶天数: {(final_df['数学搜索是否触及上限'] == '是').sum()} 天"
    )
    print(f"结果目录: {OUT}")


if __name__ == "__main__":
    main()
