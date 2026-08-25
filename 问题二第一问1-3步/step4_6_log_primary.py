# -*- coding: utf-8 -*-
"""问题二第一问：统一对数主模型试验版。"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson, jarque_bera

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
TRIAL = OUT / "step4_6_log_primary"
TRIAL.mkdir(parents=True, exist_ok=True)
CATEGORIES = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
BASE_FEATURES = ["日平均售价"] + [f"月份_{m:02d}" for m in range(2, 13)] + [f"星期_{w}" for w in range(2, 8)]
HAC_LAGS = 7


def num(value, digits=3):
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def ptxt(value):
    if value is None or not np.isfinite(float(value)):
        return "—"
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def design(frame, log_model, trend=False):
    cols = list(BASE_FEATURES)
    if trend:
        cols.append("时间趋势")
    x = frame[cols].astype(float).copy()
    if log_model:
        x["日平均售价"] = np.log(x["日平均售价"])
    return sm.add_constant(x, has_constant="add")


def robust(model):
    try:
        return model.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS, use_correction=True)
    except TypeError:
        return model.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)


def fit(frame, log_model, trend=False):
    y_raw = frame["日总销量"].astype(float)
    x = design(frame, log_model, trend)
    y = np.log(y_raw) if log_model else y_raw
    model = sm.OLS(y, x).fit()
    hac = robust(model)
    names = list(x.columns)
    idx = names.index("日平均售价")
    params = np.asarray(model.params, float)
    robust_se = np.asarray(hac.bse, float)
    robust_p = np.asarray(hac.pvalues, float)
    ci = np.asarray(hac.conf_int(), float)
    smear = float(np.exp(model.resid).mean()) if log_model else 1.0
    fitted = np.exp(model.fittedvalues) * smear if log_model else model.fittedvalues
    row = {
        "模型": "对数主模型" if log_model and not trend else "线性对照模型" if not log_model else "对数趋势敏感性",
        "售价效应": float(params[idx]),
        "稳健标准误": float(robust_se[idx]),
        "稳健p值": float(robust_p[idx]),
        "稳健95%下限": float(ci[idx, 0]),
        "稳健95%上限": float(ci[idx, 1]),
        "模型R²": float(model.rsquared),
        "调整R²": float(model.rsquared_adj),
        "模型整体F检验p值": float(model.f_pvalue),
        "样本数": int(len(frame)),
    }
    coeffs = []
    for i, name in enumerate(names):
        coeffs.append({
            "模型": row["模型"],
            "变量": "截距" if name == "const" else name,
            "系数": float(params[i]),
            "经典p值": float(model.pvalues[i]),
            "稳健标准误": float(robust_se[i]),
            "稳健p值": float(robust_p[i]),
            "稳健95%下限": float(ci[i, 0]),
            "稳健95%上限": float(ci[i, 1]),
        })
    return {
        "model": model,
        "row": row,
        "coeffs": coeffs,
        "resid": np.asarray(model.resid, float),
        "fitted": np.asarray(fitted, float),
        "smear": smear,
        "log_model": log_model,
        "trend": trend,
    }


def validate(frame, log_model, trend=False):
    cut = min(max(int(len(frame) * 0.8), 30), len(frame) - 10)
    train = frame.iloc[:cut]
    test = frame.iloc[cut:]
    spec = fit(train, log_model, trend)
    pred_response = np.asarray(spec["model"].predict(design(test, log_model, trend)), float)
    pred = np.exp(pred_response) * spec["smear"] if log_model else pred_response
    actual = test["日总销量"].to_numpy(float)
    error = actual - pred
    return {
        "验证起点": str(test["销售日期"].iloc[0].date()),
        "验证RMSE": float(np.sqrt(np.mean(error ** 2))),
        "验证MAE": float(np.mean(np.abs(error))),
        "验证MAPE": float(np.mean(np.abs(error) / actual) * 100),
    }


def diagnose(category, frame, spec):
    resid = spec["resid"]
    n = len(resid)
    lag = min(7, n // 5)
    lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
    _, _, _, bp_p = het_breuschpagan(resid, design(frame, spec["log_model"], spec["trend"]))
    _, jb_p, skew, kurtosis = jarque_bera(resid)
    alerts = []
    if float(lb["lb_pvalue"].iloc[0]) < 0.05:
        alerts.append("残差存在短期自相关")
    if bp_p < 0.05:
        alerts.append("残差方差可能不恒定")
    time_corr = float(spearmanr(np.arange(n), resid).statistic)
    if abs(time_corr) >= 0.20:
        alerts.append("残差仍有时间趋势")
    return {
        "分类": category,
        "杜宾沃森统计量": float(durbin_watson(resid)),
        "七阶滞后检验p值": float(lb["lb_pvalue"].iloc[0]),
        "异方差检验p值": float(bp_p),
        "正态性检验p值": float(jb_p),
        "残差偏度": float(skew),
        "残差峰度": float(kurtosis),
        "残差时间秩相关": time_corr,
        "诊断提示": "；".join(alerts) if alerts else "未发现需要优先处理的诊断信号",
    }


panel = pd.read_csv(OUT / "品类日销售面板.csv", encoding="utf-8-sig")
panel["销售日期"] = pd.to_datetime(panel["销售日期"])
panel = panel.dropna(subset=["日总销量", "日平均售价"]).copy()
if (panel["日总销量"] <= 0).any() or (panel["日平均售价"] <= 0).any():
    raise ValueError("销量和售价必须为正")

main_rows = []
linear_rows = []
trend_rows = []
main_coeffs = []
main_specs = {}
linear_specs = {}
trend_specs = {}
fit_rows = []
diagnostic_rows = []

for category in CATEGORIES:
    frame = panel.loc[panel["分类名称"] == category].sort_values("销售日期").copy()
    frame["时间趋势"] = np.arange(len(frame), dtype=float)

    main = fit(frame, True, False)
    main.update(validate(frame, True, False))
    main["分类"] = category
    main_rows.append({"分类": category, **main["row"], **{k: v for k, v in main.items() if k in ["验证起点", "验证RMSE", "验证MAE", "验证MAPE"]}})
    main_coeffs.extend({"分类": category, **item} for item in main["coeffs"])
    main_specs[category] = main

    linear = fit(frame, False, False)
    linear.update(validate(frame, False, False))
    linear["分类"] = category
    linear_rows.append({"分类": category, **linear["row"], **{k: v for k, v in linear.items() if k in ["验证起点", "验证RMSE", "验证MAE", "验证MAPE"]}})
    linear_specs[category] = linear

    trend = fit(frame, True, True)
    trend.update(validate(frame, True, True))
    trend["分类"] = category
    trend_rows.append({"分类": category, **trend["row"], **{k: v for k, v in trend.items() if k in ["验证起点", "验证RMSE", "验证MAE", "验证MAPE"]}})
    trend_specs[category] = trend

    diagnostic_rows.append(diagnose(category, frame, main))
    for i in range(len(frame)):
        fit_rows.append({
            "销售日期": frame["销售日期"].iloc[i].date().isoformat(),
            "分类": category,
            "实际日销量": float(frame["日总销量"].iloc[i]),
            "主模型拟合日销量": float(main["fitted"][i]),
            "主模型销量尺度残差": float(frame["日总销量"].iloc[i] - main["fitted"][i]),
            "主模型残差": float(main["resid"][i]),
        })

main_df = pd.DataFrame(main_rows)
linear_df = pd.DataFrame(linear_rows)
trend_df = pd.DataFrame(trend_rows)
coeff_df = pd.DataFrame(main_coeffs)
diagnostic_df = pd.DataFrame(diagnostic_rows)
fit_df = pd.DataFrame(fit_rows)

_, q, _, _ = multipletests(main_df["稳健p值"].to_numpy(float), alpha=0.05, method="fdr_bh")
main_df["校正后q值"] = q
main_df["售价检验结论"] = np.where(q < 0.05, "通过多重检验校正", "未通过多重检验校正")

comparison_df = pd.concat([
    main_df.assign(比较用途="主模型"),
    linear_df.assign(比较用途="线性对照"),
    trend_df.assign(比较用途="时间趋势敏感性"),
], ignore_index=True)

local_rows = []
for _, row in main_df.iterrows():
    category = row["分类"]
    frame = panel.loc[panel["分类名称"] == category]
    mean_price = float(frame["日平均售价"].mean())
    mean_quantity = float(frame["日总销量"].mean())
    local_change = mean_price / (float(row["售价效应"]) * mean_quantity)
    local_rows.append({
        "分类": category,
        "参考日平均售价": mean_price,
        "参考日销量": mean_quantity,
        "价格弹性": float(row["售价效应"]),
        "销量增加1千克对应的局部价格变化": local_change,
        "校正后q值": float(row["校正后q值"]),
        "解释边界": "统一对数需求模型的局部反推，不是成本加成定价结论",
    })
local_df = pd.DataFrame(local_rows)

main_df.to_csv(TRIAL / "主模型_对数多元回归.csv", index=False, encoding="utf-8-sig")
linear_df.to_csv(TRIAL / "对照模型_线性多元回归.csv", index=False, encoding="utf-8-sig")
trend_df.to_csv(TRIAL / "敏感性_加入时间趋势.csv", index=False, encoding="utf-8-sig")
comparison_df.to_csv(TRIAL / "模型形式比较.csv", index=False, encoding="utf-8-sig")
coeff_df.to_csv(TRIAL / "主模型_系数明细.csv", index=False, encoding="utf-8-sig")
diagnostic_df.to_csv(TRIAL / "主模型_残差诊断.csv", index=False, encoding="utf-8-sig")
fit_df.to_csv(TRIAL / "主模型_拟合值.csv", index=False, encoding="utf-8-sig")
local_df.to_csv(TRIAL / "销量增加1千克的局部价格变化.csv", index=False, encoding="utf-8-sig")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# 三种形式的原始销量验证误差
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CATEGORIES))
width = 0.25
for offset, name, frame in [
    (-width, "对数主模型", main_df),
    (0, "线性对照模型", linear_df),
    (width, "对数趋势敏感性", trend_df),
]:
    ax.bar(x + offset, frame["验证RMSE"], width, label=name)
ax.set_xticks(x, CATEGORIES)
ax.set_ylabel("后20%日期验证集 RMSE（千克）")
ax.set_title("统一对数主模型与对照模型的原始销量预测误差")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(TRIAL / "模型形式比较.png", dpi=160)
plt.close(fig)

# 对数主模型价格弹性
fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(CATEGORIES))
values = main_df["售价效应"].to_numpy()
lower = main_df["稳健95%下限"].to_numpy()
upper = main_df["稳健95%上限"].to_numpy()
ax.errorbar(values, y, xerr=[values - lower, upper - values], fmt="o", color="#2F5D8C", ecolor="#7A9EBC", capsize=3)
ax.axvline(0, color="#B22222", linestyle="--", linewidth=0.8)
ax.set_yticks(y, CATEGORIES)
ax.set_title("统一对数主模型的价格弹性及稳健95%区间")
ax.set_xlabel("价格弹性")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(TRIAL / "主模型_价格弹性.png", dpi=160)
plt.close(fig)

# 实际值和拟合值
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, category in zip(axes.flat, CATEGORIES):
    sub = fit_df.loc[fit_df["分类"] == category]
    actual = sub["实际日销量"].to_numpy()
    fitted = sub["主模型拟合日销量"].to_numpy()
    limit = max(float(actual.max()), float(fitted.max())) * 1.05
    ax.scatter(actual, fitted, s=8, alpha=0.42, color="#4C72B0")
    ax.plot([0, limit], [0, limit], color="#B22222", linestyle="--", linewidth=0.9)
    row = main_df.loc[main_df["分类"] == category].iloc[0]
    ax.set_title(category)
    ax.set_xlabel("实际日销量（千克）")
    ax.set_ylabel("对数主模型拟合值（千克）")
    ax.text(0.04, 0.94, f"验证RMSE={num(row['验证RMSE'], 1)}\n调整R²={num(row['调整R²'], 3)}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.grid(alpha=0.18)
fig.suptitle("统一对数主模型的实际值与拟合值", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(TRIAL / "主模型_实际值拟合值.png", dpi=160)
plt.close(fig)

# 残差时间序列
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, category in zip(axes.flat, CATEGORIES):
    sub = fit_df.loc[fit_df["分类"] == category]
    ax.plot(pd.to_datetime(sub["销售日期"]), sub["主模型残差"], linewidth=0.65, color="#2F5D8C")
    ax.axhline(0, color="#B22222", linestyle="--", linewidth=0.8)
    ax.set_title(category)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(alpha=0.18)
fig.suptitle("统一对数主模型的残差时间序列", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(TRIAL / "主模型_残差时间序列.png", dpi=160)
plt.close(fig)

lines = [
    "# 问题二第一问第四至第六步正式结果",
    "",
    "本目录是问题二第一问第四至第六步的当前正式主模型结果。六个品类统一采用对数—对数多元回归，线性多元回归作为对照，时间趋势模型作为敏感性分析。",
    "",
    "## 建模口径",
    "",
    "六个品类统一采用对数—对数多元回归作为主模型，控制月份和星期；线性多元回归作为对照；另增加时间趋势敏感性模型。",
    "",
    "主模型的价格系数解释为价格弹性：价格上涨1%，预测销量平均变化约为该系数对应的百分比。显著性使用七阶稳健标准误，并对六个品类的售价检验进行多重检验校正。",
    "",
    "| 品类 | 价格弹性 | 校正后q值 | 验证RMSE | 调整R² |",
    "|---|---:|---:|---:|---:|",
]
for _, row in main_df.iterrows():
    lines.append(f"| {row['分类']} | {num(row['售价效应'], 4)} | {ptxt(row['校正后q值'])} | {num(row['验证RMSE'], 2)} | {num(row['调整R²'], 3)} |")
lines += [
    "",
    "## 局部价格反推",
    "",
    "在各品类平均价格和平均销量附近，销量增加1千克对应的价格变化见单独结果表。负值表示需求模型局部反推需要降低价格；这不是成本加成定价。",
    "",
    "## 诊断边界",
    "",
    "主模型仍需结合残差诊断解释。如果存在自相关或异方差，应使用稳健标准误；当前模型仍然不能单独证明价格造成销量变化。",
]
(TRIAL / "正式结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("统一对数主模型试验完成")
for _, row in main_df.iterrows():
    print(f"{row['分类']}：弹性={num(row['售价效应'], 4)}，校正后q={ptxt(row['校正后q值'])}，验证RMSE={num(row['验证RMSE'], 2)}")
