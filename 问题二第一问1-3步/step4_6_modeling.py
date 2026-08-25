# -*- coding: utf-8 -*-
"""问题二第一问第四至第六步：回归、检验、诊断和图表。"""

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
RUN = OUT / "step4_6"
RUN.mkdir(parents=True, exist_ok=True)
CATEGORIES = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
FEATURES = ["日平均售价"] + [f"月份_{m:02d}" for m in range(2, 13)] + [f"星期_{w}" for w in range(2, 8)]
HAC_LAGS = 7


def number(value, digits=3):
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def pvalue(value):
    if value is None or not np.isfinite(float(value)):
        return "—"
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def design(frame, log_price):
    x = frame[FEATURES].astype(float).copy()
    if log_price:
        x["日平均售价"] = np.log(x["日平均售价"])
    return sm.add_constant(x, has_constant="add")


def hac_cov(model):
    try:
        return model.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS, use_correction=True)
    except TypeError:
        return model.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)


def fit_model(frame, log_model):
    y_raw = frame["日总销量"].astype(float)
    x = design(frame, log_model)
    y = np.log(y_raw) if log_model else y_raw
    model = sm.OLS(y, x).fit()
    hac = hac_cov(model)
    names = list(x.columns)
    price_i = names.index("日平均售价")
    params = np.asarray(model.params, dtype=float)
    robust_params = np.asarray(hac.params, dtype=float)
    robust_se = np.asarray(hac.bse, dtype=float)
    robust_p = np.asarray(hac.pvalues, dtype=float)
    robust_ci = np.asarray(hac.conf_int(), dtype=float)
    smearing = float(np.exp(model.resid).mean()) if log_model else 1.0
    fitted = np.exp(model.fittedvalues) * smearing if log_model else model.fittedvalues
    try:
        vif = float(variance_inflation_factor(x.drop(columns=["const"]).to_numpy(), 0))
    except Exception:
        vif = np.nan
    model_name = "对数模型" if log_model else "线性模型"
    row = {
        "模型": model_name,
        "响应尺度": "对数销量" if log_model else "日总销量",
        "售价效应类型": "售价弹性" if log_model else "售价系数",
        "售价效应": float(params[price_i]),
        "稳健标准误": float(robust_se[price_i]),
        "稳健p值": float(robust_p[price_i]),
        "稳健95%下限": float(robust_ci[price_i, 0]),
        "稳健95%上限": float(robust_ci[price_i, 1]),
        "经典p值": float(model.pvalues[price_i]),
        "模型R²": float(model.rsquared),
        "调整R²": float(model.rsquared_adj),
        "模型AIC": float(model.aic),
        "模型BIC": float(model.bic),
        "模型整体F检验p值": float(model.f_pvalue),
        "售价方差膨胀因子": vif,
        "样本数": int(len(frame)),
    }
    coeffs = []
    for i, name in enumerate(names):
        coeffs.append({
            "模型": model_name,
            "变量": "截距" if name == "const" else name,
            "系数": float(params[i]),
            "经典标准误": float(model.bse[i]),
            "经典p值": float(model.pvalues[i]),
            "稳健标准误": float(robust_se[i]),
            "稳健p值": float(robust_p[i]),
            "稳健95%下限": float(robust_ci[i, 0]),
            "稳健95%上限": float(robust_ci[i, 1]),
        })
    return {
        "model": model,
        "row": row,
        "coeffs": coeffs,
        "resid": np.asarray(model.resid, dtype=float),
        "fitted_sales": np.asarray(fitted, dtype=float),
        "smearing": smearing,
        "log_model": log_model,
    }


def validate(frame, log_model):
    cut = min(max(int(len(frame) * 0.8), 30), len(frame) - 10)
    train = frame.iloc[:cut]
    test = frame.iloc[cut:]
    spec = fit_model(train, log_model)
    pred_response = np.asarray(spec["model"].predict(design(test, log_model)), dtype=float)
    pred = np.exp(pred_response) * spec["smearing"] if log_model else pred_response
    actual = test["日总销量"].to_numpy(dtype=float)
    error = actual - pred
    return {
        "验证起点": str(test["销售日期"].iloc[0].date()),
        "验证样本数": int(len(test)),
        "验证MAE": float(np.mean(np.abs(error))),
        "验证RMSE": float(np.sqrt(np.mean(error ** 2))),
        "验证MAPE": float(np.mean(np.abs(error) / actual) * 100),
    }


def diagnostics(category, frame, spec, selected):
    resid = spec["resid"]
    n = len(resid)
    lag = min(7, n // 5)
    lag2 = min(14, n // 5)
    lags = sorted({x for x in [lag, lag2] if x >= 2})
    if lags:
        lb = acorr_ljungbox(resid, lags=lags, return_df=True)
        ljung7 = float(lb["lb_pvalue"].iloc[0])
        ljung14 = float(lb["lb_pvalue"].iloc[-1])
    else:
        ljung7 = np.nan
        ljung14 = np.nan
    x = design(frame, spec["log_model"])
    _, _, _, bp_p = het_breuschpagan(resid, x)
    _, jb_p, skew, kurtosis = jarque_bera(resid)
    lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    time_corr = float(spearmanr(np.arange(n), resid).statistic)
    try:
        max_studentized = float(np.max(np.abs(spec["model"].get_influence().resid_studentized_internal)))
    except Exception:
        max_studentized = np.nan
    alerts = []
    if selected["稳健p值"] >= 0.05:
        alerts.append("售价效应未通过稳健显著性检验")
    if np.isfinite(ljung7) and ljung7 < 0.05:
        alerts.append("残差存在短期自相关")
    if np.isfinite(bp_p) and bp_p < 0.05:
        alerts.append("残差方差可能不恒定")
    if np.isfinite(time_corr) and abs(time_corr) >= 0.20:
        alerts.append("残差仍有时间趋势")
    return {
        "分类": category,
        "选择模型": selected["模型"],
        "样本数": n,
        "杜宾沃森统计量": float(durbin_watson(resid)),
        "残差一阶相关": lag1,
        "七阶滞后检验p值": ljung7,
        "十四阶滞后检验p值": ljung14,
        "异方差检验p值": float(bp_p),
        "正态性检验p值": float(jb_p),
        "残差偏度": float(skew),
        "残差峰度": float(kurtosis),
        "残差时间秩相关": time_corr,
        "最大内部学生化残差": max_studentized,
        "诊断提示": "；".join(alerts) if alerts else "未发现需要优先处理的诊断信号",
    }


panel = pd.read_csv(OUT / "品类日销售面板.csv", encoding="utf-8-sig")
panel["销售日期"] = pd.to_datetime(panel["销售日期"])
missing = [col for col in FEATURES if col not in panel.columns]
if missing:
    raise ValueError(f"缺少交接说明中的特征列: {missing}")
panel = panel.dropna(subset=["日总销量", "日平均售价"]).copy()
if (panel["日总销量"] <= 0).any() or (panel["日平均售价"] <= 0).any():
    raise ValueError("销量和售价必须为正")
panel = panel.sort_values(["分类名称", "销售日期"]).reset_index(drop=True)

model_rows = []
price_rows = []
coeff_rows = []
all_specs = {}
for category in CATEGORIES:
    frame = panel.loc[panel["分类名称"] == category].copy()
    if len(frame) < 100:
        raise ValueError(f"{category}样本量过少")
    all_specs[category] = {}
    for log_model in [False, True]:
        spec = fit_model(frame, log_model)
        valid = validate(frame, log_model)
        row = {"分类": category, **spec["row"], **valid}
        model_rows.append(row)
        price_rows.append({"分类": category, **spec["row"], **valid})
        coeff_rows.extend({"分类": category, **item} for item in spec["coeffs"])
        all_specs[category][spec["row"]["模型"]] = spec

pvals = np.array([r["稳健p值"] if np.isfinite(r["稳健p值"]) else 1.0 for r in price_rows])
_, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
for row, q in zip(price_rows, qvals):
    row["FDR校正后q值"] = float(q)
    row["售价检验结论"] = "通过5%校正阈值" if q < 0.05 else "未通过5%校正阈值"
for row in model_rows:
    match = next(x for x in price_rows if x["分类"] == row["分类"] and x["模型"] == row["模型"])
    row["FDR校正后q值"] = match["FDR校正后q值"]
    row["售价检验结论"] = match["售价检验结论"]

selected_rows = []
selected_specs = {}
for category in CATEGORIES:
    candidates = [x for x in model_rows if x["分类"] == category]
    selected = dict(min(candidates, key=lambda x: (x["验证RMSE"], x["验证MAE"])))
    selected["选择依据"] = "后20%日期验证集原始销量RMSE优先，MAE次优先"
    selected_rows.append(selected)
    selected_specs[category] = all_specs[category][selected["模型"]]

fit_rows = []
diagnostic_rows = []
for selected in selected_rows:
    category = selected["分类"]
    frame = panel.loc[panel["分类名称"] == category].sort_values("销售日期").copy()
    spec = selected_specs[category]
    original_residual = frame["日总销量"].to_numpy(float) - spec["fitted_sales"]
    diagnostic_rows.append(diagnostics(category, frame, spec, selected))
    for i in range(len(frame)):
        fit_rows.append({
            "销售日期": frame["销售日期"].iloc[i].date().isoformat(),
            "分类": category,
            "选择模型": selected["模型"],
            "实际日销量": float(frame["日总销量"].iloc[i]),
            "拟合日销量": float(spec["fitted_sales"][i]),
            "销量尺度残差": float(original_residual[i]),
            "模型残差": float(spec["resid"][i]),
        })

model_df = pd.DataFrame(model_rows)
price_df = pd.DataFrame(price_rows)
selected_df = pd.DataFrame(selected_rows)
coeff_df = pd.DataFrame(coeff_rows)
diagnostic_df = pd.DataFrame(diagnostic_rows)
fit_df = pd.DataFrame(fit_rows)

# 根据已选需求模型，在各品类平均价格和平均销量附近反推销量增加1千克所需的局部价格变化。
# 这只是逆需求关系的局部敏感度，不把它当作成本加成定价结果。
local_price_rows = []
for selected in selected_rows:
    category = selected["分类"]
    frame = panel.loc[panel["分类名称"] == category]
    reference_price = float(frame["日平均售价"].mean())
    reference_quantity = float(frame["日总销量"].mean())
    beta = float(selected["售价效应"])
    if selected["模型"] == "线性模型":
        price_change = 1.0 / beta
        method = "线性需求关系：局部价格变化=1/售价系数"
    else:
        price_change = reference_price / (beta * reference_quantity)
        method = "对数需求关系：在平均点处局部价格变化=平均价格/(价格弹性×平均销量)"
    local_price_rows.append({
        "分类": category,
        "选择模型": selected["模型"],
        "参考日平均售价": reference_price,
        "参考日销量": reference_quantity,
        "售价效应": beta,
        "销量增加1千克对应的局部价格变化": price_change,
        "稳健p值": selected["稳健p值"],
        "校正后q值": selected["FDR校正后q值"],
        "计算说明": method,
        "解释边界": "需求模型的局部反推，不是成本加成定价结论",
    })
local_price_df = pd.DataFrame(local_price_rows)
model_df.to_csv(RUN / "step4_模型比较.csv", index=False, encoding="utf-8-sig")
price_df.to_csv(RUN / "step4_价格效应.csv", index=False, encoding="utf-8-sig")
coeff_df.to_csv(RUN / "step4_系数明细.csv", index=False, encoding="utf-8-sig")
selected_df.to_csv(RUN / "step6_模型选择.csv", index=False, encoding="utf-8-sig")
diagnostic_df.to_csv(RUN / "step5_残差诊断.csv", index=False, encoding="utf-8-sig")
fit_df.to_csv(RUN / "step6_拟合值.csv", index=False, encoding="utf-8-sig")
local_price_df.to_csv(RUN / "step6_销量增加1千克的局部价格变化.csv", index=False, encoding="utf-8-sig")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# Step4 模型比较图
fig, ax = plt.subplots(figsize=(12, 6))
xpos = np.arange(len(CATEGORIES))
width = 0.36
level = [model_df.loc[(model_df["分类"] == c) & (model_df["模型"] == "线性模型"), "验证RMSE"].iloc[0] for c in CATEGORIES]
log = [model_df.loc[(model_df["分类"] == c) & (model_df["模型"] == "对数模型"), "验证RMSE"].iloc[0] for c in CATEGORIES]
ax.bar(xpos - width / 2, level, width, label="线性模型")
ax.bar(xpos + width / 2, log, width, label="对数模型")
ax.set_xticks(xpos, CATEGORIES)
ax.set_ylabel("后20%日期验证集 RMSE（千克）")
ax.set_title("Step4 两种模型的原始销量预测误差比较")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(RUN / "step4_模型比较.png", dpi=160)
plt.close(fig)

# Step5 残差图
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, category in zip(axes.flat, CATEGORIES):
    sub = fit_df.loc[fit_df["分类"] == category]
    ax.plot(pd.to_datetime(sub["销售日期"]), sub["模型残差"], linewidth=0.65, color="#2F5D8C")
    ax.axhline(0, color="#B22222", linestyle="--", linewidth=0.8)
    selected = selected_df.loc[selected_df["分类"] == category].iloc[0]
    ax.set_title(f"{category}（{selected['模型']}）", fontsize=11)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(alpha=0.18)
fig.suptitle("Step5 选择模型的残差时间序列", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(RUN / "step5_残差时间序列.png", dpi=160)
plt.close(fig)

# Step6 实际值与拟合值
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, category in zip(axes.flat, CATEGORIES):
    sub = fit_df.loc[fit_df["分类"] == category]
    actual = sub["实际日销量"].to_numpy()
    fitted = sub["拟合日销量"].to_numpy()
    limit = max(float(actual.max()), float(fitted.max())) * 1.05
    ax.scatter(actual, fitted, s=8, alpha=0.42, color="#4C72B0")
    ax.plot([0, limit], [0, limit], color="#B22222", linestyle="--", linewidth=0.9)
    selected = selected_df.loc[selected_df["分类"] == category].iloc[0]
    ax.set_title(f"{category}（{selected['模型']}）", fontsize=11)
    ax.set_xlabel("实际日销量（千克）")
    ax.set_ylabel("拟合日销量（千克）")
    ax.text(0.04, 0.94, f"验证RMSE={number(selected['验证RMSE'], 1)}\n调整R²={number(selected['调整R²'], 3)}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.grid(alpha=0.18)
fig.suptitle("Step6 选择模型的实际值与拟合值", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(RUN / "step6_实际值拟合值.png", dpi=160)
plt.close(fig)

# Step6 价格效应图
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, model_name, title in zip(axes, ["线性模型", "对数模型"], ["线性模型：每提高1元/千克的销量变化", "对数模型：价格弹性"]):
    sub = price_df.loc[price_df["模型"] == model_name].set_index("分类").loc[CATEGORIES].reset_index()
    values = sub["售价效应"].to_numpy()
    lower = sub["稳健95%下限"].to_numpy()
    upper = sub["稳健95%上限"].to_numpy()
    y = np.arange(len(CATEGORIES))
    ax.errorbar(values, y, xerr=[values - lower, upper - values], fmt="o", color="#2F5D8C", ecolor="#7A9EBC", capsize=3)
    ax.axvline(0, color="#B22222", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, CATEGORIES)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
fig.suptitle("Step6 售价效应的稳健95%区间", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(RUN / "step6_价格效应.png", dpi=160)
plt.close(fig)

# 中文交接说明
lines = [
    "# 问题二第一问 Step4-6 结果说明",
    "",
    "本阶段严格沿用 Step1-3 的面板：剔除退货和打折销售，按品类—日期分析，没有重新改变清洗口径。",
    "",
    "## 第四步：回归建模",
    "",
    "对六个品类分别建立线性模型和对数模型，控制月份与星期。标准误采用七阶稳健修正。两种模型不直接比较不同响应尺度的决定系数，而是按时间顺序使用后20%的日期做原始销量验证，优先选择验证误差更小的模型。",
    "",
    "| 品类 | 选择模型 | 验证RMSE | 调整R² | 售价效应 | 稳健p值 | 校正后q值 |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for _, row in selected_df.iterrows():
    lines.append(f"| {row['分类']} | {row['模型']} | {number(row['验证RMSE'], 2)} | {number(row['调整R²'], 3)} | {number(row['售价效应'], 4)} | {pvalue(row['稳健p值'])} | {pvalue(row['FDR校正后q值'])} |")
lines += [
    "",
    "## 销量增加一千克时的局部价格反推",
    "",
    "下表根据已选择的需求模型，在各品类平均价格和平均销量附近反推销量增加一千克对应的局部价格变化。负值表示需要降低价格才能支持销量增加；这不是成本加成定价结论。",
    "",
    "| 品类 | 选择模型 | 参考日平均售价 | 参考日销量 | 销量增加1千克对应的局部价格变化 |",
    "|---|---|---:|---:|---:|",
]
for _, row in local_price_df.iterrows():
    lines.append(f"| {row['分类']} | {row['选择模型']} | {number(row['参考日平均售价'], 2)} | {number(row['参考日销量'], 2)} | {number(row['销量增加1千克对应的局部价格变化'], 4)} |")
lines += [
    "",
    "成本加成定价还需要可靠的单位成本、损耗率和目标加成率；当前结果只能说明需求模型下的局部价格敏感度。",
    "",
    "## 第五步：残差诊断",
    "",
    "| 品类 | 选择模型 | 杜宾—沃森统计量 | 七阶滞后检验p值 | 异方差检验p值 | 诊断提示 |",
    "|---|---|---:|---:|---:|---|",
]
for _, row in diagnostic_df.iterrows():
    lines.append(f"| {row['分类']} | {row['选择模型']} | {number(row['杜宾沃森统计量'], 3)} | {pvalue(row['七阶滞后检验p值'])} | {pvalue(row['异方差检验p值'])} | {row['诊断提示']} |")
lines += [
    "",
    "## 第六步：图表与文件",
    "",
    "模型比较、价格效应、系数明细、残差诊断、模型选择和拟合值均已输出到本目录；同时生成模型比较图、残差时间序列图、实际值—拟合值图和价格效应区间图。",
    "",
    "## 解释边界",
    "",
    "售价是品类日销售额除以品类日销量得到的加权平均售价，可能同时受到单品结构、库存、上架、促销和供给变化影响。因此本阶段只能支持价格—销量关系的统计描述，不能单独证明价格造成销量变化。",
]
(RUN / "Step4-6_结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Step4-6完成")
for _, row in selected_df.iterrows():
    print(f"{row['分类']}：{row['模型']}，售价效应={number(row['售价效应'], 4)}，稳健p={pvalue(row['稳健p值'])}，验证RMSE={number(row['验证RMSE'], 2)}")
