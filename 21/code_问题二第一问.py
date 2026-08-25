# -*- coding: utf-8 -*-
"""
问题二 第一小问：分析各蔬菜品类的销售总量与成本加成定价的关系
方法：需求价格弹性模型
  加成率 = 售价/批发价 - 1（销售额加权聚合到 品类×日期）
  朴素OLS：ln(日销量) ~ ln(1+加成率) + 打折占比 + 月份 + 星期   （展示内生性）
  2SLS  ：ln(日销量) ~ ln(售价)，用 ln(批发价) 作工具变量        （识别真实需求弹性）
说明：控制成本后，价格弹性 = 加成率弹性（ln(1+R)=lnP-lnC）
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

SRC = r"D:\数学建模\c23"
OUT = r"D:\数学建模\21"
os.makedirs(OUT, exist_ok=True)

# ============ 1. 读取清洗后数据 ============
df1 = pd.read_excel(f"{SRC}\\附件1_清洗.xlsx", dtype=str)
df2 = pd.read_excel(f"{SRC}\\附件2_清洗.xlsx")
df3 = pd.read_excel(f"{SRC}\\附件3_清洗.xlsx")

df1["单品编码"] = df1["单品编码"].astype(str)
df3["单品编码"] = df3["单品编码"].astype(str)
df3["批发价格(元/千克)"] = pd.to_numeric(df3["批发价格(元/千克)"], errors="coerce")
df3["日期"] = pd.to_datetime(df3["日期"])
df2["单品编码"] = df2["单品编码"].astype(str)
df2["销售日期"] = pd.to_datetime(df2["销售日期"])

# ============ 2. 三表对齐 ============
df = df2.merge(df3[["日期", "单品编码", "批发价格(元/千克)"]],
               left_on=["销售日期", "单品编码"], right_on=["日期", "单品编码"], how="left")
df = df.merge(df1[["单品编码", "分类名称"]], on="单品编码", how="left")

# ============ 3. 分析阶段过滤：剔除退货，保留打折 ============
df = df[df["销售类型"] == "销售"].copy()
df = df[df["销量(千克)"] > 0].copy()
df["销售额"] = df["销量(千克)"] * df["销售单价(元/千克)"]
df["成本额"] = df["销量(千克)"] * df["批发价格(元/千克)"]

# ============ 4. 聚合到 品类×日期 ============
g = df.groupby(["销售日期", "分类名称"])
panel = pd.DataFrame({
    "日销量": g["销量(千克)"].sum(),
    "日销售额": g["销售额"].sum(),
    "日成本": g["成本额"].sum(),
    "打折销量": g.apply(lambda s: s.loc[s["是否打折销售"] == "是", "销量(千克)"].sum()),
}).reset_index()
panel["加成率"] = panel["日销售额"] / panel["日成本"] - 1
panel["打折占比"] = panel["打折销量"] / panel["日销量"]
panel = panel[panel["日成本"] > 0].copy()

# ============ 5. 特征 ============
panel["月份"] = panel["销售日期"].dt.month
panel["星期"] = panel["销售日期"].dt.weekday + 1
panel["lnQ"] = np.log(panel["日销量"])
panel["lnR"] = np.log1p(panel["加成率"])          # ln(1+加成率)
panel["lnP"] = np.log(panel["日销售额"] / panel["日销量"])   # 加权售价
panel["lnC"] = np.log(panel["日成本"] / panel["日销量"])     # 加权批发价

panel.to_csv(f"{OUT}\\结果_品类日面板.csv", index=False, encoding="utf-8-sig")

# ============ 6. 描述统计 ============
cats = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
desc_rows = []
for c in cats:
    s = panel[panel["分类名称"] == c]
    desc_rows.append({
        "品类": c, "样本天数": len(s),
        "平均日销量(kg)": round(s["日销量"].mean(), 2),
        "平均加成率": round(s["加成率"].mean(), 4),
        "加成率标准差": round(s["加成率"].std(), 4),
        "平均售价(元/kg)": round((s["日销售额"].sum()/s["日销量"].sum()), 2),
        "平均批发价(元/kg)": round((s["日成本"].sum()/s["日销量"].sum()), 2),
        "r(lnQ,lnR)": round(s["lnQ"].corr(s["lnR"]), 3),
    })
pd.DataFrame(desc_rows).to_csv(f"{OUT}\\结果_描述统计.csv", index=False, encoding="utf-8-sig")

# ============ 7. 建模函数 ============
def build_ctrl(sub):
    """控制变量: 打折占比 + 月份哑变量 + 星期哑变量"""
    ctrl = pd.concat([
        sub[["打折占比"]].rename(columns={"打折占比": "D"}),
        pd.get_dummies(sub["月份"], prefix="M", drop_first=True),
        pd.get_dummies(sub["星期"], prefix="W", drop_first=True),
    ], axis=1).astype(float)
    return ctrl

def ols_reg(sub):
    """朴素OLS: lnQ ~ lnR + 控制变量"""
    sub = sub.reset_index(drop=True)
    X = sm.add_constant(pd.concat([sub[["lnR"]], build_ctrl(sub)], axis=1))
    m = sm.OLS(sub["lnQ"].astype(float), X).fit(cov_type="HC1")
    return m

def iv2sls(sub):
    """2SLS: lnQ ~ lnP + 控制变量, 工具变量 lnC"""
    sub = sub.reset_index(drop=True)
    y = sub["lnQ"].astype(float)
    lnP = sub["lnP"].astype(float)
    lnC = sub["lnC"].astype(float)
    ctrl = build_ctrl(sub)
    # 第一阶段: lnP ~ lnC + 控制
    X1 = sm.add_constant(pd.concat([lnC.rename("lnC"), ctrl], axis=1))
    fs = sm.OLS(lnP, X1).fit(cov_type="HC1")
    lnP_hat = fs.predict(X1)
    F1 = float(fs.fvalue)
    # 第二阶段: lnQ ~ lnP_hat + 控制
    X2 = sm.add_constant(pd.concat([lnP_hat.rename("lnP"), ctrl], axis=1))
    ss = sm.OLS(y, X2).fit()
    b = float(ss.params["lnP"])
    # 正确 2SLS 标准误(用原始 lnP 的残差)
    X2o = sm.add_constant(pd.concat([lnP.rename("lnP"), ctrl], axis=1))
    resid = y.values - X2o.values @ ss.params.values
    n, k = X2.shape
    sigma2 = float(resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X2.values.T @ X2.values)
    se = float(np.sqrt(cov[1, 1]))
    t = b / se
    p = 2 * sps.t.sf(abs(t), df=n - k)
    r2 = float(ss.rsquared)
    return dict(beta=b, se=se, t=t, p=p, F1=F1, r2=r2, n=n, fs=fs, ss=ss)

# ============ 8. 分品类回归(OLS 与 2SLS 对比) ============
rows = []
models = {}
for c in cats:
    sub = panel[panel["分类名称"] == c]
    m_ols = ols_reg(sub)
    m_iv = iv2sls(sub)
    models[c] = (m_ols, m_iv)
    rows.append({
        "品类": c,
        "样本数": m_iv["n"],
        "一阶段F值": round(m_iv["F1"], 1),
        "OLS系数(有偏)": round(m_ols.params["lnR"], 3),
        "OLS_p值": round(m_ols.pvalues["lnR"], 4),
        "IV弹性β(需求)": round(m_iv["beta"], 3),
        "IV标准误": round(m_iv["se"], 3),
        "IV_p值": round(m_iv["p"], 4),
        "IV_95%CI下": round(m_iv["beta"] - 1.96 * m_iv["se"], 3),
        "IV_95%CI上": round(m_iv["beta"] + 1.96 * m_iv["se"], 3),
        "显著性": "***" if m_iv["p"] < 0.01 else ("**" if m_iv["p"] < 0.05 else ("*" if m_iv["p"] < 0.1 else "不显著")),
    })
coef = pd.DataFrame(rows)
coef.to_csv(f"{OUT}\\结果_弹性系数表.csv", index=False, encoding="utf-8-sig")

# ============ 9. 散点图(OLS vs IV 对比) ============
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, c in enumerate(cats):
    sub = panel[panel["分类名称"] == c]
    m_ols, m_iv = models[c]
    ax = axes[i]
    x = sub["lnR"].values
    y = sub["lnQ"].values
    ax.scatter(x, y, s=7, alpha=0.4, color="#4C72B0")
    xs = np.linspace(x.min(), x.max(), 100)
    xm, ym = x.mean(), y.mean()
    # OLS 拟合线(有偏, 通常偏正)
    b_ols = m_ols.params["lnR"]
    ax.plot(xs, ym + b_ols * (xs - xm), "--", color="#C44E52", lw=1.6, label=f"OLS β={b_ols:.2f}(有偏)")
    # IV 拟合线(需求曲线)
    b_iv = m_iv["beta"]
    ax.plot(xs, ym + b_iv * (xs - xm), "-", color="#55A868", lw=2.2, label=f"2SLS β={b_iv:.2f}(需求)")
    ax.set_title(f"{c}  需求弹性 β={b_iv:.3f}", fontsize=12)
    ax.set_xlabel("ln(1+加成率)")
    ax.set_ylabel("ln(日销量)")
    ax.legend(fontsize=8)
plt.suptitle("各品类: 销售总量与成本加成率的关系  (2SLS工具变量校正内生性)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUT}\\图_分品类弹性拟合.png", dpi=130)
plt.close()

# ============ 10. 稳健性1: 剔除打折后 OLS ============
panel_nd = panel[panel["打折占比"] < 0.01].copy()
rows_nd = []
for c in cats:
    sub = panel_nd[panel_nd["分类名称"] == c]
    if len(sub) < 50:
        continue
    m = ols_reg(sub)
    rows_nd.append({"品类": c, "样本数": int(m.nobs),
                    "OLS系数": round(m.params["lnR"], 3), "p值": round(m.pvalues["lnR"], 4),
                    "R²": round(m.rsquared, 3)})
pd.DataFrame(rows_nd).to_csv(f"{OUT}\\结果_稳健性_剔除打折.csv", index=False, encoding="utf-8-sig")

# ============ 11. 稳健性2: 面板固定效应(合并样本+品类虚拟变量) ============
dummies = pd.concat([
    pd.get_dummies(panel["月份"], prefix="M", drop_first=True),
    pd.get_dummies(panel["星期"], prefix="W", drop_first=True),
    pd.get_dummies(panel["分类名称"], prefix="C", drop_first=True),
], axis=1).astype(float)
Xfe = pd.concat([
    panel[["lnP", "lnC", "打折占比"]].rename(columns={"打折占比": "D"}),
    dummies], axis=1).astype(float)
yfe = panel["lnQ"].astype(float)
# 第一阶段 lnP ~ lnC + 控制
X1fe = sm.add_constant(pd.concat([Xfe[["lnC"]], Xfe.drop(columns=["lnP", "lnC"])], axis=1))
fsfe = sm.OLS(Xfe["lnP"], X1fe).fit(cov_type="HC1")
lnP_hat = fsfe.predict(X1fe)
# 第二阶段 lnQ ~ lnP_hat + 控制
X2fe = sm.add_constant(pd.concat([lnP_hat.rename("lnP"), Xfe.drop(columns=["lnP", "lnC"])], axis=1))
ssfe = sm.OLS(yfe, X2fe).fit()
b_fe = float(ssfe.params["lnP"])
X2o = sm.add_constant(pd.concat([Xfe[["lnP"]], Xfe.drop(columns=["lnP", "lnC"])], axis=1))
resid = yfe.values - X2o.values @ ssfe.params.values
n, k = X2fe.shape
sigma2 = float(resid @ resid) / (n - k)
cov = sigma2 * np.linalg.inv(X2fe.values.T @ X2fe.values)
se_fe = float(np.sqrt(cov[1, 1]))

# ============ 12. 回归报告 ============
with open(f"{OUT}\\结果_回归报告.txt", "w", encoding="utf-8") as f:
    f.write("=" * 72 + "\n")
    f.write("问题二第一问 · 各品类销售总量与成本加成定价的关系\n")
    f.write("因变量: ln(日销量)  模型: 对数-对数弹性模型\n")
    f.write("控制变量: 打折占比 + 月份哑变量 + 星期哑变量 (稳健标准误HC1)\n")
    f.write("=" * 72 + "\n\n")
    f.write("【核心结论】2SLS工具变量法(工具变量=批发价)识别的需求弹性:\n")
    f.write(coef.to_string(index=False))
    f.write("\n\n")
    f.write("=" * 72 + "\n")
    f.write("【分品类 2SLS 完整结果】\n")
    for c in cats:
        m_ols, m_iv = models[c]
        f.write(f"\n### {c}   (一阶段F={m_iv['F1']:.1f})\n")
        f.write("--- 朴素OLS(有偏) ---\n")
        f.write(m_ols.summary().as_text())
        f.write("\n")
    f.write("=" * 72 + "\n")
    f.write("【稳健性: 面板固定效应 2SLS (合并6品类, 加品类虚拟变量)】\n")
    f.write(f"平均需求弹性 = {b_fe:.4f}  (se={se_fe:.4f})\n")
    f.write(f"一阶段F = {float(fsfe.tvalues['lnC'] ** 2):.1f}\n")

print("DONE")
