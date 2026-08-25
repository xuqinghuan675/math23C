# -*- coding: utf-8 -*-
"""
问题二 第一小问 —— Step1~3 数据清洗、聚合、特征工程、散点图
按小组决策执行：
  Step1 剔除打折(是否打折销售=是) + 剔除退货(销售类型=退货) + 匹配批发价 + 按(日期,分类)聚合
  Step2 生成月份/星期及哑变量
  Step3 分品类画散点图(日平均售价 vs 日总销量)
输出到 output/ 目录
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"D:\数学建模\c23"
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

log = []
def w(s=""):
    log.append(str(s))

# ============ 读取 ============
df1 = pd.read_excel(f"{BASE}\\附件1.xlsx", dtype=str)          # 单品 -> 分类
df3 = pd.read_excel(f"{BASE}\\附件3.xlsx", dtype=str)          # 日期+单品 -> 批发价
df3["批发价格(元/千克)"] = pd.to_numeric(df3["批发价格(元/千克)"], errors="coerce")
cols2 = ["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)", "销售类型", "是否打折销售"]
df2 = pd.read_excel(f"{BASE}\\附件2.xlsx", usecols=cols2)

w("=" * 60)
w("原始数据")
w(f"  附件2 原始行数: {len(df2)}")
w(f"  附件2 唯一日期数: {df2['销售日期'].nunique()}")

# ============ 清洗 ============
# 统一单品编码为字符串
df1["单品编码"] = df1["单品编码"].astype(str)
df3["单品编码"] = df3["单品编码"].astype(str)
df2["单品编码"] = df2["单品编码"].astype(str)
df3["日期"] = pd.to_datetime(df3["日期"])

n0 = len(df2)
df2 = df2[df2["销售类型"] == "销售"].copy()
w(f"  剔除退货(销售类型=退货)后: {len(df2)} (去掉 {n0-len(df2)})")

n1 = len(df2)
df2 = df2[df2["是否打折销售"] == "否"].copy()
w(f"  剔除打折(是否打折销售=是)后: {len(df2)} (去掉 {n1-len(df2)})")

# ============ 匹配分类(附件1) ============
before = len(df2)
df2 = df2.merge(df1[["单品编码", "分类名称"]], on="单品编码", how="left")
assert len(df2) == before, "分类合并导致行数变化"
w(f"  匹配分类后, 分类名称缺失数: {df2['分类名称'].isna().sum()}")

# ============ 匹配批发价(附件3, 按日期+单品) —— 备用 ============
before = len(df2)
df2 = df2.merge(df3, left_on=["销售日期", "单品编码"], right_on=["日期", "单品编码"],
                how="left", suffixes=("", "_wp"))
assert len(df2) == before, "批发价合并导致行数变化"
wp_missing = df2["批发价格(元/千克)"].isna().mean() * 100
w(f"  匹配批发价后, 批发价缺失比例: {wp_missing:.2f}%")
df2 = df2.drop(columns=["日期"], errors="ignore")

# ============ 聚合: 按(销售日期, 分类名称) ============
df2["销售额"] = df2["销量(千克)"] * df2["销售单价(元/千克)"]
df2["成本额"] = df2["销量(千克)"] * df2["批发价格(元/千克)"]

def wmean(rev, qty):
    return (rev.sum() / qty.sum()) if qty.sum() > 0 else np.nan

panel = df2.groupby(["销售日期", "分类名称"]).agg(
    日总销量=("销量(千克)", "sum"),
    日销售额=("销售额", "sum"),
    条数=("单品编码", "size"),
    单品数=("单品编码", "nunique"),
    日批发成本=("成本额", "sum"),
    批发价覆盖销量=("批发价格(元/千克)", lambda s: s.notna().sum()),
).reset_index()

panel["日平均售价"] = panel["日销售额"] / panel["日总销量"]
panel["日加权批发价"] = panel["日批发成本"] / panel["批发价覆盖销量"]  # 仅覆盖到的部分, 备用

w("=" * 60)
w("聚合结果(品类 x 日期)")
w(f"  面板形状: {panel.shape}")
w(f"  唯一日期数: {panel['销售日期'].nunique()}")
w(f"  各分类样本数:")
w(panel["分类名称"].value_counts().to_string())

# ============ Step2 特征工程 ============
panel["月份"] = panel["销售日期"].dt.month
panel["星期"] = panel["销售日期"].dt.weekday + 1  # 1=周一 ... 7=周日

# 月份哑变量: 以1月为基准, 生成2~12月
for m in range(2, 13):
    panel[f"月份_{m:02d}"] = (panel["月份"] == m).astype(int)
# 星期哑变量: 以周一(1)为基准, 生成周二~周日(2~7)
for wd in range(2, 8):
    panel[f"星期_{wd}"] = (panel["星期"] == wd).astype(int)

# ============ Step3 分品类散点图 ============
cats = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
corr_rows = []
for i, cat in enumerate(cats):
    sub = panel[panel["分类名称"] == cat]
    ax = axes[i]
    ax.scatter(sub["日平均售价"], sub["日总销量"], s=8, alpha=0.5, color="#4C72B0")
    ax.set_title(cat, fontsize=13)
    ax.set_xlabel("日平均售价 (元/千克)")
    ax.set_ylabel("日总销量 (千克)")
    # 相关系数(原尺度 与 对数尺度)
    r_lin = sub["日平均售价"].corr(sub["日总销量"])
    r_log = np.log(sub["日平均售价"]).corr(np.log(sub["日总销量"]))
    ax.text(0.03, 0.97, f"r(线性)={r_lin:.3f}\nr(对数)={r_log:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    corr_rows.append((cat, len(sub), sub["日平均售价"].mean(),
                      sub["日总销量"].mean(), r_lin, r_log))

plt.suptitle("各蔬菜品类: 日平均售价 vs 日总销量 (已剔除打折/退货)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUT}\\分品类散点图.png", dpi=130)
plt.close()

w("=" * 60)
w("各品类相关系数汇总")
w("  品类 | 样本数 | 平均售价 | 平均日销量 | r(线性) | r(对数)")
for c, n, mp, mq, rl, rg in corr_rows:
    w(f"  {c} | {n} | {mp:.2f} | {mq:.1f} | {rl:.3f} | {rg:.3f}")

# ============ 保存 ============
panel = panel.sort_values(["分类名称", "销售日期"]).reset_index(drop=True)
panel.to_csv(f"{OUT}\\品类日销售面板.csv", index=False, encoding="utf-8-sig")
w(f"  已保存: output/品类日销售面板.csv ({len(panel)} 行, {panel.shape[1]} 列)")

report = "\n".join(log)
with open(f"{OUT}\\summary.txt", "w", encoding="utf-8") as fp:
    fp.write(report)
print("DONE")
