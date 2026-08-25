# -*- coding: utf-8 -*-
"""数据质量审计：缺失值、异常值、重复值、日期连续性"""
import pandas as pd
import numpy as np

BASE = r"D:\数学建模\c23"
out = []
def w(s=""):
    out.append(str(s))

# ---------- 附件1 ----------
df1 = pd.read_excel(f"{BASE}\\附件1.xlsx", dtype=str)
w("=" * 60)
w("附件1 (品类及单品信息)")
w(f"  缺失值:\n{df1.isna().sum().to_string()}")
w(f"  单品编码重复数: {df1['单品编码'].duplicated().sum()}")
w(f"  单品编码去重后数量: {df1['单品编码'].nunique()} (应=251)")
w(f"  单品名称重复数: {df1['单品名称'].duplicated().sum()}")

# ---------- 附件3 ----------
df3 = pd.read_excel(f"{BASE}\\附件3.xlsx", dtype=str)
df3["批发价格(元/千克)"] = pd.to_numeric(df3["批发价格(元/千克)"], errors="coerce")
df3["日期"] = pd.to_datetime(df3["日期"])
w("=" * 60)
w("附件3 (批发价格)")
w(f"  缺失值:\n{df3.isna().sum().to_string()}")
w(f"  批发价 NaN 数: {df3['批发价格(元/千克)'].isna().sum()}")
w(f"  批发价 <=0 数: {(df3['批发价格(元/千克)'] <= 0).sum()}")
w(f"  批发价 <0 数: {(df3['批发价格(元/千克)'] < 0).sum()}")
w(f"  (日期,单品) 重复组合数: {df3.duplicated(subset=['日期','单品编码']).sum()}")
w(f"  批发价 分布: min={df3['批发价格(元/千克)'].min():.3f}  "
  f"p50={df3['批发价格(元/千克)'].median():.3f}  "
  f"max={df3['批发价格(元/千克)'].max():.3f}")

# ---------- 附件2 ----------
cols2 = ["销售日期", "单品编码", "销量(千克)", "销售单价(元/千克)", "销售类型", "是否打折销售"]
df2 = pd.read_excel(f"{BASE}\\附件2.xlsx", usecols=cols2)
w("=" * 60)
w("附件2 (销售流水)")
w(f"  缺失值:\n{df2.isna().sum().to_string()}")
w(f"  销量 <=0 数: {(df2['销量(千克)'] <= 0).sum()}")
w(f"  销量 <0 数: {(df2['销量(千克)'] < 0).sum()}")
w(f"  销量 =0 数: {(df2['销量(千克)'] == 0).sum()}")
w(f"  销售单价 <=0 数: {(df2['销售单价(元/千克)'] <= 0).sum()}")
w(f"  销售单价 <0 数: {(df2['销售单价(元/千克)'] < 0).sum()}")
w(f"  完全重复行数: {df2.duplicated().sum()}")
w(f"  销售类型取值: {df2['销售类型'].value_counts(dropna=False).to_dict()}")
w(f"  是否打折销售取值: {df2['是否打折销售'].value_counts(dropna=False).to_dict()}")
w(f"  销量 分布: p1={df2['销量(千克)'].quantile(0.01):.3f}  "
  f"p50={df2['销量(千克)'].quantile(0.5):.3f}  "
  f"p99={df2['销量(千克)'].quantile(0.99):.3f}  max={df2['销量(千克)'].max():.3f}")
w(f"  单价 分布: p1={df2['销售单价(元/千克)'].quantile(0.01):.3f}  "
  f"p50={df2['销售单价(元/千克)'].quantile(0.5):.3f}  "
  f"p99={df2['销售单价(元/千克)'].quantile(0.99):.3f}  max={df2['销售单价(元/千克)'].max():.3f}")

# ---------- 面板日期连续性 ----------
panel = pd.read_csv(f"{BASE}\\output\\品类日销售面板.csv")
panel["销售日期"] = pd.to_datetime(panel["销售日期"])
w("=" * 60)
w("面板 (品类x日期)")
w(f"  日平均售价 NaN 数: {panel['日平均售价'].isna().sum()}")
w(f"  日加权批发价 NaN 数: {panel['日加权批发价'].isna().sum()}")
w(f"  日总销量 =0 数: {(panel['日总销量'] == 0).sum()}")
w("  各分类日期是否连续(缺天数):")
for cat, sub in panel.groupby("分类名称"):
    dts = sorted(sub["销售日期"].unique())
    full = pd.date_range(dts[0], dts[-1], freq="D")
    missing = len(set(full) - set(dts))
    w(f"    {cat}: 首={dts[0].date()} 末={dts[-1].date()}  缺{missing}天")

report = "\n".join(out)
with open(f"{BASE}\\output\\qa_audit.txt", "w", encoding="utf-8") as fp:
    fp.write(report)
print("DONE")
