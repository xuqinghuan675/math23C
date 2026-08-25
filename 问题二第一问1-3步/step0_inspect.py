# -*- coding: utf-8 -*-
"""探查三张附件的结构，输出到 inspect_report.txt（UTF-8）"""
import pandas as pd

BASE = r"D:\数学建模\c23"
out = []

def w(s=""):
    out.append(str(s))

# ---------- 附件1 ----------
f1 = f"{BASE}\\附件1.xlsx"
df1 = pd.read_excel(f1, dtype=str)
w("=" * 60)
w("附件1 (品类及单品信息)")
w(f"  形状: {df1.shape}")
w(f"  列名: {list(df1.columns)}")
w("  前5行:")
w(df1.head().to_string())
if '品类名称' in df1.columns:
    w("  品类名称(去重):")
    for v in df1['品类名称'].dropna().unique():
        w(f"    - {v}")
w(f"  品类数量: {df1['品类名称'].nunique() if '品类名称' in df1.columns else 'N/A'}")
w(f"  单品数量: {df1['单品编码'].nunique() if '单品编码' in df1.columns else 'N/A'}")

# ---------- 附件3 ----------
f3 = f"{BASE}\\附件3.xlsx"
df3 = pd.read_excel(f3, dtype=str)
w("=" * 60)
w("附件3 (批发价格)")
w(f"  形状: {df3.shape}")
w(f"  列名: {list(df3.columns)}")
w("  前5行:")
w(df3.head().to_string())
w(f"  单品编码数量: {df3.iloc[:, 1].nunique() if df3.shape[1] >= 2 else 'N/A'}")

# ---------- 附件2 (抽样) ----------
f2 = f"{BASE}\\附件2.xlsx"
df2 = pd.read_excel(f2, nrows=20)
w("=" * 60)
w("附件2 (销售流水明细, 抽样前20行)")
w(f"  列名: {list(df2.columns)}")
w("  各列类型:")
w(df2.dtypes.to_string())
w("  前10行:")
w(df2.head(10).to_string())

# 打折列取值
disc_col = None
for c in df2.columns:
    if '打折' in str(c):
        disc_col = c
w("=" * 60)
w(f"打折列名: {disc_col}")
if disc_col:
    w(f"  取值分布(抽样): {df2[disc_col].value_counts(dropna=False).to_dict()}")

# 完整读取附件2统计(全量)
w("=" * 60)
w("开始全量读取附件2以统计行数/日期范围/打折分布(可能需要1-3分钟)...")
df2full = pd.read_excel(f2)
w(f"  全量形状: {df2full.shape}")
date_col = df2full.columns[0]
w(f"  首列(日期) 类型: {df2full[date_col].dtype}")
w(f"  日期范围: {df2full[date_col].min()} ~ {df2full[date_col].max()}")
if disc_col:
    w(f"  打折取值分布(全量):")
    w(df2full[disc_col].value_counts(dropna=False).to_string())
w("  缺失值统计:")
w(df2full.isnull().sum().to_string())

report = "\n".join(out)
with open(f"{BASE}\\output_inspect.txt", "w", encoding="utf-8") as fp:
    fp.write(report)
print("DONE, len=", len(report))
