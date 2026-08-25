# -*- coding: utf-8 -*-
"""按品类生成整体、月份、季度、日期和节假日的多面板对比图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
PANEL_PATH = OUT / "品类日销售面板.csv"
FIG_DIR = OUT / "分组对比图"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


def holiday_map():
    result = {}

    def add(start, end, name):
        for date in pd.date_range(start, end, freq="D"):
            result[date.date()] = name

    add("2020-10-01", "2020-10-08", "国庆及中秋")
    add("2021-02-11", "2021-02-17", "春节")
    add("2021-04-03", "2021-04-05", "清明")
    add("2021-05-01", "2021-05-05", "劳动节")
    add("2021-06-12", "2021-06-14", "端午")
    add("2021-09-19", "2021-09-21", "中秋")
    add("2021-10-01", "2021-10-07", "国庆")
    add("2022-01-31", "2022-02-06", "春节")
    add("2022-04-03", "2022-04-05", "清明")
    add("2022-04-30", "2022-05-04", "劳动节")
    add("2022-06-03", "2022-06-05", "端午")
    add("2022-09-10", "2022-09-12", "中秋")
    add("2022-10-01", "2022-10-07", "国庆")
    add("2023-01-21", "2023-01-27", "春节")
    add("2023-04-05", "2023-04-05", "清明")
    add("2023-04-29", "2023-05-03", "劳动节")
    add("2023-06-22", "2023-06-24", "端午")
    return result


def season_name(month):
    if month in (3, 4, 5):
        return "春季"
    if month in (6, 7, 8):
        return "夏季"
    if month in (9, 10, 11):
        return "秋季"
    return "冬季"


def corr_pair(frame):
    if len(frame) < 3:
        return np.nan, np.nan
    linear = frame["日平均售价"].corr(frame["日总销量"])
    log_corr = np.log(frame["日平均售价"]).corr(np.log(frame["日总销量"]))
    return float(linear), float(log_corr)


def group_summary(frame, group_type, group_name):
    linear, log_corr = corr_pair(frame)
    return {
        "分类": frame["分类名称"].iloc[0],
        "分组类型": group_type,
        "分组": str(group_name),
        "样本天数": int(len(frame)),
        "平均售价": float(frame["日平均售价"].mean()),
        "平均日销量": float(frame["日总销量"].mean()),
        "销量中位数": float(frame["日总销量"].median()),
        "销量标准差": float(frame["日总销量"].std(ddof=1)) if len(frame) > 1 else np.nan,
        "线性相关": linear,
        "对数相关": log_corr,
    }


def plot_twin_lines(ax, x, quantity, price, xlabel, xticks=None, labels=None):
    price_ax = ax.twinx()
    line1 = ax.plot(x, quantity, marker="o", markersize=3.5, linewidth=1.5, color=BLUE, label="平均日销量")
    line2 = price_ax.plot(x, price, marker="s", markersize=3.5, linewidth=1.4, color=ORANGE, label="平均售价")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("平均日销量（千克）", color=BLUE)
    price_ax.set_ylabel("平均售价（元/千克）", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    price_ax.tick_params(axis="y", labelcolor=ORANGE)
    if xticks is not None:
        ax.set_xticks(xticks, labels if labels is not None else xticks)
    ax.grid(axis="y", alpha=0.18)
    price_ax.spines["top"].set_visible(False)
    handles = line1 + line2
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=False, fontsize=8)
    return price_ax


def fmt(value, digits=1):
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


panel = pd.read_csv(PANEL_PATH, encoding="utf-8-sig")
panel["销售日期"] = pd.to_datetime(panel["销售日期"])
panel["月份"] = panel["销售日期"].dt.month
panel["季度"] = ((panel["月份"] - 1) // 3 + 1).astype(int)
panel["季节"] = panel["月份"].map(season_name)
holiday_names = holiday_map()
panel["节假日"] = panel["销售日期"].dt.date.map(holiday_names).fillna("非节假日")

# 固定度量和统一刻度：所有品类都使用同一套销量、售价定义和同类面板刻度。
global_quantity_max = float(panel["日总销量"].max()) * 1.05
global_price_min = max(0.0, float(panel["日平均售价"].min()) * 0.95)
global_price_max = float(panel["日平均售价"].max()) * 1.05
month_global = panel.groupby(["分类名称", "月份"]).agg(平均日销量=("日总销量", "mean"), 平均售价=("日平均售价", "mean"))
quarter_global = panel.groupby(["分类名称", "季度"]).agg(平均日销量=("日总销量", "mean"), 平均售价=("日平均售价", "mean"))
season_global = panel.groupby(["分类名称", "季节"]).agg(平均日销量=("日总销量", "mean"), 平均售价=("日平均售价", "mean"))
holiday_global = panel.groupby(["分类名称", "节假日"]).agg(平均日销量=("日总销量", "mean"), 平均售价=("日平均售价", "mean"))
group_quantity_max = float(max(month_global["平均日销量"].max(), quarter_global["平均日销量"].max(), season_global["平均日销量"].max(), holiday_global["平均日销量"].max())) * 1.10
group_price_max = float(max(month_global["平均售价"].max(), quarter_global["平均售价"].max(), season_global["平均售价"].max(), holiday_global["平均售价"].max())) * 1.10

summary_rows = []
for category in CATEGORIES:
    sub = panel.loc[panel["分类名称"] == category].sort_values("销售日期").copy()
    summary_rows.append(group_summary(sub, "整体", "全部日期"))
    for month, frame in sub.groupby("月份"):
        summary_rows.append(group_summary(frame, "月份", f"{int(month)}月"))
    for quarter, frame in sub.groupby("季度"):
        summary_rows.append(group_summary(frame, "季度", f"第{int(quarter)}季度"))
    for season, frame in sub.groupby("季节"):
        summary_rows.append(group_summary(frame, "季节", season))
    for holiday, frame in sub.groupby("节假日"):
        summary_rows.append(group_summary(frame, "节假日", holiday))

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "分组统计汇总.csv", index=False, encoding="utf-8-sig")

(OUT / "节假日口径说明.md").write_text(
    "# 节假日分组口径\n\n"
    "图中节假日按照2020年至2023年公开法定节假日期间划分，包含春节、清明、劳动节、端午、中秋和国庆；"
    "没有落入这些日期的观察日归为非节假日。节假日分组只用于描述性对比，不表示节假日造成销量变化。\n",
    encoding="utf-8",
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

for category in CATEGORIES:
    sub = panel.loc[panel["分类名称"] == category].sort_values("销售日期").copy()
    month = sub.groupby("月份", as_index=False).agg(平均日销量=("日总销量", "mean"), 平均售价=("日平均售价", "mean"))
    quarter = sub.groupby("季度", as_index=False).agg(平均日销量=("日总销量", "mean"), 平均售价=("日平均售价", "mean"))
    season = sub.groupby("季节", as_index=False).agg(平均日销量=("日总销量", "mean"), 平均售价=("日平均售价", "mean"))
    season["排序"] = season["季节"].map({"春季": 0, "夏季": 1, "秋季": 2, "冬季": 3})
    season = season.sort_values("排序")
    holiday = sub.groupby("节假日", as_index=False).agg(平均日销量=("日总销量", "mean"), 平均售价=("日平均售价", "mean"), 样本天数=("日总销量", "size"))
    holiday["排序"] = holiday["节假日"].map({"非节假日": 0}).fillna(1)
    holiday = holiday.sort_values(["排序", "节假日"])

    overall_linear, overall_log = corr_pair(sub)
    month_high = month.loc[month["平均日销量"].idxmax()]
    quarter_high = quarter.loc[quarter["平均日销量"].idxmax()]
    holiday_rows = {row["节假日"]: row for _, row in holiday.iterrows()}

    fig = plt.figure(figsize=(16, 17))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.05], hspace=0.42, wspace=0.30)
    axes = [
        fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[3, 0]), fig.add_subplot(grid[3, 1]),
    ]

    # A：整体，不分月份和日期
    ax = axes[0]
    ax.scatter(sub["日平均售价"], sub["日总销量"], s=10, alpha=0.35, color=BLUE, edgecolors="none")
    if len(sub) >= 2:
        coef = np.polyfit(sub["日平均售价"], sub["日总销量"], 1)
        xline = np.linspace(float(sub["日平均售价"].min()), float(sub["日平均售价"].max()), 100)
        ax.plot(xline, np.polyval(coef, xline), color=ORANGE, linewidth=1.5)
    ax.set_title("A  整体关系（不分时段）")
    ax.set_xlabel("日平均售价（元/千克）")
    ax.set_ylabel("日总销量（千克）")
    ax.set_xlim(global_price_min, global_price_max)
    ax.set_ylim(0, global_quantity_max)
    ax.text(0.04, 0.95, f"线性相关={fmt(overall_linear, 3)}\n对数相关={fmt(overall_log, 3)}\n样本天数={len(sub)}", transform=ax.transAxes, va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.grid(alpha=0.18)

    # B：月份
    ax = axes[1]
    month_price_ax = plot_twin_lines(ax, month["月份"], month["平均日销量"], month["平均售价"], "月份", list(range(1, 13)), [f"{x}月" for x in range(1, 13)])
    ax.set_title("B  月份对比")
    ax.set_ylim(0, group_quantity_max)
    month_price_ax.set_ylim(0, group_price_max)
    ax.tick_params(axis="x", rotation=35)

    # C：季度
    ax = axes[2]
    quarter_price_ax = plot_twin_lines(ax, quarter["季度"], quarter["平均日销量"], quarter["平均售价"], "季度", [1, 2, 3, 4], ["一季度", "二季度", "三季度", "四季度"])
    ax.set_title("C  季度对比")
    ax.set_ylim(0, group_quantity_max)
    quarter_price_ax.set_ylim(0, group_price_max)

    # D：季节
    ax = axes[3]
    season_price_ax = plot_twin_lines(ax, season["季节"], season["平均日销量"], season["平均售价"], "季节", ["春季", "夏季", "秋季", "冬季"], ["春季", "夏季", "秋季", "冬季"])
    ax.set_title("D  季节对比")
    ax.set_ylim(0, group_quantity_max)
    season_price_ax.set_ylim(0, group_price_max)

    # E：日期趋势
    ax = axes[4]
    price_ax = ax.twinx()
    ax.plot(sub["销售日期"], sub["日总销量"], color=BLUE, alpha=0.22, linewidth=0.55)
    ax.plot(sub["销售日期"], sub["日总销量"].rolling(7, min_periods=1).mean(), color=BLUE, linewidth=1.4, label="销量七日移动平均")
    price_ax.plot(sub["销售日期"], sub["日平均售价"], color=ORANGE, alpha=0.78, linewidth=0.8, label="日平均售价")
    ax.set_title("E  逐日趋势")
    ax.set_xlabel("日期")
    ax.set_ylabel("日销量（千克）", color=BLUE)
    price_ax.set_ylabel("日平均售价（元/千克）", color=ORANGE)
    ax.set_ylim(0, global_quantity_max)
    price_ax.set_ylim(global_price_min, global_price_max)
    ax.tick_params(axis="y", labelcolor=BLUE)
    price_ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.18)
    price_ax.spines["top"].set_visible(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = price_ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=8)

    # F：节假日
    ax = axes[5]
    x = np.arange(len(holiday))
    width = 0.36
    bars = ax.bar(x - width / 2, holiday["平均日销量"], width, color=BLUE, label="平均日销量")
    price_ax = ax.twinx()
    price_ax.bar(x + width / 2, holiday["平均售价"], width, color=ORANGE, label="平均售价")
    labels = holiday["节假日"].tolist()
    ax.set_xticks(x, labels)
    ax.set_title("F  节假日与非节假日对比")
    ax.set_ylabel("平均日销量（千克）", color=BLUE)
    price_ax.set_ylabel("平均售价（元/千克）", color=ORANGE)
    ax.set_ylim(0, group_quantity_max)
    price_ax.set_ylim(0, group_price_max)
    ax.tick_params(axis="y", labelcolor=BLUE)
    price_ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.grid(axis="y", alpha=0.18)
    price_ax.spines["top"].set_visible(False)
    ax.legend([bars, price_ax.patches[0]], ["平均日销量", "平均售价"], loc="upper left", frameon=False, fontsize=8)

    # G：季节销量—售价表
    ax = axes[6]
    ax.axis("off")
    season_table_rows = []
    for _, row in season.iterrows():
        count = int((sub["季节"] == row["季节"]).sum())
        season_table_rows.append([row["季节"], fmt(row["平均日销量"], 1), fmt(row["平均售价"], 2), str(count)])
    season_table = ax.table(cellText=season_table_rows, colLabels=["季节", "平均销量（千克）", "平均售价（元/千克）", "样本天数"], loc="center", cellLoc="center", colLoc="center")
    season_table.auto_set_font_size(False)
    season_table.set_fontsize(8)
    season_table.scale(1.0, 1.65)
    ax.set_title("G  同一季节的销量与售价对比表", pad=12)

    # H：关键指标汇总表
    ax = axes[7]
    ax.axis("off")
    month_count = int((sub["月份"] == int(month_high["月份"])).sum())
    quarter_count = int((sub["季度"] == int(quarter_high["季度"])).sum())
    table_rows = [
        ["整体", "全部日期", fmt(sub["日总销量"].mean(), 1), fmt(sub["日平均售价"].mean(), 2), str(len(sub))],
        ["销量最高月份", f"{int(month_high['月份'])}月", fmt(month_high["平均日销量"], 1), fmt(month_high["平均售价"], 2), str(month_count)],
        ["销量最高季度", f"第{int(quarter_high['季度'])}季度", fmt(quarter_high["平均日销量"], 1), fmt(quarter_high["平均售价"], 2), str(quarter_count)],
    ]
    if "节假日" in holiday_rows and "非节假日" in holiday_rows:
        table_rows.extend([
            ["节假日", "节假日", fmt(holiday_rows["节假日"]["平均日销量"], 1), fmt(holiday_rows["节假日"]["平均售价"], 2), str(int(holiday_rows["节假日"]["样本天数"]))],
            ["非节假日", "非节假日", fmt(holiday_rows["非节假日"]["平均日销量"], 1), fmt(holiday_rows["非节假日"]["平均售价"], 2), str(int(holiday_rows["非节假日"]["样本天数"]))],
        ])
    table = ax.table(cellText=table_rows, colLabels=["对比项目", "分组", "平均销量（千克）", "平均售价（元/千克）", "样本天数"], loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.65)
    ax.set_title("H  关键指标汇总表", pad=12)

    fig.suptitle(f"{category}：整体、月份、季度、季节、日期与节假日对比", fontsize=16, y=0.995)
    fig.text(0.5, 0.012, "注：所有面板固定使用日总销量和加权日平均售价；月份、季度、季节和节假日展示分组均值，日期展示逐日值。", ha="center", fontsize=8, color=GRAY)
    fig.savefig(FIG_DIR / f"{category}_分组对比.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{category}_分组对比.pdf", bbox_inches="tight")
    plt.close(fig)

print("六个品类分组对比图生成完成")
for category in CATEGORIES:
    print(f"{category}: {FIG_DIR / f'{category}_分组对比.png'}")
