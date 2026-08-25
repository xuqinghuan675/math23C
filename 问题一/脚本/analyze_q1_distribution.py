from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


PROBLEM_DIR = Path(__file__).resolve().parents[1]
OUT = PROBLEM_DIR / "结果"
ITEM_PATH = OUT / "daily_item_panel.csv"
CATEGORY_PATH = OUT / "daily_category_panel.csv"
FIG_DIR = PROBLEM_DIR / "图表"


def quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q, interpolation="linear"))


def safe_cv(mean: float, std: float) -> float | None:
    return None if mean <= 0 else std / mean


def summarize(series: pd.Series) -> dict[str, float | int | None]:
    values = series.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return {
        "n_days": int(values.notna().sum()),
        "mean_kg": mean,
        "median_kg": float(values.median()),
        "std_kg": std,
        "cv": safe_cv(mean, std),
        "min_kg": float(values.min()),
        "q1_kg": q1,
        "q3_kg": q3,
        "iqr_kg": q3 - q1,
        "p95_kg": quantile(values, 0.95),
        "max_kg": float(values.max()),
        "skewness": float(skew(values, bias=False)) if len(values) >= 3 else None,
        "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)) if len(values) >= 4 else None,
        "zero_days": int((values == 0).sum()),
        "positive_days": int((values > 0).sum()),
        "zero_rate": float((values == 0).mean()),
    }


def round_frame(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    numeric_cols = frame.select_dtypes(include=[np.number]).columns
    result = frame.copy()
    result[numeric_cols] = result[numeric_cols].round(digits)
    return result


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    item = pd.read_csv(ITEM_PATH, encoding="utf-8-sig")
    category = pd.read_csv(CATEGORY_PATH, encoding="utf-8-sig")
    item["sales_date"] = pd.to_datetime(item["sales_date"])
    category["sales_date"] = pd.to_datetime(category["sales_date"])

    observed_item = item[item["date_has_records"] == 1].copy()
    observed_category = category[category["date_has_records"] == 1].copy()
    observed_product_codes = set(
        item.loc[item["product_has_records"] == 1, "product_code"].unique()
    )
    catalog_product_map = item[["product_code", "category_name"]].drop_duplicates()
    catalog_counts = catalog_product_map.groupby("category_name")["product_code"].nunique()

    # Item-level descriptive statistics.
    item_summary_rows: list[dict] = []
    for (code, name, category_name), group in observed_item[
        observed_item["product_code"].isin(observed_product_codes)
    ].groupby(
        ["product_code", "product_name", "category_name"], sort=False
    ):
        stats = summarize(group["net_sales_kg"])
        stats.update(
            {
                "product_code": code,
                "product_name": name,
                "category_name": category_name,
                "total_net_sales_kg": float(group["net_sales_kg"].sum()),
                "observed_sales_days": int((group["net_sales_kg"] > 0).sum()),
                "return_days": int((group["return_rows"] > 0).sum()),
            }
        )
        item_summary_rows.append(stats)
    item_summary = pd.DataFrame(item_summary_rows)
    item_summary = item_summary.sort_values(
        ["total_net_sales_kg", "mean_kg"], ascending=False
    ).reset_index(drop=True)
    item_summary["total_share"] = item_summary["total_net_sales_kg"] / item_summary[
        "total_net_sales_kg"
    ].sum()
    item_summary["total_rank"] = np.arange(1, len(item_summary) + 1)
    save_csv(round_frame(item_summary), "item_distribution_summary.csv")

    # Category-level descriptive statistics.
    category_summary_rows: list[dict] = []
    for category_name, group in observed_category.groupby("category_name", sort=False):
        stats = summarize(group["net_sales_kg"])
        stats.update(
            {
                "category_name": category_name,
                "total_net_sales_kg": float(group["net_sales_kg"].sum()),
                "mean_active_product_count": float(group["active_product_count"].mean()),
                "max_active_product_count": int(group["active_product_count"].max()),
            }
        )
        category_summary_rows.append(stats)
    category_summary = pd.DataFrame(category_summary_rows)
    category_summary = category_summary.sort_values(
        "total_net_sales_kg", ascending=False
    ).reset_index(drop=True)
    category_summary["total_share"] = category_summary["total_net_sales_kg"] / category_summary[
        "total_net_sales_kg"
    ].sum()
    category_summary["total_rank"] = np.arange(1, len(category_summary) + 1)
    save_csv(round_frame(category_summary), "category_distribution_summary.csv")

    # Monthly profile and index: relative to category's observed-day mean.
    observed_category["year"] = observed_category["sales_date"].dt.year
    observed_category["month"] = observed_category["sales_date"].dt.month
    observed_category["year_month"] = observed_category["sales_date"].dt.to_period("M").astype(str)
    month_profile = (
        observed_category.groupby(["category_name", "month"], as_index=False)["net_sales_kg"]
        .agg(month_mean_kg="mean", month_median_kg="median", month_std_kg="std", n_days="count")
    )
    category_means = observed_category.groupby("category_name")["net_sales_kg"].mean()
    month_profile["month_index"] = month_profile.apply(
        lambda row: row["month_mean_kg"] / category_means[row["category_name"]], axis=1
    )
    save_csv(round_frame(month_profile), "category_month_profile.csv")

    # Year-by-month profile for checking repeatability across years.
    year_month_profile = (
        observed_category.groupby(["category_name", "year", "month"], as_index=False)["net_sales_kg"]
        .agg(mean_kg="mean", median_kg="median", n_days="count")
    )
    year_month_profile["month_index_within_category"] = year_month_profile.apply(
        lambda row: row["mean_kg"] / category_means[row["category_name"]], axis=1
    )
    save_csv(round_frame(year_month_profile), "category_year_month_profile.csv")

    # Weekday profile: Monday=0, Sunday=6; provide Chinese labels for output.
    weekday_names = {
        0: "周一",
        1: "周二",
        2: "周三",
        3: "周四",
        4: "周五",
        5: "周六",
        6: "周日",
    }
    observed_category["weekday"] = observed_category["sales_date"].dt.weekday
    observed_category["weekday_name"] = observed_category["weekday"].map(weekday_names)
    weekday_profile = (
        observed_category.groupby(["category_name", "weekday", "weekday_name"], as_index=False)[
            "net_sales_kg"
        ]
        .agg(weekday_mean_kg="mean", weekday_median_kg="median", weekday_std_kg="std", n_days="count")
    )
    weekday_profile["weekday_index"] = weekday_profile.apply(
        lambda row: row["weekday_mean_kg"] / category_means[row["category_name"]], axis=1
    )
    save_csv(round_frame(weekday_profile), "category_weekday_profile.csv")

    # Product contribution inside each category.
    item_summary["category_total_net_sales_kg"] = item_summary.groupby("category_name")[
        "total_net_sales_kg"
    ].transform("sum")
    item_summary["category_share"] = item_summary["total_net_sales_kg"] / item_summary[
        "category_total_net_sales_kg"
    ]
    item_summary["category_rank"] = item_summary.groupby("category_name")[
        "total_net_sales_kg"
    ].rank(method="first", ascending=False).astype(int)
    save_csv(round_frame(item_summary), "item_distribution_summary.csv")

    concentration_rows: list[dict] = []
    for category_name, group in item_summary.groupby("category_name", sort=False):
        shares = group.sort_values("category_share", ascending=False)["category_share"].to_numpy()
        total = float(shares.sum())
        hhi = float(np.sum(np.square(shares)))
        row = {
            "category_name": category_name,
            "product_count_in_catalog": int(catalog_counts[category_name]),
            "product_count_with_observed_sales": int(len(group)),
            "top1_share": float(shares[:1].sum() / total),
            "top3_share": float(shares[:3].sum() / total),
            "top5_share": float(shares[:5].sum() / total),
            "top10_share": float(shares[:10].sum() / total),
            "hhi_on_category_shares": hhi,
        }
        concentration_rows.append(row)
    concentration = pd.DataFrame(concentration_rows).sort_values("top3_share", ascending=False)
    save_csv(round_frame(concentration), "category_concentration.csv")

    # Overall daily total and date-level completeness summary.
    overall_daily = (
        observed_category.groupby("sales_date", as_index=False)["net_sales_kg"].sum()
        .rename(columns={"net_sales_kg": "all_category_net_sales_kg"})
    )
    overall_daily["month"] = overall_daily["sales_date"].dt.month
    overall_daily["weekday"] = overall_daily["sales_date"].dt.weekday
    save_csv(round_frame(overall_daily), "overall_daily_sales.csv")

    # Figures: enough to inspect the patterns, without flooding output.
    category_order = category_summary["category_name"].tolist()
    pivot_category = observed_category.pivot(
        index="sales_date", columns="category_name", values="net_sales_kg"
    ).reindex(columns=category_order)

    fig, ax = plt.subplots(figsize=(13, 6))
    for name in category_order:
        ax.plot(pivot_category.index, pivot_category[name], linewidth=0.8, alpha=0.8, label=name)
    ax.set_title("各蔬菜品类日净销售量趋势")
    ax.set_xlabel("日期")
    ax.set_ylabel("净销售量（千克）")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "category_daily_trends.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    box_values = [pivot_category[name].dropna().to_numpy() for name in category_order]
    ax.boxplot(box_values, tick_labels=category_order, showfliers=False)
    ax.set_title("各蔬菜品类日净销售量箱线图（隐藏极端点以便比较主体分布）")
    ax.set_xlabel("品类")
    ax.set_ylabel("净销售量（千克）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "category_boxplot.png", dpi=180)
    plt.close(fig)

    month_matrix = month_profile.pivot(index="category_name", columns="month", values="month_index").reindex(
        category_order
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(month_matrix.to_numpy(), aspect="auto", cmap="RdYlBu_r", vmin=0.4, vmax=1.8)
    ax.set_title("各品类月份销量指数（相对本品类有记录日均值）")
    ax.set_yticks(range(len(month_matrix.index)), month_matrix.index)
    ax.set_xticks(range(len(month_matrix.columns)), [f"{int(x)}月" for x in month_matrix.columns])
    for i in range(month_matrix.shape[0]):
        for j in range(month_matrix.shape[1]):
            value = month_matrix.iloc[i, j]
            if not pd.isna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="销量指数")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "category_month_heatmap.png", dpi=180)
    plt.close(fig)

    top_items = item_summary.head(20).sort_values("total_net_sales_kg")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_items["product_name"], top_items["total_net_sales_kg"], color="#4472C4")
    ax.set_title("单品净销售量排名（前20名）")
    ax.set_xlabel("累计净销售量（千克）")
    ax.set_ylabel("单品")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top20_product_total_sales.png", dpi=180)
    plt.close(fig)

    # JSON report with the key methodological counts and top findings.
    top_month_rows = []
    for category_name in category_order:
        rows = month_profile[month_profile["category_name"] == category_name].sort_values(
            "month_index", ascending=False
        )
        best = rows.iloc[0]
        worst = rows.iloc[-1]
        top_month_rows.append(
            {
                "category_name": category_name,
                "highest_month": int(best["month"]),
                "highest_month_index": float(best["month_index"]),
                "lowest_month": int(worst["month"]),
                "lowest_month_index": float(worst["month_index"]),
            }
        )
    report = {
        "method": {
            "measure": "net_sales_kg",
            "observed_dates_used": int(observed_category["sales_date"].nunique()),
            "missing_full_dates_excluded": int(category["date_has_records"].eq(0).sum() / category["category_name"].nunique()),
            "zero_for_missing_product_on_observed_date": True,
            "descriptive_statistics": [
                "mean",
                "median",
                "std",
                "cv",
                "quartiles",
                "p95",
                "skewness",
                "excess_kurtosis",
                "zero_rate",
            ],
        },
        "category_order_by_total": category_order,
        "top_products": item_summary.head(10)[
            ["product_code", "product_name", "category_name", "total_net_sales_kg", "total_share"]
        ].to_dict(orient="records"),
        "category_concentration": concentration.to_dict(orient="records"),
        "month_extremes": top_month_rows,
        "figures": [
            str(path.relative_to(PROBLEM_DIR)) for path in sorted(FIG_DIR.glob("*.png"))
        ],
    }
    (OUT / "distribution_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
