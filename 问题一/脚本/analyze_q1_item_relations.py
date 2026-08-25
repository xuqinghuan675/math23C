from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t


PROBLEM_DIR = Path(__file__).resolve().parents[1]
OUT = PROBLEM_DIR / "结果"
FIG_DIR = PROBLEM_DIR / "图表"
ITEM_PATH = OUT / "daily_item_panel.csv"


PRIMARY_MIN_POSITIVE_DAYS = 30
MIN_JOINT_POSITIVE_DAYS = 15
SENSITIVITY_THRESHOLDS = [30, 60, 90]


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running = 1.0
    for position in range(m - 1, -1, -1):
        rank = position + 1
        running = min(running, sorted_p[position] * m / rank)
        adjusted_sorted[position] = running
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return float(r), float(r)
    clipped = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(clipped)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = 1.959963984540054
    return float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))


def make_design(dates: pd.DatetimeIndex) -> np.ndarray:
    frame = pd.DataFrame({"date": dates})
    frame["trend_years"] = (frame["date"] - frame["date"].min()).dt.days / 365.25
    month = pd.get_dummies(frame["date"].dt.month, prefix="month", drop_first=True, dtype=float)
    weekday = pd.get_dummies(frame["date"].dt.weekday, prefix="weekday", drop_first=True, dtype=float)
    design = pd.concat(
        [
            pd.Series(1.0, index=frame.index, name="intercept"),
            frame[["trend_years"]],
            month,
            weekday,
        ],
        axis=1,
    )
    return design.to_numpy(dtype=float)


def residualize_matrix(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, axis=0, method="average")
    return np.corrcoef(ranks, rowvar=False)


def p_values_from_correlation(matrix: np.ndarray, n: int) -> np.ndarray:
    clipped = np.clip(matrix, -0.999999999999, 0.999999999999)
    statistic = clipped * np.sqrt((n - 2) / np.maximum(1.0 - clipped**2, 1e-15))
    p_values = 2.0 * t.sf(np.abs(statistic), df=n - 2)
    np.fill_diagonal(p_values, 0.0)
    return p_values


def add_matrix_values(
    rows: list[dict],
    codes: list[str],
    metadata: pd.DataFrame,
    raw_matrix: np.ndarray,
    raw_p: np.ndarray,
    residual_matrix: np.ndarray,
    residual_p: np.ndarray,
    positive_days: np.ndarray,
    values: np.ndarray,
) -> pd.DataFrame:
    raw_p_upper = raw_p[np.triu_indices(len(codes), k=1)]
    residual_p_upper = residual_p[np.triu_indices(len(codes), k=1)]
    raw_fdr_upper = bh_adjust(raw_p_upper)
    residual_fdr_upper = bh_adjust(residual_p_upper)
    raw_fdr_matrix = np.zeros_like(raw_p)
    residual_fdr_matrix = np.zeros_like(residual_p)
    upper = np.triu_indices(len(codes), k=1)
    raw_fdr_matrix[upper] = raw_fdr_upper
    raw_fdr_matrix[(upper[1], upper[0])] = raw_fdr_upper
    residual_fdr_matrix[upper] = residual_fdr_upper
    residual_fdr_matrix[(upper[1], upper[0])] = residual_fdr_upper

    positive_mask = values > 0
    for i, j in zip(*upper):
        code_a, code_b = codes[i], codes[j]
        joint_positive = int(np.sum(positive_mask[:, i] & positive_mask[:, j]))
        union_positive = int(np.sum(positive_mask[:, i] | positive_mask[:, j]))
        jaccard = joint_positive / union_positive if union_positive else 0.0
        raw_low, raw_high = fisher_ci(float(raw_matrix[i, j]), len(values))
        residual_low, residual_high = fisher_ci(float(residual_matrix[i, j]), len(values))
        metadata_a = metadata.loc[code_a]
        metadata_b = metadata.loc[code_b]
        rows.append(
            {
                "product_code_a": code_a,
                "product_name_a": metadata_a["product_name"],
                "category_a": metadata_a["category_name"],
                "product_code_b": code_b,
                "product_name_b": metadata_b["product_name"],
                "category_b": metadata_b["category_name"],
                "same_category": metadata_a["category_name"] == metadata_b["category_name"],
                "n_days": int(len(values)),
                "positive_days_a": int(positive_days[i]),
                "positive_days_b": int(positive_days[j]),
                "min_positive_days": int(min(positive_days[i], positive_days[j])),
                "joint_positive_days": joint_positive,
                "union_positive_days": union_positive,
                "jaccard_positive_days": jaccard,
                "raw_spearman_rho": float(raw_matrix[i, j]),
                "raw_spearman_p": float(raw_p[i, j]),
                "raw_spearman_p_fdr": float(raw_fdr_matrix[i, j]),
                "raw_spearman_ci_low": raw_low,
                "raw_spearman_ci_high": raw_high,
                "residual_spearman_rho": float(residual_matrix[i, j]),
                "residual_spearman_p": float(residual_p[i, j]),
                "residual_spearman_p_fdr": float(residual_fdr_matrix[i, j]),
                "residual_spearman_ci_low": residual_low,
                "residual_spearman_ci_high": residual_high,
                "residual_change": float(residual_matrix[i, j] - raw_matrix[i, j]),
            }
        )
    return pd.DataFrame(rows)


def period_name(year: int) -> str:
    if year == 2020:
        return "2020H2"
    if year == 2023:
        return "2023H1"
    return str(year)


def period_stability(
    candidates: pd.DataFrame,
    values: pd.DataFrame,
    residual_values: pd.DataFrame,
    dates: pd.DatetimeIndex,
    max_candidates: int = 100,
) -> pd.DataFrame:
    candidates = candidates.sort_values(
        ["residual_spearman_p_fdr", "residual_spearman_rho"], ascending=[True, False]
    ).head(max_candidates)
    date_periods = pd.Series([period_name(int(year)) for year in dates.year], index=dates)
    period_masks = {name: date_periods.eq(name).to_numpy() for name in date_periods.unique()}
    rows: list[dict] = []
    for row in candidates.itertuples(index=False):
        code_a = row.product_code_a
        code_b = row.product_code_b
        rhos: list[float] = []
        period_details: dict[str, float | None] = {}
        for name, mask in period_masks.items():
            x = values.loc[mask, code_a].to_numpy(dtype=float)
            y = values.loc[mask, code_b].to_numpy(dtype=float)
            rx = residual_values.loc[mask, code_a].to_numpy(dtype=float)
            ry = residual_values.loc[mask, code_b].to_numpy(dtype=float)
            if np.std(x) == 0 or np.std(y) == 0:
                rho = np.nan
            else:
                rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
            if np.std(rx) == 0 or np.std(ry) == 0:
                residual_rho = np.nan
            else:
                residual_rho = float(np.corrcoef(rankdata(rx), rankdata(ry))[0, 1])
            period_details[f"{name}_raw_rho"] = rho
            period_details[f"{name}_residual_rho"] = residual_rho
            if np.isfinite(residual_rho):
                rhos.append(residual_rho)
        signs = np.sign(np.asarray(rhos))
        nonzero = signs[signs != 0]
        rows.append(
            {
                "product_code_a": code_a,
                "product_name_a": row.product_name_a,
                "category_a": row.category_a,
                "product_code_b": code_b,
                "product_name_b": row.product_name_b,
                "category_b": row.category_b,
                "full_sample_residual_rho": row.residual_spearman_rho,
                "full_sample_residual_fdr": row.residual_spearman_p_fdr,
                "period_count_valid": int(len(rhos)),
                "period_sign_consistent": bool(len(nonzero) > 0 and np.all(nonzero == nonzero[0])),
                "period_min_abs_residual_rho": float(np.min(np.abs(rhos))) if rhos else np.nan,
                "period_median_abs_residual_rho": float(np.median(np.abs(rhos))) if rhos else np.nan,
                "periods_abs_residual_rho_ge_0_3": int(np.sum(np.abs(rhos) >= 0.3)),
                **period_details,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    item = pd.read_csv(ITEM_PATH, encoding="utf-8-sig")
    item["sales_date"] = pd.to_datetime(item["sales_date"])
    item = item[item["date_has_records"] == 1].copy()
    values = (
        item.pivot(index="sales_date", columns="product_code", values="net_sales_kg")
        .sort_index()
        .fillna(0.0)
    )
    metadata = item[["product_code", "product_name", "category_name"]].drop_duplicates().set_index(
        "product_code"
    )
    values = values.loc[:, sorted(values.columns)]
    metadata = metadata.loc[values.columns]
    dates = pd.DatetimeIndex(values.index)
    positive_days_series = (values > 0).sum(axis=0)
    eligible_codes = positive_days_series[positive_days_series >= PRIMARY_MIN_POSITIVE_DAYS].index.tolist()
    values_eligible = values[eligible_codes]
    positive_days = positive_days_series.loc[eligible_codes].to_numpy(dtype=int)
    design = make_design(dates)
    residual_values_array = residualize_matrix(values_eligible.to_numpy(dtype=float), design)
    residual_values = pd.DataFrame(
        residual_values_array, index=values_eligible.index, columns=eligible_codes
    )

    raw_matrix = correlation_matrix(values_eligible.to_numpy(dtype=float))
    residual_matrix = correlation_matrix(residual_values_array)
    raw_p = p_values_from_correlation(raw_matrix, len(values_eligible))
    residual_p = p_values_from_correlation(residual_matrix, len(values_eligible))

    pair_rows: list[dict] = []
    pairs = add_matrix_values(
        pair_rows,
        eligible_codes,
        metadata,
        raw_matrix,
        raw_p,
        residual_matrix,
        residual_p,
        positive_days,
        values_eligible.to_numpy(dtype=float),
    )
    pairs["eligible_for_overlap_screen"] = pairs["joint_positive_days"] >= MIN_JOINT_POSITIVE_DAYS
    pairs["raw_strong"] = pairs["raw_spearman_rho"].abs() >= 0.5
    pairs["residual_strong"] = pairs["residual_spearman_rho"].abs() >= 0.5
    pairs["residual_significant_fdr"] = pairs["residual_spearman_p_fdr"] < 0.05
    pairs["robust_residual_relation"] = (
        pairs["eligible_for_overlap_screen"]
        & pairs["residual_strong"]
        & pairs["residual_significant_fdr"]
    )
    pairs = pairs.sort_values(
        ["residual_spearman_rho", "jaccard_positive_days"], ascending=[False, False]
    ).reset_index(drop=True)
    pairs.round(8).to_csv(OUT / "item_relation_primary.csv", index=False, encoding="utf-8-sig")

    screened = pairs[pairs["eligible_for_overlap_screen"]].copy()
    top_positive = screened.sort_values(
        ["residual_spearman_rho", "jaccard_positive_days"], ascending=[False, False]
    ).head(50)
    top_negative = screened.sort_values(
        ["residual_spearman_rho", "jaccard_positive_days"], ascending=[True, False]
    ).head(50)
    top_edges = pd.concat([top_positive, top_negative]).drop_duplicates(
        subset=["product_code_a", "product_code_b"]
    )
    top_edges.round(8).to_csv(OUT / "item_relation_top_edges.csv", index=False, encoding="utf-8-sig")

    # Sensitivity: reapply FDR within each minimum positive-day threshold.
    sensitivity_rows: list[dict] = []
    for threshold in SENSITIVITY_THRESHOLDS:
        sub = pairs[pairs["min_positive_days"] >= threshold].copy()
        sub["residual_fdr_threshold_specific"] = bh_adjust(sub["residual_spearman_p"].to_numpy())
        sub["raw_fdr_threshold_specific"] = bh_adjust(sub["raw_spearman_p"].to_numpy())
        overlap = sub[sub["joint_positive_days"] >= MIN_JOINT_POSITIVE_DAYS]
        robust = overlap[
            (overlap["residual_spearman_rho"].abs() >= 0.5)
            & (overlap["residual_fdr_threshold_specific"] < 0.05)
        ]
        top = robust.sort_values("residual_spearman_rho", ascending=False).head(3)
        sensitivity_rows.append(
            {
                "min_positive_days_threshold": threshold,
                "eligible_product_count": int((positive_days_series >= threshold).sum()),
                "eligible_pair_count": int(len(sub)),
                "pair_count_with_joint_positive_ge_15": int(len(overlap)),
                "residual_fdr_lt_0_05_count": int(
                    (overlap["residual_fdr_threshold_specific"] < 0.05).sum()
                ),
                "robust_abs_residual_rho_ge_0_5_count": int(len(robust)),
                "top_robust_pairs": "; ".join(
                    f"{r.product_name_a}-{r.product_name_b} ({r.residual_spearman_rho:.3f})"
                    for r in top.itertuples()
                ),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT / "item_relation_sensitivity.csv", index=False, encoding="utf-8-sig")

    # Same-category versus cross-category summary, based on the primary all-pair FDR family.
    relation_summary = (
        pairs.groupby("same_category", as_index=False)
        .agg(
            pair_count=("same_category", "size"),
            median_abs_raw_rho=("raw_spearman_rho", lambda x: float(x.abs().median())),
            median_abs_residual_rho=("residual_spearman_rho", lambda x: float(x.abs().median())),
            raw_abs_rho_ge_0_5_count=("raw_strong", "sum"),
            residual_abs_rho_ge_0_5_count=("residual_strong", "sum"),
            residual_fdr_lt_0_05_count=("residual_significant_fdr", "sum"),
            robust_relation_count=("robust_residual_relation", "sum"),
            median_joint_positive_days=("joint_positive_days", "median"),
            median_jaccard=("jaccard_positive_days", "median"),
        )
    )
    relation_summary["relation_scope"] = relation_summary["same_category"].map(
        {True: "同品类内部", False: "跨品类"}
    )
    relation_summary.to_csv(OUT / "item_relation_scope_summary.csv", index=False, encoding="utf-8-sig")

    # Stability of the strongest 100 screened candidates across the four available time blocks.
    candidates = screened[screened["robust_residual_relation"]].copy()
    stability = period_stability(
        candidates,
        values_eligible,
        residual_values,
        dates,
        max_candidates=100,
    )
    stability.round(8).to_csv(OUT / "item_relation_top100_period_stability.csv", index=False, encoding="utf-8-sig")

    # Plot top positive and negative residual edges after overlap screen.
    plot_rows = pd.concat(
        [
            screened.sort_values("residual_spearman_rho", ascending=False).head(12),
            screened.sort_values("residual_spearman_rho", ascending=True).head(12),
        ]
    ).drop_duplicates(subset=["product_code_a", "product_code_b"])
    plot_rows = plot_rows.sort_values("residual_spearman_rho")
    labels = [
        f"{row.product_name_a}—{row.product_name_b}"
        for row in plot_rows.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = ["#4472C4" if value >= 0 else "#ED7D31" for value in plot_rows["residual_spearman_rho"]]
    ax.barh(labels, plot_rows["residual_spearman_rho"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("单品两两关联：控制月份、星期和趋势后的 Spearman 相关（代表性边）")
    ax.set_xlabel("残差 Spearman 相关系数")
    ax.set_ylabel("单品组合")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "item_relation_top_edges.png", dpi=180)
    plt.close(fig)

    # Distribution of residual correlations by scope.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_data = [
        pairs.loc[pairs["same_category"], "residual_spearman_rho"],
        pairs.loc[~pairs["same_category"], "residual_spearman_rho"],
    ]
    ax.boxplot(plot_data, tick_labels=["同品类内部", "跨品类"], showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("单品残差相关系数分布")
    ax.set_ylabel("残差 Spearman 相关系数")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "item_relation_scope_boxplot.png", dpi=180)
    plt.close(fig)

    raw_robust = pairs[
        pairs["eligible_for_overlap_screen"]
        & pairs["raw_strong"]
        & (pairs["raw_spearman_p_fdr"] < 0.05)
    ]
    residual_robust = screened[screened["robust_residual_relation"]]
    stable_top100 = stability[
        stability["period_sign_consistent"]
        & (stability["period_count_valid"] >= 3)
        & (stability["periods_abs_residual_rho_ge_0_3"] >= 3)
    ]
    report = {
        "method": {
            "observed_dates": int(len(values)),
            "catalog_products_in_panel": int(values.shape[1]),
            "products_with_any_positive_sales": int((positive_days_series > 0).sum()),
            "primary_min_positive_days": PRIMARY_MIN_POSITIVE_DAYS,
            "primary_eligible_products": int(len(eligible_codes)),
            "primary_pair_count": int(len(pairs)),
            "minimum_joint_positive_days_screen": MIN_JOINT_POSITIVE_DAYS,
            "primary_correlation": "Spearman rank correlation on daily net sales",
            "controlled_correlation": "Spearman rank correlation after residualizing month, weekday and linear trend",
            "multiple_testing": "Benjamini-Hochberg FDR across all primary eligible pairs",
            "sparsity_note": "zero sales on an observed store day are retained as zero; listing or stockout status is unavailable",
            "interpretation_boundary": "association, not causation",
        },
        "counts": {
            "primary_pairs_with_joint_positive_ge_15": int(len(screened)),
            "raw_strong_after_overlap_and_fdr": int(len(raw_robust)),
            "residual_strong_after_overlap_and_fdr": int(len(residual_robust)),
            "residual_fdr_lt_0_05_all_pairs": int((pairs["residual_spearman_p_fdr"] < 0.05).sum()),
        },
        "top_positive_residual_pairs": screened.sort_values(
            "residual_spearman_rho", ascending=False
        )
        .head(10)[
            [
                "product_name_a",
                "category_a",
                "product_name_b",
                "category_b",
                "joint_positive_days",
                "jaccard_positive_days",
                "raw_spearman_rho",
                "residual_spearman_rho",
                "residual_spearman_p_fdr",
            ]
        ]
        .to_dict(orient="records"),
        "top_negative_residual_pairs": screened.sort_values(
            "residual_spearman_rho", ascending=True
        )
        .head(10)[
            [
                "product_name_a",
                "category_a",
                "product_name_b",
                "category_b",
                "joint_positive_days",
                "jaccard_positive_days",
                "raw_spearman_rho",
                "residual_spearman_rho",
                "residual_spearman_p_fdr",
            ]
        ]
        .to_dict(orient="records"),
        "scope_summary": relation_summary.to_dict(orient="records"),
        "top100_stability": {
            "candidate_count": int(len(stability)),
            "stable_count": int(len(stable_top100)),
            "stable_definition": "at least 3 valid periods, same residual-correlation sign, and |rho| >= 0.3 in at least 3 periods",
        },
        "figures": [
            "图表/item_relation_top_edges.png",
            "图表/item_relation_scope_boxplot.png",
        ],
    }
    (OUT / "item_relation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
