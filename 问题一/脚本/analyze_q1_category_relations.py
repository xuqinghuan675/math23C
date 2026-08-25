from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROBLEM_DIR = Path(__file__).resolve().parents[1]
OUT = PROBLEM_DIR / "结果"
FIG_DIR = PROBLEM_DIR / "图表"
CATEGORY_PATH = OUT / "daily_category_panel.csv"


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH adjusted p-values in the original order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m, 0, -1):
        idx = rank - 1
        value = p[order[idx]] * m / rank
        running = min(running, value)
        adjusted_sorted[idx] = running
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Approximate confidence interval for a correlation using Fisher's z."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (float(r), float(r))
    r_clip = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r_clip)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = 1.959963984540054
    return tuple(float(x) for x in (np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)))


def make_design(dates: pd.DatetimeIndex) -> np.ndarray:
    """Intercept + linear trend + month dummies + weekday dummies."""
    frame = pd.DataFrame({"date": dates})
    frame["trend_years"] = (frame["date"] - frame["date"].min()).dt.days / 365.25
    month = pd.get_dummies(frame["date"].dt.month, prefix="month", drop_first=True, dtype=float)
    weekday = pd.get_dummies(frame["date"].dt.weekday, prefix="weekday", drop_first=True, dtype=float)
    design = pd.concat(
        [pd.Series(1.0, index=frame.index, name="intercept"), frame[["trend_years"]], month, weekday],
        axis=1,
    )
    return design.to_numpy(dtype=float)


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def corr_row(
    x: np.ndarray,
    y: np.ndarray,
    name_x: str,
    name_y: str,
    family: str,
) -> dict[str, float | int | str]:
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    pearson_low, pearson_high = fisher_ci(float(pearson_r), len(x))
    spearman_low, spearman_high = fisher_ci(float(spearman_rho), len(x))
    return {
        "category_a": name_x,
        "category_b": name_y,
        "n_days": int(len(x)),
        "family": family,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "pearson_ci_low": pearson_low,
        "pearson_ci_high": pearson_high,
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "spearman_ci_low": spearman_low,
        "spearman_ci_high": spearman_high,
    }


def add_adjusted_p_values(frame: pd.DataFrame, p_col: str, out_col: str) -> pd.DataFrame:
    result = frame.copy()
    result[out_col] = benjamini_hochberg(result[p_col].tolist())
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    raw = pd.read_csv(CATEGORY_PATH, encoding="utf-8-sig")
    raw["sales_date"] = pd.to_datetime(raw["sales_date"])
    raw = raw[raw["date_has_records"] == 1].copy()
    pivot = (
        raw.pivot(index="sales_date", columns="category_name", values="net_sales_kg")
        .sort_index()
    )
    if pivot.isna().any().any():
        raise ValueError("unexpected missing category values on an observed date")
    category_order = list(pivot.columns)
    dates = pd.DatetimeIndex(pivot.index)
    design = make_design(dates)

    raw_rows: list[dict] = []
    residual_rows: list[dict] = []
    residual_values: dict[str, np.ndarray] = {}
    for category_name in category_order:
        residual_values[category_name] = residualize(
            pivot[category_name].to_numpy(dtype=float), design
        )

    for category_a, category_b in combinations(category_order, 2):
        raw_rows.append(
            corr_row(
                pivot[category_a].to_numpy(dtype=float),
                pivot[category_b].to_numpy(dtype=float),
                category_a,
                category_b,
                "raw_daily_net_sales",
            )
        )
        residual_rows.append(
            corr_row(
                residual_values[category_a],
                residual_values[category_b],
                category_a,
                category_b,
                "residual_after_month_weekday_trend",
            )
        )

    raw_results = pd.DataFrame(raw_rows)
    raw_results = add_adjusted_p_values(raw_results, "spearman_p", "spearman_p_fdr")
    raw_results = add_adjusted_p_values(raw_results, "pearson_p", "pearson_p_fdr")
    residual_results = pd.DataFrame(residual_rows)
    residual_results = add_adjusted_p_values(
        residual_results, "spearman_p", "spearman_p_fdr"
    )
    residual_results = add_adjusted_p_values(
        residual_results, "pearson_p", "pearson_p_fdr"
    )

    # Compare raw and residual Spearman relationships in one table.
    comparison = raw_results[
        [
            "category_a",
            "category_b",
            "n_days",
            "spearman_rho",
            "spearman_p",
            "spearman_p_fdr",
            "pearson_r",
            "pearson_p",
            "pearson_p_fdr",
        ]
    ].rename(
        columns={
            "spearman_rho": "raw_spearman_rho",
            "spearman_p": "raw_spearman_p",
            "spearman_p_fdr": "raw_spearman_p_fdr",
            "pearson_r": "raw_pearson_r",
            "pearson_p": "raw_pearson_p",
            "pearson_p_fdr": "raw_pearson_p_fdr",
        }
    )
    comparison = comparison.merge(
        residual_results[
            [
                "category_a",
                "category_b",
                "spearman_rho",
                "spearman_p",
                "spearman_p_fdr",
                "pearson_r",
                "pearson_p",
                "pearson_p_fdr",
            ]
        ].rename(
            columns={
                "spearman_rho": "residual_spearman_rho",
                "spearman_p": "residual_spearman_p",
                "spearman_p_fdr": "residual_spearman_p_fdr",
                "pearson_r": "residual_pearson_r",
                "pearson_p": "residual_pearson_p",
                "pearson_p_fdr": "residual_pearson_p_fdr",
            }
        ),
        on=["category_a", "category_b"],
        how="left",
    )
    comparison["spearman_change_after_controls"] = (
        comparison["residual_spearman_rho"] - comparison["raw_spearman_rho"]
    )
    comparison["raw_abs_spearman"] = comparison["raw_spearman_rho"].abs()
    comparison["residual_abs_spearman"] = comparison["residual_spearman_rho"].abs()
    comparison = comparison.sort_values(
        ["residual_abs_spearman", "raw_abs_spearman"], ascending=False
    ).reset_index(drop=True)

    # Stability by calendar year. Partial years are kept explicit and reported.
    annual_rows: list[dict] = []
    years = sorted(dates.year.unique())
    for year in years:
        year_mask = dates.year == year
        year_pivot = pivot.loc[year_mask]
        for category_a, category_b in combinations(category_order, 2):
            x = year_pivot[category_a].to_numpy(dtype=float)
            y = year_pivot[category_b].to_numpy(dtype=float)
            rho, p_value = spearmanr(x, y)
            annual_rows.append(
                {
                    "year": int(year),
                    "category_a": category_a,
                    "category_b": category_b,
                    "n_days": int(len(x)),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_value),
                }
            )
    annual = pd.DataFrame(annual_rows)

    stability_rows: list[dict] = []
    for (category_a, category_b), group in annual.groupby(
        ["category_a", "category_b"], sort=False
    ):
        values = group.sort_values("year")["spearman_rho"].to_numpy(dtype=float)
        signs = np.sign(values)
        nonzero_signs = signs[signs != 0]
        sign_consistent = bool(
            len(nonzero_signs) > 0 and np.all(nonzero_signs == nonzero_signs[0])
        )
        stability_rows.append(
            {
                "category_a": category_a,
                "category_b": category_b,
                "year_count": int(len(values)),
                "rho_min": float(values.min()),
                "rho_max": float(values.max()),
                "rho_mean": float(values.mean()),
                "rho_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "sign_consistent": sign_consistent,
                "all_year_abs_rho_ge_0_3": bool(np.all(np.abs(values) >= 0.3)),
                "all_year_abs_rho_ge_0_5": bool(np.all(np.abs(values) >= 0.5)),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["sign_consistent", "rho_mean"], ascending=[False, False]
    )

    # Build symmetric matrices for visual inspection.
    def symmetric_matrix(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
        matrix = pd.DataFrame(np.eye(len(category_order)), index=category_order, columns=category_order)
        for row in frame.itertuples(index=False):
            matrix.loc[row.category_a, row.category_b] = getattr(row, value_col)
            matrix.loc[row.category_b, row.category_a] = getattr(row, value_col)
        return matrix

    raw_matrix = symmetric_matrix(raw_results, "spearman_rho")
    residual_matrix = symmetric_matrix(residual_results, "spearman_rho")
    raw_pearson_matrix = symmetric_matrix(raw_results, "pearson_r")

    def save_frame(frame: pd.DataFrame, name: str) -> None:
        frame.round(8).to_csv(OUT / name, index=False, encoding="utf-8-sig")

    save_frame(raw_results, "category_relation_raw.csv")
    save_frame(residual_results, "category_relation_residual.csv")
    save_frame(comparison, "category_relation_comparison.csv")
    save_frame(annual, "category_relation_by_year.csv")
    save_frame(stability, "category_relation_stability.csv")
    raw_matrix.round(8).to_csv(OUT / "category_spearman_matrix_raw.csv", encoding="utf-8-sig")
    residual_matrix.round(8).to_csv(
        OUT / "category_spearman_matrix_residual.csv", encoding="utf-8-sig"
    )
    raw_pearson_matrix.round(8).to_csv(
        OUT / "category_pearson_matrix_raw.csv", encoding="utf-8-sig"
    )

    def plot_heatmap(matrix: pd.DataFrame, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6.8))
        im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="相关系数")
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=180)
        plt.close(fig)

    plot_heatmap(raw_matrix, "品类日净销量 Spearman 相关系数（原始）", "category_spearman_raw.png")
    plot_heatmap(
        residual_matrix,
        "品类日净销量 Spearman 相关系数（控制月份、星期和趋势后）",
        "category_spearman_residual.png",
    )

    # A compact report with data and selection counts.
    raw_strong = raw_results[raw_results["spearman_rho"].abs() >= 0.5]
    raw_fdr = raw_results[raw_results["spearman_p_fdr"] < 0.05]
    residual_strong = residual_results[residual_results["spearman_rho"].abs() >= 0.5]
    residual_fdr = residual_results[residual_results["spearman_p_fdr"] < 0.05]
    report = {
        "method": {
            "n_observed_dates": int(len(pivot)),
            "category_count": int(len(category_order)),
            "pair_count": int(len(raw_results)),
            "primary_correlation": "Spearman rank correlation",
            "linear_reference": "Pearson correlation",
            "controls": "intercept + linear time trend + month fixed effects + weekday fixed effects",
            "multiple_testing": "Benjamini-Hochberg FDR within each 15-pair family",
            "confidence_interval": "approximate 95% Fisher-z interval",
            "interpretation_boundary": "association, not causation",
        },
        "raw": {
            "strong_abs_spearman_ge_0_5_count": int(len(raw_strong)),
            "spearman_fdr_lt_0_05_count": int(len(raw_fdr)),
            "top_pairs": raw_results.sort_values("spearman_rho", ascending=False)
            .head(5)[["category_a", "category_b", "spearman_rho", "spearman_p_fdr"]]
            .to_dict(orient="records"),
            "bottom_pairs": raw_results.sort_values("spearman_rho", ascending=True)
            .head(5)[["category_a", "category_b", "spearman_rho", "spearman_p_fdr"]]
            .to_dict(orient="records"),
        },
        "residual": {
            "strong_abs_spearman_ge_0_5_count": int(len(residual_strong)),
            "spearman_fdr_lt_0_05_count": int(len(residual_fdr)),
            "top_pairs": residual_results.sort_values("spearman_rho", ascending=False)
            .head(5)[["category_a", "category_b", "spearman_rho", "spearman_p_fdr"]]
            .to_dict(orient="records"),
            "bottom_pairs": residual_results.sort_values("spearman_rho", ascending=True)
            .head(5)[["category_a", "category_b", "spearman_rho", "spearman_p_fdr"]]
            .to_dict(orient="records"),
        },
        "stability": {
            "periods": years,
            "sign_consistent_count": int(stability["sign_consistent"].sum()),
            "all_year_abs_rho_ge_0_3_count": int(stability["all_year_abs_rho_ge_0_3"].sum()),
            "all_year_abs_rho_ge_0_5_count": int(stability["all_year_abs_rho_ge_0_5"].sum()),
        },
        "figures": [
            "图表/category_spearman_raw.png",
            "图表/category_spearman_residual.png",
        ],
    }
    (OUT / "category_relation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
