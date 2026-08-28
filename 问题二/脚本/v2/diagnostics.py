# -*- coding: utf-8 -*-
"""端到端需求回测、校准和最终验收所需诊断。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CATEGORIES, OUTPUT_DIR
from .cost_models import forecast_cost
from .demand_models import DemandFit, fit_demand_model, make_folds, predict_point
from .price_response import PriceFit, fit_price_model, response_multiplier


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / max(np.abs(actual).sum(), 1e-9))


def run_end_to_end_backtest(
    normal_panel: pd.DataFrame,
    enriched_panel: pd.DataFrame,
    selected_demand_models: dict[str, str],
    selected_price_fits: dict[str, PriceFit],
    selected_cost: dict[str, dict[str, str]],
    last_date: pd.Timestamp,
    demand_summary: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    detail_rows: list[dict[str, Any]] = []
    folds = make_folds(last_date)
    for cat in CATEGORIES:
        for fold in folds:
            cutoff = pd.Timestamp(fold["截止日"])
            train_normal = normal_panel[(normal_panel["品类"] == cat) & (normal_panel["销售日期"] <= cutoff)].copy()
            train_enriched = enriched_panel[(enriched_panel["品类"] == cat) & (enriched_panel["销售日期"] <= cutoff)].copy()
            test = enriched_panel[(enriched_panel["品类"] == cat) & (enriched_panel["销售日期"] > cutoff) & (enriched_panel["销售日期"] <= cutoff + pd.Timedelta(days=7))].copy()
            test = test.dropna(subset=["正常销售量", "正常销售售价"])
            if len(train_normal) < 120 or len(train_enriched) < 120 or len(test) < 2:
                continue
            model = fit_demand_model(train_normal, cat, selected_demand_models[cat], "正常销售量", use_prequential_residual=False)
            price_template = selected_price_fits[cat]
            try:
                price_fit = fit_price_model(train_enriched, cat, price_template.model_name, price_template.source_name, "正常销售量", rng, bootstrap_reps=0)
            except (ValueError, np.linalg.LinAlgError):
                price_fit = None
            choice = selected_cost[cat]
            cost_col = "销量加权进价" if choice["成本口径"] == "销量加权成本" else "固定篮子成本指数"
            cost_train = train_enriched.dropna(subset=[cost_col])
            cost_pred = forecast_cost(cost_train, pd.DatetimeIndex(test["销售日期"]), choice["成本预测方法"], cost_col)
            base = predict_point(model, test["销售日期"])
            if price_fit is None:
                adjusted = base.copy()
            else:
                adjusted = np.array(
                    [base[i] * response_multiplier(price_fit, date, price, cost_pred.loc[pd.Timestamp(date).normalize()]) for i, (date, price) in enumerate(zip(test["销售日期"], test["正常销售售价"]))],
                    dtype=float,
                )
            actual_normal = test["正常销售量"].to_numpy(float)
            actual_net = test["净销售量"].to_numpy(float)
            for target_name, actual in [("正常销售量", actual_normal), ("净销售量", actual_net)]:
                prediction = adjusted
                detail_rows.append(
                    {
                        "品类": cat,
                        "需求口径": target_name,
                        "需求模型": selected_demand_models[cat],
                        "价格响应模型": price_template.model_name,
                        "成本预测方法": choice["成本预测方法"],
                        "成本口径": choice["成本口径"],
                        "折组": fold["折组"],
                        "训练截止日": cutoff.date().isoformat(),
                        "测试开始日": pd.Timestamp(test["销售日期"].min()).date().isoformat(),
                        "测试结束日": pd.Timestamp(test["销售日期"].max()).date().isoformat(),
                        "训练样本数": int(len(train_normal)),
                        "测试样本数": int(len(test)),
                        "实际总量": float(actual.sum()),
                        "实际绝对量总和": float(np.abs(actual).sum()),
                        "预测总量": float(prediction.sum()),
                        "绝对误差总量": float(np.abs(actual - prediction).sum()),
                        "端到端加权绝对百分比误差": wape(actual, prediction),
                        "端到端平均绝对误差": float(np.mean(np.abs(actual - prediction))),
                        "价格响应系数": float(price_fit.coefficient) if price_fit is not None else 0.0,
                        "训练最大日期": pd.Timestamp(train_normal["销售日期"].max()).date().isoformat(),
                        "泄漏检查": "通过" if pd.Timestamp(train_normal["销售日期"].max()) < pd.Timestamp(test["销售日期"].min()) else "失败",
                    }
                )
    detail_df = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for (cat, target), sub in detail_df.groupby(["品类", "需求口径"], sort=False):
        actual_abs_total = sub["实际绝对量总和"].sum()
        absolute_error_total = sub["绝对误差总量"].sum()
        end_wape = float(absolute_error_total / max(actual_abs_total, 1e-9))
        fold_wape = sub["端到端加权绝对百分比误差"].to_numpy(float)
        baseline_rows = demand_summary[(demand_summary["品类"] == cat) & (demand_summary["口径"] == ("正常销售" if target == "正常销售量" else "全量净需求"))]
        baseline_wape = float(baseline_rows.loc[baseline_rows["需求模型"] == sub.iloc[0]["需求模型"], "池化加权绝对百分比误差"].iloc[0]) if not baseline_rows.loc[baseline_rows["需求模型"] == sub.iloc[0]["需求模型"]].empty else np.nan
        summary_rows.append(
            {
                "品类": cat,
                "需求口径": target,
                "端到端池化加权绝对百分比误差": end_wape,
                "端到端折均加权绝对百分比误差": float(fold_wape.mean()),
                "端到端折中位加权绝对百分比误差": float(np.median(fold_wape)),
                "单独需求模型池化误差": baseline_wape,
                "端到端相对单独误差变化": end_wape / baseline_wape - 1.0 if np.isfinite(baseline_wape) and baseline_wape > 0 else np.nan,
                "是否明显劣化": "是" if np.isfinite(baseline_wape) and end_wape > 1.05 * baseline_wape else "否",
                "回测折数": int(len(sub)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    calibration_rows: list[dict[str, Any]] = []
    calibration: dict[str, float] = {}
    for cat in CATEGORIES:
        sub = detail_df[(detail_df["品类"] == cat) & (detail_df["需求口径"] == "正常销售量")]
        if sub.empty or sub["预测总量"].sum() <= 0:
            factor = 1.0
        else:
            factor = float(np.clip(sub["实际总量"].sum() / sub["预测总量"].sum(), 0.5, 2.0))
        calibration[cat] = factor
        summary_cat = summary_df[(summary_df["品类"] == cat) & (summary_df["需求口径"] == "正常销售量")]
        calibration_rows.append(
            {
                "品类": cat,
                "端到端需求校准系数": factor,
                "校准前端到端误差": float(summary_cat["端到端池化加权绝对百分比误差"].iloc[0]) if not summary_cat.empty else np.nan,
                "校准后用途": "未来七日正常参考需求中心乘以该系数",
                "是否因明显劣化而校准": str(summary_cat["是否明显劣化"].iloc[0]) if not summary_cat.empty else "否",
            }
        )
    calibration_df = pd.DataFrame(calibration_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(OUTPUT_DIR / "05_端到端需求回测明细.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "05_端到端需求回测汇总.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(OUTPUT_DIR / "05_需求校准系数.csv", index=False, encoding="utf-8-sig")
    return detail_df, summary_df, calibration_df, calibration
