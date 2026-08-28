# -*- coding: utf-8 -*-
"""成本加成与销量关系、稳健区间、子时期稳定性和安慰剂检验。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import CATEGORIES, OUTPUT_DIR, PRICE_RESPONSE_MODELS, BOOTSTRAP_REPS, QUICK_BOOTSTRAP_REPS
from .demand_models import make_folds


STRUCTURE_CONTROL_COLUMNS = [
    "前一日销量集中度HHI",
    "前一日前三单品销量占比",
    "前一日当前权重与基准权重距离",
]


@dataclass
class PriceFit:
    category: str
    model_name: str
    source_name: str
    target_col: str
    coefficient: float
    fwl_coefficient: float
    se7: float
    se14: float
    p7: float
    lower90: float
    upper90: float
    lower95: float
    upper95: float
    bootstrap7: np.ndarray
    bootstrap14: np.ndarray
    residual_y: np.ndarray
    reference_map: dict[tuple[int, int], float]
    reference_fallback: float
    controls_medians: dict[str, float]
    origin: pd.Timestamp
    coefficients: np.ndarray
    design_names: list[str]
    sample_count: int
    markup_iqr: float
    price_iqr: float


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gram = x.T @ x
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(x, y, rcond=None)[0]


def _safe_p(value: float) -> float:
    return float(math.erfc(abs(float(value)) / math.sqrt(2.0)))


def _reference_map(frame: pd.DataFrame, price_col: str, cost_col: str) -> tuple[dict[tuple[int, int], float], float]:
    valid = frame[(frame[price_col] > 0) & (frame[cost_col] > 0)].copy()
    valid["实际加成"] = valid[price_col] / valid[cost_col] - 1.0
    valid = valid[np.isfinite(valid["实际加成"]) & (valid["实际加成"] > -0.95)]
    fallback = float(valid["实际加成"].median()) if len(valid) else 0.5
    group = valid.groupby(["星期", "月份"], observed=True)["实际加成"].median()
    mapping = {(int(k[0]), int(k[1])): float(v) for k, v in group.items()}
    return mapping, fallback


def _prepare_features(
    frame: pd.DataFrame,
    category: str,
    source_name: str,
    model_name: str,
    target_col: str,
    reference_map: dict[tuple[int, int], float] | None = None,
    reference_fallback: float | None = None,
    controls_medians: dict[str, float] | None = None,
    origin: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[tuple[int, int], float], float, dict[str, float], pd.Timestamp]:
    data = frame[frame["品类"] == category].copy()
    if source_name == "固定篮子":
        price_col, cost_col = "固定篮子价格指数", "固定篮子成本指数"
    else:
        price_col, cost_col = "正常销售售价", "正常销售进价"
    data["价格"] = pd.to_numeric(data[price_col], errors="coerce")
    data["成本"] = pd.to_numeric(data[cost_col], errors="coerce")
    data["目标"] = pd.to_numeric(data[target_col], errors="coerce")
    data = data[(data["价格"] > 0) & (data["成本"] > 0) & (data["目标"] > 0)].copy()
    data["实际加成"] = data["价格"] / data["成本"] - 1.0
    if reference_map is None or reference_fallback is None:
        reference_map, reference_fallback = _reference_map(data, "价格", "成本")
    data["参考加成"] = [reference_map.get((int(w), int(m)), reference_fallback) for w, m in zip(data["星期"], data["月份"])]
    data["加成偏离"] = data["实际加成"] - data["参考加成"]
    data["参考价格"] = data["成本"] * (1.0 + data["参考加成"])
    data["价格偏离"] = data["价格"] - data["参考价格"]
    data["对数加成偏离"] = np.log1p(np.maximum(data["实际加成"], -0.95)) - np.log1p(np.maximum(data["参考加成"], -0.95))
    data["对数价格比"] = np.log(np.maximum(data["价格"] / data["参考价格"], 1e-8))
    if controls_medians is None:
        controls_medians = {}
        for col in ["折扣销量占比", *STRUCTURE_CONTROL_COLUMNS]:
            value = pd.to_numeric(data.get(col, pd.Series(dtype=float)), errors="coerce").median()
            controls_medians[col] = float(value) if np.isfinite(value) else 0.0
    for col, value in controls_medians.items():
        if col not in data:
            data[col] = value
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(value)
    data["对数成本"] = np.log(np.maximum(data["成本"].to_numpy(float), 1e-6))
    if origin is None:
        origin = pd.Timestamp(data["销售日期"].min()).normalize()
    trend = (pd.to_datetime(data["销售日期"]).dt.normalize() - origin).dt.days / 365.25
    data["时间趋势"] = trend.astype(float)
    design = pd.DataFrame(index=data.index)
    design["常数项"] = 1.0
    names = ["常数项"]
    for value in range(2, 8):
        name = f"星期{value}"
        design[name] = (data["星期"].astype(int) == value).astype(float)
        names.append(name)
    for value in range(2, 13):
        name = f"月份{value}"
        design[name] = (data["月份"].astype(int) == value).astype(float)
        names.append(name)
    design["时间趋势"] = data["时间趋势"].to_numpy(float)
    design["对数成本"] = data["对数成本"].to_numpy(float)
    for col in ["折扣销量占比", *STRUCTURE_CONTROL_COLUMNS]:
        design[col] = data[col].to_numpy(float)
    names += ["时间趋势", "对数成本", "折扣销量占比", *STRUCTURE_CONTROL_COLUMNS]
    if model_name == "半对数加成":
        data["价格变量"] = data["加成偏离"]
    elif model_name == "对数加成":
        data["价格变量"] = data["对数加成偏离"]
    elif model_name == "半对数售价偏离":
        data["价格变量"] = data["价格偏离"]
    elif model_name == "对数售价比":
        data["价格变量"] = data["对数价格比"]
    else:
        raise ValueError(f"未知价格响应模型: {model_name}")
    design["价格变量"] = data["价格变量"].to_numpy(float)
    names.append("价格变量")
    valid = np.isfinite(design.to_numpy(float)).all(axis=1) & np.isfinite(data["目标"].to_numpy(float))
    data = data.loc[valid].copy()
    design_array = design.loc[valid, names].to_numpy(float)
    return data, design_array, names, reference_map, float(reference_fallback), controls_medians, origin


def _hac_covariance(x: np.ndarray, residual: np.ndarray, lag: int) -> np.ndarray:
    gram = x.T @ x
    try:
        inv = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(gram)
    scores = x * residual[:, None]
    meat = scores.T @ scores
    lag = min(max(0, int(lag)), max(0, len(residual) - 1))
    for current_lag in range(1, lag + 1):
        weight = 1.0 - current_lag / (lag + 1.0)
        cross = scores[current_lag:].T @ scores[:-current_lag]
        meat += weight * (cross + cross.T)
    return inv @ meat @ inv


def _fwl(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    if controls.shape[1] == 0:
        return float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
    control_beta_y = _ols(controls, y)
    control_beta_x = _ols(controls, x)
    y_res = y - controls @ control_beta_y
    x_res = x - controls @ control_beta_x
    return float(np.dot(x_res, y_res) / max(np.dot(x_res, x_res), 1e-12))


def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(1, n), dtype=int)
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    values: list[int] = []
    for start in starts:
        values.extend(((start + np.arange(block_length)) % n).tolist())
    return np.asarray(values[:n], dtype=int)


def _bootstrap_coefficients(x: np.ndarray, y: np.ndarray, reps: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if reps <= 0:
        return np.array([], dtype=float)
    values = []
    for _ in range(reps):
        indices = _moving_block_indices(len(y), block_length, rng)
        beta = _ols(x[indices], y[indices])
        values.append(float(beta[-1]))
    return np.asarray(values, dtype=float)


def fit_price_model(
    frame: pd.DataFrame,
    category: str,
    model_name: str,
    source_name: str,
    target_col: str,
    rng: np.random.Generator,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> PriceFit:
    data, x, names, ref_map, ref_fallback, medians, origin = _prepare_features(
        frame, category, source_name, model_name, target_col
    )
    y = np.log(np.maximum(data["目标"].to_numpy(float), 1e-6))
    coefficients = _ols(x, y)
    residual = y - x @ coefficients
    cov7 = _hac_covariance(x, residual, 7)
    cov14 = _hac_covariance(x, residual, 14)
    se7 = float(np.sqrt(max(cov7[-1, -1], 0.0)))
    se14 = float(np.sqrt(max(cov14[-1, -1], 0.0)))
    coefficient = float(coefficients[-1])
    bootstrap7 = _bootstrap_coefficients(x, y, bootstrap_reps, 7, rng)
    bootstrap14 = _bootstrap_coefficients(x, y, bootstrap_reps, 14, rng)
    se_boot = float(np.std(bootstrap7, ddof=1)) if len(bootstrap7) > 1 else se7
    lower90 = coefficient - 1.645 * se_boot
    upper90 = coefficient + 1.645 * se_boot
    lower95 = coefficient - 1.96 * se_boot
    upper95 = coefficient + 1.96 * se_boot
    controls = x[:, :-1]
    fwl_coefficient = _fwl(y, x[:, -1], controls)
    markup_iqr = float(np.quantile(data["实际加成"], 0.75) - np.quantile(data["实际加成"], 0.25))
    price_iqr = float(np.quantile(data["价格"], 0.75) - np.quantile(data["价格"], 0.25))
    return PriceFit(
        category=category,
        model_name=model_name,
        source_name=source_name,
        target_col=target_col,
        coefficient=coefficient,
        fwl_coefficient=fwl_coefficient,
        se7=se7,
        se14=se14,
        p7=_safe_p(coefficient / se7) if se7 > 0 else 1.0,
        lower90=float(lower90),
        upper90=float(upper90),
        lower95=float(lower95),
        upper95=float(upper95),
        bootstrap7=bootstrap7,
        bootstrap14=bootstrap14,
        residual_y=residual,
        reference_map=ref_map,
        reference_fallback=ref_fallback,
        controls_medians=medians,
        origin=origin,
        coefficients=coefficients,
        design_names=names,
        sample_count=int(len(data)),
        markup_iqr=markup_iqr,
        price_iqr=price_iqr,
    )


def _predict_log(fit: PriceFit, frame: pd.DataFrame, category: str) -> np.ndarray:
    data, x, names, _, _, _, _ = _prepare_features(
        frame,
        category,
        fit.source_name,
        fit.model_name,
        fit.target_col,
        fit.reference_map,
        fit.reference_fallback,
        fit.controls_medians,
        fit.origin,
    )
    if names != fit.design_names:
        raise ValueError("价格响应模型训练与预测的设计矩阵不一致")
    return x @ fit.coefficients


def _candidate_summary(fit: PriceFit) -> dict[str, Any]:
    return {
        "品类": fit.category,
        "价格口径": fit.source_name,
        "价格响应模型": fit.model_name,
        "样本数": fit.sample_count,
        "价格响应系数": fit.coefficient,
        "FWL系数": fit.fwl_coefficient,
        "稳健标准误7": fit.se7,
        "稳健标准误14": fit.se14,
        "稳健概率值": fit.p7,
        "百分之九十区间下限": fit.lower90,
        "百分之九十区间上限": fit.upper90,
        "百分之九十五区间下限": fit.lower95,
        "百分之九十五区间上限": fit.upper95,
        "百分之七自助法标准差": float(np.std(fit.bootstrap7, ddof=1)) if len(fit.bootstrap7) > 1 else np.nan,
        "百分之十四自助法标准差": float(np.std(fit.bootstrap14, ddof=1)) if len(fit.bootstrap14) > 1 else np.nan,
        "百分之七自助法负向比例": float(np.mean(fit.bootstrap7 < 0)) if len(fit.bootstrap7) else np.nan,
        "百分之十四自助法负向比例": float(np.mean(fit.bootstrap14 < 0)) if len(fit.bootstrap14) else np.nan,
        "加成有效四分位距": fit.markup_iqr,
        "售价有效四分位距": fit.price_iqr,
    }


def _choose_candidate(candidates: list[PriceFit], backtest_rows: list[dict[str, Any]], category: str, source_name: str) -> PriceFit:
    sub = [row for row in backtest_rows if row["品类"] == category and row["价格口径"] == source_name]
    if not sub:
        return candidates[0]
    summary = []
    for fit in candidates:
        rows = [row for row in sub if row["价格响应模型"] == fit.model_name and np.isfinite(row["回测误差"])]
        if rows:
            values = np.asarray([row["回测误差"] for row in rows], dtype=float)
            summary.append((fit, float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0))
    if not summary:
        return candidates[0]
    best = min(summary, key=lambda x: x[1])
    eligible = [item for item in summary if item[1] <= best[1] + best[2]]
    eligible.sort(key=lambda x: (PRICE_RESPONSE_MODELS.index(x[0].model_name), x[1]))
    return eligible[0][0]


def _price_backtest(
    frame: pd.DataFrame,
    category: str,
    target_col: str,
    last_date: pd.Timestamp,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    folds = make_folds(last_date)
    # 候选函数在全样本摘要中全部拟合；滚动回测对主函数执行，避免把同一批
    # 高维控制变量重复拟合上千次，同时保留每个候选的可复核全样本结果。
    backtest_models = ["半对数加成"]
    for source_name in ["销量加权", "固定篮子"]:
        for model_name in backtest_models:
            for fold in folds:
                cutoff = pd.Timestamp(fold["截止日"])
                train = frame[(frame["品类"] == category) & (frame["销售日期"] <= cutoff)]
                test = frame[(frame["品类"] == category) & (frame["销售日期"] > cutoff) & (frame["销售日期"] <= cutoff + pd.Timedelta(days=7))]
                price_col = "固定篮子价格指数" if source_name == "固定篮子" else "正常销售售价"
                cost_col = "固定篮子成本指数" if source_name == "固定篮子" else "正常销售进价"
                test = test.dropna(subset=[price_col, cost_col, target_col])
                try:
                    fit = fit_price_model(train, category, model_name, source_name, target_col, rng, bootstrap_reps=0)
                    if fit.sample_count < 80 or test.empty:
                        continue
                    pred_log = _predict_log(fit, test, category)
                    pred = np.exp(pred_log)
                    actual = test[target_col].to_numpy(float)
                    error = actual - pred
                    rows.append(
                        {
                            "品类": category,
                            "价格口径": source_name,
                            "价格响应模型": model_name,
                            "折组": fold["折组"],
                            "训练截止日": cutoff.date().isoformat(),
                            "测试开始日": pd.Timestamp(test["销售日期"].min()).date().isoformat(),
                            "测试结束日": pd.Timestamp(test["销售日期"].max()).date().isoformat(),
                            "训练样本数": fit.sample_count,
                            "测试样本数": int(len(test)),
                            "价格响应系数": fit.coefficient,
                            "回测误差": float(np.abs(error).sum() / max(np.abs(actual).sum(), 1e-9)),
                            "泄漏检查": "通过" if pd.Timestamp(train["销售日期"].max()) < pd.Timestamp(test["销售日期"].min()) else "失败",
                            "是否伪未来": "是" if fold["是否伪未来"] else "否",
                        }
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
    return rows


def _stability_rows(frame: pd.DataFrame, category: str, fit: PriceFit, target_col: str, rng: np.random.Generator) -> list[dict[str, Any]]:
    data = frame[(frame["品类"] == category) & (frame["销售日期"] <= pd.Timestamp("2023-06-30"))].copy()
    rows: list[dict[str, Any]] = []
    data["年份"] = pd.to_datetime(data["销售日期"]).dt.year
    data["半年度"] = np.where(pd.to_datetime(data["销售日期"]).dt.month <= 6, "上半年", "下半年")
    for label, sub in data.groupby(["年份", "半年度"], observed=True):
        try:
            sub_fit = fit_price_model(sub, category, fit.model_name, fit.source_name, target_col, rng, bootstrap_reps=0)
            rows.append({"品类": category, "子时期": f"{label[0]}年{label[1]}", "样本数": sub_fit.sample_count, "价格响应系数": sub_fit.coefficient, "稳健标准误7": sub_fit.se7, "方向": "负" if sub_fit.coefficient < 0 else "非负"})
        except (ValueError, np.linalg.LinAlgError):
            continue
    return rows


def _placebo_row(frame: pd.DataFrame, category: str, fit: PriceFit, target_col: str, rng: np.random.Generator) -> dict[str, Any]:
    data = frame[frame["品类"] == category].sort_values("销售日期").copy()
    if fit.source_name == "固定篮子":
        price_col, cost_col = "固定篮子价格指数", "固定篮子成本指数"
    else:
        price_col, cost_col = "正常销售售价", "正常销售进价"
    data["实际加成"] = data[price_col] / data[cost_col] - 1.0
    data["提前一期加成"] = data["实际加成"].shift(-1)
    data["提前一期价格"] = data[price_col].shift(-1)
    data["价格"] = data[price_col]
    data["成本"] = data[cost_col]
    data["目标"] = data[target_col]
    valid = data[["提前一期加成", "价格", "成本", "目标"]].notna().all(axis=1) & (data["提前一期加成"] > -0.95) & (data["目标"] > 0)
    data = data.loc[valid].copy()
    if len(data) < 100:
        return {"品类": category, "价格口径": fit.source_name, "价格响应模型": fit.model_name, "当期系数": fit.coefficient, "提前一期安慰剂系数": np.nan, "安慰剂是否更强": "无法判断", "安慰剂检查": "样本不足"}
    # 以提前一期价格替换当前售价，控制项仍严格按当前日期构造。
    # 拟合函数会根据这两列重新计算加成，因此不把提前一期加成直接写死。
    data["正常销售售价"] = data["提前一期价格"]
    data["正常销售进价"] = data[cost_col]
    try:
        placebo_fit = fit_price_model(data, category, "半对数加成", "销量加权", target_col, rng, bootstrap_reps=0)
        stronger = bool(abs(placebo_fit.coefficient) > abs(fit.coefficient) * 1.1)
        return {"品类": category, "价格口径": fit.source_name, "价格响应模型": fit.model_name, "当期系数": fit.coefficient, "提前一期安慰剂系数": placebo_fit.coefficient, "安慰剂是否更强": "是" if stronger else "否", "安慰剂检查": "未发现提前一期关系更强" if not stronger else "需警惕趋势或同时性污染"}
    except (ValueError, np.linalg.LinAlgError):
        return {"品类": category, "价格口径": fit.source_name, "价格响应模型": fit.model_name, "当期系数": fit.coefficient, "提前一期安慰剂系数": np.nan, "安慰剂是否更强": "无法判断", "安慰剂检查": "拟合失败"}


def run_price_response(
    panel: pd.DataFrame,
    last_date: pd.Timestamp,
    rng: np.random.Generator,
    quick: bool = False,
) -> dict[str, Any]:
    target_col = "正常销售量"
    net_target_col = "净销售量"
    bootstrap_reps = QUICK_BOOTSTRAP_REPS if quick else BOOTSTRAP_REPS
    backtest_rows: list[dict[str, Any]] = []
    all_fit_rows: list[dict[str, Any]] = []
    selected_fits: dict[str, PriceFit] = {}
    selected_full_fits: dict[str, PriceFit] = {}
    source_fits: dict[tuple[str, str, str], PriceFit] = {}
    for cat in CATEGORIES:
        cat_fits: dict[str, list[PriceFit]] = {"销量加权": [], "固定篮子": []}
        for source_name in ["销量加权", "固定篮子"]:
            for model_name in PRICE_RESPONSE_MODELS:
                try:
                    fit = fit_price_model(panel, cat, model_name, source_name, target_col, rng, bootstrap_reps=bootstrap_reps)
                    cat_fits[source_name].append(fit)
                    source_fits[(cat, source_name, model_name)] = fit
                    all_fit_rows.append(_candidate_summary(fit))
                except (ValueError, np.linalg.LinAlgError):
                    continue
        backtest_rows.extend(_price_backtest(panel, cat, target_col, last_date, rng))
        coverage = float(panel.loc[panel["品类"] == cat, "固定篮子覆盖率"].median()) if "固定篮子覆盖率" in panel else 0.0
        source_choice = "固定篮子" if coverage >= 0.70 and cat_fits["固定篮子"] else "销量加权"
        selected = _choose_candidate(cat_fits[source_choice], backtest_rows, cat, source_choice)
        selected_fits[cat] = selected
        try:
            full_fit = fit_price_model(panel, cat, selected.model_name, selected.source_name, net_target_col, rng, bootstrap_reps=bootstrap_reps)
        except (ValueError, np.linalg.LinAlgError):
            full_fit = fit_price_model(panel, cat, "半对数加成", "销量加权", net_target_col, rng, bootstrap_reps=bootstrap_reps)
        selected_full_fits[cat] = full_fit

    reliability_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    placebo_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        fit = selected_fits[cat]
        fold_betas: list[float] = []
        pseudo: dict[str, float] = {}
        for fold in make_folds(last_date):
            if fold["折组"] != "八折主要回测":
                continue
            cutoff = pd.Timestamp(fold["截止日"])
            train = panel[(panel["品类"] == cat) & (panel["销售日期"] <= cutoff)]
            try:
                fold_fit = fit_price_model(train, cat, fit.model_name, fit.source_name, target_col, rng, bootstrap_reps=0)
                fold_betas.append(fold_fit.coefficient)
                if cutoff in {pd.Timestamp("2021-06-30"), pd.Timestamp("2022-06-30")}:
                    pseudo[str(cutoff.year)] = fold_fit.coefficient
            except (ValueError, np.linalg.LinAlgError):
                continue
        negative_count = int(sum(x < 0 for x in fold_betas))
        fold_total = int(len(fold_betas))
        bootstrap_direction = float(np.mean(fit.bootstrap7 < 0)) if len(fit.bootstrap7) else np.nan
        direction_same = True
        for source_name in ["销量加权", "固定篮子"]:
            source_candidate = [source_fits[k] for k in source_fits if k[0] == cat and k[1] == source_name and k[2] == fit.model_name]
            if source_candidate:
                direction_same = direction_same and (np.sign(source_candidate[0].coefficient) == np.sign(fit.coefficient) or abs(source_candidate[0].coefficient) < 1e-12)
        placebo = _placebo_row(panel, cat, fit, target_col, rng)
        placebo_rows.append(placebo)
        placebo_pass = placebo["安慰剂是否更强"] != "是"
        pseudo_values = [value for value in pseudo.values() if np.isfinite(value)]
        pseudo_pass = len(pseudo_values) < 2 or np.sign(pseudo_values[0]) == np.sign(pseudo_values[-1])
        criteria = {
            "全样本为负": fit.coefficient < 0,
            "95%区间上限小于0或显著": fit.upper95 < 0 or (fit.p7 < 0.05 and bootstrap_direction >= 0.80),
            "主要折至少七折为负": negative_count >= min(7, max(1, fold_total)),
            "伪未来方向不冲突": pseudo_pass,
            "有效变异充分": fit.markup_iqr >= 0.02 and fit.price_iqr > 0,
            "固定篮子与销量加权方向一致": direction_same,
            "提前一期安慰剂不过强": placebo_pass,
        }
        reliable = bool(all(criteria.values()))
        failed = [name for name, passed in criteria.items() if not passed]
        reliability_rows.append(
            {
                "品类": cat,
                "主价格口径": fit.source_name,
                "主价格响应模型": fit.model_name,
                "价格响应系数": fit.coefficient,
                "参考点弹性": fit.coefficient * (1.0 + fit.reference_fallback) if fit.model_name in {"半对数加成", "对数加成"} else fit.coefficient,
                "稳健标准误7": fit.se7,
                "稳健标准误14": fit.se14,
                "稳健概率值": fit.p7,
                "百分之九十五区间下限": fit.lower95,
                "百分之九十五区间上限": fit.upper95,
                "主要折负向数": negative_count,
                "主要折总数": fold_total,
                "伪未来2021系数": pseudo.get("2021", np.nan),
                "伪未来2022系数": pseudo.get("2022", np.nan),
                "自助法负向比例": bootstrap_direction,
                "加成有效变异四分位距": fit.markup_iqr,
                "售价有效变异四分位距": fit.price_iqr,
                "价格关系是否可靠": "是" if reliable else "否",
                "可靠性判定": "通过全部已实施门槛" if reliable else "未通过：" + "、".join(failed),
                "价格关系说明": "在控制日期、成本和商品结构后，历史数据中观察到成本加成偏离与正常销量之间的条件关联。" if reliable else "价格关系不可识别，最终不用于精细调价。",
                "候选未来价格支持检查": "待联合优化后复核",
            }
        )
        stability_rows.extend(_stability_rows(panel, cat, fit, target_col, rng))
        for source_name in ["销量加权", "固定篮子"]:
            fits = [source_fits[k] for k in source_fits if k[0] == cat and k[1] == source_name]
            for source_fit in fits:
                support_rows.append({"品类": cat, "价格口径": source_name, "价格响应模型": source_fit.model_name, "历史加成百分之一分位": np.nan, "历史加成百分之九十九分位": np.nan, "历史售价百分之一分位": np.nan, "历史售价百分之九十九分位": np.nan, "说明": "候选价格的最终支持区间在联合优化模块中填充"})

    relation_df = pd.DataFrame(reliability_rows)
    candidate_df = pd.DataFrame(all_fit_rows)
    backtest_df = pd.DataFrame(backtest_rows)
    stability_df = pd.DataFrame(stability_rows)
    placebo_df = pd.DataFrame(placebo_rows)
    support_df = pd.DataFrame(support_rows)
    relation_df.to_csv(OUTPUT_DIR / "04_价格关系可靠性.csv", index=False, encoding="utf-8-sig")
    candidate_df.to_csv(OUTPUT_DIR / "04_价格响应汇总.csv", index=False, encoding="utf-8-sig")
    backtest_df.to_csv(OUTPUT_DIR / "04_价格响应回测明细.csv", index=False, encoding="utf-8-sig")
    stability_df.to_csv(OUTPUT_DIR / "04_价格响应子时期稳定性.csv", index=False, encoding="utf-8-sig")
    placebo_df.to_csv(OUTPUT_DIR / "04_价格响应安慰剂检验.csv", index=False, encoding="utf-8-sig")
    support_df.to_csv(OUTPUT_DIR / "04_价格支持范围.csv", index=False, encoding="utf-8-sig")
    return {
        "selected": selected_fits,
        "selected_full": selected_full_fits,
        "source_fits": source_fits,
        "relation": relation_df,
        "candidate": candidate_df,
        "backtest": backtest_df,
        "stability": stability_df,
        "placebo": placebo_df,
        "support": support_df,
    }


def reference_markup(fit: PriceFit, date: pd.Timestamp) -> float:
    date = pd.Timestamp(date)
    return float(fit.reference_map.get((date.weekday() + 1, date.month), fit.reference_fallback))


def response_multiplier(fit: PriceFit, date: pd.Timestamp, price: float, cost: float) -> float:
    if price <= 0 or cost <= 0:
        return 1.0
    ref = reference_markup(fit, date)
    markup = price / cost - 1.0
    if fit.model_name == "半对数加成":
        variable = markup - ref
    elif fit.model_name == "对数加成":
        variable = np.log1p(max(markup, -0.95)) - np.log1p(max(ref, -0.95))
    elif fit.model_name == "半对数售价偏离":
        variable = price - cost * (1.0 + ref)
    elif fit.model_name == "对数售价比":
        variable = np.log(max(price / max(cost * (1.0 + ref), 1e-8), 1e-8))
    else:
        variable = 0.0
    return float(np.exp(np.clip(fit.coefficient * variable, -10.0, 10.0)))


def item_fixed_effect_robustness(item_daily: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """按单品—日期正常销售面板做固定效应稳健性检查。"""
    rows: list[dict[str, Any]] = []
    category_reference: dict[tuple[str, int, int], float] = {}
    valid_panel = panel[panel["正常销售量"] > 0].copy()
    for (cat, weekday, month), sub in valid_panel.groupby(["品类", "星期", "月份"], observed=True):
        category_reference[(cat, int(weekday), int(month))] = float(sub["正常销售加成率"].median())
    for cat in CATEGORIES:
        data = item_daily[item_daily["品类"] == cat].copy()
        if data.empty:
            continue
        data["销售日期"] = pd.to_datetime(data["销售日期"]).dt.normalize()
        data["星期"] = data["销售日期"].dt.weekday + 1
        data["月份"] = data["销售日期"].dt.month
        data["参考加成"] = [category_reference.get((cat, int(w), int(m)), 0.5) for w, m in zip(data["星期"], data["月份"])]
        data["加成偏离"] = data["单品加成率"] - data["参考加成"]
        data["目标"] = np.log(np.maximum(data["正常销售量"].astype(float), 1e-6))
        data["对数成本"] = np.log(np.maximum(data["当日批发价"].astype(float), 1e-6))
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["目标", "加成偏离", "对数成本"])
        if len(data) < 150:
            continue
        controls = pd.DataFrame({"常数项": 1.0}, index=data.index)
        item_dummies = pd.get_dummies(data["单品编码"].astype(str), prefix="单品", drop_first=True, dtype=float)
        controls = pd.concat([controls, item_dummies], axis=1)
        for value in range(2, 8):
            controls[f"星期{value}"] = (data["星期"] == value).astype(float)
        for value in range(2, 13):
            controls[f"月份{value}"] = (data["月份"] == value).astype(float)
        controls["时间趋势"] = (data["销售日期"] - data["销售日期"].min()).dt.days / 365.25
        controls["对数成本"] = data["对数成本"]
        controls["折扣占比"] = data["折扣占比"].fillna(0.0)
        x = np.column_stack([controls.to_numpy(float), data["加成偏离"].to_numpy(float)])
        y = data["目标"].to_numpy(float)
        beta = _ols(x, y)
        residual = y - x @ beta
        covariance = _hac_covariance(x, residual, 7)
        se = float(np.sqrt(max(covariance[-1, -1], 0.0)))
        rows.append(
            {
                "敏感性类型": "单品固定效应",
                "品类": cat,
                "价格响应系数": float(beta[-1]),
                "稳健标准误7": se,
                "样本数": int(len(data)),
                "方向": "负" if beta[-1] < 0 else "非负",
                "可靠性": "仅作稳健性检验",
                "说明": "控制单品固定效应、星期、月份、趋势、批发成本和折扣状态；缺失单品—日期不填零。",
            }
        )
    return pd.DataFrame(rows)
