# -*- coding: utf-8 -*-
"""2023 C题问题二：高维约束内嵌梯度优化实验。

目标不是替换当前正式方案，而是检验：
1. 将价格、补货约束直接编码进参数化，使整个搜索过程始终可行；
2. 不降低搜索维度，将每个物理变量用 R 个冗余非线性参数表示；
3. 对 42 个“日期×品类”单元联合做梯度优化；
4. 比较 R=1/4/16/64/256 时的收益、稳定性和边界行为。

真实经营变量仍是每个单元的加成率 m 和补货量 q；优化变量 z 始终保留为高维。
解码仅用于计算目标函数：
    z -> h(z) -> (m, q) -> (price, demand, profit)

所有价格强制位于正常销售历史 IQR；补货量强制位于 [0, q_max]。
茄类、食用菌因价格关系证据不足，价格固定为历史中位加成，只优化补货量。
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Q2 = HERE.parent
HYBRID_PATH = Q2 / "脚本" / "求解问题二_分层稳健.py"
_spec = importlib.util.spec_from_file_location("q2_hybrid", HYBRID_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("无法载入分层稳健求解器")
hybrid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hybrid)
base = hybrid.base

OUT = Q2 / "实验" / "高维结果"
OUT.mkdir(parents=True, exist_ok=True)

R_LEVELS = [1, 4, 16, 64, 256]
SEEDS = [11, 29, 47]
SAMPLE_COUNT = 1200
STEPS = 750
LEARNING_RATE = 0.035
EPS = 1e-9

# 两种评分：pure 用于验证高维表示是否能找回纯利润边界；
# robust 体现“评分 + 惩罚”的经营版本。
SCORE_PROFILES = {
    "pure": {"risk": 0.0, "history": 0.0, "smooth": 0.0},
    "robust": {"risk": 0.08, "history": 6.0, "smooth": 1.5},
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x))


def decoder_hidden(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """高维非线性解码；每个维度都直接参与，不先做降维学习。"""
    r = z.shape[-1]
    scale = math.sqrt(float(r))
    h = (np.tanh(z).sum(axis=-1) + 0.20 * np.sin(z).sum(axis=-1)) / scale
    dh_dz = ((1.0 - np.tanh(z) ** 2) + 0.20 * np.cos(z)) / scale
    return h, dh_dz


def decode(
    z: np.ndarray,
    low_markup: np.ndarray,
    high_markup: np.ndarray,
    q_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    hm, dhm = decoder_hidden(z[:, 0, :])
    hq, dhq = decoder_hidden(z[:, 1, :])
    sm = sigmoid(hm)
    sq = sigmoid(hq)
    markup_range = high_markup - low_markup
    markup = low_markup + markup_range * sm
    order = q_max * sq
    cache = {
        "dhm": dhm,
        "dhq": dhq,
        "dm_dh": markup_range * sm * (1.0 - sm),
        "dq_dh": q_max * sq * (1.0 - sq),
    }
    return markup, order, cache


def evaluate_physical(
    markup: np.ndarray,
    order: np.ndarray,
    data: dict,
    profile: dict,
    need_grad: bool = True,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    cost = data["cost"]
    loss = data["loss"]
    ref_price = data["ref_price"]
    beta = data["beta"]
    base_samples = data["base_samples"]
    reliable = data["reliable"]
    median_markup = data["median_markup"]
    half_iqr = data["half_iqr"]
    cat_index = data["cat_index"]
    date_index = data["date_index"]

    price = cost * (1.0 + markup)
    ratio = np.maximum(price / ref_price, 1e-8)
    demand = base_samples * np.power(ratio[:, None], beta[:, None])
    sellable = order * (1.0 - loss)
    demand_limited = demand <= sellable[:, None]
    sales = np.minimum(demand, sellable[:, None])
    profit = price[:, None] * sales - cost[:, None] * order[:, None]

    mean_profit = profit.mean(axis=1)
    total_profit = float(mean_profit.sum())
    center = profit - mean_profit[:, None]
    std_profit = np.sqrt(np.mean(center * center, axis=1) + EPS)
    risk_term = float(std_profit.sum())

    normalized = np.zeros_like(markup)
    movable = reliable & (half_iqr > EPS)
    normalized[movable] = (
        (markup[movable] - median_markup[movable]) / half_iqr[movable]
    )
    history_term = float(np.sum(normalized[movable] ** 4))

    smooth_term = 0.0
    if profile["smooth"] > 0:
        for c in range(len(base.CATEGORIES)):
            ids = np.where((cat_index == c) & reliable)[0]
            ids = ids[np.argsort(date_index[ids])]
            if len(ids) >= 2:
                scale = max(float(np.median(half_iqr[ids])), 1e-4)
                diffs = np.diff(markup[ids]) / scale
                smooth_term += float(np.sum(diffs * diffs))

    score = (
        total_profit
        - profile["risk"] * risk_term
        - profile["history"] * history_term
        - profile["smooth"] * smooth_term
    )

    fill_rate = np.divide(
        sales.mean(axis=1),
        demand.mean(axis=1),
        out=np.zeros_like(mean_profit),
        where=demand.mean(axis=1) > EPS,
    )
    result = {
        "score": float(score),
        "raw_profit": total_profit,
        "risk_term": risk_term,
        "history_term": history_term,
        "smooth_term": smooth_term,
        "price": price,
        "demand_mean": demand.mean(axis=1),
        "sales_mean": sales.mean(axis=1),
        "fill_rate": fill_rate,
        "cell_profit": mean_profit,
        "profit_std": std_profit,
    }
    if not need_grad:
        return result, None, None

    # 固定 beta 下的分段解析梯度。
    dprofit_dp = np.where(
        demand_limited,
        demand * (1.0 + beta[:, None]),
        sellable[:, None],
    )
    dprofit_dq = np.where(
        demand_limited,
        -cost[:, None],
        (price * (1.0 - loss) - cost)[:, None],
    )
    dmean_dp = dprofit_dp.mean(axis=1)
    dmean_dq = dprofit_dq.mean(axis=1)

    # std 的梯度：d std = E[(pi-mu)dpi] / std。
    dstd_dp = np.mean(center * dprofit_dp, axis=1) / std_profit
    dstd_dq = np.mean(center * dprofit_dq, axis=1) / std_profit

    dscore_dp = dmean_dp - profile["risk"] * dstd_dp
    dscore_dq = dmean_dq - profile["risk"] * dstd_dq
    dscore_dm = dscore_dp * cost

    if profile["history"] > 0:
        hist_grad = np.zeros_like(markup)
        hist_grad[movable] = (
            4.0 * normalized[movable] ** 3 / half_iqr[movable]
        )
        dscore_dm -= profile["history"] * hist_grad

    if profile["smooth"] > 0:
        smooth_grad = np.zeros_like(markup)
        for c in range(len(base.CATEGORIES)):
            ids = np.where((cat_index == c) & reliable)[0]
            ids = ids[np.argsort(date_index[ids])]
            if len(ids) < 2:
                continue
            scale = max(float(np.median(half_iqr[ids])), 1e-4)
            inv = 1.0 / (scale * scale)
            for left, right in zip(ids[:-1], ids[1:]):
                diff = markup[right] - markup[left]
                smooth_grad[left] += -2.0 * diff * inv
                smooth_grad[right] += 2.0 * diff * inv
        dscore_dm -= profile["smooth"] * smooth_grad

    return result, dscore_dm, dscore_dq


def init_latent(
    rng: np.random.Generator,
    r: int,
    low: np.ndarray,
    high: np.ndarray,
    q_max: np.ndarray,
    baseline_markup: np.ndarray,
    baseline_order: np.ndarray,
) -> np.ndarray:
    n = len(low)
    z = np.zeros((n, 2, r), dtype=float)
    m_range = high - low
    m_fraction = np.where(
        m_range > EPS,
        (np.clip(baseline_markup, low, high) - low) / np.maximum(m_range, EPS),
        0.5,
    )
    q_fraction = np.clip(baseline_order / np.maximum(q_max, EPS), 0.02, 0.98)
    target_m = logit(m_fraction)
    target_q = logit(q_fraction)
    # 在零附近 tanh(z)+0.2sin(z) ~= 1.2z，因此用此近似初始化。
    z[:, 0, :] = target_m[:, None] / (1.20 * math.sqrt(float(r)))
    z[:, 1, :] = target_q[:, None] / (1.20 * math.sqrt(float(r)))
    z += rng.normal(0.0, 0.035, size=z.shape)
    return z


def optimize_one(
    data: dict,
    profile_name: str,
    r: int,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    profile = SCORE_PROFILES[profile_name]
    rng = np.random.default_rng(seed + r * 1000)
    z = init_latent(
        rng,
        r,
        data["low_markup"],
        data["high_markup"],
        data["q_max"],
        data["baseline_markup"],
        data["baseline_order"],
    )
    m1 = np.zeros_like(z)
    m2 = np.zeros_like(z)
    b1, b2 = 0.9, 0.999
    history: list[dict] = []
    best_score = -np.inf
    best_z = z.copy()

    for step in range(1, STEPS + 1):
        markup, order, cache = decode(
            z, data["low_markup"], data["high_markup"], data["q_max"]
        )
        result, grad_m, grad_q = evaluate_physical(markup, order, data, profile)
        assert grad_m is not None and grad_q is not None
        grad_z = np.zeros_like(z)
        grad_z[:, 0, :] = (
            grad_m[:, None]
            * cache["dm_dh"][:, None]
            * cache["dhm"]
        )
        grad_z[:, 1, :] = (
            grad_q[:, None]
            * cache["dq_dh"][:, None]
            * cache["dhq"]
        )
        grad_norm = float(np.sqrt(np.mean(grad_z * grad_z)))

        # 最大化 score，因此沿正梯度做 Adam。
        m1 = b1 * m1 + (1.0 - b1) * grad_z
        m2 = b2 * m2 + (1.0 - b2) * (grad_z * grad_z)
        m1_hat = m1 / (1.0 - b1**step)
        m2_hat = m2 / (1.0 - b2**step)
        lr = LEARNING_RATE * (0.35 + 0.65 * (1.0 - step / STEPS))
        z += lr * m1_hat / (np.sqrt(m2_hat) + 1e-8)
        z = np.clip(z, -8.0, 8.0)

        if result["score"] > best_score:
            best_score = result["score"]
            best_z = z.copy()
        if step == 1 or step % 50 == 0 or step == STEPS:
            history.append(
                {
                    "profile": profile_name,
                    "R": r,
                    "seed": seed,
                    "step": step,
                    "score": result["score"],
                    "raw_profit": result["raw_profit"],
                    "grad_rms": grad_norm,
                }
            )

    markup, order, _ = decode(
        best_z, data["low_markup"], data["high_markup"], data["q_max"]
    )
    result, _, _ = evaluate_physical(markup, order, data, profile, need_grad=False)
    return result, markup, order, history


def build_experiment_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    (
        merged,
        panel_all,
        panel_normal,
        audit,
        category_loss,
        _item_loss,
        first_date,
    ) = base.read_source_data()
    markup_info = base.markup_summary(panel_all, panel_normal)
    normal_detail, normal_summary, normal_selected, normal_specs = base.demand_backtest(
        panel_normal, "正常销售"
    )
    _cost_detail, _cost_summary, selected_cost_methods = base.cost_backtest(panel_all)
    _base_summary, base_specs = hybrid.base_demand_backtest(panel_all)
    reliability_df, reliable_map = hybrid.reliability_table(
        panel_normal,
        normal_detail,
        normal_selected,
        normal_specs,
    )

    baseline_path = Q2 / "结果" / "七天六品类最终策略_分层稳健.csv"
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig")
    baseline["日期"] = pd.to_datetime(baseline["日期"])

    cells = []
    samples = []
    rng = np.random.default_rng(20230826 + 991)
    for c_idx, cat in enumerate(base.CATEGORIES):
        cost_frame = panel_all[panel_all["品类"] == cat].sort_values("销售日期")
        costs = base.cost_forecast(
            cost_frame, base.FUTURE_DATES, selected_cost_methods[cat]
        )
        info = markup_info[cat]["正常销售"]
        median = float(info["中位数"])
        q25 = float(info["百分之二十五分位"])
        q75 = float(info["百分之七十五分位"])
        spec = normal_specs[cat]
        beta_point = float(spec["价格系数"])
        beta_low = float(spec["稳健区间下限"])
        reliable = bool(reliable_map[cat])
        # 对可靠类别采用置信区间下界（更负）作为稳健弹性；不可靠类别固定价格且 beta=0。
        beta_used = beta_low if reliable else 0.0

        for d_idx, date in enumerate(base.FUTURE_DATES):
            future_cost = float(costs.loc[date])
            ref_price = float(future_cost * (1.0 + median))
            row = baseline[
                (baseline["日期"] == date) & (baseline["品类"] == cat)
            ].iloc[0]
            baseline_price = float(row["建议售价"])
            baseline_order = float(row["建议补货量"])
            baseline_markup = baseline_price / future_cost - 1.0
            residual_draws = rng.choice(
                np.asarray(base_specs[cat]["残差"], dtype=float),
                size=SAMPLE_COUNT,
                replace=True,
            )
            b = hybrid.base_demand_samples(
                base_specs[cat], date, residual_draws, first_date
            )
            low = q25 if reliable else median
            high = q75 if reliable else median
            low_price = future_cost * (1.0 + low)
            low_demand = b * (max(low_price / ref_price, 1e-8) ** beta_used)
            q99 = float(np.quantile(low_demand, 0.995) / (1.0 - category_loss[cat]))
            q_max = max(1.8 * baseline_order, 1.45 * q99, 1.0)
            cells.append(
                {
                    "日期": pd.Timestamp(date),
                    "品类": cat,
                    "cat_index": c_idx,
                    "date_index": d_idx,
                    "cost": future_cost,
                    "loss": float(category_loss[cat]),
                    "ref_price": ref_price,
                    "beta": beta_used,
                    "beta_point": beta_point,
                    "reliable": reliable,
                    "low_markup": low,
                    "high_markup": high,
                    "median_markup": median,
                    "half_iqr": max((q75 - q25) / 2.0, 1e-6),
                    "q_max": q_max,
                    "baseline_price": baseline_price,
                    "baseline_markup": baseline_markup,
                    "baseline_order": baseline_order,
                }
            )
            samples.append(b)

    cell_df = pd.DataFrame(cells)
    data = {
        "cost": cell_df["cost"].to_numpy(float),
        "loss": cell_df["loss"].to_numpy(float),
        "ref_price": cell_df["ref_price"].to_numpy(float),
        "beta": cell_df["beta"].to_numpy(float),
        "reliable": cell_df["reliable"].to_numpy(bool),
        "low_markup": cell_df["low_markup"].to_numpy(float),
        "high_markup": cell_df["high_markup"].to_numpy(float),
        "median_markup": cell_df["median_markup"].to_numpy(float),
        "half_iqr": cell_df["half_iqr"].to_numpy(float),
        "q_max": cell_df["q_max"].to_numpy(float),
        "baseline_markup": cell_df["baseline_markup"].to_numpy(float),
        "baseline_order": cell_df["baseline_order"].to_numpy(float),
        "cat_index": cell_df["cat_index"].to_numpy(int),
        "date_index": cell_df["date_index"].to_numpy(int),
        "base_samples": np.vstack(samples).astype(float),
    }
    return data, cell_df, reliability_df


def baseline_evaluation(data: dict, profile_name: str) -> dict:
    return evaluate_physical(
        data["baseline_markup"],
        data["baseline_order"],
        data,
        SCORE_PROFILES[profile_name],
        need_grad=False,
    )[0]


def main() -> None:
    data, cells, reliability_df = build_experiment_data()
    run_rows = []
    history_rows = []
    best_payload: dict[str, dict] = {}

    for profile_name in SCORE_PROFILES:
        base_eval = baseline_evaluation(data, profile_name)
        for r in R_LEVELS:
            for seed in SEEDS:
                result, markup, order, history = optimize_one(
                    data, profile_name, r, seed
                )
                price = data["cost"] * (1.0 + markup)
                upper = data["high_markup"]
                lower = data["low_markup"]
                movable = data["reliable"] & ((upper - lower) > EPS)
                boundary = np.zeros(len(markup), dtype=bool)
                boundary[movable] = (
                    np.abs(markup[movable] - upper[movable]) <= 0.005
                ) | (np.abs(markup[movable] - lower[movable]) <= 0.005)
                run_rows.append(
                    {
                        "profile": profile_name,
                        "R": r,
                        "总搜索维度": int(len(markup) * 2 * r),
                        "seed": seed,
                        "score": result["score"],
                        "raw_profit": result["raw_profit"],
                        "相对当前同口径利润变化": result["raw_profit"] - base_eval["raw_profit"],
                        "risk_term": result["risk_term"],
                        "history_term": result["history_term"],
                        "smooth_term": result["smooth_term"],
                        "边界单元数": int(boundary.sum()),
                        "平均满足率": float(np.mean(result["fill_rate"])),
                    }
                )
                history_rows.extend(history)
                key = profile_name
                if key not in best_payload or result["score"] > best_payload[key]["result"]["score"]:
                    best_payload[key] = {
                        "result": result,
                        "markup": markup.copy(),
                        "order": order.copy(),
                        "R": r,
                        "seed": seed,
                        "baseline": base_eval,
                        "price": price.copy(),
                    }

    runs = pd.DataFrame(run_rows)
    runs.to_csv(OUT / "全部运行.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(
        OUT / "梯度轨迹.csv", index=False, encoding="utf-8-sig"
    )

    summary = (
        runs.groupby(["profile", "R", "总搜索维度"], as_index=False)
        .agg(
            平均score=("score", "mean"),
            最佳score=("score", "max"),
            平均原始利润=("raw_profit", "mean"),
            最佳原始利润=("raw_profit", "max"),
            原始利润标准差=("raw_profit", "std"),
            平均边界单元数=("边界单元数", "mean"),
            平均满足率=("平均满足率", "mean"),
        )
        .sort_values(["profile", "R"])
    )
    summary.to_csv(OUT / "升维对比汇总.csv", index=False, encoding="utf-8-sig")

    robust = best_payload["robust"]
    robust_result = robust["result"]
    strategy = cells[["日期", "品类", "cost", "loss", "beta_point", "reliable"]].copy()
    strategy["日期"] = strategy["日期"].dt.strftime("%Y-%m-%d")
    strategy = strategy.rename(
        columns={
            "cost": "预测批发价",
            "loss": "损耗率",
            "beta_point": "正常销售价格弹性",
            "reliable": "价格关系可靠",
        }
    )
    strategy["搜索R"] = robust["R"]
    strategy["搜索总维度"] = len(strategy) * 2 * robust["R"]
    strategy["建议加成率"] = robust["markup"]
    strategy["建议售价"] = robust["price"]
    strategy["建议补货量"] = robust["order"]
    strategy["预测需求量"] = robust_result["demand_mean"]
    strategy["预计满足量"] = robust_result["sales_mean"]
    strategy["预计满足率"] = robust_result["fill_rate"]
    strategy["预计利润"] = robust_result["cell_profit"]
    strategy["当前方案售价"] = cells["baseline_price"].to_numpy(float)
    strategy["当前方案补货量"] = cells["baseline_order"].to_numpy(float)
    strategy.to_csv(OUT / "最佳稳健高维策略.csv", index=False, encoding="utf-8-sig")
    reliability_df.to_csv(OUT / "价格可靠性.csv", index=False, encoding="utf-8-sig")

    pure = best_payload["pure"]
    report = f"""# 高维约束内嵌梯度优化实验

## 实验定义

- 真实决策单元：42 个日期×品类；每个单元含加成率、补货量，共 84 个物理变量。
- 高维表示：每个物理变量使用 R 个非线性冗余参数，搜索维度为 84R；R ∈ {R_LEVELS}。
- 优化过程中始终在高维 z 空间更新，不把 z 降回低维做局部搜索。
- 加成率通过 sigmoid 硬编码在正常销售历史 IQR 内；补货量硬编码在 [0,q_max]。
- 价格关系不可靠的茄类、食用菌固定为历史中位加成，只优化补货量。
- 使用解析分段梯度 + Adam，多随机种子比较。

## 两个评分

1. pure：只最大化同一随机需求样本下的期望利润，用来检验高维参数化能否找到纯利润解。
2. robust：期望利润 - 0.08×利润波动 - 6×历史偏离四次惩罚 - 1.5×同品类跨日价格跳变惩罚。

惩罚只影响 robust 的评分；同时单独报告未扣惩罚的原始期望利润，避免把评分改善伪装成利润改善。

## 结果摘要

当前分层稳健方案在本实验统一随机样本与稳健弹性口径下：
- pure 原始期望利润：{pure['baseline']['raw_profit']:.2f} 元；
- robust 原始期望利润：{robust['baseline']['raw_profit']:.2f} 元。

高维 pure 最佳：
- R={pure['R']}，总搜索维度={84*pure['R']}；
- 原始期望利润={pure['result']['raw_profit']:.2f} 元；
- score={pure['result']['score']:.2f}。

高维 robust 最佳：
- R={robust['R']}，总搜索维度={84*robust['R']}；
- 原始期望利润={robust['result']['raw_profit']:.2f} 元；
- robust score={robust['result']['score']:.2f}；
- 平均需求满足率={np.mean(robust['result']['fill_rate']):.3f}。

完整维度—随机种子结果见 `升维对比汇总.csv` 与 `全部运行.csv`。

## 如何解释

该实验不能仅凭“维度更高”宣称更优。真正要观察的是：随着 R 增加，最佳/平均 score 是否提高、随机种子方差是否下降、最终经营方案是否更稳定。如果高维只增加冗余而没有改变这些指标，则说明本题的收益几何本身较简单；如果高维稳定改善，则说明过参数化确实改变了梯度搜索几何。
"""
    (OUT / "实验说明.md").write_text(report, encoding="utf-8")

    meta = {
        "R_LEVELS": R_LEVELS,
        "SEEDS": SEEDS,
        "SAMPLE_COUNT": SAMPLE_COUNT,
        "STEPS": STEPS,
        "LEARNING_RATE": LEARNING_RATE,
        "SCORE_PROFILES": SCORE_PROFILES,
        "best_pure": {"R": pure["R"], "seed": pure["seed"]},
        "best_robust": {"R": robust["R"], "seed": robust["seed"]},
    }
    (OUT / "实验配置.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("高维约束内嵌优化实验完成")
    print(summary.to_string(index=False))
    print(
        f"best robust: R={robust['R']}, dim={84*robust['R']}, "
        f"profit={robust['result']['raw_profit']:.2f}, score={robust['result']['score']:.2f}"
    )


if __name__ == "__main__":
    main()
