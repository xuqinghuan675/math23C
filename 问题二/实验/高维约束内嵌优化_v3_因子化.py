# -*- coding: utf-8 -*-
"""问题二高维实验 v3：共享因子化高维搜索。

v2 证明“独立复制每个物理变量”不会带来真正的升维收益，因为物理梯度方向几乎不变。
v3 改为一个共享的高维因子化表示：

    h = U v / sqrt(R),  U∈R^(84×R), v∈R^R

h 的前42维控制加成率，后42维控制补货量。加成率和补货量再通过 sigmoid
硬编码到各自合法区间，因此整个梯度搜索过程中不存在非法经营方案。

对于任意 R>=1，映射都能覆盖同一个 h∈R^84 全空间；不同 R 的物理解空间相同。
但 U、v 的高维冗余和共享因子会改变 Jacobian、梯度预条件和可绕行方向，因而更接近
“维度拉高但始终在高维空间搜索”的设想。
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "高维约束内嵌优化.py"
_spec = importlib.util.spec_from_file_location("highdim_base", V1_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("无法载入高维约束内嵌优化.py")
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

OUT = HERE / "高维结果_v3_因子化"
OUT.mkdir(parents=True, exist_ok=True)

R_LEVELS = [1, 4, 16, 64, 256]
SEEDS = [11, 29, 47]
STEPS = 750
LR_U = 0.025
LR_V = 0.012
EPS = 1e-9


def physical_from_hidden(h: np.ndarray, data: dict):
    n = len(data["cost"])
    hm = h[:n]
    hq = h[n:]
    sm = exp.sigmoid(hm)
    sq = exp.sigmoid(hq)
    m_range = data["high_markup"] - data["low_markup"]
    markup = data["low_markup"] + m_range * sm
    order = data["q_max"] * sq
    cache = {
        "dm_dh": m_range * sm * (1.0 - sm),
        "dq_dh": data["q_max"] * sq * (1.0 - sq),
    }
    return markup, order, cache


def decode_factor(U: np.ndarray, v: np.ndarray, data: dict):
    r = U.shape[1]
    h = U @ v / math.sqrt(float(r))
    markup, order, cache = physical_from_hidden(h, data)
    return h, markup, order, cache


def target_hidden(data: dict) -> np.ndarray:
    low = data["low_markup"]
    high = data["high_markup"]
    m_range = high - low
    target_m = np.clip(data["baseline_markup"], low, high)
    fm = np.where(
        m_range > EPS,
        (target_m - low) / np.maximum(m_range, EPS),
        0.5,
    )
    fq = np.clip(data["baseline_order"] / np.maximum(data["q_max"], EPS), 1e-5, 1.0 - 1e-5)
    return np.concatenate([exp.logit(fm), exp.logit(fq)])


def init_factor(data: dict, r: int, seed: int):
    rng = np.random.default_rng(seed + 1009 * r)
    h0 = target_hidden(data)
    v = rng.normal(size=r)
    norm = float(np.linalg.norm(v))
    if norm < EPS:
        v[0] = 1.0
        norm = 1.0
    v = v / norm * math.sqrt(float(r))

    # U_base v / sqrt(R) = h0。
    U = h0[:, None] * v[None, :] / math.sqrt(float(r))

    # R>1 时加入严格位于 v 正交补空间的扰动：latent 不同，但物理解完全不变。
    noise = rng.normal(0.0, 0.08, size=U.shape)
    proj = (noise @ v) / float(v @ v)
    noise -= proj[:, None] * v[None, :]
    U += noise

    h_check = U @ v / math.sqrt(float(r))
    if float(np.max(np.abs(h_check - h0))) > 1e-8:
        raise RuntimeError("因子初始化未保持相同物理解")
    return U, v


def adam_update(param, grad, m, s, step, lr):
    b1, b2 = 0.9, 0.999
    m = b1 * m + (1.0 - b1) * grad
    s = b2 * s + (1.0 - b2) * (grad * grad)
    mh = m / (1.0 - b1**step)
    sh = s / (1.0 - b2**step)
    param = param + lr * mh / (np.sqrt(sh) + 1e-8)
    return param, m, s


def optimize_one(data: dict, profile_name: str, r: int, seed: int):
    profile = exp.SCORE_PROFILES[profile_name]
    U, v = init_factor(data, r, seed)
    mU = np.zeros_like(U)
    sU = np.zeros_like(U)
    mv = np.zeros_like(v)
    sv = np.zeros_like(v)
    best_score = -np.inf
    best_U = U.copy()
    best_v = v.copy()
    history = []

    for step in range(1, STEPS + 1):
        h, markup, order, cache = decode_factor(U, v, data)
        result, gm, gq = exp.evaluate_physical(markup, order, data, profile)
        assert gm is not None and gq is not None
        if result["score"] > best_score:
            best_score = float(result["score"])
            best_U = U.copy()
            best_v = v.copy()

        gh = np.concatenate([gm * cache["dm_dh"], gq * cache["dq_dh"]])
        scale = math.sqrt(float(r))
        gU = gh[:, None] * v[None, :] / scale
        gv = U.T @ gh / scale

        decay = 0.35 + 0.65 * (1.0 - step / STEPS)
        U, mU, sU = adam_update(U, gU, mU, sU, step, LR_U * decay)
        v, mv, sv = adam_update(v, gv, mv, sv, step, LR_V * decay)
        U = np.clip(U, -50.0, 50.0)
        v = np.clip(v, -50.0, 50.0)

        if step == 1 or step % 50 == 0 or step == STEPS:
            history.append(
                {
                    "profile": profile_name,
                    "R": r,
                    "seed": seed,
                    "step": step,
                    "score": float(result["score"]),
                    "raw_profit": float(result["raw_profit"]),
                    "hidden_norm": float(np.linalg.norm(h)),
                    "grad_hidden_rms": float(np.sqrt(np.mean(gh * gh))),
                }
            )

    h, markup, order, _ = decode_factor(best_U, best_v, data)
    result, _, _ = exp.evaluate_physical(markup, order, data, profile, need_grad=False)
    return result, markup, order, h, history


def prepare_data():
    data, cells, reliability_df = exp.build_experiment_data()

    # 与当前正式分层稳健求解器一致：IQR 边界按 0.01 加成率网格向外取整。
    reliable = data["reliable"]
    low = data["low_markup"].copy()
    high = data["high_markup"].copy()
    low[reliable] = np.floor(low[reliable] * 100.0) / 100.0
    high[reliable] = np.ceil(high[reliable] * 100.0) / 100.0
    data["low_markup"] = low
    data["high_markup"] = high
    return data, cells, reliability_df


def main():
    data, cells, reliability_df = prepare_data()
    runs = []
    trajectories = []
    best = {}

    baseline_metrics = {}
    for profile_name, profile in exp.SCORE_PROFILES.items():
        baseline_result, _, _ = exp.evaluate_physical(
            np.clip(data["baseline_markup"], data["low_markup"], data["high_markup"]),
            data["baseline_order"],
            data,
            profile,
            need_grad=False,
        )
        baseline_metrics[profile_name] = baseline_result

        for r in R_LEVELS:
            for seed in SEEDS:
                result, markup, order, h, hist = optimize_one(data, profile_name, r, seed)
                trajectories.extend(hist)
                runs.append(
                    {
                        "profile": profile_name,
                        "R": r,
                        "latent维度": int((len(h) + 1) * r),
                        "seed": seed,
                        "score": result["score"],
                        "raw_profit": result["raw_profit"],
                        "相对同口径基线利润": result["raw_profit"] - baseline_result["raw_profit"],
                        "相对同口径基线score": result["score"] - baseline_result["score"],
                        "risk_term": result["risk_term"],
                        "history_term": result["history_term"],
                        "smooth_term": result["smooth_term"],
                        "平均满足率": float(np.mean(result["fill_rate"])),
                    }
                )
                if profile_name not in best or result["score"] > best[profile_name]["result"]["score"]:
                    best[profile_name] = {
                        "R": r,
                        "seed": seed,
                        "result": result,
                        "markup": markup.copy(),
                        "order": order.copy(),
                        "h": h.copy(),
                    }

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(OUT / "全部运行.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trajectories).to_csv(OUT / "梯度轨迹.csv", index=False, encoding="utf-8-sig")

    summary = (
        runs_df.groupby(["profile", "R", "latent维度"], as_index=False)
        .agg(
            平均score=("score", "mean"),
            最佳score=("score", "max"),
            score标准差=("score", "std"),
            平均原始利润=("raw_profit", "mean"),
            最佳原始利润=("raw_profit", "max"),
            原始利润标准差=("raw_profit", "std"),
            平均满足率=("平均满足率", "mean"),
        )
        .sort_values(["profile", "R"])
    )
    summary.to_csv(OUT / "升维对比汇总.csv", index=False, encoding="utf-8-sig")

    robust = best["robust"]
    rr = robust["result"]
    strategy = cells[["日期", "品类", "cost", "loss", "beta_point", "reliable"]].copy()
    strategy["日期"] = pd.to_datetime(strategy["日期"]).dt.strftime("%Y-%m-%d")
    strategy = strategy.rename(columns={"cost": "预测批发价", "loss": "损耗率", "beta_point": "正常销售价格弹性", "reliable": "价格关系可靠"})
    strategy["R"] = robust["R"]
    strategy["latent维度"] = (84 + 1) * robust["R"]
    strategy["建议加成率"] = robust["markup"]
    strategy["建议售价"] = data["cost"] * (1.0 + robust["markup"])
    strategy["建议补货量"] = robust["order"]
    strategy["预测需求量"] = rr["demand_mean"]
    strategy["预计满足量"] = rr["sales_mean"]
    strategy["预计满足率"] = rr["fill_rate"]
    strategy["预计利润"] = rr["cell_profit"]
    strategy.to_csv(OUT / "最佳稳健因子化策略.csv", index=False, encoding="utf-8-sig")
    reliability_df.to_csv(OUT / "价格可靠性.csv", index=False, encoding="utf-8-sig")

    pure = best["pure"]
    report = f"""# 高维共享因子化实验 v3

## 结构

84 个物理变量（42 个加成率 + 42 个补货量）由共享高维因子生成：

`h = Uv/sqrt(R)`，其中 `U∈R^(84×R)`、`v∈R^R`。

随后使用 sigmoid 将 h 硬解码到合法加成率和补货区间。任意 R>=1 都覆盖完全相同的物理解空间；所有 R/seed 从完全相同的物理解开始，但 R>1 拥有更大的冗余流形和共享耦合搜索方向。

## 基线

- pure 基线利润/score：{baseline_metrics['pure']['raw_profit']:.4f}
- robust 基线原始利润：{baseline_metrics['robust']['raw_profit']:.4f}
- robust 基线score：{baseline_metrics['robust']['score']:.4f}

## 最佳结果

- pure 最佳：R={pure['R']}，latent维度={(84+1)*pure['R']}，利润={pure['result']['raw_profit']:.4f}，score={pure['result']['score']:.4f}
- robust 最佳：R={robust['R']}，latent维度={(84+1)*robust['R']}，原始利润={robust['result']['raw_profit']:.4f}，score={robust['result']['score']:.4f}

## 判据

只有在同一物理解空间、同一物理初值下，随着 R 增大出现可重复的更高 score / 更低随机种子方差，才能认为“高维搜索几何”在本题产生了实质收益。否则，高维方法最多作为通用优化框架，不应替换本题更直接的报童+局部定价方案。
"""
    (OUT / "实验说明.md").write_text(report, encoding="utf-8")
    (OUT / "实验配置.json").write_text(
        json.dumps({"R_LEVELS": R_LEVELS, "SEEDS": SEEDS, "STEPS": STEPS, "LR_U": LR_U, "LR_V": LR_V, "scores": exp.SCORE_PROFILES}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()
