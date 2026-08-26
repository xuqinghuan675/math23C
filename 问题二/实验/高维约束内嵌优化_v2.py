# -*- coding: utf-8 -*-
"""高维约束内嵌优化实验 v2：保证不同 R 拥有完全相同的物理解空间与初始物理解。

v1 的 tanh 主解码项有界，使最大 hidden 幅度随 sqrt(R) 增长，导致高维模型更容易
逼近 IQR 边界；这会把“表示容量”误当成“升维搜索几何”。v2 修正如下：

1. hidden 使用 unbounded 线性项 + 小幅正弦非线性：
       h(z) = [sum z_k + 0.15 sum sin(z_k)] / sqrt(R)
   因为含线性项，任何 R 都能覆盖完整实数 hidden，进而覆盖完全相同的 sigmoid 物理解空间。
2. 每个 R、每个随机种子都通过 Newton 校准 latent，使初始加成率/补货量与当前正式方案严格一致。
3. R>1 时仍保留大量不同的高维 latent 表示，因此可以公平观察过参数化是否改变梯度搜索几何。
4. 修正 best-state 记录时点：score 对应哪一个 z，就保存哪一个 z。
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "高维约束内嵌优化.py"
_spec = importlib.util.spec_from_file_location("highdim_v1", V1_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("无法载入高维约束内嵌优化.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

# v2 输出单独保存，保留 v1 作为实验审计。
v1.OUT = HERE / "高维结果_v2"
v1.OUT.mkdir(parents=True, exist_ok=True)


def decoder_hidden_fair(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """所有 R 都覆盖同一个完整 hidden 实数域。"""
    r = z.shape[-1]
    scale = math.sqrt(float(r))
    h = (z.sum(axis=-1) + 0.15 * np.sin(z).sum(axis=-1)) / scale
    dh_dz = (1.0 + 0.15 * np.cos(z)) / scale
    return h, dh_dz


def _latent_for_target(
    rng: np.random.Generator,
    targets: np.ndarray,
    r: int,
) -> np.ndarray:
    """随机高维表示 + 公共位移 Newton 校准，使 decoder hidden 精确等于 target。"""
    n = len(targets)
    z = rng.normal(0.0, 0.08, size=(n, r))
    scale = math.sqrt(float(r))
    delta = np.zeros(n, dtype=float)
    for _ in range(20):
        zz = z + delta[:, None]
        value = (zz.sum(axis=1) + 0.15 * np.sin(zz).sum(axis=1)) / scale
        deriv = (1.0 + 0.15 * np.cos(zz)).sum(axis=1) / scale
        delta -= (value - targets) / np.maximum(deriv, 1e-8)
    return z + delta[:, None]


def init_latent_fair(
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
        m_range > v1.EPS,
        (np.clip(baseline_markup, low, high) - low) / np.maximum(m_range, v1.EPS),
        0.5,
    )
    q_fraction = np.clip(
        baseline_order / np.maximum(q_max, v1.EPS), 0.00005, 0.99995
    )
    target_m = v1.logit(m_fraction)
    target_q = v1.logit(q_fraction)
    z[:, 0, :] = _latent_for_target(rng, target_m, r)
    z[:, 1, :] = _latent_for_target(rng, target_q, r)
    return z


def optimize_one_fair(
    data: dict,
    profile_name: str,
    r: int,
    seed: int,
):
    profile = v1.SCORE_PROFILES[profile_name]
    rng = np.random.default_rng(seed + r * 1000)
    z = init_latent_fair(
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
    history = []
    best_score = -np.inf
    best_z = z.copy()

    for step in range(1, v1.STEPS + 1):
        markup, order, cache = v1.decode(
            z, data["low_markup"], data["high_markup"], data["q_max"]
        )
        result, grad_m, grad_q = v1.evaluate_physical(markup, order, data, profile)
        assert grad_m is not None and grad_q is not None

        # 保存“当前 score 对应的当前 z”，而不是更新后的 z。
        if result["score"] > best_score:
            best_score = result["score"]
            best_z = z.copy()

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

        m1 = b1 * m1 + (1.0 - b1) * grad_z
        m2 = b2 * m2 + (1.0 - b2) * (grad_z * grad_z)
        m1_hat = m1 / (1.0 - b1**step)
        m2_hat = m2 / (1.0 - b2**step)
        lr = v1.LEARNING_RATE * (0.35 + 0.65 * (1.0 - step / v1.STEPS))
        z += lr * m1_hat / (np.sqrt(m2_hat) + 1e-8)
        z = np.clip(z, -30.0, 30.0)

        if step == 1 or step % 50 == 0 or step == v1.STEPS:
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

    markup, order, _ = v1.decode(
        best_z, data["low_markup"], data["high_markup"], data["q_max"]
    )
    result, _, _ = v1.evaluate_physical(
        markup, order, data, profile, need_grad=False
    )
    return result, markup, order, history


# 替换 v1 的运行时全局函数；其余数据口径、评分、输出逻辑完全保持一致。
v1.decoder_hidden = decoder_hidden_fair
v1.init_latent = init_latent_fair
v1.optimize_one = optimize_one_fair


def main() -> None:
    v1.main()
    fairness = """# v2 公平性修正\n\n"
    fairness += "v1 的主解码项 `tanh(z)` 有界，导致 R 越大可达到的 hidden 幅度越大；因此 v1 的维度比较混入了表示容量差异。\n\n"
    fairness += "v2 改用 `h=(Σz+0.15Σsin(z))/sqrt(R)`，所有 R 都可覆盖完整实数 hidden；同时使用 Newton 校准，让所有 R/seed 从完全相同的物理售价和补货方案起步。\n\n"
    fairness += "因此 v2 的 R 差异才主要反映高维过参数化对梯度搜索几何的影响。\n"
    (v1.OUT / "公平性修正.md").write_text(fairness, encoding="utf-8")


if __name__ == "__main__":
    main()
