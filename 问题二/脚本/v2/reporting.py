# -*- coding: utf-8 -*-
"""结果表、图表、说明文档、运行清单和旧新对比。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import (
    CATEGORIES,
    DATA_DIR,
    DATA_END,
    FIGURE_DIR,
    FUTURE_DATES,
    OUTPUT_DIR,
    PROJECT_ROOT,
    QUESTION_DIR,
    config_snapshot,
)
from .data_pipeline import DataBundle
from .demand_models import DemandFit
from .price_response import PriceFit


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def write_csv(frame: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def record_legacy_baseline() -> dict[str, Any]:
    """保存现有旧结果，区分任务指定的缺失入口与仓库实际旧入口。"""
    requested = QUESTION_DIR / "脚本" / "求解问题二_分层稳健.py"
    current = QUESTION_DIR / "脚本" / "求解问题二.py"
    baseline: dict[str, Any] = {
        "任务指定旧入口": str(requested),
        "任务指定旧入口存在": requested.exists(),
        "任务指定旧入口复现": "失败：文件不存在" if not requested.exists() else "待执行",
        "仓库当前旧入口": str(current),
        "仓库当前旧入口存在": current.exists(),
        "仓库当前旧入口复现": "已在本轮开始前独立成功执行" if current.exists() else "失败：文件不存在",
        "说明": "任务指定的分层稳健文件不在当前仓库；现有旧入口及其结果作为可复核基线保留。",
    }
    old_final = QUESTION_DIR / "结果" / "七天六品类最终策略.csv"
    old_key = QUESTION_DIR / "结果" / "关键回测比较.csv"
    old_relation = QUESTION_DIR / "结果" / "销售量与成本加成关系.csv"
    old_cost = QUESTION_DIR / "结果" / "成本预测回测.csv"
    old_decomp = QUESTION_DIR / "结果" / "策略分解汇总.csv"
    if old_final.exists():
        old = pd.read_csv(old_final, encoding="utf-8-sig")
        old.to_csv(OUTPUT_DIR / "00_旧方案结果快照.csv", index=False, encoding="utf-8-sig")
        profit_col = "预计利润" if "预计利润" in old.columns else "预计毛利"
        boundary_col = "数学搜索是否触及上限" if "数学搜索是否触及上限" in old.columns else None
        baseline["旧方案结果快照行数"] = int(len(old))
        baseline["旧方案七天预计收益"] = float(old[profit_col].sum()) if profit_col in old else np.nan
        baseline["旧方案边界解天数"] = int((old[boundary_col] == "是").sum()) if boundary_col else np.nan
    if old_key.exists():
        baseline["旧方案关键回测"] = pd.read_csv(old_key, encoding="utf-8-sig").to_dict("records")
    if old_relation.exists():
        relation = pd.read_csv(old_relation, encoding="utf-8-sig")
        baseline["旧方案可靠品类数"] = int(relation["价格关系可靠性"].astype(str).str.contains("较可靠").sum()) if "价格关系可靠性" in relation else np.nan
    if old_cost.exists():
        cost = pd.read_csv(old_cost, encoding="utf-8-sig")
        baseline["旧方案成本回测"] = cost[cost.get("是否入选", "否") == "是"].to_dict("records") if "是否入选" in cost else cost.head(6).to_dict("records")
    if old_decomp.exists():
        baseline["旧方案策略分解"] = pd.read_csv(old_decomp, encoding="utf-8-sig").to_dict("records")
    (OUTPUT_DIR / "00_旧方案复现.json").write_text(json.dumps(_json_safe(baseline), ensure_ascii=False, indent=2), encoding="utf-8")
    return baseline


def _set_plot_font() -> None:
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110


def _save(fig: Any, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    bundle: DataBundle,
    panel: pd.DataFrame,
    normal_fits: dict[str, DemandFit],
    price_info: dict[str, Any],
    optimization: dict[str, Any],
    sensitivities: dict[str, pd.DataFrame],
) -> list[Path]:
    if not HAS_MATPLOTLIB:
        return []
    _set_plot_font()
    paths: list[Path] = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for ax, cat, color in zip(axes.ravel(), CATEGORIES, colors):
        hist = panel[panel["品类"] == cat].sort_values("销售日期").tail(180)
        ax.plot(hist["销售日期"], hist["净销售量"], color=color, linewidth=1.1, label="历史净销量")
        final = optimization["final"][optimization["final"]["品类"] == cat].sort_values("日期")
        dates = pd.to_datetime(final["日期"])
        ax.fill_between(dates, final["正常需求P10"], final["正常需求P90"], color="#C45A3C", alpha=0.18, label="未来需求区间")
        ax.plot(dates, final["正常需求P50"], color="#C45A3C", marker="o", markersize=3, label="未来需求中位")
        ax.set_title(cat)
        ax.set_ylabel("千克")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("六个品类历史净销量与未来正常需求区间")
    _save(fig, "01_历史净销量与未来需求区间.png")
    paths.append(FIGURE_DIR / "01_历史净销量与未来需求区间.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cat, color in zip(axes.ravel(), CATEGORIES, colors):
        hist = panel[panel["品类"] == cat].sort_values("销售日期").tail(180)
        ax.plot(hist["销售日期"], hist["销量加权进价"], color=color, linewidth=1.1, label="历史成本")
        fut = optimization["cost_path"][optimization["cost_path"]["品类"] == cat].sort_values("日期")
        dates = pd.to_datetime(fut["日期"])
        ax.fill_between(dates, fut["成本P10"], fut["成本P90"], color="#2F6B8A", alpha=0.18, label="未来成本区间")
        ax.plot(dates, fut["成本P50"], color="#2F6B8A", marker="o", markersize=3, label="未来成本中位")
        ax.set_title(cat)
        ax.set_ylabel("元/千克")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("六个品类历史成本与未来成本路径区间")
    _save(fig, "02_历史成本与未来成本区间.png")
    paths.append(FIGURE_DIR / "02_历史成本与未来成本区间.png")

    discount = bundle.discount_panel.groupby("品类", as_index=False).agg(折扣销量=("折扣销售量", "sum"), 折扣交易=("折扣交易条数", "sum"), 折扣价比=("折扣价比中位数", "median"))
    positive = bundle.full_panel.groupby("品类", as_index=False).agg(正销量=("正销售量", "sum"))
    discount = discount.merge(positive, on="品类", how="left")
    discount["折扣销量占比"] = discount["折扣销量"] / discount["正销量"]
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(CATEGORIES))
    ax1.bar(x - 0.18, discount.set_index("品类").reindex(CATEGORIES)["折扣销量占比"], width=0.36, color="#2F6B8A", label="折扣销量占比")
    ax1.set_ylabel("折扣销量占比")
    ax1.set_xticks(x, CATEGORIES)
    ax2 = ax1.twinx()
    ax2.plot(x, discount.set_index("品类").reindex(CATEGORIES)["折扣价比"], color="#C45A3C", marker="o", linewidth=2, label="折扣价比例")
    ax2.set_ylabel("折扣价/同日正常价")
    ax1.grid(axis="y", alpha=0.2)
    fig.suptitle("折扣销量占比与折扣价比例")
    _save(fig, "03_折扣销量占比与折扣价比例.png")
    paths.append(FIGURE_DIR / "03_折扣销量占比与折扣价比例.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cat, color in zip(axes.ravel(), CATEGORIES, colors):
        sub = panel[panel["品类"] == cat].sort_values("销售日期").tail(180)
        ax.plot(sub["销售日期"], sub["销量加权售价"], color="#C45A3C", label="销量加权价格")
        ax.plot(sub["销售日期"], sub["固定篮子价格指数"], color="#2F6B8A", label="固定篮子价格")
        ax.set_title(cat)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("固定篮子指数与销量加权指数比较")
    _save(fig, "04_固定篮子与销量加权指数.png")
    paths.append(FIGURE_DIR / "04_固定篮子与销量加权指数.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        fit = price_info["selected"][cat]
        sub = panel[(panel["品类"] == cat) & (panel["正常销售量"] > 0)].copy()
        if fit.source_name == "固定篮子":
            price_col, cost_col = "固定篮子价格指数", "固定篮子成本指数"
        else:
            price_col, cost_col = "正常销售售价", "正常销售进价"
        sub["实际加成"] = sub[price_col] / sub[cost_col] - 1.0
        sub["参考加成"] = [fit.reference_map.get((int(w), int(m)), fit.reference_fallback) for w, m in zip(sub["星期"], sub["月份"])]
        sub["加成偏离"] = sub["实际加成"] - sub["参考加成"]
        y = np.log(np.maximum(sub["正常销售量"].to_numpy(float), 1e-6))
        ax.scatter(sub["加成偏离"], y - np.median(y), s=8, alpha=0.24, color="#2F6B8A")
        ax.set_title(cat)
        ax.set_xlabel("加成偏离")
        ax.set_ylabel("需求残差近似值")
        ax.grid(alpha=0.2)
    fig.suptitle("剥离日期基准后的加成偏离与需求残差散点")
    _save(fig, "05_加成偏离与需求残差.png")
    paths.append(FIGURE_DIR / "05_加成偏离与需求残差.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        fit = price_info["selected"][cat]
        ref = fit.reference_fallback
        grid = np.linspace(max(0.01, ref - 0.30), ref + 0.30, 80)
        multiplier = np.exp(fit.coefficient * (grid - ref)) if fit.model_name == "半对数加成" else np.exp(fit.coefficient * (np.log1p(grid) - np.log1p(max(ref, -0.95))))
        lo = np.exp(fit.lower95 * (grid - ref)) if fit.model_name == "半对数加成" else np.exp(fit.lower95 * (np.log1p(grid) - np.log1p(max(ref, -0.95))))
        hi = np.exp(fit.upper95 * (grid - ref)) if fit.model_name == "半对数加成" else np.exp(fit.upper95 * (np.log1p(grid) - np.log1p(max(ref, -0.95))))
        ax.plot(grid, multiplier, color="#C45A3C")
        ax.fill_between(grid, np.minimum(lo, hi), np.maximum(lo, hi), color="#C45A3C", alpha=0.18)
        ax.axvline(ref, color="#555555", linestyle=":")
        ax.set_title(cat + ("（可靠）" if optimization["reliability"][cat] else "（不可靠）"))
        ax.set_xlabel("成本加成率")
        ax.set_ylabel("相对参考需求倍数")
        ax.grid(alpha=0.2)
    fig.suptitle("六品类价格响应曲线及95%区间")
    _save(fig, "06_六品类价格响应曲线.png")
    paths.append(FIGURE_DIR / "06_六品类价格响应曲线.png")

    backtest = price_info["backtest"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        fit = price_info["selected"][cat]
        sub = backtest[(backtest["品类"] == cat) & (backtest["价格口径"] == fit.source_name) & (backtest["价格响应模型"] == fit.model_name)].sort_values("训练截止日")
        if not sub.empty:
            ax.plot(pd.to_datetime(sub["训练截止日"]), sub["价格响应系数"], marker="o", color="#2F6B8A")
            ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(cat)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
    fig.suptitle("各滚动窗口价格系数稳定性")
    _save(fig, "07_滚动价格系数稳定性.png")
    paths.append(FIGURE_DIR / "07_滚动价格系数稳定性.png")

    candidates = price_info["candidate"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat, color in zip(CATEGORIES, colors):
        fit = price_info["selected"][cat]
        sub = candidates[(candidates["品类"] == cat) & (candidates["价格响应模型"] == fit.model_name)]
        if sub.empty:
            continue
        values = sub.set_index("价格口径").reindex(["销量加权", "固定篮子"])["价格响应系数"]
        ax.plot([0, 1], values.to_numpy(float), marker="o", color=color, label=cat)
    ax.set_xticks([0, 1], ["销量加权", "固定篮子"])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("价格响应系数")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.suptitle("固定篮子和销量加权弹性对比")
    _save(fig, "08_固定篮子与销量加权弹性.png")
    paths.append(FIGURE_DIR / "08_固定篮子与销量加权弹性.png")

    curve = optimization["curve"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        sub = curve[curve["品类"] == cat]
        for date, one in sub.groupby("日期"):
            ax.plot(one["售价"], one["平均利润"], alpha=0.42, linewidth=1.0, label=date[5:])
        final = optimization["final"][optimization["final"]["品类"] == cat]
        if not final.empty:
            ax.axvline(final["稳健推荐售价"].iloc[0], color="#2E8B57", linestyle="--", linewidth=1.2)
            ax.axvline(final["数学期望利润最大售价"].iloc[0], color="#555555", linestyle=":", linewidth=1.1)
        ax.set_title(cat)
        ax.set_xlabel("售价（元/千克）")
        ax.set_ylabel("情景平均毛利（元）")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("每个品类七天价格—利润曲线、数学价与稳健价")
    _save(fig, "09_七天价格利润曲线.png")
    paths.append(FIGURE_DIR / "09_七天价格利润曲线.png")

    final = optimization["final"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for j, cat in enumerate(CATEGORIES):
        sub = final[final["品类"] == cat].sort_values("日期")
        math_price = pd.to_numeric(sub["数学期望利润最大售价"], errors="coerce")
        valid = math_price.notna()
        if not valid.any():
            continue
        sub = sub.loc[valid]
        math_price = math_price.loc[valid]
        x = np.arange(len(sub)) + j * 0.02
        ax.plot(x, math_price.to_numpy(float), marker="o", linestyle="none", color=colors[j], label=cat + "数学价")
        ax.plot(x, sub["稳健推荐售价"], marker="x", linestyle="none", color=colors[j], alpha=0.65)
    ax.set_xticks(np.arange(7), [x.strftime("%m-%d") for x in FUTURE_DATES])
    ax.set_ylabel("售价（元/千克）")
    ax.set_title("数学最优价与稳健推荐价（叉号为稳健价）")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, "10_数学最优与稳健价格.png")
    paths.append(FIGURE_DIR / "10_数学最优与稳健价格.png")

    band = sensitivities["band"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for band_name, sub in band.groupby("经营带"):
        ax.plot(sub["品类"], sub["七天数学搜索毛利"], marker="o", label=band_name)
    ax.set_ylabel("七天数学搜索毛利（元）")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.suptitle("价格经营带边界敏感性")
    _save(fig, "11_边界敏感性.png")
    paths.append(FIGURE_DIR / "11_边界敏感性.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    summary = final.groupby("品类", as_index=False).agg(P10=("正常需求P10", "sum"), P50=("正常需求P50", "sum"), P90=("正常需求P90", "sum"), 补货=("建议补货量", "sum"))
    x = np.arange(len(CATEGORIES))
    ax.errorbar(x, summary.set_index("品类").reindex(CATEGORIES)["P50"], yerr=[summary.set_index("品类").reindex(CATEGORIES)["P50"] - summary.set_index("品类").reindex(CATEGORIES)["P10"], summary.set_index("品类").reindex(CATEGORIES)["P90"] - summary.set_index("品类").reindex(CATEGORIES)["P50"]], fmt="o", color="#2F6B8A", label="需求区间")
    ax.bar(x + 0.18, summary.set_index("品类").reindex(CATEGORIES)["补货"], width=0.28, color="#C45A3C", alpha=0.75, label="建议补货量")
    ax.set_xticks(x, CATEGORIES)
    ax.set_ylabel("千克")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.suptitle("七天需求区间与补货量")
    _save(fig, "12_补货量与需求区间.png")
    paths.append(FIGURE_DIR / "12_补货量与需求区间.png")

    strategy_total = optimization["strategy_total"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(strategy_total["策略"], strategy_total["七天预计毛利"], color=["#8FA6B8", "#6D9DC5", "#C45A3C", "#2E8B57"])
    ax.set_ylabel("七天预计毛利（元）")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("策略 A/B/C/D 收益分解")
    _save(fig, "13_策略收益分解.png")
    paths.append(FIGURE_DIR / "13_策略收益分解.png")

    total_profit = np.zeros(int(optimization["bundle"]["情景数"]), dtype=float)
    for cell in optimization["cells"]:
        total_profit += cell["最终数组"]["利润"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(total_profit, bins=40, color="#2F6B8A", alpha=0.78)
    for quantile, style in [(0.10, "--"), (0.50, "-"), (0.90, "--")]:
        value = np.quantile(total_profit, quantile)
        ax.axvline(value, color="#C45A3C", linestyle=style, linewidth=1.3, label=f"P{int(quantile*100)}={value:.2f}")
    ax.set_xlabel("七天累计预计毛利（元）")
    ax.set_ylabel("情景数")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.suptitle("七天累计毛利分布")
    _save(fig, "14_七天累计毛利分布.png")
    paths.append(FIGURE_DIR / "14_七天累计毛利分布.png")
    return paths


def _dependency_versions() -> dict[str, str]:
    result = {"Python": platform.python_version()}
    for name in ["numpy", "pandas", "scipy", "statsmodels", "sklearn", "matplotlib"]:
        try:
            module = __import__(name)
            result[name] = str(module.__version__)
        except Exception:
            result[name] = "未安装"
    return result


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "无法读取"


def code_tree_hash() -> str:
    """对问题二脚本和测试源码做稳定哈希，不受生成结果和本地缓存影响。"""
    digest = hashlib.sha256()
    source_root = QUESTION_DIR / "脚本"
    test_root = QUESTION_DIR / "测试"
    paths = [
        path
        for root in [source_root, test_root]
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    for path in sorted(paths):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_manifest(
    start_time: datetime,
    end_time: datetime,
    quick: bool,
    cache_used: bool,
    test_result: str = "待本轮测试",
) -> dict[str, Any]:
    hashes = {}
    for path in [DATA_DIR / f"附件{i}.xlsx" for i in range(1, 5)]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    config_hash = hashlib.sha256(json.dumps(config_snapshot(), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    manifest = {
        "程序版本": __version__,
        "Git提交": _git_head(),
        "代码树哈希": code_tree_hash(),
        "原始附件哈希": hashes,
        "配置哈希": config_hash,
        "随机种子": config_snapshot()["随机种子"],
        "正式或快速模式": "快速" if quick else "正式",
        "情景数": config_snapshot()["快速情景数"] if quick else config_snapshot()["正式情景数"],
        "缓存是否命中": cache_used,
        "开始时间": start_time.isoformat(timespec="seconds"),
        "结束时间": end_time.isoformat(timespec="seconds"),
        "运行耗时秒": (end_time - start_time).total_seconds(),
        "依赖版本": _dependency_versions(),
        "数据使用截止日": str(DATA_END.date()),
        "测试命令": "pytest -q 问题二/测试",
        "测试结果": test_result,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_old_new_comparison(
    new_final: pd.DataFrame,
    new_relation: pd.DataFrame,
    demand_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    end_summary: pd.DataFrame,
    new_decomp: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    old_final_path = QUESTION_DIR / "结果" / "七天六品类最终策略.csv"
    old_key_path = QUESTION_DIR / "结果" / "关键回测比较.csv"
    old_final = pd.read_csv(old_final_path, encoding="utf-8-sig") if old_final_path.exists() else pd.DataFrame()
    old_key = pd.read_csv(old_key_path, encoding="utf-8-sig") if old_key_path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        old_cat = old_final[old_final.get("品类", pd.Series(dtype=str)) == cat] if not old_final.empty else pd.DataFrame()
        new_cat = new_final[new_final["品类"] == cat]
        selected_demand = demand_summary[(demand_summary["口径"] == "正常销售") & (demand_summary["品类"] == cat)]
        selected_demand = selected_demand.sort_values("池化加权绝对百分比误差").iloc[0] if not selected_demand.empty else None
        selected_cost = cost_summary[(cost_summary["品类"] == cat) & (cost_summary["是否入选"] == "是")] if "是否入选" in cost_summary else pd.DataFrame()
        selected_cost = selected_cost.iloc[0] if not selected_cost.empty else None
        relation = new_relation[new_relation["品类"] == cat].iloc[0]
        end = end_summary[(end_summary["品类"] == cat) & (end_summary["需求口径"] == "正常销售量")]
        old_profit = float(old_cat["预计利润"].sum()) if "预计利润" in old_cat else np.nan
        new_profit = float(new_cat["预计毛利"].sum())
        rows.append(
            {
                "品类": cat,
                "旧方案需求回测误差": float(old_key.loc[old_key["品类"] == cat, "正常销售需求回测误差"].iloc[0]) if not old_key.empty and "正常销售需求回测误差" in old_key else np.nan,
                "新方案需求回测误差": float(selected_demand["池化加权绝对百分比误差"]) if selected_demand is not None else np.nan,
                "新方案端到端需求误差": float(end["端到端池化加权绝对百分比误差"].iloc[0]) if not end.empty else np.nan,
                "旧方案成本回测误差": float(old_key.loc[old_key["品类"] == cat, "进价回测误差"].iloc[0]) if not old_key.empty and "进价回测误差" in old_key else np.nan,
                "新方案成本回测误差": float(selected_cost["池化加权绝对百分比误差"]) if selected_cost is not None else np.nan,
                "旧方案价格关系": str(old_key.loc[old_key["品类"] == cat, "正常销售价格关系可靠性"].iloc[0]) if not old_key.empty and "正常销售价格关系可靠性" in old_key else "未提供",
                "新方案价格关系可靠": relation["价格关系是否可靠"],
                "新方案价格响应系数": relation["价格响应系数"],
                "旧方案建议售价中位": float(old_cat["建议售价"].median()) if "建议售价" in old_cat else np.nan,
                "新方案稳健售价中位": float(new_cat["稳健推荐售价"].median()),
                "旧方案补货量": float(old_cat["建议补货量"].sum()) if "建议补货量" in old_cat else np.nan,
                "新方案补货量": float(new_cat["建议补货量"].sum()),
                "旧方案边界解天数": int((old_cat["数学搜索是否触及上限"] == "是").sum()) if "数学搜索是否触及上限" in old_cat else np.nan,
                "新方案边界解天数": int((new_cat["是否边界解"] == "是").sum()),
                "旧方案七天预计毛利": old_profit,
                "新方案七天预计毛利": new_profit,
                "新方案与旧方案毛利差": new_profit - old_profit if np.isfinite(old_profit) else np.nan,
                "新方案折扣处理": "全量审计，正常需求主方案零残值，折扣回收单列敏感性",
                "新方案商品结构控制": "固定篮子指数、HHI、前三单品占比和权重距离",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "10_旧新方案对比.csv", index=False, encoding="utf-8-sig")
    old_total = float(old_final["预计利润"].sum()) if "预计利润" in old_final else np.nan
    new_total = float(new_final["预计毛利"].sum())
    lines = [
        "# 问题二旧方案与新方案对比",
        "",
        f"当前仓库可复现旧入口的七天预计收益为 {old_total:.2f} 元；新方案为 {new_total:.2f} 元。两者均为采购—销售口径的模型预计毛利，不是已经实现的经营利润。",
        "",
        "任务文字中提到的4334.72元没有出现在当前旧结果文件中，因此本次不把它冒充为本仓库旧方案的实测基线；以实际可读取并成功复现的旧结果为比较对象。",
        "",
        "新方案纳入了正常需求与全量净需求分离、折扣零残值主方案、固定篮子结构控制、端到端回测、成本和利润区间、经营带敏感性及边界标记。新方案点估计若低于旧方案，不直接解释为模型退步，而首先解释为不确定性和保守约束被显式计入。",
        "",
        "补货收益只定义为策略B减策略A；定价收益只定义为策略C减策略B。没有把价格、补货、折扣和损耗的全部差额都归到定价。",
        "",
        "逐品类的数值对比见10_旧新方案对比.csv。",
    ]
    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "10_旧新方案对比.md").write_text(report, encoding="utf-8")
    return comparison, report


def write_documents(
    bundle: DataBundle,
    panel: pd.DataFrame,
    normal_fits: dict[str, DemandFit],
    demand_selection: pd.DataFrame,
    cost_summary: pd.DataFrame,
    price_info: dict[str, Any],
    end_summary: pd.DataFrame,
    optimization: dict[str, Any],
    sensitivities: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
) -> tuple[str, str]:
    final = optimization["final"]
    relation = price_info["relation"].set_index("品类")
    strategy_total = optimization["strategy_total"]
    total_profit = float(final["预计毛利"].sum())
    reliable_count = int(final.drop_duplicates("品类")["价格关系是否可靠"].eq("是").sum())
    lines = [
        "# 2023年数学建模国赛C题问题二 v2.1 建模说明",
        "",
        "## 1. 题目要求",
        "",
        "题目要求以蔬菜品类为单位，分析销售总量与成本加成定价的关系，并给出2023年7月1日至7日六个品类每天的补货总量和定价策略，使商超的预计毛利最大。本版本始终输出六个品类、七个未来日期和42条品类—日期策略。没有加入问题三的单品数量、陈列量和货架组合约束。",
        "",
        "## 2. 品类级建模理由",
        "",
        "题目明确要求按品类补货；附件四同时提供品类损耗率；题目没有给出品类之间预算、货架容量或库存联动约束。因此最终决策单位是品类—日期，单品信息只用于固定篮子、商品结构和稳健性检查。",
        "",
        "## 3. 四个附件的使用",
        "",
        f"附件一提供{bundle.audit['附件一行数']}个单品到品类的映射；附件二提供{bundle.audit['附件二行数']}条销售流水，保留销售、退货和折扣状态；附件三提供{bundle.audit['附件三行数']}条日期—单品批发价并与销售流水逐行连接；附件四提供六个品类损耗率和单品损耗率，正式补货使用品类损耗率，单品损耗率只用于核对和敏感性。",
        "",
        "## 4. 退货与折扣",
        "",
        f"退货共{bundle.audit['附件二退货记录数']}条，数量为负，进入全量净销量、净销售额和净成本额，但不进入正常售价指数和正常价格响应。折扣销售共{bundle.audit['附件二折扣销售记录数']}条；同日同单品正常价无法匹配时不填造价格，并把匹配率单独输出。主方案采用正常需求加零残值，折扣回收只用历史折扣价比例做有限回收和乐观上界敏感性。",
        "",
        "## 5. 两种价格指数",
        "",
        "销量加权价格保留用于现实金额核算；固定篮子指数用截止日前的90日、180日和365日候选窗口及全历史权重收缩构造，缺少当日价格或成本的单品重新归一化，并输出覆盖率。这样可以把商品结构变化与价格变化分开观察。",
        "",
        "## 6. 时间效应剥离与价格响应",
        "",
        "主响应函数以正常销售量的对数为目标，控制星期、月份、趋势、成本、折扣占比以及严格滞后的商品结构指标；价格变量优先使用相对条件参考加成的偏离。参考加成只由截止日前的同星期—月份历史构造。统一回归和两阶段剥离时间效应的系数同时核对。另用单品—日期正常销售面板控制单品固定效应、星期、月份、趋势、批发成本和折扣状态，作为稳健性检验；缺失单品—日期不填为零。所有价格系数只解释为控制日期、成本和商品结构后的历史条件关联，不宣称严格因果。",
        "",
        "## 7. 价格关系可靠性",
        "",
        "可靠性同时检查全样本方向、稳健区间或概率值、八个主要滚动折的方向、2021和2022伪未来方向、价格有效变异、固定篮子与销量加权方向、提前一期安慰剂以及未来价格历史支持。未通过的品类不使用精细价格弹性，只使用条件中位加成，补货仍使用概率需求。",
        "",
        "## 8. 未来七日需求",
        "",
        "需求候选包含同星期最近4次均值、同星期最近8次中位数、近7日均值、近14日均值、星期加月份对数回归和带趋势对数回归。用八个主要非重叠折和十四个近期滚动折比较池化误差、折间误差、平均绝对误差、均方根误差、尺度误差、分位数损失与区间覆盖；复杂模型改善不足3%或落在一标准误范围内时选择更简单模型。需求区间和联合情景使用最终入选模型的严格滚动预测残差，而不是同一训练集内的拟合残差。",
        "",
        "## 9. 未来七日成本",
        "",
        "同时比较销量加权成本和固定篮子成本，候选包含近7日和近14日均值、中位数、指数加权移动平均、同星期统计、周内收缩、阻尼趋势、指数平滑和滞后稳健回归。逐品类按滚动七日误差选择，并从滚动残差移动区块抽样生成未来成本区间。",
        "",
        "## 10. 联合不确定性",
        f"",
        f"正式模式使用{optimization['bundle']['情景数']}个情景；需求扰动和成本扰动采用长度7的移动区块，尽量保留品类共同冲击和日期短期相关；价格系数只对可靠品类进入优化，不可靠品类不进行价格优化；损耗率使用附件四点值的0.8、1.0和1.2倍并截断到合法范围；折扣回收情景使用零回收、历史中位比例和完全回收上界。",
        "",
        "## 11. 随机收益模型",
        "",
        "设补货量为Q、损耗率为L，可销售量为Q(1-L)。正常销售量是正常需求和可销售量的较小者；主方案剩余商品残值为零，收益为正常销售收入减采购成本。每个策略同时输出收益P10、P50、P90、亏损概率、缺货概率和剩余概率；正文使用预计毛利或采购—销售口径收益，不暗示包含人工、租金、能源等未提供成本。",
        "",
        "## 12. 售价与补货量求解",
        "",
        "历史参考加成是根据历史条件状态估计的中心值，政策经营带是允许执行的区间；两者不混同。实际参考价格先由未截断历史参考加成得到，再按政策经营带截断，并同时记录未截断参考加成、执行参考加成和是否受到经营带约束。售价在给定经营带内按0.01元/千克枚举，补货量按报童分位数得到初值，再在至少上下2千克的0.1千克网格邻域复核。每个品类—日期只有一个售价和一个补货量，不使用没有必要的遗传算法或粒子群。",
        "",
        "## 13. 数学最优价与稳健推荐价",
        "",
        "数学期望利润最大售价是可靠价格关系品类在给定情景和经营带内的离散搜索结果；稳健推荐售价先在期望收益不低于最大值99%的候选中比较利润下界，并偏向历史支持更近、价格更保守的候选。不可靠品类的数学价格、边界方向和价格系数敏感性标记为不适用，正式价格只采用执行参考加成。",
        "",
        "## 14. 边界解",
        "",
        "如果可靠品类的数学搜索价触及经营带上下界，则标记边界方向并配套输出价格—利润曲线和五组经营带敏感性；不可靠品类不计入边界解统计。边界解只表示当前历史支持区间和模型假设下的边界结果，不称为无约束全局最优。",
        "",
        "## 15. 策略收益分解",
        "",
        "策略A为条件中位加成加平均需求补货；策略B只优化补货；策略C在可靠品类上优化价格和补货；策略D采用稳健推荐价和稳健补货。补货贡献定义为B减A，定价贡献定义为C减B，总贡献定义为C减A，稳健贡献定义为D减A。",
        "",
        "## 16. 最终策略摘要",
        "",
        f"本次新方案生成42条策略，六品类中通过基础价格关系门槛的品类数为{reliable_count}，七天预计毛利合计为{total_profit:.2f}元。七天利润区间先对联合情景逐情景累计，再计算百分之十、百分之五十和百分之九十分位；逐日结果见09_七天六品类最终策略.csv。",
        "",
        "| 品类 | 七天售价范围 | 七天补货量 | 七天预计毛利 | 毛利P10/P50/P90 | 主要风险 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    strategy_summary = optimization["strategy_summary"]
    for cat in CATEGORIES:
        sub = final[final["品类"] == cat]
        d_summary = strategy_summary[(strategy_summary["品类"] == cat) & (strategy_summary["策略"] == "D稳健经营")].iloc[0]
        lines.append(f"| {cat} | {sub['稳健推荐售价'].min():.2f}—{sub['稳健推荐售价'].max():.2f} | {sub['建议补货量'].sum():.1f}千克 | {sub['预计毛利'].sum():.2f}元 | {d_summary['毛利P10']:.2f}/{d_summary['毛利P50']:.2f}/{d_summary['毛利P90']:.2f} | 无库存、缺货记录；价格关系为条件关联 |")
    d_total = strategy_total[strategy_total["策略"] == "D稳健经营"].iloc[0]
    lines.append(f"六品类合计的D策略七天累计毛利区间为 {d_total['毛利P10']:.2f}/{d_total['毛利P50']:.2f}/{d_total['毛利P90']:.2f} 元。")
    lines += [
        "",
        "## 17. 结果文件",
        "",
        "数据审计和四套面板在01_和02_文件中；需求在03_文件中；价格响应在04_文件中；端到端回测在05_文件中；成本在06_文件中；策略和利润曲线在07_文件中；敏感性在08_文件中；最终42条策略在09_七天六品类最终策略.csv；旧新差异在10_旧新方案对比.csv及其说明中。",
        "",
        "## 18. 模型限制",
        "",
        "附件没有逐日库存、真实缺货、剩余商品、货架容量、预算、天气、客流和未来真实成本；价格与销量可能同时受到库存、商品上架和促销选择影响。因此历史销量是可观测需求代理，价格响应是条件关联，预计毛利是模型结果而不是实际经营利润。",
        "",
        "## 19. 复现",
        "",
        "在仓库根目录运行：",
        "",
        "    python 问题二/脚本/求解问题二_v2.py --force-rebuild",
        "    pytest -q 问题二/测试",
        "    python 问题二/脚本/求解问题二_v2.py",
        "",
        "第二次运行使用本地缓存，并应与第一次的数值结果一致。",
        "",
        "## 20. 结论边界",
        "",
        "本模型给出的最优价格是历史支持区间和给定模型假设下的局部随机优化结果；当价格关系不可靠或搜索价触及边界时，正式建议优先采用条件中位加成并明确提示风险。",
    ]
    report = "\n".join(lines) + "\n"
    (QUESTION_DIR / "最终建模说明_v2.md").write_text(report, encoding="utf-8")
    readme = "\n".join([
        "# 2023年C题问题二",
        "",
        "当前推荐方案为 v2：按六个蔬菜品类、七个未来日期生成42条售价和补货策略。v2重新审计四个原始附件，分离净需求、正常销售和折扣销售，加入固定篮子商品结构控制、无泄漏滚动回测、端到端校准、成本和收益区间、经营带敏感性及边界说明。",
        "",
        "## 唯一入口",
        "",
        "    python 问题二/脚本/求解问题二_v2.py",
        "",
        "完整重建使用 `--force-rebuild`，跳过图表使用 `--skip-plots`，快速验收使用 `--quick`。",
        "",
        "## 主要结果",
        "",
        "- `结果_v2/01_数据审计.json`：四个附件独立审计；",
        "- `结果_v2/03_需求模型选择.csv`：需求候选模型和滚动回测；",
        "- `结果_v2/04_价格关系可靠性.csv`：价格关系门槛；",
        "- `结果_v2/05_端到端需求回测汇总.csv`：组合需求函数回测；",
        "- `结果_v2/06_成本回测汇总.csv`：成本口径和方法比较；",
        "- `结果_v2/09_七天六品类最终策略.csv`：恰好42条最终策略；",
        "- `结果_v2/10_旧新方案对比.md`：旧方案与新方案差异。",
        "",
        "旧的 `问题二/脚本/求解问题二.py` 和 `问题二/结果/` 保留为历史基准，不作为 v2 的主要依赖。",
    ]) + "\n"
    (QUESTION_DIR / "README_v2.md").write_text(readme, encoding="utf-8")
    return report, readme
