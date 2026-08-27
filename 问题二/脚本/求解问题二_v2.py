# -*- coding: utf-8 -*-
"""2023 年数学建模国赛 C 题问题二 v2 唯一正式入口。"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

from v2.category_indices import build_indices
from v2.config import CACHE_DIR, OUTPUT_DIR, RANDOM_SEED
from v2.cost_models import future_cost_points, run_cost_backtests
from v2.data_pipeline import _cache_is_valid, load_bundle
from v2.demand_models import run_demand_backtests
from v2.diagnostics import run_end_to_end_backtest
from v2.optimization import check_price_support, run_optimization, run_sensitivities
from v2.price_response import item_fixed_effect_robustness, run_price_response, reference_markup
from v2.reporting import (
    make_figures,
    record_legacy_baseline,
    write_documents,
    write_manifest,
    write_old_new_comparison,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题二 v2：六品类七日售价与补货联合决策")
    parser.add_argument("--force-rebuild", action="store_true", help="忽略本地数据缓存并重建")
    parser.add_argument("--skip-plots", action="store_true", help="跳过图表生成")
    parser.add_argument("--quick", action="store_true", help="使用较少情景和自助法次数进行快速验收")
    return parser.parse_args()


def _add_final_reliability_checks(
    price_info: dict,
    end_summary: pd.DataFrame,
    future_cost: pd.DataFrame,
    panel: pd.DataFrame,
) -> None:
    relation = price_info["relation"].copy()
    future_support: dict[str, bool] = {}
    for cat in relation["品类"]:
        fit = price_info["selected"][cat]
        support = {}
        from v2.optimization import _historical_support

        support = _historical_support(panel, cat)
        cost_rows = future_cost[future_cost["品类"] == cat].sort_values("日期")
        checks = []
        for _, row in cost_rows.iterrows():
            ref_price = float(row["预测批发价"]) * (1.0 + reference_markup(fit, pd.Timestamp(row["日期"])))
            _, supported, _, _, _ = check_price_support(ref_price, float(row["预测批发价"]), pd.Timestamp(row["日期"]), (0.35, 0.65), support)
            checks.append(supported)
        future_support[cat] = bool(checks) and bool(all(checks))
    relation["端到端不过度劣化"] = "待判定"
    relation["未来参考价格支持"] = ["是" if future_support.get(cat, False) else "否" for cat in relation["品类"]]
    for idx, row in relation.iterrows():
        end = end_summary[(end_summary["品类"] == row["品类"]) & (end_summary["需求口径"] == "正常销售量")]
        endpoint_pass = bool(end.empty or str(end["是否明显劣化"].iloc[0]) != "是")
        relation.loc[idx, "端到端不过度劣化"] = "是" if endpoint_pass else "否"
        base_reliable = str(row["价格关系是否可靠"]) == "是"
        final_reliable = base_reliable and endpoint_pass and future_support.get(row["品类"], False)
        relation.loc[idx, "价格关系是否可靠"] = "是" if final_reliable else "否"
        if not final_reliable:
            reason = str(row["可靠性判定"])
            additions = []
            if not endpoint_pass:
                additions.append("端到端回测明显劣化")
            if not future_support.get(row["品类"], False):
                additions.append("未来参考价格不在历史外圈")
            relation.loc[idx, "可靠性判定"] = reason + "；" + "、".join(additions)
            relation.loc[idx, "价格关系说明"] = "价格关系不可识别，最终不用于精细调价。"
    price_info["relation"] = relation
    relation.to_csv(OUTPUT_DIR / "04_价格关系可靠性.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = datetime.now().astimezone()
    cache_before = _cache_is_valid() and not args.force_rebuild
    baseline = record_legacy_baseline()

    bundle = load_bundle(force_rebuild=args.force_rebuild)
    index_cache = CACHE_DIR / "索引面板.pkl"
    index_cache_manifest = CACHE_DIR / "索引面板版本.txt"
    if not args.force_rebuild and index_cache.exists() and index_cache_manifest.exists() and index_cache_manifest.read_text(encoding="utf-8") == "index-2.1.0-20260827":
        panel, _index_df, _coverage_df, _structure_df = pd.read_pickle(index_cache, compression="gzip")
        panel.to_csv(OUTPUT_DIR / "02_品类日面板.csv", index=False, encoding="utf-8-sig")
        _index_df.to_csv(OUTPUT_DIR / "02_固定篮子价格成本指数.csv", index=False, encoding="utf-8-sig")
        _coverage_df.to_csv(OUTPUT_DIR / "02_指数覆盖率.csv", index=False, encoding="utf-8-sig")
        _structure_df.to_csv(OUTPUT_DIR / "02_商品结构指标.csv", index=False, encoding="utf-8-sig")
    else:
        panel, _index_df, _coverage_df, _structure_df = build_indices(
            bundle.full_panel, bundle.normal_panel, bundle.item_daily, write_outputs=True
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle((panel, _index_df, _coverage_df, _structure_df), index_cache, compression="gzip")
        index_cache_manifest.write_text("index-2.1.0-20260827", encoding="utf-8")
    demand_detail, demand_summary, demand_selection, normal_fits, _full_fits, selected_normal, _selected_full = run_demand_backtests(
        bundle.normal_panel, panel, bundle.last_date
    )
    price_info = run_price_response(panel, bundle.last_date, np.random.default_rng(RANDOM_SEED + 7), quick=args.quick)
    cost_detail, cost_summary, selected_cost = run_cost_backtests(panel, bundle.last_date)
    future_cost = future_cost_points(panel, selected_cost, bundle.last_date)
    future_cost.to_csv(OUTPUT_DIR / "06_未来成本点预测.csv", index=False, encoding="utf-8-sig")
    end_detail, end_summary, calibration_df, calibration = run_end_to_end_backtest(
        bundle.normal_panel,
        panel,
        selected_normal,
        price_info["selected"],
        selected_cost,
        bundle.last_date,
        demand_summary,
        np.random.default_rng(RANDOM_SEED + 19),
    )
    _add_final_reliability_checks(price_info, end_summary, future_cost, panel)

    optimization = run_optimization(
        panel,
        normal_fits,
        price_info,
        selected_cost,
        future_cost,
        bundle.category_loss,
        calibration,
        scenario_count=600 if args.quick else 6000,
        seed=RANDOM_SEED,
        last_date=bundle.last_date,
        quick=args.quick,
    )
    support_rows = []
    for cell in optimization["cells"]:
        support = optimization["support"][cell["品类"]]
        price = float(cell["最终"]["售价"])
        cost = float(cell["未来成本点值"])
        level, supported, markup_pos, price_pos, recent_distance = check_price_support(
            price, cost, pd.Timestamp(cell["日期"]), (0.35, 0.65), support
        )
        support_rows.append(
            {
                "日期": cell["日期"],
                "品类": cell["品类"],
                "候选价格": price,
                "点值成本": cost,
                "加成率": price / max(cost, 1e-9) - 1.0,
                "加成分位位置": markup_pos,
                "绝对售价分位位置": price_pos,
                "最近历史距离": recent_distance,
                "是否在历史外圈": "否" if supported else "是",
                "支持等级": level,
                "说明": "同时检查加成、相似成本状态下的绝对售价、固定篮子覆盖和最近历史距离",
            }
        )
    pd.DataFrame(support_rows).to_csv(OUTPUT_DIR / "04_价格支持范围.csv", index=False, encoding="utf-8-sig")
    sensitivities = run_sensitivities(
        optimization,
        panel,
        normal_fits,
        price_info,
        selected_cost,
        future_cost,
        bundle.category_loss,
        calibration,
        RANDOM_SEED + 101,
        bundle.last_date,
        quick=args.quick,
    )
    item_fixed_df = item_fixed_effect_robustness(bundle.item_daily, panel)
    item_fixed_df.to_csv(OUTPUT_DIR / "04_单品固定效应稳健性.csv", index=False, encoding="utf-8-sig")
    model_sensitivity = pd.read_csv(OUTPUT_DIR / "08_模型口径敏感性.csv", encoding="utf-8-sig")
    model_sensitivity = pd.concat([model_sensitivity, item_fixed_df], ignore_index=True, sort=False)
    model_sensitivity.to_csv(OUTPUT_DIR / "08_模型口径敏感性.csv", index=False, encoding="utf-8-sig")
    comparison, comparison_report = write_old_new_comparison(
        optimization["final"],
        price_info["relation"],
        demand_summary,
        cost_summary,
        end_summary,
        optimization["decomposition"],
    )
    write_documents(
        bundle,
        panel,
        normal_fits,
        demand_selection,
        cost_summary,
        price_info,
        end_summary,
        optimization,
        sensitivities,
        comparison,
    )
    if not args.skip_plots:
        make_figures(bundle, panel, normal_fits, price_info, optimization, sensitivities)
    end = datetime.now().astimezone()
    write_manifest(start, end, args.quick, cache_before)

    final = optimization["final"]
    total_profit = float(final["预计毛利"].sum())
    reliable_count = int(final.drop_duplicates("品类")["价格关系是否可靠"].eq("是").sum())
    print("问题二 v2 完成")
    print(f"四个附件数据截止日：{bundle.last_date.date()}")
    print(f"最终策略行数：{len(final)}")
    print(f"可靠价格关系品类数：{reliable_count}")
    print(f"七天预计毛利：{total_profit:.2f} 元")
    print(f"正式情景数：{optimization['bundle']['情景数']}")
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
