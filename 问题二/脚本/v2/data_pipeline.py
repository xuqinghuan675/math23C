# -*- coding: utf-8 -*-
"""四个附件的独立读取、连接、审计和面板构造。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import (
    CACHE_DIR,
    CATEGORIES,
    DATA_DIR,
    DATA_END,
    OUTPUT_DIR,
    all_source_paths,
    config_snapshot,
)


DATA_PIPELINE_VERSION = "data-2.1.0-20260827"


@dataclass
class DataBundle:
    transactions: pd.DataFrame
    full_panel: pd.DataFrame
    normal_panel: pd.DataFrame
    discount_panel: pd.DataFrame
    item_daily: pd.DataFrame
    category_loss: dict[str, float]
    item_loss: dict[str, float]
    audit: dict[str, Any]
    connection_audit: pd.DataFrame
    gap_audit: pd.DataFrame
    extreme_audit: pd.DataFrame
    first_date: pd.Timestamp
    last_date: pd.Timestamp


def _write_csv(frame: pd.DataFrame, name: str) -> None:
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
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in all_source_paths():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result


def _config_hash() -> str:
    payload = json.dumps(config_snapshot(), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_manifest() -> dict[str, Any]:
    return {
        "程序版本": __version__,
        "数据流程版本": DATA_PIPELINE_VERSION,
        "原始附件哈希": _source_hashes(),
        "配置哈希": _config_hash(),
    }


def _cache_is_valid() -> bool:
    manifest_path = CACHE_DIR / "数据缓存清单.json"
    cache_path = CACHE_DIR / "数据包.pkl"
    if not manifest_path.exists() or not cache_path.exists():
        return False
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        return saved == _cache_manifest()
    except (OSError, ValueError, TypeError):
        return False


def _save_cache(bundle: DataBundle) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(bundle, CACHE_DIR / "数据包.pkl", compression="gzip")
    (CACHE_DIR / "数据缓存清单.json").write_text(
        json.dumps(_cache_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _read_source_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw1 = pd.read_excel(DATA_DIR / "附件1.xlsx", sheet_name=0, dtype=str)
    raw2 = pd.read_excel(DATA_DIR / "附件2.xlsx", sheet_name=0)
    raw3 = pd.read_excel(DATA_DIR / "附件3.xlsx", sheet_name=0, dtype=str)
    raw4_category = pd.read_excel(DATA_DIR / "附件4.xlsx", sheet_name=0)
    raw4_item = pd.read_excel(DATA_DIR / "附件4.xlsx", sheet_name=1)
    return raw1, raw2, raw3, raw4_category, raw4_item


def _normalise_sources() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]
]:
    raw1, raw2, raw3, raw4_category, raw4_item = _read_source_files()

    item = raw1.iloc[:, [0, 2, 3]].copy()
    item.columns = ["单品编码", "分类编码", "品类"]
    for col in item.columns:
        item[col] = _clean_text(item[col])

    sales = raw2.iloc[:, :7].copy()
    sales.columns = [
        "销售日期",
        "扫码销售时间",
        "单品编码",
        "销量",
        "售价",
        "销售类型",
        "是否打折销售",
    ]
    sales["销售日期"] = pd.to_datetime(sales["销售日期"], errors="coerce")
    sales["扫码销售时间"] = pd.to_datetime(sales["扫码销售时间"], errors="coerce")
    sales["单品编码"] = _clean_text(sales["单品编码"])
    sales["销售类型"] = _clean_text(sales["销售类型"])
    sales["是否打折销售"] = _clean_text(sales["是否打折销售"])
    sales["销量"] = pd.to_numeric(sales["销量"], errors="coerce")
    sales["售价"] = pd.to_numeric(sales["售价"], errors="coerce")
    sales["原始行号"] = np.arange(1, len(sales) + 1, dtype=np.int64)

    wholesale = raw3.iloc[:, [0, 1, 2]].copy()
    wholesale.columns = ["销售日期", "单品编码", "进价"]
    wholesale["销售日期"] = pd.to_datetime(wholesale["销售日期"], errors="coerce")
    wholesale["单品编码"] = _clean_text(wholesale["单品编码"])
    wholesale["进价"] = pd.to_numeric(wholesale["进价"], errors="coerce")

    category_loss_frame = raw4_category.iloc[:, [1, 2]].copy()
    category_loss_frame.columns = ["品类", "损耗率百分数"]
    category_loss_frame["品类"] = _clean_text(category_loss_frame["品类"])
    category_loss_frame["损耗率百分数"] = pd.to_numeric(
        category_loss_frame["损耗率百分数"], errors="coerce"
    )
    category_loss = dict(
        zip(
            category_loss_frame["品类"],
            category_loss_frame["损耗率百分数"] / 100.0,
        )
    )

    item_loss_frame = raw4_item.iloc[:, [0, 2]].copy()
    item_loss_frame.columns = ["单品编码", "损耗率百分数"]
    item_loss_frame["单品编码"] = _clean_text(item_loss_frame["单品编码"])
    item_loss_frame["损耗率百分数"] = pd.to_numeric(
        item_loss_frame["损耗率百分数"], errors="coerce"
    )
    item_loss = dict(
        zip(item_loss_frame["单品编码"], item_loss_frame["损耗率百分数"] / 100.0)
    )

    return item, sales, wholesale, category_loss, item_loss


def _hour_from_scan_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    hour = parsed.dt.hour.astype(float)
    fallback = series.astype(str).str.extract(r"^(\d{1,2})")[0]
    fallback = pd.to_numeric(fallback, errors="coerce")
    return hour.fillna(fallback)


def _panel_from_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        transactions.groupby(["销售日期", "品类"], as_index=False, observed=True)
        .agg(
            正销售量=("正销售量", "sum"),
            退货量=("退货量", "sum"),
            净销售量=("销量", "sum"),
            正销售额=("正销售额", "sum"),
            退货金额=("退货金额", "sum"),
            净销售额=("销售额", "sum"),
            正销售成本额=("正销售成本额", "sum"),
            退货成本额=("退货成本额", "sum"),
            净销售成本额=("成本额", "sum"),
            折扣销售量=("折扣正销量", "sum"),
            折扣销售额=("折扣正销售额", "sum"),
            正常销售量=("正常正销量", "sum"),
            正常销售额=("正常正销售额", "sum"),
            正常销售成本额=("正常正销售成本额", "sum"),
            交易条数=("单品编码", "size"),
            单品数=("单品编码", "nunique"),
            最早扫码时间=("扫码销售时间", "min"),
            最晚扫码时间=("扫码销售时间", "max"),
        )
    )
    positive_qty = grouped["正销售量"].replace(0, np.nan)
    normal_qty = grouped["正常销售量"].replace(0, np.nan)
    grouped["销量加权售价"] = grouped["正销售额"] / positive_qty
    grouped["销量加权进价"] = grouped["正销售成本额"] / positive_qty
    grouped["正常销售售价"] = grouped["正常销售额"] / normal_qty
    grouped["正常销售进价"] = grouped["正常销售成本额"] / normal_qty
    grouped["折扣销量占比"] = grouped["折扣销售量"] / positive_qty
    grouped["折扣销售额占比"] = grouped["折扣销售额"] / grouped["正销售额"].replace(0, np.nan)
    grouped["正常销售加成率"] = grouped["正常销售售价"] / grouped["正常销售进价"] - 1.0
    grouped["星期"] = grouped["销售日期"].dt.weekday + 1
    grouped["月份"] = grouped["销售日期"].dt.month
    return grouped.sort_values(["销售日期", "品类"]).reset_index(drop=True)


def _normal_panel(transactions: pd.DataFrame) -> pd.DataFrame:
    normal = transactions[transactions["正常记录"]].copy()
    result = (
        normal.groupby(["销售日期", "品类"], as_index=False, observed=True)
        .agg(
            正常销售量=("销量", "sum"),
            正常销售额=("销售额", "sum"),
            正常销售成本额=("成本额", "sum"),
            正常交易条数=("单品编码", "size"),
            正常单品数=("单品编码", "nunique"),
            最后正常销售时间=("扫码销售时间", "max"),
        )
    )
    result["正常销售售价"] = result["正常销售额"] / result["正常销售量"]
    result["正常销售进价"] = result["正常销售成本额"] / result["正常销售量"]
    result["正常销售加成率"] = result["正常销售售价"] / result["正常销售进价"] - 1.0
    result["星期"] = result["销售日期"].dt.weekday + 1
    result["月份"] = result["销售日期"].dt.month
    return result.sort_values(["销售日期", "品类"]).reset_index(drop=True)


def _discount_panel(transactions: pd.DataFrame) -> pd.DataFrame:
    discounts = transactions[transactions["折扣记录"]].copy()
    normal = transactions[transactions["正常记录"]].copy()
    reference = (
        normal.groupby(["销售日期", "单品编码"], as_index=False, observed=True)
        .agg(同日正常价中位数=("售价", "median"), 同日正常价加权=("销售额", "sum"), 同日正常量=("销量", "sum"))
    )
    reference["同日正常价加权"] = reference["同日正常价加权"] / reference["同日正常量"]
    discounts = discounts.merge(reference, on=["销售日期", "单品编码"], how="left", validate="many_to_one")
    discounts["折扣价比"] = discounts["售价"] / discounts["同日正常价中位数"]
    discounts["是否匹配同日正常价"] = discounts["同日正常价中位数"].notna()
    result = (
        discounts.groupby(["销售日期", "品类"], as_index=False, observed=True)
        .agg(
            折扣销售量=("销量", "sum"),
            折扣销售额=("销售额", "sum"),
            折扣交易条数=("单品编码", "size"),
            折扣单品数=("单品编码", "nunique"),
            折扣同日正常价匹配条数=("是否匹配同日正常价", "sum"),
            折扣价比中位数=("折扣价比", "median"),
            折扣价比加权=("折扣价比", lambda x: float(np.nanmedian(x))),
            折扣晚于19点占比=("扫码小时", lambda x: float((x >= 19).mean())),
        )
    )
    result["折扣同日正常价匹配率"] = result["折扣同日正常价匹配条数"] / result["折扣交易条数"].replace(0, np.nan)
    result["折扣价比加权"] = result["折扣价比加权"].replace([np.inf, -np.inf], np.nan)
    return result.sort_values(["销售日期", "品类"]).reset_index(drop=True)


def _item_daily(transactions: pd.DataFrame) -> pd.DataFrame:
    normal = transactions[transactions["正常记录"]].copy()
    discounts = transactions[transactions["折扣记录"]].copy()
    result = (
        normal.groupby(["销售日期", "单品编码", "品类"], as_index=False, observed=True)
        .agg(
            正常销售量=("销量", "sum"),
            正常售价中位数=("售价", "median"),
            正常销售额=("销售额", "sum"),
            正常销售成本额=("成本额", "sum"),
            最后销售时间=("扫码销售时间", "max"),
        )
    )
    result["正常销量加权售价"] = result["正常销售额"] / result["正常销售量"]
    result["当日批发价"] = result["正常销售成本额"] / result["正常销售量"]
    result["单品加成率"] = result["正常销量加权售价"] / result["当日批发价"] - 1.0
    discount_qty = (
        discounts.groupby(["销售日期", "单品编码"], as_index=False, observed=True)
        .agg(折扣销售量=("销量", "sum"))
    )
    result = result.merge(discount_qty, on=["销售日期", "单品编码"], how="left", validate="one_to_one")
    result["折扣销售量"] = result["折扣销售量"].fillna(0.0)
    result["是否有折扣"] = np.where(result["折扣销售量"] > 0, "是", "否")
    result["折扣占比"] = result["折扣销售量"] / (result["正常销售量"] + result["折扣销售量"]).replace(0, np.nan)
    return result.sort_values(["销售日期", "品类", "单品编码"]).reset_index(drop=True)


def _build_gap_audit(
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    transactions: pd.DataFrame,
    full_panel: pd.DataFrame,
    normal_panel: pd.DataFrame,
) -> pd.DataFrame:
    calendar = pd.date_range(first_date, last_date, freq="D")
    observed = pd.DatetimeIndex(transactions["销售日期"].drop_duplicates().sort_values())
    rows = [
        {
            "范围": "全店",
            "缺失日期数": int(len(calendar.difference(observed))),
            "缺失日期": ",".join(str(x.date()) for x in calendar.difference(observed)),
            "说明": "无流水日期不作为零需求填入模型面板",
        }
    ]
    for cat in CATEGORIES:
        observed_cat = pd.DatetimeIndex(
            full_panel.loc[full_panel["品类"] == cat, "销售日期"].drop_duplicates().sort_values()
        )
        normal_cat = pd.DatetimeIndex(
            normal_panel.loc[normal_panel["品类"] == cat, "销售日期"].drop_duplicates().sort_values()
        )
        missing_cat = calendar.difference(observed_cat)
        missing_normal = calendar.difference(normal_cat)
        rows.append(
            {
                "范围": cat,
                "缺失日期数": int(len(missing_cat)),
                "缺失日期": ",".join(str(x.date()) for x in missing_cat),
                "正常销售面板缺失日期数": int(len(missing_normal)),
                "正常销售面板缺失日期": ",".join(str(x.date()) for x in missing_normal),
                "说明": "品类无流水或仅有折扣记录时保留为缺口，不静默填零",
            }
        )
    return pd.DataFrame(rows)


def _build_extreme_audit(transactions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positive = transactions[transactions["销售类型"] == "销售"].copy()
    for cat in CATEGORIES:
        sub = positive[positive["品类"] == cat]
        for variable, col in [("销售单价", "售价"), ("批发价", "进价"), ("销售量", "销量")]:
            values = sub[col].astype(float)
            q01, q99 = values.quantile([0.01, 0.99])
            extreme = sub[(values < q01) | (values > q99)]
            rows.append(
                {
                    "品类": cat,
                    "变量": variable,
                    "样本数": int(len(values)),
                    "百分之一分位": float(q01),
                    "百分之九十九分位": float(q99),
                    "区间外记录数": int(len(extreme)),
                    "区间外数量占比": float(extreme["销量"].sum() / sub["销量"].sum()) if len(sub) else np.nan,
                    "区间外销售额占比": float(extreme["销售额"].sum() / sub["销售额"].sum()) if len(sub) else np.nan,
                    "处理": "仅做影响审计，未删除原始记录",
                }
            )
    return pd.DataFrame(rows)


def _audit_and_build(
    item: pd.DataFrame,
    sales: pd.DataFrame,
    wholesale: pd.DataFrame,
    category_loss: dict[str, float],
    item_loss: dict[str, float],
) -> DataBundle:
    missing = {
        "附件一": int(item.isna().sum().sum()),
        "附件二": int(sales.isna().sum().sum()),
        "附件三": int(wholesale.isna().sum().sum()),
        "附件四品类": int(sum(pd.isna(v) for v in category_loss.values())),
        "附件四单品": int(sum(pd.isna(v) for v in item_loss.values())),
    }
    first_date = pd.Timestamp(sales["销售日期"].min()).normalize()
    last_date = pd.Timestamp(sales["销售日期"].max()).normalize()

    merged = (
        sales.merge(item, on="单品编码", how="left", validate="many_to_one")
        .merge(wholesale, on=["销售日期", "单品编码"], how="left", validate="many_to_one")
    )
    merged["扫码小时"] = _hour_from_scan_time(merged["扫码销售时间"])
    merged["销售额"] = merged["销量"] * merged["售价"]
    merged["成本额"] = merged["销量"] * merged["进价"]
    merged["正销售量"] = np.where(merged["销售类型"] == "销售", merged["销量"].clip(lower=0), 0.0)
    merged["退货量"] = np.where(merged["销售类型"] == "退货", merged["销量"], 0.0)
    merged["正销售额"] = np.where(merged["销售类型"] == "销售", merged["销售额"].clip(lower=0), 0.0)
    merged["退货金额"] = np.where(merged["销售类型"] == "退货", merged["销售额"], 0.0)
    merged["正销售成本额"] = np.where(merged["销售类型"] == "销售", merged["成本额"].clip(lower=0), 0.0)
    merged["退货成本额"] = np.where(merged["销售类型"] == "退货", merged["成本额"], 0.0)
    merged["折扣记录"] = (merged["销售类型"] == "销售") & (merged["是否打折销售"] == "是") & (merged["销量"] > 0)
    merged["正常记录"] = (merged["销售类型"] == "销售") & (merged["是否打折销售"] == "否") & (merged["销量"] > 0)
    merged["折扣正销量"] = np.where(merged["折扣记录"], merged["销量"], 0.0)
    merged["折扣正销售额"] = np.where(merged["折扣记录"], merged["销售额"], 0.0)
    merged["正常正销量"] = np.where(merged["正常记录"], merged["销量"], 0.0)
    merged["正常正销售额"] = np.where(merged["正常记录"], merged["销售额"], 0.0)
    merged["正常正销售成本额"] = np.where(merged["正常记录"], merged["成本额"], 0.0)

    full_panel = _panel_from_transactions(merged)
    normal_panel = _normal_panel(merged)
    discount_panel = _discount_panel(merged)
    item_daily = _item_daily(merged)

    discount_merge_cols = [
        "销售日期", "品类", "折扣交易条数", "折扣单品数",
        "折扣同日正常价匹配条数", "折扣价比中位数", "折扣价比加权", "折扣晚于19点占比", "折扣同日正常价匹配率",
    ]
    full_panel = full_panel.merge(
        discount_panel[discount_merge_cols], on=["销售日期", "品类"], how="left", validate="one_to_one"
    )
    normal_merge_cols = [
        "销售日期", "品类", "正常销售量", "正常销售额", "正常销售成本额", "正常交易条数", "正常单品数",
        "正常销售售价", "正常销售进价", "正常销售加成率", "最后正常销售时间",
    ]
    full_panel = full_panel.merge(
        normal_panel[normal_merge_cols], on=["销售日期", "品类"], how="left", suffixes=("", "_正常面板"), validate="one_to_one"
    )
    for col in ["折扣交易条数", "折扣单品数", "折扣同日正常价匹配条数"]:
        full_panel[col] = full_panel[col].fillna(0).astype(int)
    for col in ["折扣销售量", "折扣销售额"]:
        full_panel[col] = full_panel[col].fillna(0.0)
    full_panel["折扣销量占比"] = full_panel["折扣销售量"] / full_panel["正销售量"].replace(0, np.nan)
    full_panel["折扣销售额占比"] = full_panel["折扣销售额"] / full_panel["正销售额"].replace(0, np.nan)
    full_panel["折扣同日正常价匹配率"] = full_panel["折扣同日正常价匹配率"].fillna(np.nan)
    full_panel["正常销售面板存在"] = np.where(full_panel["正常销售量"].notna(), "是", "否")
    normal_panel = normal_panel.merge(
        full_panel[
            [
                "销售日期", "品类", "折扣销量占比", "折扣销售额占比", "折扣晚于19点占比",
            ]
        ], on=["销售日期", "品类"], how="left", validate="one_to_one"
    )

    category_match = merged["品类"].notna()
    wholesale_match = merged["进价"].notna()
    sign_error = (
        ((merged["销售类型"] == "退货") & (merged["销量"] >= 0))
        | ((merged["销售类型"] == "销售") & (merged["销量"] < 0))
    )
    exact_dup = int(sales.duplicated().sum())
    no_time_cols = ["销售日期", "单品编码", "销量", "售价", "销售类型", "是否打折销售"]
    candidate_groups = sales.groupby(no_time_cols, dropna=False, observed=True).size()
    candidate_dup_extra = int((candidate_groups[candidate_groups > 1] - 1).sum())
    calendar = pd.date_range(first_date, last_date, freq="D")
    observed_dates = pd.DatetimeIndex(sales["销售日期"].drop_duplicates())
    positive = merged[merged["销售类型"] == "销售"]
    returns = merged[merged["销售类型"] == "退货"]

    conservation_rows = []
    for cat in CATEGORIES:
        raw_cat = merged[merged["品类"] == cat]
        panel_cat = full_panel[full_panel["品类"] == cat]
        for metric, raw_col, panel_col in [
            ("净销售量", "销量", "净销售量"),
            ("净销售额", "销售额", "净销售额"),
            ("净销售成本额", "成本额", "净销售成本额"),
            ("正销售量", "正销售量", "正销售量"),
            ("正销售额", "正销售额", "正销售额"),
            ("正销售成本额", "正销售成本额", "正销售成本额"),
        ]:
            raw_value = float(raw_cat[raw_col].sum())
            panel_value = float(panel_cat[panel_col].sum())
            conservation_rows.append(
                {
                    "检查类别": "各品类守恒",
                    "品类": cat,
                    "指标": metric,
                    "原始流水汇总": raw_value,
                    "面板汇总": panel_value,
                    "差值": panel_value - raw_value,
                    "是否通过": "是" if abs(panel_value - raw_value) < 1e-8 else "否",
                }
            )

    normal_keys = merged.loc[merged["正常记录"], ["销售日期", "单品编码"]].drop_duplicates()
    discount_keys = merged.loc[merged["折扣记录"], ["销售日期", "单品编码"]]
    matched_discount_records = int(
        discount_keys.merge(normal_keys, on=["销售日期", "单品编码"], how="inner").shape[0]
    )

    connection_rows = [
        {"检查项": "附件一行数", "实际值": len(item), "是否通过": "是" if len(item) == 251 else "否", "说明": "单品信息行数"},
        {"检查项": "附件二行数", "实际值": len(sales), "是否通过": "是" if len(sales) == 878503 else "否", "说明": "销售流水行数"},
        {"检查项": "附件三行数", "实际值": len(wholesale), "是否通过": "是" if len(wholesale) == 55982 else "否", "说明": "日期—单品批发价行数"},
        {"检查项": "附件四品类覆盖数", "实际值": len(category_loss), "是否通过": "是" if set(category_loss) == set(CATEGORIES) else "否", "说明": "六个题目品类"},
        {"检查项": "附件一单品编码唯一", "实际值": int(item["单品编码"].nunique()), "是否通过": "是" if not item["单品编码"].duplicated().any() else "否", "说明": "不重复"},
        {"检查项": "附件三日期—单品唯一", "实际值": int(wholesale[["销售日期", "单品编码"]].drop_duplicates().shape[0]), "是否通过": "是" if not wholesale.duplicated(["销售日期", "单品编码"]).any() else "否", "说明": "不重复"},
        {"检查项": "销售流水品类匹配", "实际值": int(category_match.sum()), "匹配率": float(category_match.mean()), "是否通过": "是" if category_match.all() else "否", "说明": "附件二连接附件一"},
        {"检查项": "销售流水批发价匹配", "实际值": int(wholesale_match.sum()), "匹配率": float(wholesale_match.mean()), "是否通过": "是" if wholesale_match.all() else "否", "说明": "附件二连接附件三"},
        {"检查项": "销售量符号核对", "实际值": int(sign_error.sum()), "是否通过": "是" if not sign_error.any() else "否", "说明": "销售为正、退货为负"},
        {"检查项": "完整重复流水", "实际值": exact_dup, "是否通过": "是" if exact_dup == 0 else "否", "说明": "完整行重复"},
        {"检查项": "忽略扫码时间候选重复额外行", "实际值": candidate_dup_extra, "是否通过": "是", "说明": "扫码时间不同的交易不删除，仅记录"},
        {"检查项": "损耗率合法范围", "实际值": int(sum(not np.isfinite(v) or v < 0 or v >= 1 for v in category_loss.values())), "是否通过": "是" if all(np.isfinite(v) and 0 <= v < 1 for v in category_loss.values()) else "否", "说明": "零损耗率为合法边界"},
        {"检查项": "整天无流水日期", "实际值": int(len(calendar.difference(observed_dates))), "是否通过": "是", "说明": "不填成零需求"},
    ]
    connection_audit = pd.DataFrame(connection_rows + conservation_rows)
    gap_audit = _build_gap_audit(first_date, last_date, merged, full_panel, normal_panel)
    extreme_audit = _build_extreme_audit(merged)

    audit = {
        "附件一行数": int(len(item)),
        "附件二行数": int(len(sales)),
        "附件三行数": int(len(wholesale)),
        "附件四品类行数": int(len(category_loss)),
        "附件四单品行数": int(len(item_loss)),
        "缺失单元格总数": missing,
        "附件一单品数": int(item["单品编码"].nunique()),
        "附件一单品编码重复数": int(item["单品编码"].duplicated().sum()),
        "附件二销售记录数": int((sales["销售类型"] == "销售").sum()),
        "附件二退货记录数": int((sales["销售类型"] == "退货").sum()),
        "附件二折扣标记记录数": int((sales["是否打折销售"] == "是").sum()),
        "附件二折扣销售记录数": int(((sales["销售类型"] == "销售") & (sales["是否打折销售"] == "是")).sum()),
        "附件二正常销售记录数": int(len(merged[merged["正常记录"]])),
        "附件二完整重复行数": exact_dup,
        "附件二忽略扫码时间候选重复额外行数": candidate_dup_extra,
        "附件三日期单品组合数": int(wholesale[["销售日期", "单品编码"]].drop_duplicates().shape[0]),
        "销售日期起点": str(first_date.date()),
        "销售日期终点": str(last_date.date()),
        "日历天数": int(len(calendar)),
        "有流水日期数": int(sales["销售日期"].nunique()),
        "无流水日期数": int(len(calendar.difference(observed_dates))),
        "整天无流水日期": [str(x.date()) for x in calendar.difference(observed_dates)],
        "销售量千克": float(positive["销量"].sum()),
        "退货量千克": float(returns["销量"].sum()),
        "净销售量千克": float(merged["销量"].sum()),
        "销售额元": float(positive["销售额"].sum()),
        "退货金额元": float(returns["销售额"].sum()),
        "净销售额元": float(merged["销售额"].sum()),
        "销售成本额元": float(positive["成本额"].sum()),
        "退货成本额元": float(returns["成本额"].sum()),
        "净销售成本额元": float(merged["成本额"].sum()),
        "折扣销售量千克": float(positive.loc[positive["是否打折销售"] == "是", "销量"].sum()),
        "分类未匹配数": int((~category_match).sum()),
        "进价未匹配数": int((~wholesale_match).sum()),
        "销售量符号不一致数": int(sign_error.sum()),
        "全量品类日期面板行数": int(len(full_panel)),
        "正常销售品类日期面板行数": int(len(normal_panel)),
        "折扣品类日期面板行数": int(len(discount_panel)),
        "单品日期正常销售面板行数": int(len(item_daily)),
        "全量品类面板各类行数": {str(k): int(v) for k, v in full_panel["品类"].value_counts().to_dict().items()},
        "正常品类面板各类行数": {str(k): int(v) for k, v in normal_panel["品类"].value_counts().to_dict().items()},
        "折扣同日正常价匹配记录数": matched_discount_records,
        "日期—单品批发价匹配率": float(wholesale_match.mean()),
        "折扣标记占销售记录比例": float((sales["是否打折销售"] == "是").sum() / max(1, (sales["销售类型"] == "销售").sum())),
        "折扣销量占销售量比例": float(positive.loc[positive["是否打折销售"] == "是", "销量"].sum() / positive["销量"].sum()),
        "附件四零损耗率单品数": int(sum(v == 0 for v in item_loss.values())),
        "数据终点是否不晚于题目目标前一天": bool(last_date <= DATA_END),
        "各品类守恒": conservation_rows,
    }

    _write_csv(full_panel, "02_全量净需求面板.csv")
    _write_csv(normal_panel, "02_正常销售面板.csv")
    _write_csv(discount_panel, "02_折扣销售面板.csv")
    _write_csv(item_daily, "02_单品日期正常销售面板.csv")
    _write_csv(connection_audit, "01_附件连接审计.csv")
    _write_csv(gap_audit, "01_日期缺口审计.csv")
    _write_csv(extreme_audit, "01_极端值影响审计.csv")
    _write_csv(
        pd.DataFrame(
            [
                {
                    "检查项": "行数", "实际值": audit["附件一行数"], "说明": "附件一"
                },
                {"检查项": "行数", "实际值": audit["附件二行数"], "说明": "附件二"},
                {"检查项": "行数", "实际值": audit["附件三行数"], "说明": "附件三"},
                {"检查项": "行数", "实际值": audit["附件四品类行数"], "说明": "附件四品类表"},
                {"检查项": "缺失单元格总数", "实际值": sum(missing.values()), "说明": "四个附件"},
                {"检查项": "销售量符号不一致数", "实际值": audit["销售量符号不一致数"], "说明": "销售/退货符号"},
                {"检查项": "折扣销售记录数", "实际值": audit["附件二折扣销售记录数"], "说明": "合法折扣状态，未删除"},
                {"检查项": "日期—单品批发价匹配率", "实际值": audit["日期—单品批发价匹配率"], "说明": "应为100%"},
                {"检查项": "全量品类日期面板行数", "实际值": audit["全量品类日期面板行数"], "说明": "全量净需求"},
                {"检查项": "正常销售品类日期面板行数", "实际值": audit["正常销售品类日期面板行数"], "说明": "正常价格响应"},
                {"检查项": "无流水日期数", "实际值": audit["无流水日期数"], "说明": "不填为零需求"},
            ]
        ),
        "01_数据审计.csv",
    )
    audit_rows = [
        {"检查项": "附件一行数", "实际值": audit["附件一行数"], "是否通过": "是" if audit["附件一行数"] == 251 else "否", "说明": "附件一"},
        {"检查项": "附件二行数", "实际值": audit["附件二行数"], "是否通过": "是" if audit["附件二行数"] == 878503 else "否", "说明": "附件二"},
        {"检查项": "附件三行数", "实际值": audit["附件三行数"], "是否通过": "是" if audit["附件三行数"] == 55982 else "否", "说明": "附件三"},
        {"检查项": "附件四品类行数", "实际值": audit["附件四品类行数"], "是否通过": "是" if audit["附件四品类行数"] == 6 else "否", "说明": "附件四品类表"},
        {"检查项": "缺失单元格总数", "实际值": sum(missing.values()), "是否通过": "是" if sum(missing.values()) == 0 else "否", "说明": "四个附件"},
        {"检查项": "退货记录数", "实际值": audit["附件二退货记录数"], "是否通过": "是" if audit["附件二退货记录数"] > 0 else "否", "说明": "合法负销量，纳入净需求"},
        {"检查项": "折扣销售记录数", "实际值": audit["附件二折扣销售记录数"], "是否通过": "是", "说明": "合法折扣状态，未删除"},
        {"检查项": "完整重复行数", "实际值": audit["附件二完整重复行数"], "是否通过": "是" if audit["附件二完整重复行数"] == 0 else "否", "说明": "完整记录"},
        {"检查项": "忽略扫码时间候选重复额外行数", "实际值": audit["附件二忽略扫码时间候选重复额外行数"], "是否通过": "是", "说明": "不同扫码时间的交易保留"},
        {"检查项": "日期—单品批发价匹配率", "实际值": audit["日期—单品批发价匹配率"], "是否通过": "是" if audit["日期—单品批发价匹配率"] == 1 else "否", "说明": "附件二连接附件三"},
        {"检查项": "分类未匹配数", "实际值": audit["分类未匹配数"], "是否通过": "是" if audit["分类未匹配数"] == 0 else "否", "说明": "附件二连接附件一"},
        {"检查项": "销售量符号不一致数", "实际值": audit["销售量符号不一致数"], "是否通过": "是" if audit["销售量符号不一致数"] == 0 else "否", "说明": "销售为正、退货为负"},
        {"检查项": "销售日期覆盖起点", "实际值": audit["销售日期起点"], "是否通过": "是", "说明": "原始销售流水"},
        {"检查项": "销售日期覆盖终点", "实际值": audit["销售日期终点"], "是否通过": "是" if audit["数据终点是否不晚于题目目标前一天"] else "否", "说明": "不使用2023年7月1日及之后数据"},
        {"检查项": "全店无流水日期数", "实际值": audit["无流水日期数"], "是否通过": "是", "说明": "不填为零需求"},
        {"检查项": "全量品类日期面板行数", "实际值": audit["全量品类日期面板行数"], "是否通过": "是", "说明": "全量净需求面板"},
        {"检查项": "正常销售品类日期面板行数", "实际值": audit["正常销售品类日期面板行数"], "是否通过": "是", "说明": "正常价格响应面板"},
        {"检查项": "折扣同日正常价匹配记录数", "实际值": matched_discount_records, "是否通过": "是", "说明": "未匹配记录不填造价格"},
        {"检查项": "折扣同日正常价匹配率", "实际值": matched_discount_records / max(1, audit["附件二折扣销售记录数"]), "是否通过": "是", "说明": "折扣敏感性口径"},
    ]
    audit_rows.extend(connection_audit.to_dict("records"))
    _write_csv(pd.DataFrame(audit_rows), "01_数据审计.csv")
    (OUTPUT_DIR / "01_数据审计.json").write_text(
        json.dumps(_json_safe(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return DataBundle(
        transactions=merged,
        full_panel=full_panel,
        normal_panel=normal_panel,
        discount_panel=discount_panel,
        item_daily=item_daily,
        category_loss=category_loss,
        item_loss=item_loss,
        audit=audit,
        connection_audit=connection_audit,
        gap_audit=gap_audit,
        extreme_audit=extreme_audit,
        first_date=first_date,
        last_date=last_date,
    )


def load_bundle(force_rebuild: bool = False) -> DataBundle:
    """读取缓存或从四个原始附件完整重建数据包。"""
    if not force_rebuild and _cache_is_valid():
        bundle = pd.read_pickle(CACHE_DIR / "数据包.pkl", compression="gzip")
        # 缓存内也必须重新落盘审计结果，确保输出目录可独立复核。
        _write_csv(bundle.full_panel, "02_全量净需求面板.csv")
        _write_csv(bundle.normal_panel, "02_正常销售面板.csv")
        _write_csv(bundle.discount_panel, "02_折扣销售面板.csv")
        _write_csv(bundle.item_daily, "02_单品日期正常销售面板.csv")
        _write_csv(bundle.connection_audit, "01_附件连接审计.csv")
        _write_csv(bundle.gap_audit, "01_日期缺口审计.csv")
        _write_csv(bundle.extreme_audit, "01_极端值影响审计.csv")
        cached_audit_rows = [
            {"检查项": "附件一行数", "实际值": bundle.audit["附件一行数"], "是否通过": "是" if bundle.audit["附件一行数"] == 251 else "否", "说明": "附件一"},
            {"检查项": "附件二行数", "实际值": bundle.audit["附件二行数"], "是否通过": "是" if bundle.audit["附件二行数"] == 878503 else "否", "说明": "附件二"},
            {"检查项": "附件三行数", "实际值": bundle.audit["附件三行数"], "是否通过": "是" if bundle.audit["附件三行数"] == 55982 else "否", "说明": "附件三"},
            {"检查项": "附件四品类行数", "实际值": bundle.audit["附件四品类行数"], "是否通过": "是" if bundle.audit["附件四品类行数"] == 6 else "否", "说明": "附件四品类表"},
            {"检查项": "缺失单元格总数", "实际值": sum(bundle.audit["缺失单元格总数"].values()), "是否通过": "是" if sum(bundle.audit["缺失单元格总数"].values()) == 0 else "否", "说明": "四个附件"},
            {"检查项": "退货记录数", "实际值": bundle.audit["附件二退货记录数"], "是否通过": "是" if bundle.audit["附件二退货记录数"] > 0 else "否", "说明": "合法负销量，纳入净需求"},
            {"检查项": "折扣销售记录数", "实际值": bundle.audit["附件二折扣销售记录数"], "是否通过": "是", "说明": "合法折扣状态，未删除"},
            {"检查项": "完整重复行数", "实际值": bundle.audit["附件二完整重复行数"], "是否通过": "是" if bundle.audit["附件二完整重复行数"] == 0 else "否", "说明": "完整记录"},
            {"检查项": "忽略扫码时间候选重复额外行数", "实际值": bundle.audit["附件二忽略扫码时间候选重复额外行数"], "是否通过": "是", "说明": "不同扫码时间的交易保留"},
            {"检查项": "日期—单品批发价匹配率", "实际值": bundle.audit["日期—单品批发价匹配率"], "是否通过": "是" if bundle.audit["日期—单品批发价匹配率"] == 1 else "否", "说明": "附件二连接附件三"},
            {"检查项": "分类未匹配数", "实际值": bundle.audit["分类未匹配数"], "是否通过": "是" if bundle.audit["分类未匹配数"] == 0 else "否", "说明": "附件二连接附件一"},
            {"检查项": "销售量符号不一致数", "实际值": bundle.audit["销售量符号不一致数"], "是否通过": "是" if bundle.audit["销售量符号不一致数"] == 0 else "否", "说明": "销售为正、退货为负"},
            {"检查项": "销售日期覆盖起点", "实际值": bundle.audit["销售日期起点"], "是否通过": "是", "说明": "原始销售流水"},
            {"检查项": "销售日期覆盖终点", "实际值": bundle.audit["销售日期终点"], "是否通过": "是" if bundle.audit["数据终点是否不晚于题目目标前一天"] else "否", "说明": "不使用2023年7月1日及之后数据"},
            {"检查项": "全店无流水日期数", "实际值": bundle.audit["无流水日期数"], "是否通过": "是", "说明": "不填为零需求"},
            {"检查项": "全量品类日期面板行数", "实际值": bundle.audit["全量品类日期面板行数"], "是否通过": "是", "说明": "全量净需求面板"},
            {"检查项": "正常销售品类日期面板行数", "实际值": bundle.audit["正常销售品类日期面板行数"], "是否通过": "是", "说明": "正常价格响应面板"},
            {"检查项": "折扣同日正常价匹配记录数", "实际值": bundle.audit["折扣同日正常价匹配记录数"], "是否通过": "是", "说明": "未匹配记录不填造价格"},
            {"检查项": "折扣同日正常价匹配率", "实际值": bundle.audit["折扣同日正常价匹配记录数"] / max(1, bundle.audit["附件二折扣销售记录数"]), "是否通过": "是", "说明": "折扣敏感性口径"},
        ]
        cached_audit_rows.extend(bundle.connection_audit.to_dict("records"))
        _write_csv(pd.DataFrame(cached_audit_rows), "01_数据审计.csv")
        (OUTPUT_DIR / "01_数据审计.json").write_text(
            json.dumps(_json_safe(bundle.audit), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return bundle

    item, sales, wholesale, category_loss, item_loss = _normalise_sources()
    if sales["销售日期"].isna().any() or wholesale["销售日期"].isna().any():
        raise ValueError("原始附件存在无法识别的日期")
    if sales["单品编码"].isna().any() or wholesale["单品编码"].isna().any():
        raise ValueError("原始附件存在无法识别的单品编码")
    if set(category_loss) != set(CATEGORIES):
        raise ValueError("附件四未覆盖六个题目品类")
    if any(not np.isfinite(v) or v < 0 or v >= 1 for v in category_loss.values()):
        raise ValueError("附件四品类损耗率不在合法范围")
    bundle = _audit_and_build(item, sales, wholesale, category_loss, item_loss)
    _save_cache(bundle)
    return bundle
