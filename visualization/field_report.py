#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连续场评分长表 → Markdown 报告。

一句话用法::

    python -m xmetai_evaluation.visualization.field_report \
        --scores scores.csv --out reports/field_fuxi --model FuXi

覆盖流程 ``weather_field_scores``（RMSE / ACC / 预报活跃度 / 纬向谱），
指标语义集中在本模块的 ``METRIC_SEMANTICS``，出图和报告都从这里取。

报告内容都是算出来的，不是套话：执行摘要（规则化诊断，逐条带数值）、逐变量总览、
RMSE 随时效表、活跃度表、**分波段功率比表**、图表索引、口径说明。

三条刻意的克制：

* **只用数据内证据**。不引入没有来源的绝对合格线，所以报告里不会出现
  "ACC ≥ 0.6 算合格"这类判决，只在数据内部比大小、看形态。
* **不重算指标**。只消费已经落盘的表，缺行缺列如实写出来，不猜也不补。
  ``scores.csv`` 是数值主来源；逐波数谱只存在于 ``summary.csv`` 的
  ``scope=wavenumber`` 档，所以那一节从 ``summary.csv`` 取（见 :func:`spectrum_bands`）。
* **比值要连着分母一起看**。分波段那节同时给「实况占比」，因为一个占比 0.01%
  的波段有着 5 倍的功率比，也不说明模式在那一档失真。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: 指标语义表 —— **唯一**放指标语义的地方（出图、报告、诊断都从这里取）。
#:
#: * ``label``     中文名，表格与图例用；
#: * ``ref_result``       参考线的位置（``None`` 表示"没有理想值"，不画线）；
#: * ``direction`` ``lower`` 越小越好 / ``higher`` 越大越好 / ``target`` 越接近
#:                 ``ref_result`` 越好（两侧都坏）/ ``None`` 非技巧分，不做方向判断。
METRIC_SEMANTICS: Dict[str, Dict[str, object]] = {
    "rmse": {"label": "RMSE", "ref_result": None, "direction": "lower"},
    "acc": {"label": "ACC（距平相关）", "ref_result": 1.0, "direction": "higher"},
    "activity_ratio": {
        "label": "活跃度比（预报/实况）",
        "ref_result": 1.0,
        "direction": "target",
    },
    "activity_bias": {
        "label": "活跃度偏差（预报−实况）",
        "ref_result": 0.0,
        "direction": "target",
    },
    "spectrum_power_ratio": {
        "label": "纬向谱功率比",
        "ref_result": 1.0,
        "direction": "target",
    },
    # 下面两个是活跃度的**原始量**（距平标准差），不是技巧分：没有理想值，
    # 也就不画参考线，只作为 activity_ratio 的分子/分母摆在表里。
    "activity_forecast": {
        "label": "预报活跃度（距平标准差）",
        "ref_result": None,
        "direction": None,
    },
    "activity_observation": {
        "label": "实况活跃度（距平标准差）",
        "ref_result": None,
        "direction": None,
    },
}

#: 逐时效折线图默认画这四个；``activity_bias`` 与 ``activity_ratio`` 信息重复，
#: 只进表格不进图。
SKILL_METRICS = ("rmse", "acc", "activity_ratio", "spectrum_power_ratio")

#: **配置里的原始指标名** → 它在长表里展开成的指标名。
#:
#: 长表一行一个**展开后**的指标：``activity`` 拆成 ratio / bias / forecast /
#: observation 四行，``zonal_spectrum`` 只留 ``spectrum_power_ratio``（``summary``
#: 里的总量字段进了评分表，波数明细进了 ``scores_detail.csv``）。而 ``manifest``
#: 的 ``metric_options`` 记的是**原始**指标名——两边不同名。
#:
#: 缺项检查要把它们对齐，否则 ``curves.get("zonal_spectrum")`` 永远是空，**每个谱
#: 变量都会被误报成"该算没算"**。这不是假想：``weather_field_scores_era5_fuxi``
#: 那份产物里 22 条缺项有 13 条是这么来的（7 个谱变量 + 6 个活跃度变量其实都算出来了）。
#: 没登记的名字按原样查（``rmse`` / ``acc`` 这类不展开的指标就该如此）。
METRIC_EXPANSIONS: Dict[str, tuple] = {
    "activity": (
        "activity_ratio",
        "activity_bias",
        "activity_forecast",
        "activity_observation",
    ),
    "zonal_spectrum": ("spectrum_power_ratio",),
}

_CN_NUMBERS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")

#: 赤道周长（km）。波数 k 的波长按 ``40075 / k`` 换算，与参考实现
#: ``mean_spectrum_<var>.csv`` 的 ``wavelength_km`` 列同口径。
EQUATOR_CIRCUMFERENCE_KM = 40075.0

#: 分波段功率比的波数分界（闭区间，含两端）。
#:
#: 为什么要分段：长表里的 ``spectrum_power_ratio`` 是**全波数求和后**再相除，
#: 一个 0.99 的总量完全可能同时是"大尺度偏弱、小尺度偏强"互相抵消的结果。
#: 分段之后才看得出失真是**尺度选择性**的。分界想改直接改这个元组。
SPECTRUM_BANDS = ((1, 5), (6, 20), (21, 60), (61, 180), (181, 720))


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _fmt(value: object, digits: int = 3) -> str:
    """定点格式；非有限值写成 “—”（与 TS 报告一致）。"""
    if value is None or not _finite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_auto(value: object) -> str:
    """按量级自动定小数位：RMSE 这类跨 4 个数量级的量用它更可读。"""
    if value is None or not _finite(value):
        return "—"
    number = float(value)
    magnitude = abs(number)
    if magnitude >= 1000:
        return f"{number:.0f}"
    if magnitude >= 100:
        return f"{number:.1f}"
    if magnitude >= 1:
        return f"{number:.3f}"
    return f"{number:.4f}"


def _pct(value: object) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value):.1%}"


def metric_label(metric: str) -> str:
    """指标中文名；未登记的指标原样返回（不编名字）。"""
    entry = METRIC_SEMANTICS.get(str(metric))
    return str(entry["label"]) if entry else str(metric)


def metric_ref(metric: str) -> Optional[float]:
    """参考线位置；没有理想值或指标未登记时返回 ``None``。"""
    entry = METRIC_SEMANTICS.get(str(metric))
    if not entry:
        return None
    ref = entry.get("ref_result")
    return None if ref is None else float(ref)  # type: ignore[arg-type]


def _metric_order(metrics: Sequence[str]) -> List[str]:
    """已登记指标按 ``METRIC_SEMANTICS`` 的顺序排，未登记的按字典序垫后。"""
    known = [name for name in METRIC_SEMANTICS if name in metrics]
    unknown = sorted(str(name) for name in metrics if name not in METRIC_SEMANTICS)
    return known + unknown


def _series_stats(values: Sequence[float], leads: Sequence[float], unit: str) -> Dict[str, object]:
    """一条 ``变量 × 时效`` 曲线的报告用统计量。

    ``mean`` 是**逐时效值的算术平均**（每个时效本身已经是跨起报均值），
    口径就是 ``summary.csv`` 里 ``scope=overall`` 那一档——参考实现的
    ``det_summary_overall.csv`` 用的也是它，两边可以直接对。
    """
    numbers = [float(value) for value in values]
    rises = sum(1 for left, right in zip(numbers, numbers[1:]) if right > left)
    falls = sum(1 for left, right in zip(numbers, numbers[1:]) if right < left)
    first, last = numbers[0], numbers[-1]
    return {
        "first": first,
        "mid": numbers[len(numbers) // 2],
        "last": last,
        "mean": float(np.mean(numbers)),
        "first_lead": float(leads[0]),
        "mid_lead": float(leads[len(leads) // 2]),
        "last_lead": float(leads[-1]),
        "n_leads": len(numbers),
        "n_rise": rises,
        "n_fall": falls,
        "ratio": last / first if first else float("nan"),
        "unit": unit,
    }


# --------------------------------------------------------------------------- #
# 汇总与诊断
# --------------------------------------------------------------------------- #
def summarize(
    scores: pd.DataFrame,
    requested: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, object]:
    """把连续场长表压成报告需要的统计量。纯计算，不做任何 I/O。

    Args:
        scores: ``scores.csv`` 那样的长表，至少要有
            ``variable / metric / lead_h / value / status`` 五列。
        requested: 可选，``{metric: [variable, ...]}`` —— 本次**要求**算的组合。
            只有给了它才能识别"该算但没出数"的缺口（例如源文件里根本没有
            某个要素，reader 会跳过并只留一行日志）。``None`` 时缺口为空。

    Returns:
        ``variables`` / ``leads`` / ``metrics`` / ``units``（metric→variable→单位）/
        ``pivots``（metric→变量×时效透视表）/ ``per_variable`` / ``missing`` /
        ``n_rows`` / ``n_success`` / ``n_failed`` / ``failed_statuses`` /
        ``unknown_metrics``。
    """
    data = scores.copy()
    data.columns = [str(column).strip() for column in data.columns]

    for column in ("lead_h", "value"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "lead_h" in data.columns:
        data = data.dropna(subset=["lead_h"])

    n_rows = int(len(data))
    if "status" in data.columns:
        statuses = data["status"].astype(str)
    else:
        statuses = pd.Series(["success"] * n_rows, index=data.index)
    failed_statuses = {
        str(name): int(count)
        for name, count in statuses[statuses != "success"].value_counts().items()
    }
    n_failed = int(sum(failed_statuses.values()))

    # 失败行落在哪儿，决定了它是"数据没覆盖到"还是"算错了"：集中在少数变量的
    # 尾段时效上，前者；散在所有变量所有时效上，后者。
    failed_groups: List[Dict[str, object]] = []
    failed_rows = data[statuses != "success"]
    if not failed_rows.empty and {"metric", "variable"}.issubset(failed_rows.columns):
        for (metric, variable), subset in failed_rows.groupby(
            ["metric", "variable"], observed=True
        ):
            leads = pd.to_numeric(subset["lead_h"], errors="coerce").dropna()
            failed_groups.append(
                {
                    "metric": str(metric),
                    "variable": str(variable),
                    "n": int(len(subset)),
                    "lead_min": float(leads.min()) if len(leads) else None,
                    "lead_max": float(leads.max()) if len(leads) else None,
                }
            )
        failed_groups.sort(key=lambda item: (-int(item["n"]), str(item["variable"])))

    usable = data[statuses == "success"]
    if "value" in usable.columns:
        usable = usable.dropna(subset=["value"])

    variables = sorted(str(name) for name in usable.get("variable", pd.Series(dtype=object)).unique())
    leads = sorted({float(value) for value in usable.get("lead_h", pd.Series(dtype=float))})
    units: Dict[str, Dict[str, str]] = {}
    pivots: Dict[str, pd.DataFrame] = {}

    if not usable.empty:
        group_columns = ["metric", "variable"]
        for (metric, variable), subset in usable.groupby(group_columns, observed=True):
            metric, variable = str(metric), str(variable)
            texts = (
                [str(unit) for unit in subset["unit"].dropna().unique() if str(unit)]
                if "unit" in subset.columns
                else []
            )
            units.setdefault(metric, {})[variable] = texts[0] if texts else ""
        for metric, subset in usable.groupby("metric", observed=True):
            pivots[str(metric)] = subset.pivot_table(
                index="variable", columns="lead_h", values="value", observed=True
            )

    # metric -> variable -> 曲线统计量
    curves: Dict[str, Dict[str, Dict[str, object]]] = {}
    for metric, pivot in pivots.items():
        curves[metric] = {}
        for variable, row in pivot.iterrows():
            series = row.dropna()
            if series.empty:
                continue
            series = series.sort_index()
            curves[metric][str(variable)] = _series_stats(
                values=list(series.values),
                leads=[float(lead) for lead in series.index],
                unit=units.get(metric, {}).get(str(variable), ""),
            )

    per_variable: List[Dict[str, object]] = []
    for variable in variables:
        entry: Dict[str, object] = {"variable": variable, "metrics": {}}
        for metric in _metric_order(list(curves)):
            stats = curves.get(metric, {}).get(variable)
            if stats:
                entry["metrics"][metric] = stats  # type: ignore[assignment]
        per_variable.append(entry)

    missing: List[Dict[str, str]] = []
    for metric, names in (requested or {}).items():
        metric = str(metric)
        # 配置用的是原始指标名，长表用的是展开后的名字，先把两边对齐（见 METRIC_EXPANSIONS）
        present: set = set()
        for expanded in METRIC_EXPANSIONS.get(metric, (metric,)):
            present |= set(curves.get(expanded, {}))
        for name in names or []:
            if str(name) not in present:
                missing.append({"metric": metric, "variable": str(name)})

    return {
        "variables": variables,
        "leads": leads,
        "metrics": _metric_order(list(curves)),
        "units": units,
        "pivots": pivots,
        "curves": curves,
        "per_variable": per_variable,
        "missing": missing,
        "unknown_metrics": [name for name in curves if name not in METRIC_SEMANTICS],
        "n_rows": n_rows,
        "n_success": int(len(usable)),
        "n_failed": n_failed,
        "failed_statuses": failed_statuses,
        "failed_by_group": failed_groups,
    }


def _series_of(summary: Dict[str, object], metric: str) -> List[tuple]:
    """``[(variable, 曲线统计量), ...]``，按变量名排序。"""
    curves = summary["curves"].get(metric, {})  # type: ignore[union-attr]
    return sorted(curves.items())


def spectrum_bands(spectra: Optional[pd.DataFrame]) -> List[Dict[str, object]]:
    """逐变量 × 逐波段的功率比。纯计算，不做 I/O。

    这是对长表里 ``spectrum_power_ratio`` 的**必要补充**：那一列是全波数求和后的
    比值，尺度之间会互相抵消；这里按 :data:`SPECTRUM_BANDS` 分段，能看出失真是
    尺度选择性的。

    Args:
        spectra: 逐波数长表（``variable / wavenumber / field / value``），来自
            ``summary.csv`` 的 ``scope=wavenumber`` 档。``None`` / 空表返回 ``[]``。

    Returns:
        ``[{variable, band, wavelength_min_km, wavelength_max_km, forecast,
        observation, ratio, observation_share, total_ratio}, ...]``，
        按变量名、波数升序。``observation_share`` 是该波段实况功率占实况总功率的
        比例——比值再离谱，占比很小也说明不了什么。
    """
    if spectra is None or spectra.empty:
        return []
    frame = spectra.copy()
    for column in ("wavenumber", "value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["field"] = frame["field"].astype(str)
    frame["variable"] = frame["variable"].astype(str)
    frame = frame.dropna(subset=["wavenumber", "value"])

    rows: List[Dict[str, object]] = []
    for variable, subset in frame.groupby("variable", observed=True):
        wide = subset.pivot_table(
            index="wavenumber", columns="field", values="value", aggfunc="mean"
        ).sort_index()
        if "power_forecast" not in wide.columns or "power_observation" not in wide.columns:
            continue  # 只有一条谱线，比不出比值；不编一个数出来
        positive = wide[wide.index > 0]  # k=0 的功率恒为 0，也不属于任何波段
        total_forecast = float(positive["power_forecast"].sum())
        total_observation = float(positive["power_observation"].sum())
        for k_min, k_max in SPECTRUM_BANDS:
            band = positive[(positive.index >= k_min) & (positive.index <= k_max)]
            if band.empty:
                continue
            forecast = float(band["power_forecast"].sum())
            observation = float(band["power_observation"].sum())
            rows.append(
                {
                    "variable": str(variable),
                    "band": f"k={k_min}–{k_max}",
                    "wavelength_min_km": EQUATOR_CIRCUMFERENCE_KM / k_max,
                    "wavelength_max_km": EQUATOR_CIRCUMFERENCE_KM / k_min,
                    "forecast": forecast,
                    "observation": observation,
                    "ratio": forecast / observation if observation else float("nan"),
                    "observation_share": (
                        observation / total_observation if total_observation else float("nan")
                    ),
                    "total_ratio": (
                        total_forecast / total_observation if total_observation else float("nan")
                    ),
                }
            )
    return rows


def diagnose(
    summary: Dict[str, object], bands: Optional[Sequence[Dict[str, object]]] = None
) -> List[str]:
    """规则化诊断：每条结论都带具体数值，只用数据内部的证据。

    Args:
        bands: :func:`spectrum_bands` 的结果；给了才会出「分波段」那条结论。
    """
    findings: List[str] = []
    variables = summary["variables"]
    leads = summary["leads"]
    metrics = summary["metrics"]
    n_rows = summary["n_rows"]

    if not variables or not leads:
        return ["长表里没有任何可用的「变量 × 时效」数值，无法诊断。"]

    span = (
        f"时效 {leads[0]:g}–{leads[-1]:g}h 共 {len(leads)} 个"
        if len(leads) > 1
        else f"只有 1 个时效（{leads[0]:g}h）"
    )
    findings.append(
        f"结果共 {n_rows} 行（`status=success` {summary['n_success']} 行），"
        f"覆盖 {len(variables)} 个变量、{len(metrics)} 个指标，{span}。"
        f"变量与指标**不是满交叉**：每个变量只算被逐变量路由到的指标。"
    )

    if summary["n_failed"]:
        detail = "、".join(
            f"{name}={count}" for name, count in summary["failed_statuses"].items()
        )
        text = (
            f"有 {summary['n_failed']} 行不是 `success`（{detail}），"
            f"这些行的 `value` 不能直接当结论用，处置方式见 `status` 列。"
        )
        groups = summary.get("failed_by_group") or []
        if groups:
            shown = "；".join(
                f"{item['metric']}·{item['variable']} {item['n']} 行"
                + (
                    f"（时效 {item['lead_min']:g}–{item['lead_max']:g}h）"
                    if item["lead_min"] is not None
                    else ""
                )
                for item in groups[:5]
            )
            if len(groups) > 5:
                shown += f"；另有 {len(groups) - 5} 组"
            if len(groups) == 1:
                text += (
                    f"它们的分布是：{shown}——其它「指标 × 变量」组合一行不缺。"
                    f"缺口集中在单个变量上，是那一份观测没覆盖到那几段，不是算错了。"
                )
            else:
                text += (
                    f"它们的分布是：{shown}。缺口集中在少数变量上多半是观测没覆盖到；"
                    f"散在所有变量所有时效上才该回头查流程。"
                )
        findings.append(text)

    missing = summary["missing"]
    if missing:
        by_variable: Dict[str, List[str]] = {}
        for item in missing:
            by_variable.setdefault(item["variable"], []).append(item["metric"])
        names = "；".join(
            f"{variable}（缺 {'、'.join(sorted(items))}）"
            for variable, items in sorted(by_variable.items())
        )
        findings.append(
            f"**该算但没出数**：{len(missing)} 个「变量 × 指标」组合在结果里没有任何行 —— "
            f"{names}。最常见的成因是数据源里没有该要素（reader 会跳过并记日志），"
            f"具体原因看评测日志，本报告不从结果反推。"
        )

    unknown = summary["unknown_metrics"]
    if unknown:
        findings.append(
            f"结果里出现了语义表未登记的指标 {'、'.join(unknown)}，"
            f"报告只按数值原样列出，不解读含义。"
        )

    rmse_items = _series_of(summary, "rmse")
    if rmse_items:
        ranked = sorted(rmse_items, key=lambda item: item[1]["ratio"], reverse=True)
        fastest, slowest = ranked[0], ranked[-1]
        findings.append(
            f"RMSE 增长最快的是 **{fastest[0]}**"
            f"（{_fmt_auto(fastest[1]['first'])} → {_fmt_auto(fastest[1]['last'])}"
            f" {fastest[1]['unit']}，{fastest[1]['first_lead']:g}h → {fastest[1]['last_lead']:g}h"
            f" 涨到 {_fmt(fastest[1]['ratio'], 2)} 倍）；"
            f"最慢的是 {slowest[0]}（{_fmt(slowest[1]['ratio'], 2)} 倍）。"
        )
        wandering = [name for name, stats in rmse_items if stats["n_rise"] != stats["n_leads"] - 1]
        if wandering:
            findings.append(
                f"RMSE 并非逐时效单调上升的有 {len(wandering)} 个："
                f"{'、'.join(wandering)}（其余 {len(rmse_items) - len(wandering)} 个单调上升）。"
                f"误差随时效累积本该单调，拐点值得回看原始场。"
            )
        else:
            findings.append(
                f"全部 {len(rmse_items)} 个变量的 RMSE 都逐时效单调上升，"
                f"符合误差随时效累积的预期。"
            )

    acc_items = _series_of(summary, "acc")
    if acc_items:
        for name, stats in acc_items:
            drop = stats["first"] - stats["last"]
            relative = drop / stats["first"] if stats["first"] else float("nan")
            shape = (
                "逐时效单调下降"
                if stats["n_fall"] == stats["n_leads"] - 1
                else f"非单调下降（{stats['n_leads'] - 1} 个间隔里 {stats['n_fall']} 次下降）"
            )
            findings.append(
                f"ACC（{name}）{stats['first_lead']:g}h 的 {_fmt(stats['first'])}"
                f" → {stats['last_lead']:g}h 的 {_fmt(stats['last'])}"
                f"（绝对 −{_fmt(drop)}，相对 −{_pct(relative)}），全时效均值 "
                f"{_fmt(stats['mean'])}，{shape}。"
                f"距平相关没有普适合格线，这里只报形态与幅度。"
            )

    for metric in ("activity_ratio", "spectrum_power_ratio"):
        items = _series_of(summary, metric)
        if not items:
            continue
        ranked = sorted(items, key=lambda item: abs(item[1]["last"] - 1.0), reverse=True)
        worst_name, worst = ranked[0]
        low = [name for name, stats in items if stats["last"] < 1]
        direction = "偏平滑（能量不足）" if worst["last"] < 1 else "偏噪（能量过量）"
        head = (
            f"**{metric_label(metric)}**：偏离 1 最多的是 **{worst_name}** "
            f"{_fmt(worst['last'])}（{direction}，末时效 {worst['last_lead']:g}h），"
            f"{len(items)} 个变量里 {len(low)} 个 <1、{len(items) - len(low)} 个 >1。"
        )
        if len(ranked) > 1:
            head += "偏离最大的三个：" + "、".join(
                f"{name} {_fmt(stats['last'])}" for name, stats in ranked[:3]
            ) + "。"
        if metric == "spectrum_power_ratio":
            # 指向哪儿得看那一节在不在：说了"看下一节的表"而下一节根本没出现，
            # 比不提还糟。
            head += "这个比值是**全波数求和后**再相除，总量正常也可能掩盖分布失真——"
            if bands:
                head += "是哪个尺度出的问题，看下一节的「分波段功率比」表。"
            else:
                head += (
                    "本报告没拿到 `summary.csv`，出不了分波段表；"
                    "要看是哪个波段的问题，用 `--spectrum-variable` / `--spectrum-lead` "
                    "读 `scores_detail.csv` 出谱曲线。"
                )
        findings.append(head)

    if bands:
        # 排序用**绝对功率差**而不是 |比值−1|：一个只占总能量 0.001% 的波段可以把
        # 比值拉到 3 倍，却对总能量毫无影响，按比值排会把它顶到头条上——报告自己
        # 的注解又说"占比小的波段说明不了什么"，自相矛盾。
        ranked = sorted(
            bands, key=lambda item: abs(float(item["forecast"]) - float(item["observation"])),
            reverse=True,
        )
        worst = ranked[0]
        low = sum(1 for item in bands if float(item["ratio"]) < 1.0)
        text = (
            f"**分波段**看（{len({str(item['variable']) for item in bands})} 个变量 × "
            f"{len(SPECTRUM_BANDS)} 个波段共 {len(bands)} 段）：对总能量偏差贡献最大的是 "
            f"**{worst['variable']} 的 {worst['band']}**"
            f"（波长约 {_fmt(worst['wavelength_min_km'], 0)}–{_fmt(worst['wavelength_max_km'], 0)} km，"
            f"占实况总功率 {_pct(worst['observation_share'])}）："
            f"预报/实况 {_fmt(worst['ratio'], 3)}，比同变量**全波数总比** "
            f"{_fmt(worst['total_ratio'], 3)} 偏得更远。{len(bands)} 段里 {low} 段 <1、"
            f"{len(bands) - low} 段 >1。总比接近 1 不代表每个尺度都对——"
            f"方向相反的波段会互相抵消。"
        )
        # 再点名一个"比值最离谱、但占比不至于可忽略"的，尺度选择性通常在这里露头
        material = [item for item in bands if float(item["observation_share"]) >= 0.01]
        if material:
            sharp = max(material, key=lambda item: abs(float(item["ratio"]) - 1.0))
            if sharp is not worst:
                text += (
                    f"占比 ≥1% 的波段里，偏得最狠的是 {sharp['variable']} 的 "
                    f"{sharp['band']}（{_fmt(sharp['ratio'], 3)}，占 "
                    f"{_pct(sharp['observation_share'])}）。"
                )
        findings.append(text)

    # 逐指标的时效覆盖是否齐整
    for metric in metrics:
        curves = summary["curves"].get(metric, {})  # type: ignore[union-attr]
        if not curves:
            continue
        widest = max(stats["n_leads"] for stats in curves.values())
        short = {
            name: stats["n_leads"]
            for name, stats in curves.items()
            if stats["n_leads"] != widest
        }
        if short:
            detail = "、".join(f"{name} 只有 {count} 个" for name, count in sorted(short.items()))
            findings.append(
                f"`{metric}` 的时效覆盖不齐（最全 {widest} 个）：{detail}；"
                f"缺的时效在长表里没有任何行，不是 `value` 为空。"
            )

    return findings


# --------------------------------------------------------------------------- #
# manifest → 报告用信息
# --------------------------------------------------------------------------- #
def manifest_info_from(manifest: Optional[Dict[str, object]]) -> Dict[str, object]:
    """把 ``manifest.json`` 的原始 dict 翻成报告要用的字段。

    只搬运，不加工：拿不到的一律留空，由渲染层如实写"未提供"。
    """
    if not manifest:
        return {}
    resolved = manifest.get("resolved_config") or {}
    forecast = resolved.get("forecast") or {} if isinstance(resolved, dict) else {}
    observation = resolved.get("observation") or {} if isinstance(resolved, dict) else {}
    reference = resolved.get("reference") or {} if isinstance(resolved, dict) else {}
    return {
        "run_id": manifest.get("run_id"),
        "pipeline": resolved.get("pipeline") if isinstance(resolved, dict) else None,
        "protocol_id": manifest.get("protocol_id"),
        "description": manifest.get("description"),
        "model_id": manifest.get("forecast_source") or manifest.get("model_id"),
        "dataset_id": manifest.get("observation_source") or manifest.get("dataset_id"),
        "forecast_reader": forecast.get("reader") if isinstance(forecast, dict) else None,
        "observation_reader": observation.get("reader") if isinstance(observation, dict) else None,
        "reference_reader": reference.get("reader") if isinstance(reference, dict) else None,
        "start_date": resolved.get("start_date") if isinstance(resolved, dict) else None,
        "end_date": resolved.get("end_date") if isinstance(resolved, dict) else None,
        "created_at": manifest.get("created_at"),
        "n_results": manifest.get("n_results"),
        "scores_rows": manifest.get("scores_rows"),
        "details_rows": manifest.get("details_rows"),
        "statuses": manifest.get("statuses"),
        "metric_options": resolved.get("metric_options") if isinstance(resolved, dict) else None,
        "forecast_source_files": list(manifest.get("forecast_input_files") or []),
        "observation_source_files": list(manifest.get("observation_input_files") or []),
    }


def requested_from(info: Dict[str, object]) -> Dict[str, List[str]]:
    """从 ``metric_options`` 取"每个指标要求算哪些变量"。"""
    options = info.get("metric_options") or {}
    if not isinstance(options, dict):
        return {}
    requested: Dict[str, List[str]] = {}
    for metric, params in options.items():
        if isinstance(params, dict) and params.get("variables"):
            requested[str(metric)] = [str(name) for name in params["variables"]]
    return requested


def _source_of(sources: Optional[Dict[str, str]], name: str) -> str:
    """来源标注：给了路径就写路径，否则如实写"未提供"。"""
    if sources and sources.get(name):
        return f"`{sources[name]}`"
    return "未提供（调用方应在 sources 里给出文件路径）"


def _file_summary(paths: Sequence[str], limit: int = 1) -> str:
    """文件清单只写个数与首尾，避免把 60 条绝对路径全抄进报告。"""
    if not paths:
        return "未提供"
    if len(paths) <= 2 * limit:
        return "、".join(f"`{path}`" for path in paths)
    head = "、".join(f"`{path}`" for path in paths[:limit])
    tail = "、".join(f"`{path}`" for path in paths[-limit:])
    return f"{len(paths)} 个文件，{head} … {tail}"


# --------------------------------------------------------------------------- #
# 表格
# --------------------------------------------------------------------------- #
def _overview_table(summary: Dict[str, object], columns: Sequence[str]) -> str:
    header = ["变量"]
    for metric in columns:
        header += [f"{metric_label(metric)} 首", f"{metric_label(metric)} 末"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|" + "---|" * (2 * len(columns)),
    ]
    for entry in summary["per_variable"]:
        cells = [str(entry["variable"])]
        for metric in columns:
            stats = entry["metrics"].get(metric)  # type: ignore[union-attr]
            if stats:
                cells += [_fmt_auto(stats["first"]), _fmt_auto(stats["last"])]
            else:
                cells += ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rmse_table(summary: Dict[str, object]) -> str:
    lines = [
        "| 变量 | 单位 | 首时效 | 中位时效 | 末时效 | 全时效均值 | 末/首 | 单调上升 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, stats in _series_of(summary, "rmse"):
        monotone = "是" if stats["n_rise"] == stats["n_leads"] - 1 else "否"
        lines.append(
            f"| {name} | {stats['unit'] or '—'} "
            f"| {_fmt_auto(stats['first'])} | {_fmt_auto(stats['mid'])} "
            f"| {_fmt_auto(stats['last'])} | {_fmt_auto(stats['mean'])} "
            f"| {_fmt(stats['ratio'], 2)} | {monotone} |"
        )
    return "\n".join(lines)


def _band_table(bands: Sequence[Dict[str, object]]) -> str:
    lines = [
        "| 变量 | 波段 | 波长 (km) | 预报功率 | 实况功率 | 预报/实况 | 实况占比 | 全波数总比 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in bands:
        lines.append(
            f"| {item['variable']} | {item['band']} "
            f"| {_fmt(item['wavelength_min_km'], 0)}–{_fmt(item['wavelength_max_km'], 0)} "
            f"| {_fmt_auto(item['forecast'])} | {_fmt_auto(item['observation'])} "
            f"| {_fmt(item['ratio'], 3)} | {_pct(item['observation_share'])} "
            f"| {_fmt(item['total_ratio'], 3)} |"
        )
    return "\n".join(lines)


def _ratio_table(summary: Dict[str, object]) -> str:
    lines = [
        "| 变量 | 活跃度比（预报/实况） | 活跃度偏差（预报−实况） "
        "| 预报活跃度 | 实况活跃度 | 谱功率比 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in summary["per_variable"]:
        stats = entry["metrics"]  # type: ignore[union-attr]
        if not any(
            key in stats
            for key in ("activity_ratio", "activity_bias", "spectrum_power_ratio")
        ):
            continue
        cells = []
        for key in (
            "activity_ratio",
            "activity_bias",
            "activity_forecast",
            "activity_observation",
            "spectrum_power_ratio",
        ):
            item = stats.get(key)
            cells.append(_fmt(item["last"]) if item else "—")
        lines.append(f"| {entry['variable']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _artifacts_section(artifacts: Optional[Dict[str, Path]]) -> List[str]:
    lines: List[str] = []
    if not artifacts:
        return ["本次未出图（调用方没有传 ``artifacts``）。"]
    for name, path in artifacts.items():
        path = Path(path)
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg"):
            lines.append(f"![{name}]({path.name})")
            lines.append("")
    return lines or ["本次没有图片产物。"]


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def build_field_report(
    scores: pd.DataFrame,
    output_path: Path,
    *,
    model_name: str = "model",
    requested: Optional[Dict[str, Sequence[str]]] = None,
    manifest_info: Optional[Dict[str, object]] = None,
    sources: Optional[Dict[str, str]] = None,
    change_description: Optional[str] = None,
    artifacts: Optional[Dict[str, Path]] = None,
    spectrum_note: Optional[str] = None,
    spectra: Optional[pd.DataFrame] = None,
) -> Path:
    """写一份连续场评测的 Markdown 报告。

    Args:
        scores: 长表（``scores.csv``）。
        output_path: 报告落盘路径。
        requested: ``{metric: [variable, ...]}``，用于识别"该算但没出数"；
            通常由 :func:`requested_from` 从 manifest 得到。
        manifest_info: :func:`manifest_info_from` 的结果；``None`` 时第一节
            只写能从头几个参数拿到的内容。
        artifacts: ``{名称: 路径}``，图片会被写成 Markdown 图片。
        spectrum_note: 谱曲线的说明（按需出图时由调用方给出）。
        spectra: 逐波数长表（``summary.csv`` 的 ``scope=wavenumber`` 档）；
            给了才会出分波段功率比表。谱曲线在 ``scores.csv`` 里根本没有，
            所以这节只能从 ``summary.csv`` 来。
    """
    output_path = Path(output_path)
    info = dict(manifest_info or {})
    summary = summarize(scores, requested=requested)
    bands = spectrum_bands(spectra)
    findings = diagnose(summary, bands=bands)
    variables = summary["variables"]
    leads = summary["leads"]
    metrics = summary["metrics"]

    # ---- 一、评测对象与数据来源 -------------------------------------------- #
    head: List[str] = []
    run_id = info.get("run_id") or "未提供"
    head.append(f"- **run_id**：`{run_id}`（来源：{_source_of(sources, 'scores')}）")
    head.append(
        f"- **评测对象**：{model_name}"
        f"（预报源 `{info.get('forecast_reader') or info.get('model_id') or '未提供'}`）"
        f"；**实况**：`{info.get('observation_reader') or info.get('dataset_id') or '未提供'}`"
        f"；**参考**：`{info.get('reference_reader') or '无'}`"
    )
    pipeline = info.get("pipeline")
    head.append(
        f"- **流程模板**：`{pipeline or '未提供'}`"
        f"（协议 `{info.get('protocol_id') or '未提供'}`）"
        + (f" —— {info['description']}" if info.get("description") else "")
    )
    if info.get("start_date") or info.get("end_date"):
        head.append(
            f"- **评测时段**：`{info.get('start_date') or '?'}` – `{info.get('end_date') or '?'}`"
            f"（配置口径的起止日期，不代表实际配对上的样本数）"
        )
    if leads:
        head.append(
            f"- **实际结果**：{len(variables)} 个变量 × {len(metrics)} 个指标"
            f"（{'、'.join(metrics)}），时效 {leads[0]:g}–{leads[-1]:g}h 共 {len(leads)} 个，"
            f"长表 {summary['n_rows']} 行"
        )
    head.append(f"- **预报来源文件**：{_file_summary(info.get('forecast_source_files') or [])}")
    head.append(f"- **实况来源文件**：{_file_summary(info.get('observation_source_files') or [])}")
    if info.get("created_at"):
        head.append(f"- **评测生成时间**：`{info['created_at']}`")
    if change_description:
        head.append(f"- **本次改动**：{change_description}")
    head.append(f"- **报告生成时间**：{datetime.now():%Y-%m-%d %H:%M}")
    note = (
        "> 本节内容全部取自产物目录的 `manifest.json`，路径是**评测机上的绝对路径**，"
        "本机通常不存在，只作溯源用。"
    )

    # ---- 二、执行摘要 ------------------------------------------------------ #
    digest = [f"{index}. {text}" for index, text in enumerate(findings, start=1)]

    # ---- 三、逐变量总览 ---------------------------------------------------- #
    overview_columns = [
        name for name in SKILL_METRICS if summary["curves"].get(name)
    ]
    overview = [
        _overview_table(summary, overview_columns),
        "",
        "> 「首 / 末」是该变量该指标**自身**的最小 / 最大时效，各指标不必相同；"
        "`—` 表示该变量没有这个指标（逐变量路由的结果，不一定是缺数据）。"
        "不同变量的单位不同，跨行比大小之前先看单位列。",
    ]

    # ---- 四、RMSE 随时效 --------------------------------------------------- #
    mean_note = (
        "> 「全时效均值」是各时效值的算术平均（每个时效本身已是跨起报均值），"
        "口径同 `summary.csv` 的 `scope=overall`，也对得上参考实现的 "
        "`det_summary_overall.csv`。"
    )
    rmse = [
        _rmse_table(summary),
        "",
        mean_note,
        "> 「末/首」是末时效与首时效的倍数，单位无关，可跨变量比较；"
        "「单调上升」要求相邻时效两两递增。",
    ]

    # ---- 五、活跃度与纬向谱 ------------------------------------------------ #
    activity: List[str] = [_ratio_table(summary), ""]
    activity.append(
        "> 活跃度 = 距平（相对气候态）的标准差。活跃度比 = 预报/实况，"
        "**>1 偏噪、<1 偏平滑**；偏差 = 预报−实况，与比同号但量纲随变量走。"
        "谱功率比是**全波数求和后**的比值，总量对了也可能分布失真。"
    )
    if bands:
        activity += [
            "",
            # 数的是**真出了数的**波段，不是 SPECTRUM_BANDS 的长度：谱线短的时候
            # 高波数那几段是空的（比如 max_wavenumber 只算到 30），标题写 5 个波段
            # 而表里只有 3 行，就成了报告自己打自己的脸。
            f"### 分波段功率比（{len({str(item['band']) for item in bands})} 个波段）",
            "",
            _band_table(bands),
            "",
            "> 波长按赤道周长 40075 km 换算（k=1 → 40075 km），与 `mean_spectrum_<var>.csv` "
            "的 `wavelength_km` 同口径；波段分界见 `field_report.SPECTRUM_BANDS`。"
            "**「实况占比」小的波段，比值再离谱也说明不了什么**——先看它占多少能量。"
            "「全波数总比」一行内是常数，重复写出来是为了和分波段比值并排比；"
            "注意它和上面那张表的「谱功率比」列**不是同一口径**：上面那列逐时效，"
            "末时效会明显小于这里——这里是跨起报、跨时效平均后的全天候值。",
        ]
    if spectrum_note:
        activity += ["", spectrum_note]

    # ---- 六、图表 ---------------------------------------------------------- #
    figures = _artifacts_section(artifacts)

    # ---- 七、口径与注意事项 ------------------------------------------------ #
    tail = [
        "- **单位随数据源走**：`*_phys` 布局下 z500 的 RMSE 单位是 `m^2/s^2`（**不除 g**），"
        "q 是 `g/kg`，其余是无量纲的 `1`。所以 RMSE 图按单位分面；"
        "不同面之间、以及 `m^2/s^2` 与 `m` 的布局之间都不可直接比大小。",
        "- **`n_valid` 的含义不统一**：`rmse`/`acc`/`activity_*` 行是格点数"
        "（全球 0.25° = 1038240），而 `spectrum_power_ratio` 行是**参与平均的二维场个数**"
        "（单起报单时效就是 1），它不是样本量，不能拿来判断结果可不可信。",
        "- **`value` 为空不等于 0**：为空说明这一个组合没算出来，看同行 `status`；"
        "本报告只对 `status=success` 的行做统计。",
        "- **`level` / `region` / `threshold` / `window_h` / `weights_id` 恒为空**，"
        "这是 `grid_valid_time` 协议的设计（默认口径只有 `variable` / `sample_unit` / `unit`），"
        "不是缺数据。",
        "- **首/末时效不是稳定的统计量**：起报数与样本区间决定了这条曲线的形态，"
        "本报告里「涨到几倍」这类说法只在当前这套输入下成立，"
        "跨批次比较要用同样的起报集合与时段。",
        "- **逐波数谱曲线不在 `scores.csv` 里**，在 `summary.csv` 的 `scope=wavenumber` 档"
        "（跨起报、跨时效平均后的逐波数功率）。报告的分波段表与谱面板都从这里取；"
        "`summary.csv` 不存在时这两块直接不出现，不从别处凑。要**指定时效**的谱，"
        "得读 `diagnostics/scores_detail.csv` 的 `group=k=<波数>` 行"
        "（单变量单时效就有 721 个波数，本仓库那份 492MB），用 "
        "`--spectrum-variable` / `--spectrum-lead` 按需出图。",
        "- 本报告**不重算任何指标**，只消费已落盘的表：数值主来源是 `scores.csv`，"
        "逐波数谱那两块来自 `summary.csv`。缺表缺行如实写出，不猜也不补。",
    ]

    sections: List[tuple] = [
        ("评测对象与数据来源", [*head, "", note]),
        ("执行摘要", digest),
        ("逐变量总览", overview),
        ("RMSE 随时效", rmse),
        ("活跃度与纬向谱", activity),
    ]
    if artifacts:
        sections.append(("图表", figures))
    sections.append(("口径与注意事项", tail))

    lines: List[str] = [f"# {model_name} 连续场检验报告", ""]
    for index, (title, body) in enumerate(sections):
        marker = _CN_NUMBERS[index] if index < len(_CN_NUMBERS) else str(index + 1)
        lines.append(f"## {marker}、{title}")
        lines.append("")
        lines.extend(body)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="把连续场评分长表画成图并写成 Markdown 报告",
        epilog="示例: python -m xmetai_evaluation.visualization.field_report "
        "--scores scores.csv --out reports/field_fuxi --model FuXi",
    )
    parser.add_argument("--scores", type=Path, required=True, help="长表 scores.csv")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    parser.add_argument("--model", default=None, help="模型名（默认取 model_id 列）")
    parser.add_argument("--manifest", type=Path, default=None, help="可选：manifest.json")
    parser.add_argument("--change", default=None, help="可选：本次改动说明")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    from visualization.field_plots import FieldScorePlotter

    scores = pd.read_csv(args.scores)
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    )
    info = manifest_info_from(manifest)
    model = args.model or str(info.get("model_id") or args.scores.stem)

    # summary.csv 是 scores.csv 的同级产物（field_summary writer 落的），有就用它出
    # 纬向谱面板与分波段功率比；没有就照常出图，不额外要一个开关。
    summary_path = args.scores.with_name("summary.csv")
    plotter = FieldScorePlotter()
    artifacts = plotter.create_report(
        scores,
        output_dir=args.out,
        summary_path=summary_path if summary_path.is_file() else None,
        model_name=model,
        manifest_info=info,
        sources={"scores": str(args.scores)},
        change_description=args.change,
    )
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
