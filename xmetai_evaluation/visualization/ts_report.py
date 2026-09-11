#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS 结果表 → Markdown 报告。

一句话用法::

    python -m xmetai_evaluation.visualization.ts_report --csv ts.csv --out reports/fgvp --model FGVP

报告按**论文实验章节**的写法组织：每一节都是「引导句 → 图表 → 定量分析」。
图就地带着图注出现（`图 N：…` / `表 N：…`，按出现顺序编号），图下必有图注、
表上必有表题，每节末尾有一段**小结**——小结里的每个论断都带具体数值，由
``analysis_*`` 系列函数从已经算好的统计量里推出来，不是套话。

章节：评测对象与对比对象 → 结论摘要 → 逐等级表现 → 随时效变化 → 误差形态与偏差结构 →
跨时效分布（箱线图）→（可选）与对比模型的比较 → 改进建议 → 口径与注意事项。

``analysis_*`` 函数只吃 ``summarize()`` / ``box_stats()`` 已经算好的数字，
不重新读表、不重新实现指标——口径只有一处，表、图、文字不可能各说各话。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: 第五节（跨时效分布）里逐字固定的三行。``assets/templates/ts.md`` 的骨架要抄同一份字节，
#: 两边差一个全角/半角字符测试就会红——所以只在**这里**定义一次。
_BOX_HEADER = "| 模型 | 降水等级 | 中位数 | IQR | Q1–Q3 | 须线范围 | 离群时效 | 时效数 |"
_BOX_SEP = "|---|---|---|---|---|---|---|---|"
_BOX_NOTE = (
    "> 每个箱体的样本是**同模型、同量级在全部时效上的 TS 取值**"
    "（无有效 TS 的时效不计入，样本数见「时效数」列）：箱体为四分位距（IQR），中线为中位数，"
    "须线延伸到 1.5×IQR 内最远的点，空心圆为离群时效。**TS 越高越优**，"
    "所以箱体**越高**表示该量级整体技巧越好、**越窄**表示跨时效越稳定；"
    "这里的统计量与第二节的「全时效平均」不是同一个量（等权平均 vs 中位数/IQR），不可互相替代。"
)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


class _Numbering:
    """图和表**按在报告里出现的顺序**编号，图注/表题由它生成。

    编号跟着出现顺序走而不是写死，是因为图会缺：某份结果表没有 POD/FAR 就没有
    性能图，单模型分支没有 multi_model_TS_delta。缺图时**连图注一起不出现**，
    序号仍然连续（图 1、图 2…），不会出现「图 3 不见了」这种断号。
    """

    def __init__(self) -> None:
        self._figures: Dict[str, int] = {}
        self._tables = 0

    def figure(self, name: str) -> int:
        number = len(self._figures) + 1
        self._figures[name] = number
        return number

    def table(self) -> int:
        self._tables += 1
        return self._tables


def _figure_block(
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    name: str,
    caption: str,
) -> List[str]:
    """图片 + 图注。产物里没有这张图（或不是图片）时返回空列表，编号也不占位。"""
    path = (artifacts or {}).get(name)
    if path is None or Path(path).suffix.lower() not in (".png", ".jpg", ".svg"):
        return []
    number = numbering.figure(name)
    return [f"![{name}]({Path(path).name})", "", f"图 {number}：{caption}", ""]


def _table_caption(numbering: _Numbering, text: str) -> List[str]:
    """表题**在表上方**（图注在下方）——这是中文论文的排版惯例。"""
    return [f"表 {numbering.table()}：{text}", ""]


def _grade_sort_key(grade: str, threshold: Optional[float]) -> float:
    if threshold is not None and np.isfinite(threshold):
        return float(threshold)
    text = str(grade).replace("≥", "").strip()
    try:
        return float(text)
    except ValueError:
        return 1e9


def summarize(data: pd.DataFrame) -> Dict[str, object]:
    """把 TS 表压成报告需要的统计量。"""
    grades: List[str] = []
    per_grade: List[Dict[str, object]] = []
    for grade, subset in data.groupby("grade", observed=True):
        thresholds = subset["threshold_mm"].dropna() if "threshold_mm" in subset else pd.Series(dtype=float)
        threshold = float(thresholds.iloc[0]) if len(thresholds) else None
        totals = {
            key: float(subset[key].sum()) if key in subset.columns else float("nan")
            for key in ("hits", "misses", "false_alarms", "n_pairs")
        }
        hits = totals["hits"]
        misses = totals["misses"]
        false_alarms = totals["false_alarms"]

        def _ratio(numerator: float, denominator: float) -> float:
            if not np.isfinite(denominator) or denominator <= 0:
                return float("nan")
            return numerator / denominator

        entry = {
            "grade": str(grade),
            "threshold": threshold,
            # 主口径：先合计列联表再算比率；计数与之一致（都是合计）
            "TS_sum": _ratio(hits, hits + misses + false_alarms),
            "POD_sum": _ratio(hits, hits + misses),
            "FAR_sum": _ratio(false_alarms, hits + false_alarms),
            "漏报率_sum": _ratio(misses, hits + misses),
            "BIAS_sum": _ratio(hits + false_alarms, hits + misses),
            # 参考口径：各时效先算比率再等权平均
            "TS_mean": float(subset["TS"].mean()),
            "TS_last": float(subset.sort_values("lead_h")["TS"].iloc[-1]),
            "hits": int(hits) if np.isfinite(hits) else 0,
            "misses": int(misses) if np.isfinite(misses) else 0,
            "false_alarms": int(false_alarms) if np.isfinite(false_alarms) else 0,
            "n_pairs": int(totals["n_pairs"]) if np.isfinite(totals["n_pairs"]) else 0,
            "n_leads": int(subset["lead_h"].nunique()),
            "events": int(hits + misses) if np.isfinite(hits) else 0,
        }
        entry["TS"] = entry["TS_sum"]
        entry["POD"] = entry["POD_sum"]
        entry["FAR"] = entry["FAR_sum"]
        entry["漏报率"] = entry["漏报率_sum"]
        entry["BIAS"] = entry["BIAS_sum"]
        grades.append(str(grade))
        per_grade.append(entry)
    per_grade.sort(key=lambda item: _grade_sort_key(item["grade"], item["threshold"]))

    leads = sorted(float(value) for value in pd.unique(data["lead_h"]))
    first, last = leads[0], leads[-1]
    overall = float(np.nanmean([entry["TS_sum"] for entry in per_grade]))
    overall_mean = float(np.nanmean([entry["TS_mean"] for entry in per_grade]))
    weakest = per_grade[0] if per_grade else None
    strongest = per_grade[-1] if per_grade else None

    decay = None
    if weakest and strongest:
        first_ts = float(data[data["lead_h"] == first]["TS"].mean())
        last_ts = float(data[data["lead_h"] == last]["TS"].mean())
        decay = {
            "first_lead": first,
            "last_lead": last,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "drop": first_ts - last_ts,
            "relative": (first_ts - last_ts) / first_ts if first_ts else np.nan,
            "per_day": (first_ts - last_ts) / max((last - first) / 24.0, 1e-9),
        }

    pivot = data.pivot_table(index="lead_h", columns="grade", values="TS", observed=True)
    return {
        "grades": per_grade,
        "leads": leads,
        "overall_ts": overall,
        "overall_ts_mean": overall_mean,
        "decay": decay,
        "pivot": pivot,
        "n_rows": int(len(data)),
    }


def diagnose(summary: Dict[str, object]) -> List[str]:
    """规则化诊断：每条结论都带具体数值。"""
    grades = summary["grades"]
    findings: List[str] = []
    if not grades:
        return ["数据里没有可用的降水等级"]

    weak = grades[0]
    strong = grades[-1]
    overall = summary["overall_ts"]
    level = "较好" if overall >= 0.3 else ("中等" if overall >= 0.2 else "偏弱")
    findings.append(
        f"整体技巧 **{level}**：各等级全时效平均 TS = {_fmt(overall)}"
        f"（参考经验分级：≥0.3 较好 / 0.2–0.3 中等 / <0.2 偏弱）。"
    )

    if len(grades) > 1:
        findings.append(
            f"技巧随量级**单调衰减**：{weak['grade']}mm 的 TS {_fmt(weak['TS'])}"
            f" → {strong['grade']}mm 的 TS {_fmt(strong['TS'])}"
            f"（高量级命中数 {strong['hits']}，样本 {strong['n_pairs']}）。"
        )

    bias_entries = [entry for entry in grades if np.isfinite(entry.get("BIAS", np.nan))]
    if bias_entries:
        worst_bias = max(bias_entries, key=lambda entry: abs(np.log(max(entry["BIAS"], 1e-6))))
        direction = "系统性高报" if worst_bias["BIAS"] > 1 else "系统性漏报"
        findings.append(
            f"偏差结构：偏离理想值 1 最多的是 {worst_bias['grade']}mm，"
            f"BIAS = {_fmt(worst_bias['BIAS'])}（{direction}）；"
            f"BIAS 是频率偏差（预报事件数/观测事件数），不是平均误差。"
        )

    far_entries = [
        entry
        for entry in grades
        if np.isfinite(entry.get("FAR", np.nan)) and np.isfinite(entry.get("漏报率", np.nan))
    ]
    if far_entries:
        entry = far_entries[-1]
        if entry["FAR"] > entry["漏报率"]:
            findings.append(
                f"误差形态偏**空报**：{entry['grade']}mm 的 FAR {_fmt(entry['FAR'])}"
                f" 高于漏报率 {_fmt(entry['漏报率'])}。"
            )
        else:
            findings.append(
                f"误差形态偏**漏报**：{entry['grade']}mm 的漏报率 {_fmt(entry['漏报率'])}"
                f" 高于 FAR {_fmt(entry['FAR'])}。"
            )

    decay = summary.get("decay")
    if decay:
        findings.append(
            f"时效衰减：TS 从 {decay['first_lead']:g}h 的 {_fmt(decay['first_ts'])}"
            f" 降到 {decay['last_lead']:g}h 的 {_fmt(decay['last_ts'])}"
            f"（绝对 -{_fmt(decay['drop'])}，约 {_fmt(decay['per_day'])}/天）。"
        )

    zero_hit = [entry for entry in grades if entry["hits"] == 0]
    if zero_hit:
        names = "、".join(entry["grade"] for entry in zero_hit)
        findings.append(
            f"{names}mm 在全部时效中**命中为 0**，因此 TS/漏报率 显示为 —（无有效配对或无事件），"
            f"这与「评分为 0.000」含义不同，解读时注意区分。"
        )
    no_event = [entry for entry in grades if entry["events"] == 0]
    if no_event:
        names = "、".join(entry["grade"] for entry in no_event)
        findings.append(
            f"{names}mm 在评估期内**观测事件合计为 0**，属于样本库的物理上限："
            f"该量级技巧**不可评定**（不具备统计学意义），不是模型缺陷，也不应当作结论使用。"
        )
    sparse = [
        entry
        for entry in grades
        if 0 < entry["events"] < 50 and entry not in no_event
    ]
    if sparse:
        names = "、".join(
            f"{entry['grade']}（{entry['events']} 次事件）" for entry in sparse
        )
        findings.append(f"样本量过少、方差大，仅供参照：{names}。")
    return findings


def _grade_table(grades: Sequence[Dict[str, object]]) -> str:
    """（见 _source_of 与表格生成）"""
    header = (
        "| 降水等级 | 阈值(mm) | TS | POD | FAR | 漏报率 | BIAS "
        "| TS(等权平均) | 命中 | 漏报 | 空报 | 样本合计 |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for entry in grades:
        lines.append(
            f"| {entry['grade']} | {_fmt(entry['threshold'], 3)} "
            f"| {_fmt(entry['TS'])} {_ts_mark(entry['TS'])}"
            f" | {_fmt(entry['POD'])} | {_fmt(entry['FAR'])} | {_fmt(entry['漏报率'])}"
            f" | {_fmt(entry['BIAS'])} | {_fmt(entry['TS_mean'])}"
            f" | {entry['hits']} | {entry['misses']} | {entry['false_alarms']}"
            f" | {entry['n_pairs']} |"
        )
    return "\n".join(lines)


def _ts_mark(value: float) -> str:
    """技巧分级标记：≥0.3 好 / 0.2–0.3 中 / <0.2 弱。"""
    if not np.isfinite(value):
        return "—"
    if value >= 0.3:
        return "🟢"
    if value >= 0.2:
        return "🟡"
    return "🔴"


def _source_of(sources: Optional[Dict[str, str]], name: str) -> str:
    """报告头里的来源标注：给了文件路径就写路径，否则如实写"未提供"。"""
    if sources and sources.get(name):
        return f"`{sources[name]}`"
    return "未提供（调用方应在 sources 里给出文件路径）"


def business_conclusion(grades: Sequence[Dict[str, object]]) -> str:
    """把技巧水平翻译成业务语言（基于数据自动生成，不是套话）。"""
    def _entry(threshold: float):
        for item in grades:
            if item.get("threshold") is not None and abs(item["threshold"] - threshold) < 1e-6:
                return item
        return None

    light = _entry(0.1)
    moderate = _entry(10.0) or _entry(13.0)
    heavy = _entry(25.0) or _entry(50.0)

    parts: List[str] = []
    if light:
        usable = "可作为有无降水的定性参考" if light["TS"] >= 0.3 else "仅能给出粗略的有雨/无雨趋势"
        parts.append(f"小雨（≥0.1mm）TS {_fmt(light['TS'])}，{usable}")
    if moderate:
        parts.append(
            f"中雨（≥{_fmt(moderate['threshold'], 0)}mm）TS {_fmt(moderate['TS'])}，"
            + ("可作量级参考" if moderate["TS"] >= 0.2 else "已基本失去量级判别能力")
        )
    if heavy:
        parts.append(f"强降水（≥{_fmt(heavy['threshold'], 0)}mm）TS {_fmt(heavy['TS'])}")

    conclusion = "；".join(parts)
    if heavy and np.isfinite(heavy["TS"]) and heavy["TS"] < 0.15:
        conclusion += "。**结论定性：当前模型不宜直接用于强降水预警，需先做后处理订正。**"
    elif conclusion:
        conclusion += "。"
    return conclusion or "数据不足以给出业务定性。"


def recommendations(grades: Sequence[Dict[str, object]]) -> List[str]:
    """按数据条件给出的改进建议（只在条件成立时出现）。"""
    items: List[str] = []
    light = next((item for item in grades if item.get("threshold") == 0.1), None)
    heavy = next(
        (item for item in grades if item.get("threshold") in (25.0, 50.0)), None
    )

    if light and np.isfinite(light["BIAS"]) and light["BIAS"] > 1.3 and light["FAR"] > 0.45:
        items.append(
            f"小雨量级 BIAS {_fmt(light['BIAS'])}、FAR {_fmt(light['FAR'])} → "
            f"模型倾向「到处报小雨」、空间上过度平滑。损失函数可引入**焦点损失（Focal Loss）**"
            f"降低无雨背景点权重，并在频次上做**频率匹配**订正。"
        )
    if heavy and np.isfinite(heavy["TS"]) and heavy["TS"] < 0.15:
        items.append(
            f"强降水（≥{_fmt(heavy['threshold'], 0)}mm）TS {_fmt(heavy['TS'])} → "
            f"高频细节丢失。可引入**空间谱损失（Spectral Loss）**恢复小尺度结构，"
            f"并对极端事件做**样本重加权**以缓解长尾稀疏。"
        )
    sparse = [item for item in grades if item["events"] < 50]
    if sparse:
        names = "、".join(item["grade"] for item in sparse)
        items.append(
            f"{names}mm 的事件数不足 50（合计 {sum(item['events'] for item in sparse)} 次）→ "
            f"先延长评估期或扩大统计区域补齐样本，再评该量级技巧；"
            f"在样本补齐前不要据此调参。"
        )
    items.append(
        "后处理路径：分位数映射（QM）或频率匹配（FM）可同时压缩 BIAS 与空报；"
        "对强降水优先保证不漏报（提高 POD），再压 FAR。"
    )
    return items


# --------------------------------------------------------------------------- #
# 各节小结
#
# 这些函数是报告的「分析」部分：只吃 summarize() / box_stats() 已经算好的数字，
# **不重新读表、不重新实现指标**——口径只有一处，表、图、文字就不可能各说各话。
#
# 一条硬规矩：每句话都得带具体数值，或者是个能被数据证伪的判断。
# 写不出结论时如实说「数据不足以判断」，不拿「表现良好」这类话凑数。
# --------------------------------------------------------------------------- #
def analysis_by_grade(grades: Sequence[Dict[str, object]]) -> str:
    """第二节小结：量级维度——技巧分级、单调性、断崖在哪一档、样本随量级的收缩。"""
    usable = [entry for entry in grades if np.isfinite(entry.get("TS", np.nan))]
    if not usable:
        return "各降水等级的 TS 均不可定义（无有效配对或无命中），无法做量级间比较。"

    marks = [_ts_mark(entry["TS"]) for entry in usable]
    tally = "、".join(
        f"{mark} {marks.count(mark)} 档" for mark in ("🟢", "🟡", "🔴") if marks.count(mark)
    )
    parts = [f"{len(usable)} 个可评定的降水等级里 {tally}。"]

    steps = [
        (usable[index], usable[index + 1], usable[index]["TS"] - usable[index + 1]["TS"])
        for index in range(len(usable) - 1)
    ]
    rises = [step for step in steps if step[2] < -0.005]
    if rises:
        where = "、".join(f"{left['grade']}→{right['grade']}" for left, right, _ in rises)
        parts.append(
            f"TS 并非随量级单调下降：{where} 处出现回升，说明该处的样本波动已经盖过技巧的真实差异，"
            f"这个位置上的名次不可当真。"
        )
    elif steps:
        parts.append(
            f"TS 随量级**单调下降**，从 {usable[0]['grade']}mm 的 {_fmt(usable[0]['TS'])} "
            f"一路降到 {usable[-1]['grade']}mm 的 {_fmt(usable[-1]['TS'])}。"
        )
    if steps:
        left, right, drop = max(steps, key=lambda step: step[2])
        if drop > 0:
            parts.append(
                f"相邻量级之间落差最大的是 **{left['grade']}→{right['grade']}**"
                f"（{_fmt(left['TS'])} → {_fmt(right['TS'])}，掉 {_fmt(drop)}），"
                f"技巧的断崖就出现在这一档之间。"
            )

    business = [entry for entry in usable if entry["TS"] >= 0.2]
    if business:
        best = business[-1]
        parts.append(
            f"以 TS ≥ 0.2（🟡 及以上）作为最低可用门槛，可用量级最高到 {best['grade']}mm"
            f"（TS {_fmt(best['TS'])}），再往上不具备定量判别能力。"
        )
    else:
        parts.append("没有任何量级达到 TS ≥ 0.2，全部等级都不满足最低可用门槛。")

    first, last = usable[0], usable[-1]
    if first["events"] > 0 and 0 < last["events"] < first["events"]:
        ratio = first["events"] / last["events"]
        parts.append(
            f"观测事件数从 {first['grade']}mm 的 {first['events']:,} 次收缩到 "
            f"{last['grade']}mm 的 {last['events']:,} 次"
            + (f"（约 1/{ratio:.0f}）" if ratio >= 10 else "")
            + "，高量级的统计不确定性随之放大，其 TS 的小数位不宜过度解读。"
        )
    return "".join(parts)


def analysis_by_lead(summary: Dict[str, object], *, floor: float = 0.10) -> str:
    """第三节小结：时效维度——有效时效上限、单调性、平台期、衰减速度。"""
    pivot = summary.get("pivot")
    leads = list(summary.get("leads") or [])
    if pivot is None or len(pivot.index) < 2 or len(leads) < 2:
        return "时效点不足两个，无法判断技巧的时效衰减。"

    limits: List[tuple] = []
    for grade in pivot.columns:
        series = pivot[grade].dropna()
        below = series[series < floor]
        limits.append((str(grade), float(below.index.min()) if len(below) else None))

    parts: List[str] = []
    broken = [(grade, lead) for grade, lead in limits if lead is not None]
    kept = [grade for grade, lead in limits if lead is None]
    if broken:
        detail = "、".join(f"{grade}mm 在 {lead:g}h" for grade, lead in broken)
        parts.append(
            f"以 **TS 首次跌破 {floor:g}** 作为「有效时效上限」：{detail} 已跌破"
            + (f"；{'、'.join(f'{grade}mm' for grade in kept)} 全程未跌破。" if kept else "。")
        )
    else:
        parts.append(
            f"全部等级到 {leads[-1]:g}h 都未跌破 {floor:g}，时效衰减不构成本次评估的限制因素。"
        )

    curve = pivot.mean(axis=1).dropna()
    if len(curve) >= 2:
        steps = curve.diff().dropna()
        rises = steps[steps > 0.005]
        if rises.empty:
            parts.append(
                f"各等级等权平均的 TS 随时效**单调下降**"
                f"（{curve.index[0]:g}h 的 {_fmt(curve.iloc[0])} → "
                f"{curve.index[-1]:g}h 的 {_fmt(curve.iloc[-1])}）。"
            )
        else:
            parts.append(
                f"等权平均 TS 总体随时效下降，但在 {float(rises.idxmax()):g}h 出现 "
                f"+{_fmt(float(rises.max()))} 的回升，属于抽样波动而非真实的技巧反转。"
            )
        tail = steps.tail(3)
        if len(tail) == 3 and bool((tail.abs() < 0.005).all()):
            parts.append(
                f"最后三个时效区间（{float(tail.index[0]):g}–{curve.index[-1]:g}h）的相邻变化都小于 0.005，"
                f"技巧已进入平台期：再往后延长时间不会带来等比例的进一步损失。"
            )

    decay = summary.get("decay") or {}
    if decay and np.isfinite(decay.get("relative", np.nan)):
        judge = (
            "相对衰减已超过 50%，长时效技巧明显退化，结论应限定在较短时效内。"
            if decay["relative"] > 0.5
            else "相对衰减未超过 50%，长时效仍保有一定技巧。"
        )
        parts.append(
            f"整体来看，等权平均 TS 从 {decay['first_lead']:g}h 的 {_fmt(decay['first_ts'])} 降到 "
            f"{decay['last_lead']:g}h 的 {_fmt(decay['last_ts'])}：绝对衰减 {_fmt(decay['drop'])}、"
            f"相对衰减 {decay['relative']:.0%}、约 {_fmt(decay['per_day'])}/天。{judge}"
        )

    speeds = []
    for grade in pivot.columns:
        series = pivot[grade].dropna()
        if len(series) >= 2 and series.iloc[0] > 0:
            speeds.append((str(grade), float((series.iloc[0] - series.iloc[-1]) / series.iloc[0])))
    speeds = [item for item in speeds if item[1] > 0]
    if speeds:
        grade, relative = max(speeds, key=lambda item: item[1])
        parts.append(f"逐等级比较，相对衰减最快的是 {grade}mm（{relative:.0%}）。")
    return "".join(parts)


def analysis_error_structure(grades: Sequence[Dict[str, object]]) -> str:
    """第四节小结：误差形态——空报与漏报谁主导、在哪一档反转、BIAS 的走向。"""
    parts: List[str] = []
    pairs = [
        entry for entry in grades
        if np.isfinite(entry.get("FAR", np.nan)) and np.isfinite(entry.get("漏报率", np.nan))
    ]
    if not pairs:
        parts.append("FAR 与漏报率在各等级上都不可用，无法判断误差形态。")
    else:
        dominant = [entry["FAR"] > entry["漏报率"] for entry in pairs]
        flips = [
            index for index in range(1, len(dominant)) if dominant[index] != dominant[index - 1]
        ]
        if not flips:
            kind = "空报" if dominant[0] else "漏报"
            span = (
                f"{pairs[0]['grade']}–{pairs[-1]['grade']}mm"
                if len(pairs) > 1
                else f"{pairs[0]['grade']}mm"
            )
            parts.append(f"{span} 全部以**{kind}为主**，误差形态在量级之间没有反转。")
        else:
            index = flips[0]
            before, after = pairs[index - 1], pairs[index]
            parts.append(
                f"误差形态在 {before['grade']} 与 {after['grade']} 之间**发生反转**："
                f"{before['grade']} 以空报为主（FAR {_fmt(before['FAR'])} > 漏报率 {_fmt(before['漏报率'])}），"
                f"{after['grade']} 转为漏报为主（漏报率 {_fmt(after['漏报率'])} > FAR {_fmt(after['FAR'])}）。"
            )
            if len(flips) > 1:
                parts.append(
                    f"整条量级链上共有 {len(flips)} 处反转，说明该模型没有贯穿各量级的一致误差倾向，"
                    f"订正策略需要分档设定。"
                )
        entry = max(pairs, key=lambda item: abs(item["FAR"] - item["漏报率"]))
        kind = "空报" if entry["FAR"] > entry["漏报率"] else "漏报"
        parts.append(
            f"空报与漏报相差最悬殊的是 {entry['grade']}mm"
            f"（FAR {_fmt(entry['FAR'])} 对漏报率 {_fmt(entry['漏报率'])}，{kind}主导）。"
        )

    biases = [entry for entry in grades if np.isfinite(entry.get("BIAS", np.nan))]
    if biases:
        low, high = biases[0], biases[-1]
        closest = min(biases, key=lambda item: abs(item["BIAS"] - 1.0))
        drift = high["BIAS"] - low["BIAS"]
        trend = (
            "随量级升高而下降" if drift < -0.05
            else ("随量级升高反而上升" if drift > 0.05 else "在各量级之间大致持平")
        )
        parts.append(
            f"BIAS 从 {low['grade']}mm 的 {_fmt(low['BIAS'])} 变到 {high['grade']}mm 的 "
            f"{_fmt(high['BIAS'])}（{trend}）；最接近理想值 1 的是 {closest['grade']}mm"
            f"（{_fmt(closest['BIAS'])}）。"
        )
        if low["BIAS"] > 1.1 and high["BIAS"] < 0.9:
            parts.append(
                "**小雨偏多、强降水偏少**，这是空间过度平滑的典型征兆："
                "模式把小尺度强中心抹成了大范围弱降水，既抬高小雨的空报，也吃掉强降水的命中。"
            )
    return "".join(parts) or "误差形态与偏差结构均不可评定。"


def analysis_stability(
    models: Dict[str, pd.DataFrame], primary: str, metric: str = "TS"
) -> str:
    """第五节小结：跨时效稳定性——谁最稳、离群时效落在哪、两种口径的名次是否一致。"""
    per_model: Dict[str, List[Dict[str, object]]] = {}
    for name, frame in models.items():
        rows = [row for row in box_stats_by_grade(frame, metric=metric) if row["n"]]
        if rows:
            per_model[name] = rows
    if not per_model:
        return "各模型在各量级上都没有有效样本，无法评估跨时效稳定性。"

    parts: List[str] = []
    spread = {
        name: float(np.median([row["iqr"] for row in rows]))
        for name, rows in per_model.items()
    }
    ranked = sorted(spread, key=lambda name: spread[name])
    if len(ranked) == 1:
        parts.append(
            f"单模型评估：{ranked[0]} 各量级 IQR 的中位数为 {_fmt(spread[ranked[0]])}，"
            f"这是本次评估的跨时效稳定性基准。"
        )
    elif spread[ranked[-1]] - spread[ranked[0]] < 0.01:
        parts.append(
            f"各模型的跨时效稳定性接近（IQR 中位数 {_fmt(spread[ranked[0]])}–"
            f"{_fmt(spread[ranked[-1]])}，极差不足 0.01），稳定性不构成它们之间的区分度。"
        )
    else:
        parts.append(
            f"按各量级 IQR 的中位数排序，跨时效最稳定的是 **{ranked[0]}**"
            f"（{_fmt(spread[ranked[0]])}），最不稳定的是 **{ranked[-1]}**"
            f"（{_fmt(spread[ranked[-1]])}）——箱体越窄，说明该模型各时效的技巧越均衡。"
        )

    outliers = sorted(
        {float(lead) for row in per_model.get(primary, []) for lead in row["outlier_leads"]}
    )
    if outliers and primary in models:
        all_leads = sorted(float(value) for value in pd.unique(models[primary]["lead_h"]))
        middle = (all_leads[0] + all_leads[-1]) / 2
        early = sum(1 for lead in outliers if lead <= middle)
        where = (
            "短时效段" if early * 2 > len(outliers)
            else ("长时效段" if early * 2 < len(outliers) else "首尾两端")
        )
        shown = "、".join(f"{lead:g}h" for lead in outliers[:6])
        parts.append(
            f"{primary} 的离群时效共 {len(outliers)} 个（{shown}"
            + ("…" if len(outliers) > 6 else "")
            + f"），偏向{where}——这些时效的技巧与其余时效明显脱节，"
            f"只看等权平均会把这种脱节抹平。"
        )
    elif primary in models:
        parts.append(f"{primary} 的各量级都没有离群时效，TS 的跨时效分布连续、没有突变的时效点。")

    flips = []
    for grade in ordered_grades(list(models.values())):
        medians: Dict[str, float] = {}
        means: Dict[str, float] = {}
        for name, frame in models.items():
            subset = frame[frame["grade"].astype(str) == grade]
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            if values.empty:
                continue
            medians[name] = float(values.median())
            means[name] = float(values.mean())
        if len(medians) < 2:
            continue
        by_median = max(medians, key=medians.get)
        by_mean = max(means, key=means.get)
        if by_median != by_mean:
            flips.append((grade, by_median, by_mean))
    if flips:
        detail = "、".join(
            f"{grade}（中位数 {left} 领先、等权平均 {right} 领先）" for grade, left, right in flips
        )
        parts.append(
            f"**注意口径**：{detail}。本节表里的统计量是中位数/IQR，第二节表里的是等权平均，"
            f"两者给出不同的领先者是正常现象（前者抗离群、后者被离群时效拉动），不可互相替代。"
        )
    return "".join(parts)


def analysis_comparison(
    data: pd.DataFrame, comparisons: Dict[str, pd.DataFrame], lead: float
) -> str:
    """第六节小结：对比维度——谁在哪些量级领先、名次是否随量级翻转、差距分布。

    ``comparisons`` 既可以是多个对比模型，也可以是单个基准——单基准时
    「领先者随量级翻转」照样能看出两者的互补性，所以两种分支共用这一个函数。
    """
    frames: Dict[str, pd.DataFrame] = {"本模型": data, **comparisons}
    table: Dict[str, Dict[str, float]] = {}
    for grade in ordered_grades(list(frames.values())):
        row: Dict[str, float] = {}
        for name, frame in frames.items():
            subset = frame[
                (pd.to_numeric(frame["lead_h"], errors="coerce") == lead)
                & (frame["grade"].astype(str) == grade)
            ]
            values = pd.to_numeric(subset["TS"], errors="coerce").dropna()
            if len(values):
                row[name] = float(values.iloc[0])
        if len(row) >= 2:
            table[grade] = row
    if not table:
        return f"{lead:g}h 时效上没有两个及以上对象同时有有效 TS，无法比较。"

    parts: List[str] = []
    # 差距小于 0.005 视为并列——和 `_baseline_table` 里 ↑/↓/≈ 的门槛同一个数，
    # 免得表里写着「≈」、正文却说某一方领先。
    def _tie(row: Dict[str, float]) -> bool:
        order = sorted(row.values(), reverse=True)
        return len(order) > 1 and order[0] - order[1] < 0.005

    tied = [grade for grade, row in table.items() if _tie(row)]
    winners = {
        grade: max(row, key=row.get) for grade, row in table.items() if grade not in tied
    }
    if not winners:
        parts.append(
            f"各对象在 {lead:g}h、各量级上的差距都小于 0.005，视为并列，无法据此分出高下。"
        )
    else:
        counts: Dict[str, int] = {}
        for winner in winners.values():
            counts[winner] = counts.get(winner, 0) + 1
        if len(counts) == 1:
            only = next(iter(counts))
            parts.append(
                f"{lead:g}h 时效上，**{only}** 在全部 {len(winners)} 个可分出高下的量级上都领先。"
            )
        else:
            detail = "、".join(f"{grade} 由{winner}领先" for grade, winner in winners.items())
            parts.append(f"{lead:g}h 时效上的领先者随量级变化：{detail}。")
    if tied:
        # 并列要说清是"谁和谁并列"——只写"各对象并列"会把差距明显的第三名也算进去
        detail = "；".join(
            f"{grade} 的前两名 "
            + "与".join(
                f"{name}（{table[grade][name]:.3f}）"
                for name in sorted(table[grade], key=table[grade].get, reverse=True)[:2]
            )
            for grade in tied
        )
        parts.append(f"{detail} 差距不足 0.005，分不出领先者。")

    sequence = list(winners.items())
    crossovers = [
        index for index in range(1, len(sequence)) if sequence[index][1] != sequence[index - 1][1]
    ]
    if crossovers:
        index = crossovers[0]
        before, after = sequence[index - 1], sequence[index]
        parts.append(
            f"名次在 {before[0]} 与 {after[0]} 之间**发生翻转**（领先者由{before[1]}变为"
            f"{after[1]}）——两者的优势区间不同，只报一个跨量级平均值会把这种互补性掩盖掉。"
        )
        if len(crossovers) > 1:
            parts.append(f"整条量级链上共出现 {len(crossovers)} 处领先者交替。")

    gaps: Dict[str, float] = {}
    for grade, row in table.items():
        if "本模型" in row:
            others = [value for name, value in row.items() if name != "本模型"]
            if others:
                gaps[grade] = row["本模型"] - max(others)
    # 各量级的差距都一样（比如全部并列）时，这句话不含信息，不如不说
    if gaps and max(gaps.values()) - min(gaps.values()) >= 0.005:
        best = max(gaps, key=gaps.get)
        worst = min(gaps, key=gaps.get)
        parts.append(
            f"以最强的对比对象为参照，本模型相对领先最多的是 {best}（{gaps[best]:+.3f}）、"
            f"相对落后最多的是 {worst}（{gaps[worst]:+.3f}）。"
        )

    # 名次只在分得出高下的量级上算——并列的量级硬排名次就是在报告噪声
    ranks = []
    for grade in winners:
        row = table[grade]
        if "本模型" in row:
            better = sum(1 for value in row.values() if value > row["本模型"] + 1e-12)
            ranks.append(better + 1)
    if ranks:
        widest = max(len(row) for row in table.values())
        parts.append(f"本模型在 {len(ranks)} 个量级上的平均名次为 {np.mean(ranks):.1f}/{widest}。")
    return "".join(parts)


def _lead_table(pivot: pd.DataFrame) -> str:
    columns = [str(column) for column in pivot.columns]
    lines = ["| 时效(h) | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for lead, row in pivot.iterrows():
        values = " | ".join(_fmt(row[column]) for column in pivot.columns)
        lines.append(f"| {lead:g} | {values} |")
    return "\n".join(lines)


def _baseline_table(data: pd.DataFrame, baseline: pd.DataFrame, lead: float) -> str:
    left = data[data["lead_h"] == lead].set_index("grade")["TS"]
    right = baseline[baseline["lead_h"] == lead].set_index("grade")["TS"]
    lines = [
        "| 降水等级 | 本模型 TS | 对比模型 TS | 差值 |",
        "|---|---|---|---|",
    ]
    for grade in left.index:
        if grade in right.index:
            delta = float(left[grade]) - float(right[grade])
            flag = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
            lines.append(
                f"| {grade} | {_fmt(left[grade])} | {_fmt(right[grade])} | {flag} {delta:+.3f} |"
            )
    return "\n".join(lines)


def box_stats(data: pd.DataFrame, metric: str = "TS") -> Dict[str, object]:
    """一个箱体的 Tukey 统计量——**与 matplotlib 的 boxplot 同口径**。

    样本是"同一模型、同一量级在全部时效上的取值"，一行一个时效。
    放进 ``ts_report`` 而不是画图模块，是为了让**报告表格和图上箱体同源**：
    两边都调这个函数，就不存在"表里的中位数和图上的线对不上"。

    返回 ``{"n", "q1", "median", "q3", "iqr", "whisker_lo", "whisker_hi",
    "outlier_leads", "outlier_values"}``；没有有效值时 ``n=0``、统计量为 ``nan``、列表为空。

    Raises:
        KeyError: 表里没有 ``metric`` 列。
    """
    if metric not in data.columns:
        raise KeyError(f"结果表里没有指标列 '{metric}'")

    values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
    leads = (
        pd.to_numeric(data["lead_h"], errors="coerce").to_numpy(dtype=float)
        if "lead_h" in data.columns
        else np.full(values.shape, np.nan)
    )
    # 与 prepare_ts_dataframe 的 dropna(subset=["lead_h","TS"]) 同口径：非有限值一律不算样本
    keep = np.isfinite(values)
    values, leads = values[keep], leads[keep]

    if values.size == 0:
        nan = float("nan")
        return {
            "n": 0, "q1": nan, "median": nan, "q3": nan, "iqr": nan,
            "whisker_lo": nan, "whisker_hi": nan,
            "outlier_leads": [], "outlier_values": [],
        }

    # 分位数用 np.percentile 的默认参数（线性插值）——matplotlib 的 cbook.boxplot_stats
    # 用的也是它，所以口径天然一致；千万别传 method=，一传两边就可能对不上。
    q1, median, q3 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    # 须线画到"栅栏内最远的点"，不是把极值夹到栅栏上；栅栏内一个点都没有（全是离群点）
    # 时退回四分位——这是 matplotlib 的规则，不是常见的 max(min, Q1-1.5*IQR) 写法。
    inside = values[(values >= low_fence) & (values <= high_fence)]
    outliers = (values < low_fence) | (values > high_fence)
    return {
        "n": int(values.size),
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "whisker_lo": float(inside.min()) if inside.size else q1,
        "whisker_hi": float(inside.max()) if inside.size else q3,
        "outlier_leads": [float(value) for value in leads[outliers]],
        "outlier_values": [float(value) for value in values[outliers]],
    }


def box_stats_by_grade(data: pd.DataFrame, metric: str = "TS") -> List[Dict[str, object]]:
    """``[{"grade", "threshold", **box_stats(...)}]``，按阈值升序。"""
    rows: List[Dict[str, object]] = []
    for grade, subset in data.groupby("grade", observed=True):
        thresholds = (
            subset["threshold_mm"].dropna()
            if "threshold_mm" in subset.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "grade": str(grade),
                "threshold": float(thresholds.iloc[0]) if len(thresholds) else None,
                **box_stats(subset, metric=metric),
            }
        )
    rows.sort(key=lambda row: _grade_sort_key(row["grade"], row["threshold"]))
    return rows


def ordered_grades(frames: Sequence[pd.DataFrame]) -> List[str]:
    """多张结果表里出现过的降水等级**并集**，按 ``threshold_mm`` 升序。

    缺 ``threshold_mm`` 时退回解析 ``"≥X"`` 文本（``_grade_sort_key`` 已经带了这条路）。
    各模型的等级集合未必一样（``ts_multi_fuxi_ens`` 就只有 5 档），所以要取并集。
    """
    thresholds: Dict[str, float] = {}
    for frame in frames:
        if "grade" not in frame.columns:
            continue
        first = (
            frame.groupby("grade", observed=True)["threshold_mm"].first()
            if "threshold_mm" in frame.columns
            else pd.Series(dtype=float)
        )
        for grade in frame["grade"].astype(str):
            if grade in thresholds:
                continue
            value = first.get(grade, float("nan")) if len(first) else float("nan")
            thresholds[grade] = float(value) if pd.notna(value) else float("nan")
    return sorted(thresholds, key=lambda grade: _grade_sort_key(grade, thresholds[grade]))


def collect_box_models(
    model_name: str,
    data: pd.DataFrame,
    *,
    baseline_df: Optional[pd.DataFrame] = None,
    baseline_name: Optional[str] = None,
    baselines: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """箱线图画哪些模型：本模型在前，然后是单基准、各对比模型。

    图和表都调这一个函数，免得两边的模型集合悄悄分叉。
    """
    models: Dict[str, pd.DataFrame] = {model_name: data}
    if baseline_df is not None and baseline_name:
        models[baseline_name] = baseline_df
    models.update(baselines or {})
    return models


def _box_table(models: Dict[str, pd.DataFrame], metric: str = "TS") -> str:
    """一行一个（模型 × 降水等级）：量级升序，同量级内按 ``models`` 的顺序。"""
    names = list(models)
    grades = ordered_grades(list(models.values()))
    stats = {
        name: {row["grade"]: row for row in box_stats_by_grade(frame, metric=metric)}
        for name, frame in models.items()
    }

    lines = [_BOX_HEADER, _BOX_SEP]
    for grade in grades:
        for name in names:
            row = stats[name].get(grade)
            if row is None or row["n"] == 0:
                # 该模型没有这个量级（或全无有效值）：照出一行全 —，和图上那个空槽位对齐
                cells = ["—"] * 6 + ["0"]
            else:
                leads = "、".join(f"{lead:g}h" for lead in sorted(row["outlier_leads"])) or "—"
                cells = [
                    _fmt(row["median"]),
                    _fmt(row["iqr"]),
                    f"{_fmt(row['q1'])}–{_fmt(row['q3'])}",
                    f"{_fmt(row['whisker_lo'])}–{_fmt(row['whisker_hi'])}",
                    leads,
                    str(row["n"]),
                ]
            lines.append(f"| {name} | {grade} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _multi_baseline_table(
    data: pd.DataFrame, baselines: Dict[str, pd.DataFrame], lead: float
) -> str:
    """一行一个降水等级，一列一个模型（本模型 + 各对比模型）。"""
    names = list(baselines.keys())
    left = data[data["lead_h"] == lead].set_index("grade")["TS"]
    columns = {
        name: frame[frame["lead_h"] == lead].set_index("grade")["TS"]
        for name, frame in baselines.items()
    }
    header = "| 降水等级 | 本模型 | " + " | ".join(names) + " | 本模型名次 |"
    lines = [header, "|---|" + "---|" * (len(names) + 2)]
    for grade in left.index:
        values = {name: columns[name].get(grade, float("nan")) for name in names}
        pool = {**{name: float(value) for name, value in values.items() if np.isfinite(value)},
                "本模型": float(left[grade]) if np.isfinite(left[grade]) else float("nan")}
        ranked = [name for name, value in pool.items() if np.isfinite(value)]
        # 并列同名次（competition ranking）：名次 = 比自己高的对象个数 + 1。
        # 若按 sorted 的下标算，数值相同的两家谁在前全看字典插入顺序，
        # 表里会出现「3/3」而正文按同一套数据算出「并列第 2」。
        rank = (
            1 + sum(1 for value in pool.values() if value > pool["本模型"])
            if np.isfinite(pool["本模型"]) else 0
        )
        cells = " | ".join(_fmt(values[name]) for name in names)
        lines.append(
            f"| {grade} | {_fmt(left[grade])} | {cells} | {rank}/{len(ranked) if ranked else 0} |"
        )
    return "\n".join(lines)


def _figure_captions(
    summary: Dict[str, object], *, window: Optional[float], lead: float, model_name: str
) -> Dict[str, str]:
    """每张图的图注——说明**怎么读这张图**，而不是重复图的标题。"""
    leads = summary["leads"]
    span = f"{leads[0]:g}–{leads[-1]:g}h"
    window_text = f"{window:g}h 累积窗口" if window else "本次评估窗口"
    return {
        "metrics_by_threshold": (
            f"{lead:g}h 时效（{window_text}）各降水等级的 TS、POD、FAR 与 BIAS 对比"
        ),
        "TS_vs_lead": (
            f"{span} 各降水等级的 TS 曲线；线越靠上技巧越好，"
            f"虚线表示该等级事件数不足 50、仅作参照"
        ),
        "TS_heatmap": (
            f"降水等级 × 预报时效的 TS 热力图（{window_text}）；"
            f"颜色越深技巧越高，格子里的数字随底色自动切黑/白"
        ),
        "POD_vs_lead": (
            f"{span} 各降水等级的命中率 POD；POD 低意味着漏报多，"
            f"但单独抬高 POD 往往以增大空报为代价，需与下一张 FAR 图联合判读"
        ),
        "FAR_vs_lead": f"{span} 各降水等级的空报率 FAR（纵轴锁定 0–1）；FAR 越低误报越少",
        "BIAS_vs_lead": (
            f"{span} 各降水等级的频率偏差 BIAS，虚线为理想值 1（纵轴自适应、不截断）；"
            f"曲线在 1 上方表示预报事件偏多、下方表示偏少"
        ),
        "performance_diagram": (
            f"{lead:g}h 时效的性能图：横轴为成功比 1−FAR、纵轴为命中率 POD，"
            f"灰色虚线为 TS 等值线、蓝色点线为 BIAS 参考线；点越靠近右上角技巧越好"
        ),
        "TS_boxplot": (
            "各模型 TS 的跨时效分布；箱体越高表示该量级整体技巧越好、"
            "越窄表示跨时效越稳定，空心圆为离群时效"
        ),
        "multi_model_TS_delta": (
            f"{span} 各对比模型相对 {model_name} 的 TS 差值"
            f"（纵轴是「对比模型 − 本模型」：正值表示对比模型领先，负值表示本模型领先）"
        ),
    }


def _section_grade(
    summary: Dict[str, object], artifacts, numbering: _Numbering, captions: Dict[str, str]
) -> List[str]:
    grades = summary["grades"]
    lines = ["## 二、逐降水等级表现（全时效平均）", ""]
    lines.append(
        "下表汇总各降水等级在整个时效范围内的表现，"
        "用于判断模型在不同量级上的技巧水平与偏差结构。"
    )
    lines.append("")
    lines += _table_caption(numbering, "各降水等级的全时效平均技巧与列联表计数（按阈值升序）")
    lines.append(_grade_table(grades))
    lines.append("")
    lines.append(
        "> 全时效平均为**各时效等权平均**（不是按样本数加权）；判断强降水能力请看 ≥25mm 及以上等级。"
    )
    lines.append("")
    lines += _figure_block(artifacts, numbering, "metrics_by_threshold", captions["metrics_by_threshold"])
    lines.append(f"**小结**：{analysis_by_grade(grades)}")
    lines.append("")
    return lines


def _section_lead(
    summary: Dict[str, object], artifacts, numbering: _Numbering, captions: Dict[str, str]
) -> List[str]:
    leads = summary["leads"]
    lines = [f"## 三、TS 随时效变化（{leads[0]:g}–{leads[-1]:g}h）", ""]
    lines.append(
        "按预报时效展开，用于确定该模型的有效预报时效——技巧掉到不可用之前能推多远。"
    )
    lines.append("")
    for name in ("TS_vs_lead", "TS_heatmap"):
        lines += _figure_block(artifacts, numbering, name, captions[name])
    lines += _table_caption(numbering, "各降水等级在各预报时效上的 TS")
    lines.append(_lead_table(summary["pivot"]))
    lines.append("")
    lines.append(
        "> 单元格是该时效该等级的 TS；同一行内横向比较看衰减速度，同一列内纵向比较看量级差异。"
    )
    lines.append("")
    lines.append(f"**小结**：{analysis_by_lead(summary)}")
    lines.append("")
    return lines


def _section_error_shape(
    summary: Dict[str, object], artifacts, numbering: _Numbering, captions: Dict[str, str]
) -> List[str]:
    lines = ["## 四、误差形态与偏差结构", ""]
    lines.append(
        "技巧高低之外还要看误差从哪来：本节从命中率、空报率与频率偏差三个方面拆解误差的来源与形态。"
    )
    lines.append("")
    for name in ("POD_vs_lead", "FAR_vs_lead", "BIAS_vs_lead", "performance_diagram"):
        lines += _figure_block(artifacts, numbering, name, captions[name])
    lines.append(f"**小结**：{analysis_error_structure(summary['grades'])}")
    lines.append("")
    return lines


def _section_boxplot(
    models: Dict[str, pd.DataFrame],
    artifacts,
    numbering: _Numbering,
    captions: Dict[str, str],
    primary: str,
) -> List[str]:
    lines = ["## 五、跨时效分布（箱线图，TS）", ""]
    lines.append(
        "把同一模型、同一量级在各时效上的 TS 视为一个样本，用箱线图刻画其离散程度——"
        "平均技巧相同的两个模型，稳定性可能差很远。"
    )
    lines.append("")
    lines += _table_caption(numbering, "各模型 × 各降水等级 TS 的跨时效分布统计量")
    lines.append(_box_table(models))
    lines.append("")
    lines.append(_BOX_NOTE)
    lines.append("")
    lines += _figure_block(artifacts, numbering, "TS_boxplot", captions["TS_boxplot"])
    lines.append(f"**小结**：{analysis_stability(models, primary)}")
    lines.append("")
    return lines


def _section_comparison(
    data: pd.DataFrame,
    comparisons: Dict[str, pd.DataFrame],
    artifacts,
    numbering: _Numbering,
    captions: Dict[str, str],
    *,
    lead: float,
    heading: str,
    caption: str,
    note: Optional[str],
) -> List[str]:
    lines = [heading, ""]
    lines.append("以下比较在同一时效、同一阈值口径下进行。")
    lines.append("")
    lines += _table_caption(numbering, caption)
    lines.append(_table_body_of(comparisons, data, lead))
    lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    lines += _figure_block(artifacts, numbering, "multi_model_TS_delta", captions["multi_model_TS_delta"])
    lines.append(f"**小结**：{analysis_comparison(data, comparisons, lead)}")
    lines.append("")
    return lines


def _table_body_of(
    comparisons: Dict[str, pd.DataFrame], data: pd.DataFrame, lead: float
) -> str:
    """单基准是多模型的一个特例——表体按对比对象的个数选，别让两个分支各写一遍。"""
    if len(comparisons) == 1:
        name, frame = next(iter(comparisons.items()))
        return _baseline_table(data, frame, lead)
    return _multi_baseline_table(data, comparisons, lead)


def build_ts_report(
    data: pd.DataFrame,
    output_path: Path,
    *,
    model_name: str = "model",
    baseline_df: Optional[pd.DataFrame] = None,
    baseline_name: Optional[str] = None,
    baselines: Optional[Dict[str, pd.DataFrame]] = None,
    sources: Optional[Dict[str, str]] = None,
    change_description: Optional[str] = None,
    artifacts: Optional[Dict[str, Path]] = None,
    lead_h: Optional[float] = None,
    box_models: Optional[Dict[str, pd.DataFrame]] = None,
) -> Path:
    """写一份 Markdown 报告。

    每一节都是「引导句 → 图表 → 小结」：图就地带着 `图 N：` 图注出现，
    表上方带 `表 N：` 表题，节末一段带具体数值的分析。末尾不再有「图表」索引节——
    图都在正文里，再列一遍只是重复，还会让人以为是两张不同的图。

    Args:
        box_models: 第五节箱线图的模型集合（本模型 + 各对比模型）。
            由 ``create_report`` 算好传进来，与它画图用的是**同一个 dict**；
            不传则按 ``model_name`` + ``baseline_df``/``baseline_name`` + ``baselines``
            自己拼一份（直接调本函数的场景，比如测试）。
    """
    output_path = Path(output_path)
    summary = summarize(data)
    findings = diagnose(summary)
    grades = summary["grades"]
    leads = summary["leads"]
    window = None
    if "window_h" in data.columns and data["window_h"].notna().any():
        window = float(data["window_h"].dropna().iloc[0])
    lead = float(lead_h) if lead_h is not None else leads[0]

    # 模型集合由 create_report 算好传进来（它同时拿去画图）；直接调 build_ts_report 的
    # 调用方没给，就按同样的规则自己拼一份，别让表和图用两套模型集合。
    models = box_models if box_models is not None else collect_box_models(
        model_name,
        data,
        baseline_df=baseline_df,
        baseline_name=baseline_name,
        baselines=baselines,
    )
    # 单基准和多模型走同一条比较路径——单基准只是"一个对比对象"的特例
    comparisons: Dict[str, pd.DataFrame] = dict(baselines or {})
    if baseline_df is not None:
        comparisons.setdefault(baseline_name or "基准", baseline_df)

    numbering = _Numbering()
    captions = _figure_captions(summary, window=window, lead=lead, model_name=model_name)

    lines: List[str] = []
    lines.append(f"# {model_name} 降水分类检验报告")
    lines.append("")

    lines.append("## 评测对象与对比对象")
    lines.append("")
    lines.append(f"- **评测对象**：{model_name}（来源：{_source_of(sources, model_name)}）")
    if comparisons:
        joined = "、".join(
            f"{name}（来源：{_source_of(sources, name)}）" for name in comparisons
        )
        lines.append(f"- **对比对象**：{joined}")
        lines.append(
            "- **可比性**：以上对象应使用同一时段、同一站点集、同一窗口与阈值口径；"
            "若来源文件的口径不同，结论不可直接横向比较。"
        )
    else:
        lines.append("- **对比对象**：无（单模型评估）")
    if change_description:
        lines.append(f"- **本次改动**：{change_description}")
    lines.append(f"- **生成时间**：{datetime.now():%Y-%m-%d %H:%M}")
    lines.append(
        f"- 评估口径：站点分类检验（"
        + (f"{window:g}h 累积窗口，" if window else "")
        + f"时效 {leads[0]:g}–{leads[-1]:g}h 共 {len(leads)} 个，{summary['n_rows']} 条记录）"
    )
    lines.append(
        "- 指标定义：TS = hits/(hits+misses+false_alarms)，POD = hits/(hits+misses)，"
        "FAR = false_alarms/(hits+false_alarms)，漏报率 = 1−POD，"
        "BIAS = (hits+false_alarms)/(hits+misses)（频率偏差）"
    )
    lines.append("")

    lines.append("## 一、结论摘要")
    lines.append("")
    for index, finding in enumerate(findings, start=1):
        lines.append(f"{index}. {finding}")
    lines.append("")
    lines.append(f"**业务定性**：{business_conclusion(grades)}")
    lines.append("")

    lines += _section_grade(summary, artifacts, numbering, captions)
    lines += _section_lead(summary, artifacts, numbering, captions)
    lines += _section_error_shape(summary, artifacts, numbering, captions)
    lines += _section_boxplot(models, artifacts, numbering, captions, model_name)

    if comparisons:
        if baselines:
            heading = f"## 六、与对比模型的比较（{lead:g}h，TS）"
            caption = f"{lead:g}h 时效上各模型的 TS 与本模型名次"
            note: Optional[str] = (
                "> 名次按该时效的 TS 排序（仅统计有有效值的模型）；"
                "不同数据的起报时段/时效范围要一致才可直接比较。"
            )
        else:
            heading = f"## 六、与基准 {baseline_name or '基准'} 的对比（{lead:g}h）"
            caption = f"{lead:g}h 时效上本模型与基准的 TS 及差值"
            note = None
        lines += _section_comparison(
            data, comparisons, artifacts, numbering, captions,
            lead=lead, heading=heading, caption=caption, note=note,
        )

    tail_section = "七" if comparisons else "六"
    lines.append(f"## {tail_section}、改进建议")
    lines.append("")
    for index, item in enumerate(recommendations(grades), start=1):
        lines.append(f"{index}. {item}")
    lines.append("")
    tail_section = "八" if comparisons else "七"
    lines.append(f"## {tail_section}、口径与注意事项")
    lines.append("")
    lines.append("- **值缺失的含义**：TS 为 “—” 表示该等级没有有效配对或无事件，与“评分为 0”不同；")
    lines.append("- **BIAS 不是平均误差**：它是频率偏差，>1 表示预报事件偏多，<1 表示偏少；")
    lines.append("- **跨时效不要直接平均后比较模型**：正式对比应使用共同有效样本与固定验证口径；")
    lines.append("- 结果来自统一长表（`scores.csv` / `categorical_wide.csv`），逐行可追溯到具体时效与阈值。")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 TS 结果表画成图并写成 Markdown 报告",
        epilog="示例: python -m xmetai_evaluation.visualization.ts_report "
        "--csv ts_fgvp_2025.csv --out reports/fgvp --model FGVP",
    )
    parser.add_argument("--csv", type=Path, required=True, help="TS 结果表（csv）")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    parser.add_argument("--model", default=None, help="模型名（默认取文件名）")
    parser.add_argument("--baseline-csv", type=Path, default=None, help="可选：基准模型结果表")
    parser.add_argument("--baseline-name", default=None, help="基准模型名")
    parser.add_argument("--lead-h", type=float, default=None, help="基准时效（默认取最小时效）")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    from xmetai_evaluation.visualization.precipitation_plots import PrecipitationPlotter

    plt_dpi = args.dpi
    import matplotlib.pyplot as plt

    plt.rcParams["savefig.dpi"] = plt_dpi
    model = args.model or args.csv.stem
    plotter = PrecipitationPlotter()
    artifacts = plotter.create_report(
        pd.read_csv(args.csv),
        output_dir=args.out,
        model_name=model,
        lead_h=args.lead_h,
        baseline_df=pd.read_csv(args.baseline_csv) if args.baseline_csv else None,
        baseline_name=args.baseline_name,
    )
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
