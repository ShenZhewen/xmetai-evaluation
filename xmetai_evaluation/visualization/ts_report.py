#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS 结果表 → Markdown 报告。

一句话用法::

    python -m xmetai_evaluation.visualization.ts_report --csv ts.csv --out reports/fgvp --model FGVP

报告内容（都是算出来的，不是套话）：
执行摘要（规则化诊断，逐条带数值）、逐等级表、逐时效 TS 表、可选与基准的差值表、
图表索引、以及口径说明（等权平均、BIAS 的含义、无效值的写法）。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

GRADE_ORDER_HINT = ("≥0.1", "≥4", "≥10", "≥13", "≥25", "≥50", "≥100", "≥250")


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


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
        ranked = sorted(
            (name for name, value in pool.items() if np.isfinite(value)),
            key=lambda name: pool[name],
            reverse=True,
        )
        rank = ranked.index("本模型") + 1 if "本模型" in ranked else 0
        cells = " | ".join(_fmt(values[name]) for name in names)
        lines.append(
            f"| {grade} | {_fmt(left[grade])} | {cells} | {rank}/{len(ranked) if ranked else 0} |"
        )
    return "\n".join(lines)


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
) -> Path:
    """写一份 Markdown 报告。"""
    output_path = Path(output_path)
    summary = summarize(data)
    findings = diagnose(summary)
    grades = summary["grades"]
    leads = summary["leads"]
    window = None
    if "window_h" in data.columns and data["window_h"].notna().any():
        window = float(data["window_h"].dropna().iloc[0])
    lead = float(lead_h) if lead_h is not None else leads[0]

    lines: List[str] = []
    lines.append(f"# {model_name} 降水分类检验报告")
    lines.append("")

    lines.append("## 评测对象与对比对象")
    lines.append("")
    lines.append(f"- **评测对象**：{model_name}（来源：{_source_of(sources, model_name)}）")
    compare_names = [name for name in (baselines or {})]
    if baseline_df is not None and baseline_name:
        compare_names.append(baseline_name)
    if compare_names:
        joined = "、".join(
            f"{name}（来源：{_source_of(sources, name)}）" for name in compare_names
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

    lines.append("## 二、逐降水等级表现（全时效平均）")
    lines.append("")
    lines.append(_grade_table(grades))
    lines.append("")
    lines.append("> 全时效平均为**各时效等权平均**（不是按样本数加权）；判断强降水能力请看 ≥25mm 及以上等级。")
    lines.append("")

    lines.append(f"## 三、TS 随时效变化（{leads[0]:g}–{leads[-1]:g}h）")
    lines.append("")
    lines.append(_lead_table(summary["pivot"]))
    lines.append("")

    if baselines:
        lines.append(f"## 四、与对比模型的比较（{lead:g}h，TS）")
        lines.append("")
        lines.append(_multi_baseline_table(data, baselines, lead))
        lines.append("")
        lines.append(
            "> 名次按该时效的 TS 排序（仅统计有有效值的模型）；"
            "不同数据的起报时段/时效范围要一致才可直接比较。"
        )
        lines.append("")
    elif baseline_df is not None:
        baseline_data = baseline_df
        lines.append(f"## 四、与基准 {baseline_name or '基准'} 的对比（{lead:g}h）")
        lines.append("")
        lines.append(_baseline_table(data, baseline_data, lead))
        lines.append("")

    figure_section = "五" if (baselines or baseline_df is not None) else "四"
    if artifacts:
        lines.append(f"## {figure_section}、图表")
        lines.append("")
        for name, path in artifacts.items():
            if Path(path).suffix.lower() in (".png", ".jpg", ".svg"):
                lines.append(f"![{name}]({Path(path).name})")
        lines.append("")

    tail_section = "六" if (baselines or baseline_df is not None) else "五"
    lines.append(f"## {tail_section}、改进建议")
    lines.append("")
    for index, item in enumerate(recommendations(grades), start=1):
        lines.append(f"{index}. {item}")
    lines.append("")
    tail_section = "七" if (baselines or baseline_df is not None) else "六"
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
