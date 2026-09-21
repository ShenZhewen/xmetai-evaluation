#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集合预报（ensemble）批次归档 → 多模型评估检验报告。

输入是 ``weather_rmse_<模型>_ens`` 支线的**批次归档目录**（``summarize_mode``
为 ``--summarize-ens``）：``summary.csv`` + ``batch_meta.json`` + ``<YYYYMMDD>/``。
逐日文件是集合特有的 ``crps_<日期>.csv`` / ``spread_<日期>.csv`` /
``spread_rmse_ratio_<日期>.csv``，加上集合平均场口径的
``rmse_<日期>_ensmean.csv`` / ``acc_<日期>_ensmean.csv`` /
``fa_<日期>_ensmean.csv`` / ``spectrum_<日期>_<变量>.csv``。

**与 ``_single`` 支线的三条硬差别**（踩了会静默算错，见
``references/rmse-batch-evaluation.md`` §1.3）：

1. ``rmse`` / ``acc`` / ``fa`` 读的是后缀 ``_ensmean`` 的**集合平均场**，
   不是单个成员。``det_report._pick_metric_file`` 认这个后缀，于是
   ``det_report.analyze`` 能原样吃下集合归档；但**报告口径不是一回事**，
   别拿 ``generate_det_report.py`` 出集合报告，也别拿本模块出单成员报告。
2. 这里多两个集合特有指标：CRPS 与离散度（Spread、Spread/RMSE）。
3. 对数底数是 **ln**，而 ``det_report`` 的 ``spectrum_rms`` 是 **log10**
   （两者差 ×2.3026）。本模块的 §4/§5 一律自算 ln 版，**不读**
   ``base.spectrum_rms``——读了会出现同一份报告里 §4 表头与 §4.4 表
   自相矛盾 2.3026 倍。``log_rms_ln`` 与 ``_save_figure`` 直接复用
   :mod:`visualization.wave_report`，它们本就与骨架口径对齐。

**逐日文件一律严格按名字读，不做 glob 兜底**：兜底会把
``rmse_<日期>_member_000.csv`` 这类单成员文件当成集合平均静默读进来，
而集合与单成员的 ``crps`` / ``spread`` / ``spread_rmse_ratio`` 同名，
选错文件时不会有任何报错。

**不读 ``mean_*.csv`` / ``ens_summary_*.csv``**：那些是**该模型全部日期**口径，
混进来会出现「表里 349 天、图里 27 天」的错位。

格式契约是 ``skills/xmetai-evaluation/assets/templates/weather_rmse_ens.md``：
章节用 ``# 1.``…``# 7.`` + 附录，**表不编号**，只有附录六张图带 ``图 N：``
且必须 1–6 连续（``render_ens`` 末尾自检）。骨架**不含球谐带功率**
（那是 ``weather_rmse_wave`` 的活），所以本模块也不读 ``spherical_bands_*.csv``。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.det_plots import DetPlotter, model_colors
from visualization.precipitation_plots import setup_chinese_font
from visualization.det_report import (
    Archive,
    Bundle,
    LEAD_BANDS,
    _CN_NUM,
    _bullets,
    _date_span,
    _fmt,
    _lead_span,
    _pick_metric_file,
    _read_lead_frame,
    _relative,
    _table,
    analyze,
    load_spectrum,
    paired_tests,
)
from visualization.wave_report import _save_figure, log_rms_ln


#: 综合相对 RMSE 的基线（各模型几何均值 = 100）
BASELINE = 100.0

#: 共同日期少于这个数就在 §1 / §6 / §7 点名提示检验力不足
SMALL_SAMPLE_DATES = 30

#: 纬向谱偏差的口径标签，§3 / §4 / §5 共用一套措辞
SPECTRUM_DEVIATION_LABEL = "纬向谱偏差"

#: §6 指标族名 → 是否越大越好（不在表里的族默认越小越好）
_FAMILY_DIRECTION = {"ACC": True}


# ====================================================================== 加载


def _daily_path(root: Path, stem: str, date: str) -> Path:
    """集合归档里 ``<stem>_<日期>.csv`` 的**唯一**合法路径（不兜底）。"""
    return Path(root) / date / f"{stem}_{date}.csv"


def load_daily_frames(
    root: Path, stem: str, dates: Sequence[str], name: str = ""
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """逐日 ``<stem>_<日期>.csv`` → ``(按 lead 平均的表, {日期: 原表})``。

    两份都返回：出图与总表要按 lead 平均的表，配对检验要**逐日样本**
    （先平均再检验等于把样本量从 N 天压成 1 个点）。

    Raises:
        FileNotFoundError: 某个共同日期缺这个文件——不静默跳过。
        ValueError: 各日期的列集合不一致——跨日平均会按「有几天算几天」
            静默缩样本，让同一行里两个模型的样本量不同。
    """
    frames: Dict[str, pd.DataFrame] = {}
    columns: Optional[List[str]] = None
    for date in dates:
        path = _daily_path(root, stem, date)
        if not path.is_file():
            raise FileNotFoundError(
                f"{name or root}: 缺 {path.name}（{date} 没算 {stem}？集合归档应当有）"
            )
        frame = _read_lead_frame(path)
        own = sorted(str(c) for c in frame.columns)
        if columns is None:
            columns = own
        elif own != columns:
            raise ValueError(
                f"{name or root}: {stem} 逐日文件的列不一致——"
                f"{date} 是 {own}，其余日期是 {columns}；"
                f"跨日平均会静默缩样本，不做"
            )
        frames[date] = frame
    if not frames:
        raise ValueError(f"{name or root}: 没有可读的 {stem} 逐日文件")
    stacked = pd.concat(frames.values(), axis=0)
    return stacked.groupby(level=0).mean().sort_index(), frames


def load_spread_rmse_ratio(
    root: Path, dates: Sequence[str], name: str = ""
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """逐日 Spread/RMSE → ``({日期: lead × 变量}, 来源标签)``。

    优先读归档自带的 ``spread_rmse_ratio_<日期>.csv``；**任一**共同日期缺它，
    就整批退到 ``spread_<日期>.csv ÷ rmse_<日期>_ensmean.csv`` 现算——
    不混口径（一半读归档、一半现算，两个来源的小数位与分母处理未必一致）。
    """
    if all(_daily_path(root, "spread_rmse_ratio", date).is_file() for date in dates):
        frames = {
            date: _read_lead_frame(_daily_path(root, "spread_rmse_ratio", date))
            for date in dates
        }
        return frames, "归档自带 spread_rmse_ratio_<日期>.csv"

    frames = {}
    for date in dates:
        spread_path = _daily_path(root, "spread", date)
        if not spread_path.is_file():
            raise FileNotFoundError(f"{name or root}: 缺 {spread_path.name}")
        rmse_path = _pick_metric_file(Path(root) / date, "rmse", date)
        if rmse_path is None:
            raise FileNotFoundError(f"{name or root}: {date} 目录下没有 rmse_*.csv")
        spread = _read_lead_frame(spread_path)
        rmse = _read_lead_frame(rmse_path)
        shared = [c for c in spread.columns if c in rmse.columns]
        if not shared:
            raise ValueError(
                f"{name or root}: {date} 的 spread 与 rmse 没有同名变量；"
                f"spread 有 {list(spread.columns)}，rmse 有 {list(rmse.columns)}"
            )
        frames[date] = spread[shared] / rmse[shared].replace(0.0, np.nan)
    return frames, "由 spread ÷ 集合均值 rmse 现算"


def _per_date_means(frames: Mapping[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """``{日期: lead × 变量}`` → ``{变量: 日期 → 该变量在全部 lead 上的均值}``。"""
    if not frames:
        return {}
    variables = sorted({str(c) for frame in frames.values() for c in frame.columns})
    values: Dict[str, Dict[str, float]] = {variable: {} for variable in variables}
    for date, frame in frames.items():
        for variable in variables:
            if variable not in frame.columns:
                continue
            numbers = pd.to_numeric(frame[variable], errors="coerce")
            values[variable][date] = float(numbers.mean())
    return {
        variable: pd.Series(series).sort_index()
        for variable, series in values.items()
        if series
    }


def _shared_columns(frames: Mapping[str, pd.DataFrame]) -> List[str]:
    """各模型都有的列名（字典序）。"""
    shared: Optional[set] = None
    for frame in frames.values():
        own = {str(c) for c in frame.columns}
        shared = own if shared is None else (shared & own)
    return sorted(shared or set())


def _unique_summary_value(archive: Archive, column: str) -> Optional[int]:
    """``summary.csv`` 某列的唯一数值；列缺失或有多值就返回 None。"""
    if column not in archive.summary.columns:
        return None
    values = pd.to_numeric(archive.summary[column], errors="coerce").dropna().unique()
    return int(values[0]) if len(values) == 1 else None


def _acc_per_variable(
    root: Path, dates: Sequence[str], variables: Sequence[str], name: str = ""
) -> Dict[str, pd.Series]:
    """逐日 ``acc_<日期>_ensmean.csv`` → ``{变量: 日期 → 全 lead 平均 ACC(%)}``。

    ``det_report._acc_by_date`` 只算一个变量（报告只要 ``acc_variable``），
    §6 的 ACC 指标族要逐变量，所以这里重算一遍。``analyze`` 已经在入口乘过
    ``BASELINE``，本函数同样乘——**下游不要再乘一次**。
    """
    values: Dict[str, Dict[str, float]] = {str(v): {} for v in variables}
    for date in dates:
        path = _pick_metric_file(Path(root) / date, "acc", date)
        if path is None:
            raise FileNotFoundError(f"{name or root}: {date} 目录下没有 acc_*.csv")
        raw = _read_lead_frame(path)
        for variable in variables:
            if variable not in raw.columns:
                raise ValueError(
                    f"{name or root}: {path.name} 里没有 {variable} 列；"
                    f"现有列: {list(raw.columns)}"
                )
            numbers = pd.to_numeric(raw[variable], errors="coerce")
            values[variable][date] = BASELINE * float(numbers.mean())
    return {variable: pd.Series(series).sort_index() for variable, series in values.items()}


# ================================================================== 统计检验


def _dominant(frame: pd.DataFrame, names: Sequence[str], *, higher_is_better: bool = False) -> str:
    """``日期 × 模型`` 表里显著优于**其余全部**模型的那个模型；没有就「无显著差异」。

    判定用 Holm 校正后的 p（``paired_tests`` 已经在做）。只要有一个模型压过
    其余全部，这一项就归它；否则整项记无显著差异——于是「各模型优胜项数 +
    无显著差异数 = 该项总数」，不会重复计数。
    """
    tests = paired_tests(frame, higher_is_better=higher_is_better)
    if not tests:
        return "无显著差异"
    for name in names:
        involved = [t for t in tests if name in (t.a, t.b)]
        if involved and all(t.significant and t.better == name for t in involved):
            return str(name)
    return "无显著差异"


def _family_table(
    families: Mapping[str, Mapping[str, pd.DataFrame]],
    names: Sequence[str],
) -> pd.DataFrame:
    """``{指标族: {项名: 日期 × 模型}}`` → 每族每模型的显著优胜项计数。

    Returns:
        index = 指标族名（字典序），columns = 模型名 + ``无显著差异``。
    """
    rows = {}
    for family in sorted(families):
        higher = _FAMILY_DIRECTION.get(family, False)
        counts = {str(name): 0 for name in names}
        ties = 0
        for _, frame in sorted(families[family].items()):
            winner = _dominant(frame, names, higher_is_better=higher)
            if winner == "无显著差异":
                ties += 1
            else:
                counts[winner] += 1
        rows[family] = {**counts, "无显著差异": ties}
    return pd.DataFrame(rows).T


def _log_deviation(ratio: pd.DataFrame) -> pd.Series:
    """逐 lead 的 ``|ln(ratio)|`` 在变量上的平均（ratio ≤ 0 的样本丢弃）。"""
    return np.log(ratio.where(ratio > 0)).abs().mean(axis=1)


def _log_deviation_scalar(ratio: pd.DataFrame) -> float:
    """跨变量 × lead 的 ``mean|ln(Spread/RMSE)|``（§4 表头的「Spread偏差(绝对对数)」）。"""
    values = _log_deviation(ratio).dropna()
    return float(values.mean()) if not values.empty else float("nan")


# ================================================================== 汇总分析


class EnsBundle(NamedTuple):
    """``det_report.Bundle`` + 集合特有量。

    第一个字段是 ``base``，``names`` / ``dates`` / ``archives`` 等经 property
    透传，于是各 ``_section_*`` 里 ``bundle.xxx`` 的写法和 ``Bundle`` 一致。
    """

    base: Bundle
    members: Dict[str, Optional[int]]
    leads: Dict[str, Optional[int]]
    member_label: str
    crps_by_lead: Dict[str, pd.DataFrame]  # 模型 -> lead × 变量
    crps_per_date: Dict[str, Dict[str, pd.Series]]  # 模型 -> 变量 -> 日期
    crps_variables: List[str]
    crps_mean: pd.Series  # 模型 -> CRPS
    acc_variables: List[str]
    acc_per_date: Dict[str, Dict[str, pd.Series]]
    spread_ratio_by_lead: Dict[str, pd.DataFrame]  # 模型 -> lead × 变量
    spread_ratio_per_date: Dict[str, Dict[str, pd.Series]]
    spread_variables: List[str]
    spread_ratio_source: str
    spread_log_deviation: pd.Series  # 模型 -> mean|ln(Spread/RMSE)|
    spectrum_by_variable: Dict[str, Dict[str, pd.DataFrame]]  # 变量 -> 模型 -> 波数×{pred,obs}
    spectrum_deviation: pd.DataFrame  # 谱变量 × 模型（%）
    spectrum_deviation_scalar: pd.Series  # 模型 -> %（谱变量平均）
    family_wins: pd.DataFrame  # 指标族 × (模型..., 无显著差异)
    family_totals: Dict[str, int]  # 指标族 -> 项数
    small_sample: bool

    # --- 透传，让 _section_* 能把 EnsBundle 当 Bundle 用 ---
    @property
    def names(self) -> List[str]:
        return self.base.names

    @property
    def dates(self) -> List[str]:
        return self.base.dates

    @property
    def archives(self) -> Dict[str, Archive]:
        return self.base.archives

    @property
    def common_variables(self) -> List[str]:
        return self.base.common_variables

    @property
    def fa_variables(self) -> List[str]:
        return self.base.fa_variables

    @property
    def spectrum_variables(self) -> List[str]:
        return self.base.spectrum_variables

    @property
    def acc_variable(self) -> str:
        return self.base.acc_variable


def _common_dates_or_raise(archives: Sequence[Archive]) -> List[str]:
    """共同日期 < 2 时点名报错：说清谁和谁没有交集，而不是只说「共同日期不够」。"""
    sets = {a.name: set(a.summary["init_date"]) for a in archives}
    shared = set.intersection(*sets.values()) if sets else set()
    if len(shared) >= 2:
        return sorted(shared)

    spans = "；".join(
        f"{name} {_date_span(sorted(own))}（{len(own)} 天）" for name, own in sets.items()
    )
    order = list(sets)
    pairs = [
        f"{a}∩{b} {len(sets[a] & sets[b])} 天"
        for index, a in enumerate(order)
        for b in order[index + 1:]
    ]
    raise ValueError(
        f"共同日期只有 {len(shared)} 天，做不了配对检验。\n"
        f"  各模型自有的起报日期：{spans}\n"
        f"  两两交集：{'；'.join(pairs)}\n"
        f"  没有共同日期的模型之间无法比较——先补齐数据，"
        f"或把它们拆成有交集的两组、各出一份报告。"
    )


def analyze_ens(archives: Sequence[Archive]) -> EnsBundle:
    """算齐集合报告要的一切。

    Raises:
        ValueError: 共同日期 < 2（点名说清谁和谁没交集）、缺逐日文件、
            各日期的列集合不一致。
    """
    _common_dates_or_raise(archives)
    base = analyze(archives)
    names = base.names
    dates = base.dates

    members = {a.name: _unique_summary_value(a, "n_members") for a in archives}
    leads = {a.name: _unique_summary_value(a, "n_leads") for a in archives}

    # --- 集合特有指标：CRPS 与离散度（逐日读，不用 summary 里 precomputed 的列）---
    crps_by_lead: Dict[str, pd.DataFrame] = {}
    crps_per_date: Dict[str, Dict[str, pd.Series]] = {}
    spread_ratio_by_lead: Dict[str, pd.DataFrame] = {}
    spread_ratio_per_date: Dict[str, Dict[str, pd.Series]] = {}
    spread_sources = set()
    for archive in archives:
        by_lead, frames = load_daily_frames(archive.root, "crps", dates, archive.name)
        crps_by_lead[archive.name] = by_lead
        crps_per_date[archive.name] = _per_date_means(frames)

        ratio_frames, source = load_spread_rmse_ratio(archive.root, dates, archive.name)
        spread_sources.add(source)
        stacked = pd.concat(ratio_frames.values(), axis=0)
        spread_ratio_by_lead[archive.name] = stacked.groupby(level=0).mean().sort_index()
        spread_ratio_per_date[archive.name] = _per_date_means(ratio_frames)

    crps_variables = _shared_columns(crps_by_lead)
    spread_variables = _shared_columns(spread_ratio_by_lead)
    if not crps_variables:
        raise ValueError("这些模型的 crps 逐日文件没有共享变量，算不出 CRPS 指标")
    if not spread_variables:
        raise ValueError("这些模型的离散度比值没有共享变量，算不出 Spread/RMSE")

    # --- ACC 逐变量（§6 的 ACC 指标族要逐变量，base 只给 acc_variable）---
    acc_columns = _shared_columns(base.acc_by_lead)
    acc_per_date = {
        archive.name: _acc_per_variable(archive.root, dates, acc_columns, archive.name)
        for archive in archives
    }

    # --- 纬向谱：逐变量、逐模型，口径与 §4.4 / 附 6 一致（ln）---
    spectrum_by_variable: Dict[str, Dict[str, pd.DataFrame]] = {}
    spectrum_rows: Dict[str, Dict[str, float]] = {name: {} for name in names}
    for variable in base.spectrum_variables:
        per_model: Dict[str, pd.DataFrame] = {}
        for archive in archives:
            pred, obs = load_spectrum(archive.root, dates, variable, archive.name)
            per_model[archive.name] = pd.DataFrame(
                {"pred": pred.mean(axis=1), "obs": obs.mean(axis=1)}
            )
            spectrum_rows[archive.name][variable] = float(log_rms_ln(pred, obs).mean())
        spectrum_by_variable[variable] = per_model

    # pd.DataFrame({模型: {谱变量: 值}}) → index = 谱变量、columns = 模型
    spectrum_deviation = pd.DataFrame(spectrum_rows).reindex(base.spectrum_variables)
    spectrum_deviation_scalar = spectrum_deviation.mean(axis=0)

    crps_mean = pd.Series(
        {name: float(crps_by_lead[name].mean(axis=1).mean()) for name in names}
    )
    spread_log_deviation = pd.Series(
        {name: _log_deviation_scalar(spread_ratio_by_lead[name]) for name in names}
    )

    # --- §6 指标族：项 = 变量，样本 = 共同日期（逐日配对）---
    families: Dict[str, Dict[str, pd.DataFrame]] = {
        "RMSE": dict(base.per_variable_rmse),
        "CRPS": {
            variable: pd.DataFrame({name: crps_per_date[name][variable] for name in names})
            for variable in crps_variables
        },
        "ACC": {
            variable: pd.DataFrame({name: acc_per_date[name][variable] for name in names})
            for variable in sorted(acc_per_date[names[0]])
        },
        "FA ratio 偏差": {
            variable: pd.DataFrame(
                {name: (base.fa_ratio_by_lead[name][variable] - 1.0).abs() for name in names}
            )
            for variable in base.fa_variables
        },
        "Spread/RMSE 距1": {
            variable: pd.DataFrame(
                {name: (spread_ratio_per_date[name][variable] - 1.0).abs() for name in names}
            )
            for variable in spread_variables
        },
    }
    # 无该项数据的指标族不出现在表里（骨架 §6 的要求）
    families = {
        family: items
        for family, items in families.items()
        if items and all(len(frame.columns) == len(names) for frame in items.values())
    }
    family_wins = _family_table(families, names)
    family_totals = {family: len(items) for family, items in families.items()}

    distinct_members = sorted({m for m in members.values() if m is not None})
    if not distinct_members:
        member_label = "—"
    elif len(distinct_members) == 1:
        member_label = str(distinct_members[0])
    else:
        member_label = "/".join(str(m) for m in distinct_members)

    return EnsBundle(
        base=base,
        members=members,
        leads=leads,
        member_label=member_label,
        crps_by_lead=crps_by_lead,
        crps_per_date=crps_per_date,
        crps_variables=crps_variables,
        crps_mean=crps_mean,
        acc_variables=sorted(acc_per_date[names[0]]),
        acc_per_date=acc_per_date,
        spread_ratio_by_lead=spread_ratio_by_lead,
        spread_ratio_per_date=spread_ratio_per_date,
        spread_variables=spread_variables,
        spread_ratio_source="；".join(sorted(spread_sources)),
        spread_log_deviation=spread_log_deviation,
        spectrum_by_variable=spectrum_by_variable,
        spectrum_deviation=spectrum_deviation,
        spectrum_deviation_scalar=spectrum_deviation_scalar,
        family_wins=family_wins,
        family_totals=family_totals,
        small_sample=len(dates) < SMALL_SAMPLE_DATES,
    )


# ====================================================================== 出图


def _spectrum_panels(
    out_dir: Path,
    filename: str,
    spectra: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    variables: Sequence[str],
    names: Sequence[str],
    suptitle: str,
) -> Path:
    """一个谱变量一个双对数子图，子图内 N 条模型预测谱 + 一条共享观测谱。

    ``DetPlotter.plot_model_spectrum`` 只收**单个**变量、且自建一张图，
    产不出骨架「附 6」要的「{谱变量数}个变量{模型数}模型曲线」单张 PNG，
    所以这里自绘。观测谱只画一次——所有模型跑的是同一批共同日期、同一个
    ERA5 观测，各模型的 ``obs`` 列本就逐点相同。
    """
    setup_chinese_font()
    variables = [str(v) for v in variables]
    if not variables:
        raise ValueError("没有可画的谱变量")
    colors = model_colors([str(n) for n in names])

    ncols = max(1, min(2, len(variables)))
    nrows = int(math.ceil(len(variables) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    axes = axes.ravel()

    for index, variable in enumerate(variables):
        axis = axes[index]
        observed_drawn = False
        for name in names:
            frame = spectra.get(variable, {}).get(name)
            if frame is None or frame.empty:
                continue
            data = frame.copy()
            data.index = pd.to_numeric(data.index, errors="coerce")
            data = data[data.index.notna() & (data.index > 0)].sort_index()
            pred = pd.to_numeric(data["pred"], errors="coerce")
            axis.plot(pred.index, pred.values, color=colors[str(name)],
                      linewidth=1.6, label=str(name))
            if not observed_drawn:
                obs = pd.to_numeric(data["obs"], errors="coerce")
                axis.plot(obs.index, obs.values, color="#000000", linestyle="--",
                          linewidth=1.4, label="观测")
                observed_drawn = True
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(variable, fontsize=11)
        axis.set_xlabel("纬向波数 k", fontsize=9)
        axis.set_ylabel("功率", fontsize=9)
        axis.grid(alpha=0.3, which="both", linewidth=0.6)
        axis.tick_params(labelsize=8)
    for axis in axes[len(variables):]:
        axis.axis("off")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(suptitle, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(figure, Path(out_dir) / filename)


# ====================================================================== 渲染


def _cn(number: int) -> str:
    """中文数字（骨架要求 ``{模型数}`` 填「两」而不是「2」）。"""
    return _CN_NUM.get(int(number), str(int(number)))


def _named(names: Sequence[str]) -> str:
    return "、".join(str(n) for n in names)


def _lead_bands(leads: Sequence[float]) -> List[Tuple[str, List[float]]]:
    """按 ``LEAD_BANDS`` 切实际存在的 lead，段名用段内首末值现写（空段不输出）。"""
    finite = [float(x) for x in leads if math.isfinite(float(x))]
    bands: List[Tuple[str, List[float]]] = []
    for low, high in LEAD_BANDS:
        subset = [lead for lead in finite if low < lead <= high]
        if subset:
            bands.append((f"{min(subset):g}–{max(subset):g}h", subset))
    return bands


def _row_relative(values: Mapping[str, float]) -> pd.Series:
    """把一行（模型 → 值）按各模型几何均值重标成 100 基线。"""
    frame = pd.DataFrame({str(k): [v] for k, v in values.items()}, index=["row"])
    return _relative(frame).iloc[0]


def _band_values(frame: pd.DataFrame, subset: Sequence[float]) -> pd.DataFrame:
    """取 ``frame`` 中 index 落在 ``subset``（一组 lead）里的行。"""
    return frame.loc[frame.index.isin(list(subset))]


def _band_relative(base: Bundle, subset: Sequence[float], names: Sequence[str]) -> pd.Series:
    """一个时效段的综合相对 RMSE（以段内各模型的几何均值为 100）。

    §5 的表和 §5.1 的结论**必须**都走这个函数。曾经两处各算各的——表里重标成
    100 基线、结论里直接印原始值，同一段同一模型出现 97.242 与 97.518 两个数，
    而骨架那句「综合RMSE以{模型数}模型的几何均值为100」只对前者成立。
    """
    return _row_relative({
        name: float(_band_values(base.lead_relative, subset)[name].mean())
        for name in names
    })


def _failures_phrase(bundle: EnsBundle) -> str:
    broken = {
        name: len(bundle.archives[name].meta.get("failures") or [])
        for name in bundle.names
    }
    broken = {name: count for name, count in broken.items() if count}
    if not broken:
        return "所有模型的 batch_meta 都记录 n_ok = n_dates，没有失败起报点"
    return "；".join(f"{name} 有 {count} 个起报点失败" for name, count in broken.items())


def _section_conclusion(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    best = base.relative_composite.idxmin()
    items = [
        f"{_cn(len(names))}个集合归档完整性检查通过，共同日期 {len(bundle.dates)} 天"
        f"（{_date_span(bundle.dates)}）；{_failures_phrase(bundle)}。",
        f"综合相对 RMSE 最好 {best}（{_fmt(base.relative_composite[best])}），"
        f"最差 {base.relative_composite.idxmax()}"
        f"（{_fmt(base.relative_composite.max())}）；"
        f"ACC 最高 {base.acc_mean.idxmax()}（{_fmt(base.acc_mean.max())}%）。",
        f"CRPS 最低 {bundle.crps_mean.idxmin()}（{_fmt(bundle.crps_mean.min())}），"
        f"最高 {bundle.crps_mean.idxmax()}（{_fmt(bundle.crps_mean.max())}）；"
        f"离散度校准最好 {bundle.spread_log_deviation.idxmin()}"
        f"（平均 |ln(Spread/RMSE)| = {_fmt(bundle.spread_log_deviation.min(), 4)}）。",
        f"幅度真实性（全程 FA ratio 绝对偏差）最好 {base.fa_bias.idxmin()}"
        f"（{_fmt(base.fa_bias.min(), 4)} pp），最差 {base.fa_bias.idxmax()}"
        f"（{_fmt(base.fa_bias.max(), 4)} pp）。",
        f"{SPECTRUM_DEVIATION_LABEL}（100×RMS[ln(pred/obs)]，"
        f"{len(base.spectrum_variables)}个谱变量平均）最小 "
        f"{bundle.spectrum_deviation_scalar.idxmin()}"
        f"（{_fmt(bundle.spectrum_deviation_scalar.min(), 2)}），最大 "
        f"{bundle.spectrum_deviation_scalar.idxmax()}"
        f"（{_fmt(bundle.spectrum_deviation_scalar.max(), 2)}）。",
        f"集合配置：成员数 {bundle.member_label}；"
        f"逐日指标覆盖 RMSE（{len(base.common_variables)}个变量）、"
        f"CRPS（{_named(bundle.crps_variables)}）、"
        f"ACC（{_named(bundle.acc_variables)}）、"
        f"FA（{len(base.fa_variables)}个变量）与离散度"
        f"（{len(bundle.spread_variables)}个变量）。",
        f"异常变量检查：{base.anomaly_model} 的 {base.anomaly_variable} RMSE 为其余模型中位数的 "
        f"{_fmt(base.anomaly_ratio, 2)} 倍"
        + ("，超出 3× 阈值，详见 4.2。" if base.anomaly_ratio >= 3.0 else "，未超 3× 阈值。"),
        f"选型取向：整体精度看综合相对 RMSE（{best}），"
        f"集合可靠性看 CRPS（{bundle.crps_mean.idxmin()}）与离散度"
        f"（{bundle.spread_log_deviation.idxmin()}），"
        f"幅度与谱真实性看 FA 偏差（{base.fa_bias.idxmin()}）与"
        f"{SPECTRUM_DEVIATION_LABEL}（{bundle.spectrum_deviation_scalar.idxmin()}），"
        f"几者不一定指向同一模型。",
    ]
    if bundle.small_sample:
        items.append(
            f"共同日期只有 {len(bundle.dates)} 天（少于 {SMALL_SAMPLE_DATES} 天），"
            f"配对 Wilcoxon 检验的检验力很低，「无显著差异」多半是样本不够、"
            f"而不是两者真的等价；本报告的排名只作方向性参考。"
        )
    return ["# 1. 结论摘要", ""] + [f"• {item}" for item in items]


def _section_samples(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    lines = ["# 2. 样本口径与数据质量", "", "**样本口径**", ""]

    rows = []
    for name in sorted(names):
        archive = base.archives[name]
        meta = archive.meta
        own = set(archive.summary["init_date"])
        common = own & set(base.dates)
        n_dates = meta.get("n_dates")
        n_ok = meta.get("n_ok")
        failed = len(meta.get("failures") or [])
        member = bundle.members.get(name)
        lead = bundle.leads.get(name)
        rows.append([
            name,
            str(n_dates) if n_dates is not None else "—",
            str(n_ok) if n_ok is not None else "—",
            str(len(archive.dates)),
            str(len(common)),
            str(member) if member is not None else "—",
            str(lead) if lead is not None else "—",
            f"{failed} 个起报点失败" if failed else "无",
        ])
    any_failed = sum(1 for name in names if (base.archives[name].meta.get("failures") or []))
    rows.append([
        "共同有效样本", "—", str(len(base.dates)), "—", str(len(base.dates)),
        bundle.member_label,
        str(len(base.rmse_by_lead[names[0]].index)),
        f"{any_failed} 个模型有失败起报点" if any_failed else "无",
    ])
    lines += _table(
        ["**模型**", "**请求日期**", "**成功日期**", "**日期目录**",
         "**有效共同日期**", "**成员数**", "**Lead数**", "**失败情况**"],
        rows,
    )
    lines += [""]
    lines += _bullets([
        f"正文与附录统一使用 {len(base.dates)} 个共同日期（{_date_span(base.dates)}）；"
        f"各模型的非共同日期不参与任何统计与曲线，因此各表样本量一致、可直接横向比较。",
        f"「请求日期」「成功日期」取自各模型 ``batch_meta.json`` 的 "
        f"``n_dates`` / ``n_ok``，「日期目录」是磁盘上真实存在的 ``<YYYYMMDD>/`` 个数。",
        f"离散度比值来源：{bundle.spread_ratio_source}。",
    ] + _quality_notes(bundle))
    return lines


def _quality_notes(bundle: EnsBundle) -> List[str]:
    """§2 的数据质量逐条说明。"""
    base = bundle.base
    notes: List[str] = []

    for name in sorted(bundle.names):
        meta = base.archives[name].meta
        n_dates = meta.get("n_dates")
        n_ok = meta.get("n_ok")
        if n_dates is None or n_ok is None:
            continue
        ratio = meta.get("missing_ratio")
        percent = _fmt(BASELINE * float(ratio), 2) if ratio is not None else "—"
        tail = "，**已超过** batch 配置的缺失阈值" if meta.get("missing_threshold_exceeded") \
            else "，未超过缺失阈值"
        notes.append(
            f"{name}：请求 {n_dates} 个起报日期、成功 {n_ok} 个"
            f"（缺失率 {percent}%）{tail}；未成功的起报点不进入任何统计。"
        )

    declared = {
        name: (base.archives[name].meta.get("options") or {}).get("var_metrics") or {}
        for name in sorted(bundle.names)
    }
    if any(declared.values()):
        counts = "、".join(f"{name} {len(v)} 个" for name, v in declared.items())
        notes.append(
            f"变量覆盖按各模型 ``batch_meta.json → options.var_metrics`` 的声明口径："
            f"{counts}。报告只比较共同日期上**都真的算出来**的变量"
            f"（RMSE {len(base.common_variables)} 个：{_named(base.common_variables)}；"
            f"ACC {len(bundle.acc_variables)} 个：{_named(bundle.acc_variables)}；"
            f"FA {len(base.fa_variables)} 个：{_named(base.fa_variables)}）。"
        )
        gaps = sorted({
            f"{name}.{variable}"
            for name, var_metrics in declared.items()
            for variable in var_metrics
            if variable not in base.common_variables
        })
        if gaps:
            notes.append(
                f"声明了 RMSE 但没进入本次共同比较的变量有 {len(gaps)} 个"
                f"（如 {'、'.join(gaps[:4])}）——这是各批 ``variables`` / ``var_metrics`` "
                f"配置不同导致的**口径差异**，不是数据缺陷；"
                f"要评这些变量得先把各批配置对齐后重跑。"
            )

    fa_gap = sorted(set(base.common_variables) - set(base.fa_variables))
    notes.append(
        f"指标列数在各模型间不完全一致也是同一原因：CRPS 只有 "
        f"{_named(bundle.crps_variables)}；ACC 只有 {_named(bundle.acc_variables)}"
        + (f"；FA 没有 {_named(fa_gap)}" if fa_gap else "")
        + f"——集合归档的 ``crps`` / ``fa`` 是逐变量配置决定算不算的。"
    )

    if len({m for m in bundle.members.values() if m is not None}) > 1:
        detail = "、".join(f"{name} {bundle.members[name]} 成员" for name in sorted(bundle.names))
        notes.append(
            f"各模型成员数不一致（{detail}）——CRPS 与离散度对成员数敏感，"
            f"成员数不同的模型之间比较这两个指标要格外小心。"
        )
    else:
        notes.append(
            f"各模型成员数一致（{bundle.member_label} 成员），"
            f"CRPS 与离散度在成员数口径上可直接横向比较。"
        )

    if bundle.small_sample:
        notes.append(
            f"共同日期仅 {len(base.dates)} 天，低于 {SMALL_SAMPLE_DATES} 天的经验下限，"
            f"配对检验与分时效段均值都不稳定，**不要**据此写结论。"
        )
    return notes


def _section_metrics(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    first, last, count = _lead_span(base.rmse_by_lead[bundle.names[0]].index)
    lines = ["# 3. 评估指标与解释", "", "**指标口径**", ""]

    rows = [
        ["ACC", "集合平均场与观测的距平相关系数，逐日先按 lead 平均再跨日平均（百分数）",
         "越大越好"],
        ["CRPS", "连续分级概率评分，集合分布与观测的差距，越低越好", "越小越好"],
        ["FA ratio 偏差",
         f"Forecast Activity 的 pred/obs 比值，全程 {first}–{last}h 平均后取 |ratio−1|×100（pp）",
         "越小越好"],
        ["RMSE", "集合平均场与观测的均方根误差", "越小越好"],
        ["Spread/RMSE", "集合成员标准差 ÷ 集合平均场 RMSE，理想值为 1", "越接近 1 越好"],
        ["Spread偏差(绝对对数)",
         "跨变量与 lead 的 mean|ln(Spread/RMSE)|，与「距1」是同一直觉的对数口径",
         "越小越好"],
        ["综合相对RMSE",
         f"各变量 RMSE 相对{len(bundle.names)}模型几何均值的 100 倍，再跨变量等权平均",
         "越小越好"],
        [SPECTRUM_DEVIATION_LABEL,
         "100 × 各谱变量 RMS_k[ln(pred/obs)] 的平均，k 为纬向波数 1–720", "越小越好"],
    ]
    rows.sort(key=lambda row: row[0])
    lines += _table(["**指标**", "**定义/口径**", "**方向**"], rows)

    lines += [
        "",
        f"报告类型为集合预报（ensemble，{bundle.member_label} 成员）。"
        f"RMSE、ACC、FA 一律是**集合平均场**口径（逐日 ``*_ensmean.csv``），不是单个成员；"
        f"CRPS 与离散度用的是整个成员集合；两者不可混读。",
        "",
        f"所有对数一律取**自然对数 ln**。本报告的「{SPECTRUM_DEVIATION_LABEL}」与 "
        f"``weather_rmse_single`` 报告里的「频谱对数 RMS」**不同源**"
        f"（那边是 log10，数值相差 2.3026 倍），跨报告比大小前先统一底数。",
        "",
        f"逐日指标覆盖 {count} 个 lead（{first}–{last}h）；FA 只做全程统计、不分时效段"
        f"（避免与降水二分类的 False Alarm 混淆），"
        f"RMSE、CRPS、ACC 与 Spread/RMSE 参与分时效分析。",
        "",
        f"显著性一律用共同日期（{len(base.dates)} 天）上的逐日配对 Wilcoxon 符号秩检验，"
        f"多重比较用 Holm 逐步校正，Holm p < 0.05 才记「显著」。"
        f"「显著优胜」指该模型在同一项上显著优于其余**全部**模型。",
    ]
    return lines


def _section_overall(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    order = base.relative_composite.sort_values().index.tolist()
    rank = base.relative_composite.rank(method="min")

    lines = ["# 4. 总体评估结果", "", f"**{_cn(len(names))}模型总体指标**", ""]
    rows = []
    for name in order:
        rows.append([
            f"{name}（第{int(rank[name])}）",
            _fmt(base.relative_composite[name]),
            _fmt(bundle.crps_mean[name]),
            _fmt(base.acc_mean[name]),
            _fmt(bundle.spread_log_deviation[name], 4),
            _fmt(base.fa_bias[name], 4),
            _fmt(bundle.spectrum_deviation_scalar[name], 2),
        ])
    lines += _table(
        ["**模型（相对排序）**", "**综合相对RMSE**", "**CRPS**", "**ACC**",
         "**Spread偏差(绝对对数)**", "**FA偏差(全程)**", "**纬向谱偏差**"],
        rows,
    )

    lines += ["", "## 4.1 准确度与可靠性分项结论", ""]
    lines += _bullets(_overall_notes(bundle))

    lines += ["", "## 4.2 逐变量RMSE", "", "**分变量集合均值RMSE**", ""]
    rmse_rows = []
    for variable in base.common_variables:
        frame = base.per_variable_rmse[variable]
        rmse_rows.append(
            [variable]
            + [_fmt(frame[name].mean(), 4) for name in names]
            + [_dominant(frame, names)]
        )
    lines += _table(["**变量**"] + list(names) + ["**显著优胜**"], rmse_rows)

    lines += ["", "## 4.3 离散度校准", "", "**分变量平均 Spread/RMSE 与距 1 偏差**", ""]
    spread_rows = []
    for variable in bundle.spread_variables:
        ratio = pd.DataFrame(
            {name: bundle.spread_ratio_by_lead[name][variable] for name in names}
        )
        mean = ratio.mean(axis=0)
        linear = (ratio - 1.0).abs().mean(axis=0)
        spread_rows.append(
            [variable]
            + [f"{name} {_fmt(mean[name], 4)}" for name in names]
            + [f"{name} {_fmt(linear[name], 4)}" for name in names]
            + [str(linear.idxmin())]
        )
    lines += _table(
        ["**变量**"]
        + [f"**{name} 平均Spread/RMSE**" for name in names]
        + [f"**{name} 距1**" for name in names]
        + ["**更接近1**"],
        spread_rows,
    )
    lines += [""]
    lines += _bullets(_spread_notes(bundle))

    lines += ["", "## 4.4 全程FA ratio与谱结构", ""]
    first, last, _ = _lead_span(base.rmse_by_lead[names[0]].index)
    lines.append(
        f"FA仅指Forecast Activity，全程{first}–{last}h统计，不进行时效分段，"
        f"以避免与降水False Alarm混淆。"
    )
    lines += ["", "**全程FA ratio绝对偏差**", ""]
    fa_rows = []
    for variable in base.fa_variables:
        values = {
            name: BASELINE * float((base.fa_ratio_by_lead[name][variable] - 1.0).abs().mean())
            for name in names
        }
        fa_rows.append(
            [variable]
            + [f"{name} {_fmt(values[name], 4)}" for name in names]
            + [min(values, key=lambda key: values[key])]
        )
    lines += _table(
        ["**变量**"]
        + [f"**{name} 平均绝对ratio偏差**" for name in names]
        + ["**更接近1**"],
        fa_rows,
    )

    lines += ["", "**纬向谱（越大表示与观测谱的幅度结构偏差越大）**", ""]
    spectrum_rows = []
    for variable in base.spectrum_variables:
        row = bundle.spectrum_deviation.loc[variable]
        spectrum_rows.append(
            [variable]
            + [f"{name} {_fmt(row[name], 2)}" for name in names]
            + [str(row.idxmin())]
        )
    lines += _table(
        ["**谱变量**"]
        + [f"**{name} 纬向谱绝对对数偏差**" for name in names]
        + ["**更优**"],
        spectrum_rows,
    )
    return lines


def _overall_notes(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    best = base.relative_composite.idxmin()
    worst = base.relative_composite.idxmax()
    items = [
        f"综合相对 RMSE 以{_cn(len(bundle.names))}模型的几何均值为 100："
        f"{best} 最低（{_fmt(base.relative_composite[best])}），"
        f"{worst} 最高（{_fmt(base.relative_composite[worst])}），"
        f"两者相差 {_fmt(base.relative_composite[worst] - base.relative_composite[best])}。",
        f"CRPS 最低 {bundle.crps_mean.idxmin()}（{_fmt(bundle.crps_mean.min())}），"
        f"最高 {bundle.crps_mean.idxmax()}（{_fmt(bundle.crps_mean.max())}）；"
        f"CRPS 同时惩罚偏差与离散度不足，是与 RMSE 相互印证的一条独立证据。",
        f"ACC 最高 {base.acc_mean.idxmax()}（{_fmt(base.acc_mean.max())}%），"
        f"最低 {base.acc_mean.idxmin()}（{_fmt(base.acc_mean.min())}%）。",
        f"离散度平均绝对对数偏差最小 {bundle.spread_log_deviation.idxmin()}"
        f"（{_fmt(bundle.spread_log_deviation.min(), 4)}），"
        f"最大 {bundle.spread_log_deviation.idxmax()}"
        f"（{_fmt(bundle.spread_log_deviation.max(), 4)}）；"
        f"越接近 0 表示集合离散度与自身误差越匹配。",
        f"FA ratio 绝对偏差最小 {base.fa_bias.idxmin()}"
        f"（{_fmt(base.fa_bias.min(), 4)} pp），最大 {base.fa_bias.idxmax()}"
        f"（{_fmt(base.fa_bias.max(), 4)} pp）；FA 只反映活动幅度是否被系统性高估或低估，"
        f"与 RMSE 好坏无关。",
        f"{SPECTRUM_DEVIATION_LABEL}最小 {bundle.spectrum_deviation_scalar.idxmin()}"
        f"（{_fmt(bundle.spectrum_deviation_scalar.min(), 2)}），"
        f"最大 {bundle.spectrum_deviation_scalar.idxmax()}"
        f"（{_fmt(bundle.spectrum_deviation_scalar.max(), 2)}）；"
        f"数值越大表示预测谱的幅度结构偏离观测谱越多。",
    ]
    if best != bundle.crps_mean.idxmin():
        items.append(
            f"口径分歧：综合相对 RMSE 最好的是 {best}，而 CRPS 最低的是 "
            f"{bundle.crps_mean.idxmin()}——集合平均场精度与集合分布质量不是同一件事，"
            f"选型时要说明以哪一个为准。"
        )
    if bundle.small_sample:
        items.append(
            f"共同日期只有 {len(bundle.dates)} 天，上述排序的方向性高于显著性，不要当成定论。"
        )
    return items


def _spread_notes(bundle: EnsBundle) -> List[str]:
    items = []
    for name in bundle.names:
        ratio = bundle.spread_ratio_by_lead[name]
        values = pd.to_numeric(ratio.stack(), errors="coerce").dropna()
        if values.empty:
            continue
        under = BASELINE * float((values < 1.0).mean())
        median = float(values.median())
        # 这句判语必须跟着数字走：离散度偏小是「过于自信」，偏大是「过于保守」。
        # 写死成前者时，中位数 1.32、0% 低于 1 的模型也会被说成「过于自信」。
        verdict = "离散度偏小、集合过于自信" if median < 1.0 else "离散度偏大、集合偏保守"
        if 1.0 < under < 99.0:
            # 中位数只说多数在哪一侧；两侧都有样本时不能读成单向系统性偏差。
            verdict += "，但两侧都有样本，不是单向系统性偏差"
        items.append(
            f"{name}：Spread/RMSE 跨变量 × lead 的中位数为 {_fmt(median, 4)}，"
            f"{_fmt(under, 2)}% 的样本低于 1（{verdict}）。"
        )
    items.append(
        "「平均Spread/RMSE」是逐 lead 比值先跨日期平均、再在 lead 上平均；"
        "「距1」是这条逐 lead 平均曲线在每个 lead 上离 1 的距离的平均，"
        "衡量整体偏移幅度，**不区分正负**——上下波动会被抵消掉一部分。"
    )
    items.append(
        f"要看不会抵消的波动幅度，用第 4 节表头与第 5 节的「Spread偏差(绝对对数)」"
        f"（mean|ln(Spread/RMSE)|）；它对小离散度的欠校准更敏感，"
        f"与本节的线性距 1 互补而不重复。"
    )
    return items


def _section_lead(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    bands = _lead_bands(list(base.rmse_by_lead[names[0]].index))

    lines = [
        "# 5. 分时效表现", "",
        f"分时效分析只覆盖RMSE、CRPS、ACC和Spread/RMSE；FA不参与分段。"
        f"综合RMSE以{_cn(len(names))}模型的几何均值为100。",
        "",
        f"**分时效段{_cn(len(names))}模型指标**",
        "",
    ]
    rows = []
    for label, subset in bands:
        relative = _band_relative(base, subset, names)
        for name in sorted(names):
            crps = _band_values(bundle.crps_by_lead[name], subset)
            acc = _band_values(base.acc_by_lead[name], subset)
            ratio = _band_values(bundle.spread_ratio_by_lead[name], subset)
            rows.append([
                label, name, _fmt(relative[name]),
                _fmt(crps.mean(axis=1).mean()),
                _fmt(acc.mean(axis=1).mean()),
                _fmt(_log_deviation(ratio).mean(), 4),
            ])
    lines += _table(
        ["**时效**", "**模型**", "**相对综合RMSE**", "**CRPS**",
         "**ACC**", "**Spread绝对对数偏差**"],
        rows,
    )
    lines += ["", "## 5.1 时效结论", ""]
    lines += _bullets(_lead_notes(bundle, bands))
    return lines


def _lead_notes(bundle: EnsBundle, bands: Sequence[Tuple[str, List[float]]]) -> List[str]:
    base = bundle.base
    names = bundle.names
    items = []
    for label, subset in bands:
        values = _band_relative(base, subset, names)
        best, worst = values.idxmin(), values.idxmax()
        items.append(
            f"{label}：综合相对 RMSE 最低 {best}（{_fmt(values[best])}），"
            f"最高 {worst}（{_fmt(values[worst])}）。"
        )

    finite = [float(x) for x in base.rmse_by_lead[names[0]].index if math.isfinite(float(x))]
    if finite:
        middle = (min(finite) + max(finite)) / 2.0
        drop = {}
        for name in names:
            acc = base.acc_by_lead[name]
            early = float(acc[acc.index <= middle].mean(axis=1).mean())
            late = float(acc[acc.index > middle].mean(axis=1).mean())
            drop[name] = early - late
        steepest = max(drop, key=lambda key: drop[key])
        flattest = min(drop, key=lambda key: drop[key])
        items.append(
            f"以 {min(finite):g}–{max(finite):g}h 的中点为界，ACC 从短时效到长时效"
            f"掉得最多的是 {steepest}（{_fmt(drop[steepest])} 个百分点），"
            f"掉得最少的是 {flattest}（{_fmt(drop[flattest])} 个百分点）。"
        )

    saturated = [
        name for name in names
        if float(base.acc_by_lead[name].iloc[0].mean()) >= 99.0
    ]
    if saturated:
        items.append(
            f"{_named(saturated)} 在首时效的 ACC 已 ≥ 99%，短时效上 ACC 接近饱和、"
            f"区分度低，短时效的模型差异要看 RMSE 与 CRPS。"
        )
    items.append(
        "各时效段的相对综合 RMSE 都以**该段内**各模型的几何均值为 100，"
        "因此只能在同一行里横向比模型，不能跨行比大小。"
    )
    return items


def _section_tests(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    lines = [
        "# 6. 统计检验", "",
        f"主检验为共同日期（{len(base.dates)} 天）上的**配对 Wilcoxon 符号秩检验**，"
        f"每个指标项上把所有模型两两配对；同一项内的多重比较用 **Holm 逐步校正**，"
        f"Holm p < 0.05 才记「显著」。检验用的是逐日样本而不是平均后的单点，"
        f"所以样本量等于共同日期数。RMSE / CRPS / FA 偏差 / Spread-RMSE 距 1 都是"
        f"越小越好，ACC 越大越好。「指标族」= 同一类指标、项 = 变量。",
        "",
        "**分指标族显著优胜项汇总**",
        "",
    ]
    rows = []
    for family in bundle.family_wins.index:
        record = bundle.family_wins.loc[family]
        rows.append(
            [family]
            + [f"{name} {int(record[name])}" for name in names]
            + [str(int(record["无显著差异"]))]
        )
    lines += _table(
        ["**指标族**"]
        + [f"**{name} 显著优胜项**" for name in names]
        + ["**无显著差异**"],
        rows,
    )
    lines += [""]
    lines += _bullets(_test_notes(bundle))
    return lines


def _test_notes(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    total_dates = len(base.dates)
    items = []
    for family in bundle.family_wins.index:
        record = bundle.family_wins.loc[family]
        wins = {name: int(record[name]) for name in names}
        total = bundle.family_totals.get(family, sum(wins.values()) + int(record["无显著差异"]))
        dominant = max(wins, key=lambda key: wins[key])
        if wins[dominant] == 0:
            items.append(
                f"{family}：{total} 个项上没有任何模型显著优于其余全部模型，"
                f"全部记入「无显著差异」——{_cn(len(names))}者在这些项上的差异看不出来。"
            )
        else:
            items.append(
                f"{family}：共 {total} 个项，{dominant} 拿下 {wins[dominant]} 个显著优胜项；"
                f"其余 {total - wins[dominant]} 个项上没有模型能压过其余全部。"
            )

    order = base.relative_composite.sort_values().index.tolist()
    items.append(
        f"综合相对 RMSE 的逐日配对检验（{' vs '.join(order[:2])}）基于 {total_dates} "
        f"个共同日期，是本报告最接近「整体精度」的单条检验。"
    )
    if bundle.small_sample:
        items.append(
            f"共同日期仅 {total_dates} 天：Wilcoxon 符号秩检验在 n < {SMALL_SAMPLE_DATES} "
            f"时几乎检验不出差异（n = {total_dates} 时理论最小两侧 p 值约 "
            f"{2.0 / (2 ** total_dates):.2e}，但需要全部样本同号才可能达到），"
            f"「无显著差异」不能读成「两者等价」。"
        )
    items.append(
        "本表的「显著优胜」要求该模型在**同一项上显著优于其余全部模型**，"
        "因此「各模型优胜项数 + 无显著差异 = 该项总数」，不会重复计数；"
        "只有两两检验、没有全局检验，结论是两两之间的方向性判断。"
    )
    return items


def _section_risks(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    items = [
        f"样本量：全部统计基于 {len(base.dates)} 个共同日期"
        f"（{_date_span(base.dates)}），分时效段的每个点又只有其中一个子集，"
        f"越靠后的时效样本越少，末端时效的曲线波动要谨慎解读。",
        f"口径边界：正文与附录只使用共同日期。各模型归档里还有大量非共同日期"
        f"（逐日目录数远多于共同日期的那些模型尤其明显），本报告**没有**使用它们，"
        f"因此不要把本报告的数字与各模型单批归档自身的汇总数直接对比。",
        f"集合平均场口径：RMSE / ACC / FA 都来自逐日 ``*_ensmean.csv``，"
        f"衡量的是集合平均场而不是最优成员；择优成员带来的收益在本报告中不可见。",
        f"对数底数：本报告的纬向谱偏差用 ln，``weather_rmse_single`` 报告的"
        f"「频谱对数 RMS」用 log10，数值相差 2.3026 倍，跨报告比大小前先统一底数。",
    ]
    if bundle.small_sample:
        items.append(
            f"**共同日期只有 {len(base.dates)} 天**，本报告不构成任何模型优劣的结论，"
            f"只能用于验证链路与看曲线形状。"
        )
    gaps = sorted(set(base.fa_variables) ^ set(base.common_variables))
    if gaps:
        items.append(
            f"指标覆盖不齐：FA 变量为 {_named(base.fa_variables)}，"
            f"RMSE 变量为 {_named(base.common_variables)}，差集为 {_named(gaps)}——"
            f"某模型在缺项指标上没有证据，不要把「没算」当成「算得差」。"
        )
    ratios = {
        name: meta.get("missing_ratio")
        for name in names
        for meta in [base.archives[name].meta]
    }
    if any(isinstance(r, (int, float)) and float(r) > 0.5 for r in ratios.values()):
        worst = max(
            names,
            key=lambda name: float(base.archives[name].meta.get("missing_ratio") or 0.0),
        )
        items.append(
            f"数据完整性：{worst} 的起报点缺失率最高"
            f"（{_fmt(BASELINE * float(base.archives[worst].meta.get('missing_ratio') or 0.0), 2)}%），"
            f"它覆盖的日期段明显短于其余模型，共同日期被最短的那一批限定。"
        )
    items.append(
        f"验收建议：本报告作为**方向性**依据；正式验收前把各批 ``periods`` 对齐到"
        f"同一段起报日期、补齐缺失变量后重跑，再以重跑结果为准。"
    )
    return ["# 7. 风险与验收建议", ""] + [f"• {item}" for item in items]


def _appendix(bundle: EnsBundle, figures: Mapping[str, str]) -> List[str]:
    base = bundle.base
    names = bundle.names
    shown = _named(names)
    count = _cn(len(names))
    first, last, leads = _lead_span(base.rmse_by_lead[names[0]].index)
    crps_shown = _named(bundle.crps_variables)
    acc_shown = _named(bundle.acc_variables)

    return [
        "# 附录：集合预报曲线图",
        "",
        f"本附录使用{count}个模型共同的{len(base.dates)}个日期"
        f"（{base.dates[0]} 至 {base.dates[-1]}）。"
        f"以下每张图均包含{shown}{count}条模型曲线。"
        f"正文统计表与图中曲线使用同一批共同日期。",
        "",
        f"## 附 1 集合均值RMSE：{len(base.common_variables)}个变量{count}模型曲线",
        "",
        f"每幅子图包含{shown}{count}条曲线。",
        "",
        f"![ensemble_rmse_vs_lead]({figures['ensemble_rmse_vs_lead']})",
        "",
        f"图 1：{len(base.common_variables)}个变量上{shown}的集合均值 RMSE 随 lead 变化"
        f"（{len(base.dates)} 个共同日期平均）。",
        "",
        f"## 附 2 Spread / Ensemble-mean RMSE：{len(bundle.spread_variables)}个变量{count}模型曲线",
        "",
        "纵轴为集合成员标准差与集合均值RMSE之比，**理想值为1**，图中画 y=1 参考线。",
        "",
        f"![ensemble_spread_rmse_ratio_vs_lead]({figures['ensemble_spread_rmse_ratio_vs_lead']})",
        "",
        f"图 2：{len(bundle.spread_variables)}个变量的 Spread / Ensemble-mean RMSE 随 lead "
        f"变化（{len(base.dates)} 个共同日期平均），虚线为理想值 1。",
        "",
        f"## 附 3 {crps_shown} CRPS：{count}模型曲线",
        "",
        f"纵轴越低越好；{count}条曲线对应{shown}。",
        "",
        f"![ensemble_crps_vs_lead]({figures['ensemble_crps_vs_lead']})",
        "",
        f"图 3：{crps_shown} 的 CRPS 随 lead 变化（{len(base.dates)} 个共同日期平均），"
        f"越低越好。",
        "",
        f"## 附 4 {acc_shown} ACC：{count}模型曲线",
        "",
        f"纵轴越高越好；{count}条曲线对应{shown}。",
        "",
        f"![ensemble_acc_vs_lead]({figures['ensemble_acc_vs_lead']})",
        "",
        f"图 4：{acc_shown} 的 ACC 随 lead 变化（{len(base.dates)} 个共同日期平均），"
        f"越高越好。",
        "",
        f"## 附 5 FA（Forecast Activity）：{len(base.fa_variables)}个变量、"
        f"全程{first}–{last}h {count}模型曲线",
        "",
        f"每个子图均包含{count}条模型曲线，并覆盖全部{leads}个lead（{first}–{last}h）；"
        f"ratio越接近1越好。",
        "",
        f"![ensemble_fa_ratio_vs_lead]({figures['ensemble_fa_ratio_vs_lead']})",
        "",
        f"图 5：{_named(base.fa_variables)} 的 FA（Forecast Activity）ratio 随 lead 变化，"
        f"覆盖全程 {first}–{last}h 的 {leads} 个 lead，虚线为理想值 1。",
        "",
        f"## 附 6 纬向功率谱：{len(base.spectrum_variables)}个变量{count}模型曲线",
        "",
        f"双对数坐标；{count}条颜色曲线对应{shown}的预测谱。观测谱为黑色虚线。",
        "",
        f"![ensemble_spectrum]({figures['ensemble_spectrum']})",
        "",
        f"图 6：{_named(base.spectrum_variables)} 的纬向功率谱"
        f"（双对数，{len(base.dates)} 个共同日期平均），"
        f"每条颜色曲线为一个模型的预测谱，黑色虚线为观测谱。",
    ]


def _check_figure_contract(text: str, count: int = 6) -> None:
    """骨架的编号契约：``图 N：`` 必须 1..count 各出现一次，且首现位置递增。"""
    positions = []
    for index in range(1, count + 1):
        marker = f"图 {index}："
        found = text.count(marker)
        if found != 1:
            raise ValueError(f"图注契约被破坏：{marker} 出现 {found} 次，应当恰好 1 次")
        positions.append(text.index(marker))
    if positions != sorted(positions):
        raise ValueError(f"图注契约被破坏：图 1..{count} 的首现位置不是递增的")


def render_ens(bundle: EnsBundle, figures: Mapping[str, str], *,
               archive: Optional[str] = None) -> str:
    """把 :class:`EnsBundle` 渲染成骨架形状的完整 Markdown。"""
    base = bundle.base
    names = bundle.names
    shown = _named(sorted(names))
    count = _cn(len(names))
    first, last, leads = _lead_span(base.rmse_by_lead[names[0]].index)

    lines: List[str] = [
        f"# XMETAI 集合预报{count}模型评估检验报告（{bundle.member_label}成员）",
        "",
        f"**归档：{archive or _named(names)}**",
        "",
        f"报告日期：{base.dates[-1]}    报告类型：集合预报（{bundle.member_label}成员）",
        "",
        f"**评估对象：{shown}**",
        "",
        "## 报告定位与衔接",
        "",
        f"报告类型：集合预报（ensemble，{bundle.member_label}成员）。"
        f"本报告覆盖 {len(base.dates)} 个共同日期（{_date_span(base.dates)}）上{shown}的"
        f"集合平均场精度、集合分布质量（CRPS）与离散度校准；确定性的单成员版本见 "
        f"`weather_rmse_single` 报告，球谐带功率谱补充见 `weather_rmse_wave` 报告，"
        f"三份报告口径不同、数字不可互相搬运。",
        "",
        f"本报告中的FA指Forecast Activity，统一使用全程{first}–{last}h、{leads}个lead，"
        f"不对FA做短中长时效分段；分时效分析用于RMSE、CRPS、ACC与Spread/RMSE。",
        "",
    ]
    for section in (
        _section_conclusion(bundle),
        _section_samples(bundle),
        _section_metrics(bundle),
        _section_overall(bundle),
        _section_lead(bundle),
        _section_tests(bundle),
        _section_risks(bundle),
        _appendix(bundle, figures),
    ):
        lines += section
        lines.append("")

    text = "\n".join(lines)
    _check_figure_contract(text)
    return text


# ====================================================================== 入口


def build_ens_report(archives: Sequence[Archive], out_dir: Path, *,
                     archive: Optional[str] = None) -> Dict[str, Path]:
    """出图 + 写 ``REPORT.md``，返回产物名 → 路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = analyze_ens(archives)
    names = bundle.names
    base = bundle.base
    count = _cn(len(names))
    first, last, _ = _lead_span(base.rmse_by_lead[names[0]].index)
    date_note = f"{len(base.dates)} 个共同日期平均（{_date_span(base.dates)}）"

    # 插入顺序 = 图号顺序，与骨架「附 1」…「附 6」一一对应
    figures = {
        "ensemble_rmse_vs_lead": "ensemble_rmse_vs_lead.png",
        "ensemble_spread_rmse_ratio_vs_lead": "ensemble_spread_rmse_ratio_vs_lead.png",
        "ensemble_crps_vs_lead": "ensemble_crps_vs_lead.png",
        "ensemble_acc_vs_lead": "ensemble_acc_vs_lead.png",
        "ensemble_fa_ratio_vs_lead": "ensemble_fa_ratio_vs_lead.png",
        "ensemble_spectrum": "ensemble_spectrum.png",
    }

    plotter = DetPlotter()

    # 图 1：集合均值 RMSE
    path = out_dir / figures["ensemble_rmse_vs_lead"]
    plotter.plot_model_panels(
        base.rmse_by_lead, base.common_variables, ylabel="RMSE",
        suptitle=f"{count}模型集合均值 RMSE（{date_note}）", save_path=path,
    )
    plt.close("all")

    # 图 2：Spread / RMSE（理想值 1）
    path = out_dir / figures["ensemble_spread_rmse_ratio_vs_lead"]
    plotter.plot_model_panels(
        bundle.spread_ratio_by_lead, bundle.spread_variables,
        ylabel="Spread / RMSE", ref_line=1.0,
        suptitle=f"{count}模型 Spread / Ensemble-mean RMSE（理想值 1，{date_note}）",
        save_path=path,
    )
    plt.close("all")

    # 图 3：CRPS
    path = out_dir / figures["ensemble_crps_vs_lead"]
    plotter.plot_model_panels(
        bundle.crps_by_lead, bundle.crps_variables, ylabel="CRPS",
        suptitle=f"{count}模型 CRPS（{date_note}）", save_path=path,
    )
    plt.close("all")

    # 图 4：ACC
    path = out_dir / figures["ensemble_acc_vs_lead"]
    plotter.plot_model_panels(
        {name: base.acc_by_lead[name][bundle.acc_variables] for name in names},
        bundle.acc_variables, ylabel="ACC (%)",
        suptitle=f"{count}模型 ACC（{date_note}）", save_path=path,
    )
    plt.close("all")

    # 图 5：FA ratio（理想值 1）
    path = out_dir / figures["ensemble_fa_ratio_vs_lead"]
    plotter.plot_model_panels(
        base.fa_ratio_by_lead, base.fa_variables,
        ylabel="FA ratio (pred / obs)", ref_line=1.0,
        suptitle=f"{count}模型 FA（Forecast Activity）ratio，全程 {first}–{last}h"
                 f"（理想值 1，{date_note}）",
        save_path=path,
    )
    plt.close("all")

    # 图 6：纬向功率谱（一变量一子图，模型曲线 + 共享观测谱）
    _spectrum_panels(
        out_dir, figures["ensemble_spectrum"], bundle.spectrum_by_variable,
        variables=base.spectrum_variables, names=names,
        suptitle=f"{count}模型纬向功率谱（{date_note}）",
    )

    text = render_ens(bundle, figures, archive=archive)
    report_path = out_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"已保存: {report_path}")

    artifacts: Dict[str, Path] = {"report": report_path}
    for key, filename in figures.items():
        artifacts[key] = out_dir / filename
    return artifacts
