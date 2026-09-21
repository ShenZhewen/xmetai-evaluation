#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纬向 FFT 与球谐谱补充分析报告：读评估产物 → 谱统计 → 渲染 Markdown。

输入是 ``weather_rmse_wave_<模型>`` 支线的**评估产物目录**，形状与 ``_single``
支线完全相同（``scores.csv`` + ``manifest.json`` + ``diagnostics/``）——两条支线
吃同一份产物，区别只在报告口径：这一份**专管谱**，走「总体结果 → RMSE 分变量 →
功率谱检验 → 异常与风险 → 验收建议」这条链，不重复逐变量 RMSE/ACC 明细。

格式契约是 ``skills/xmetai-evaluation/assets/templates/weather_rmse_wave.md``：
章节用 ``# 1.``…``# 7.`` + 附录，**表不编号、图带 ``图 N：`` 且连续**，中文数字
由渲染器给（``{模型数}`` 填「五」而不是「5」）。

**两套谱，别混**：

* **纬向 FFT 谱**（§2–§4、§5.4）来自 ``diagnostics/spectrum_by_init.csv``，
  逐起报一条曲线，波数 1–720。log-RMS 这里用 **ln**，而 ``det_report.analyze``
  的 ``spectrum_rms`` 用 **log10**，两者差 ×2.3026——本模块照骨架 §2.1 的定义
  走 ln，**不复用** ``spectrum_rms``。
* **球谐带功率**（§5.1–5.3）来自长表里 ``spherical_bands`` 展开的三行
  （``spherical_band_power_forecast`` / ``_observation`` / ``_ratio``），
  **频带靠 ``group`` 列区分**（``1_4`` / ``5_20`` …）。它按球谐总阶数 ``l`` 分段
  （``DEFAULT_SPHERICAL_BANDS``），与 §5.4.1 按纬向波数分的 ``WAVE_BANDS``
  不是同一套分段，不能互相换算。

球谐带这一节在**两种**情况下退化成「本批未出」的占位说明（不静默省略、也不留
空表）：配置里没有 ``spherical_bands``，或结果行缺 ``group`` 列认不出频带。
第二种情况下**绝不能把几个频带混起来平均**——那样得到的数不属于任何尺度。

占位符 ``{匹配尺度上限}`` 取 128，依据是骨架 §5.4.3 的「1–128 波数逐变量结果」
与 §5.4 的「与球谐损失的最高阶数同范围」两处措辞。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xmetai_evaluation.visualization.det_plots import DetPlotter, model_colors
from xmetai_evaluation.visualization.precipitation_plots import setup_chinese_font
from xmetai_evaluation.visualization.det_report import (
    Archive,
    BASELINE,
    Bundle,
    PairTest,
    _CN_NUM,
    _date_span,
    _fmt,
    SPECTRUM_RATIO_ABSENT,
    _lead_span,
    _model_count_word,
    _rank,
    _table,
    analyze,
    _region_reading,
    region_display,
    region_table,
    _signed,
    load_spectrum,
    metric_rows,
    paired_tests,
    probe_dates,
)

from xmetai_evaluation.metrics.specialized import DEFAULT_SPHERICAL_BANDS

#: 纬向谱的完整波数范围（``diagnostics/spectrum_by_init.csv`` 的 wavenumber 1–720）
FULL_K_MAX = 720

#: 与球谐损失的最高阶数同范围的「匹配尺度」上限
MATCHED_K_MAX = 128

#: 五个波数带：(下界, 上界, 名称)，按行星尺度 → 小尺度
WAVE_BANDS: Tuple[Tuple[int, int, str], ...] = (
    (1, 3, "行星波"),
    (4, 9, "长波"),
    (10, 20, "天气尺度"),
    (21, 60, "中尺度"),
    (61, FULL_K_MAX, "小尺度"),
)

#: 球谐带频带 ``(下界, 上界)``。直接取**算分那一侧**的常量，报告端与
#: ``metrics/specialized.py`` 的 ``spherical_bands`` 同源，不另立口径——
#: 这里写死一份，配置改了带边界报告就会跟着错。
SPHERICAL_BANDS: Tuple[Tuple[int, int], ...] = tuple(
    (int(lo), int(hi)) for lo, hi in DEFAULT_SPHERICAL_BANDS
)


def _band_tag(low: int, high: int) -> str:
    """频带标签，也是长表 ``group`` 列的取值（``1_4`` / ``5_20`` …）。"""
    return f"{low}_{high}"


def _band_label(low: int, high: int) -> str:
    """报告表格里显示的频带名。"""
    return f"{low}–{high}"

#: 球谐带功率谱的频带数（骨架里写死「五个球谐总阶数频带」）
SPHERICAL_BAND_COUNT = len(SPHERICAL_BANDS)

#: 产物里没有**可用**的球谐带功率结果时写进正文的说明。两种情况都走这里：
#: 一是配置的 ``metrics`` 里压根没有 ``spherical_bands``；二是算了，但结果行缺
#: ``group`` 列、认不出哪一行属于哪个频带（修复前的产物就是这样，重跑评测即可）。
SPHERICAL_ABSENT = (
    "产物里没有可用的球谐带功率结果（要么配置的 `metrics` 里没有 `spherical_bands`，"
    "要么结果行缺 `group` 列、认不出频带），**本批未出**。"
)

#: 谱诊断没有按纬度带切开时，「附 L」第二张表（分纬度带纬向谱）写这句。
#: 谱诊断 ``diagnostics/spectrum_by_init.csv`` 的列是
#: ``variable / init_time / wavenumber / wavelength_km / pred / obs``——**没有 region**，
#: 纬向谱跟不准到带上，这一栏就只能标未出，不能拿全球谱冒充某一带。
REGION_SPECTRUM_ABSENT = (
    "纬向谱这一项**本批未出**：谱诊断 `diagnostics/spectrum_by_init.csv` 只有 "
    "`variable / init_time / wavenumber / wavelength_km / pred / obs`，**没有 `region` 列**，"
    "谱没有跟纬度带绑在一起，拿全球谱顶替某一带会得出错的结论。要补上得让评测侧在谱诊断里"
    "也写 `region`（与 scores.csv 的 `region` 同口径），报告这边不用改。"
)


def _season_block() -> List[str]:
    """「附 S 分季节结果（可选）」——与另两份骨架同一段静态说明。

    这一块现在**没有数**：长表里没有季节维度。按骨架的约定，保留节位、写清「本批未出」
    与接上它需要先定的两条口径，而不是留一张空表。
    """
    return [
        "## 附 S 分季节结果（可选）",
        "",
        "**本块本批未出。** 产物长表里目前没有季节维度，渲染器保留节位并标注「本批未出」，",
        "不静默省略、也不留空表。",
        "",
        "要接上这一块，得先把两条口径定下来（两条都不难，选错会让数对不上）：",
        "",
        "- 季节按**起报时刻**还是**有效时刻**切——同一份检验里两者会差一个时效的长度；",
        "- DJF 跨年怎么归——12 月与次年 1、2 月要不要算同一个 DJF。",
        "",
        "定了之后这一块的形态与「附 L」一致：一张分季节表（一行一个季节，按 DJF、MAM、",
        "JJA、SON 升序，各模型各占一列，末列 `Best`），加两张图——`season_summary.png`",
        "（分季节综合相对RMSE 柱状图，`图 S1`）与 `season_rmse_vs_lead.png`（各季节",
        "综合相对RMSE 随预报时效的变化，`图 S2`）。数据同样出自长表，**不需要重跑评测**。",
        "",
        "`图 S1`/`图 S2` 两个号现在**留空**：这一块没出数就不出图，不指不存在的文件。",
    ]


def _region_block(bundle: WaveBundle, figures: Mapping[str, str]) -> List[str]:
    """「附 L 分纬度带结果（可选）」。

    与 det / ens 的差别只有一处：本报告的落脚点是谱，骨架要求这一块**同时给出 RMSE 与
    纬向谱两项**。谱诊断里没有 `region`，所以第二张表本批给不出，只能标未出——见
    :data:`REGION_SPECTRUM_ABSENT`。
    """
    base = bundle.base
    count = len(base.names)
    word = _model_count_word(count)
    lines = ["## 附 L 分纬度带结果（可选）", ""]
    if not base.region_names:
        lines += [f"**本块本批未出。** {base.region_note}", ""]
        return lines

    table = region_table(base)
    lines += [
        "本块把长表里 `region` 非空的行**单独汇总**，全球行**不参与**——全球平均与纬度带平均",
        "是两个量，混在一起算会把带间差异整个抹平。带名与边界照搬评测配置",
        "`options[\"regions\"]`：**报告不翻译、也不写死任何带名**，配置里叫什么就显示什么。",
        f"表里的相对RMSE是**对本带基线**取的（100 = 该带内的{word}模型几何均值），",
        "不是全球基线——热带和极区的绝对 RMSE 差一个量级，不除本带基线没法横向比。",
        "",
        "本报告的落脚点是谱，所以分带表**同时给出 RMSE 与纬向谱两项**：如果某个带里",
        "RMSE 排名与谱排名不一致，多半是该带的观测谱在高波数段本身接近零，",
        "谱指标在那里被放大，判读要以 RMSE 为准，并在正文点名说明。",
        "",
        "**分纬度带综合相对RMSE**",
        "",
    ]
    lines += _table(["纬度带"] + list(base.names) + ["Best"], table.rows)
    lines += [
        "",
        f"**分纬度带纬向谱 log-RMS×100（全波数 1–{FULL_K_MAX}）**",
        "",
        REGION_SPECTRUM_ABSENT,
        "",
    ]
    reading = _region_reading(
        base, table.best_global, table.best_of, table.levels,
    )
    reading += (
        " 两个口径（RMSE 与谱）的排名在高纬是否一致，这一批**核不了**："
        "分带的纬向谱没有出数（原因见上），等谱诊断带上 `region` 之后再回来对。"
    )
    lines += [reading, ""]
    lines += [
        f"![lat_band_rmse_vs_lead]({figures['lat_band_lead']})",
        "",
        f"图 L1：各纬度带综合相对RMSE随预报时效的变化"
        f"（100 = 该带内{word}模型几何均值；{_date_span(bundle.dates)} 共同日期平均）",
        "",
        f"![lat_band_summary]({figures['lat_band_summary']})",
        "",
        f"图 L2：分纬度带综合相对RMSE"
        f"（100 = 该带内{word}模型几何均值；{_date_span(bundle.dates)} 共同日期平均）",
    ]
    return lines


# ====================================================================== 谱指标


def _cn(number: int) -> str:
    """中文数字（骨架要求 ``{模型数}`` / ``{FA变量数}`` 这类填中文）。"""
    return _CN_NUM.get(int(number), str(int(number)))


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """把谱表洗成纯 float 的 波数 × 日期 表。"""
    data = frame.copy()
    data.index = pd.to_numeric(data.index, errors="coerce")
    data = data[data.index.notna()]
    data = data.apply(pd.to_numeric, errors="coerce").sort_index()
    return data


def log_rms_ln(pred: pd.DataFrame, obs: pd.DataFrame,
               kmin: int = 1, kmax: int = FULL_K_MAX) -> pd.Series:
    """逐日 ``100 × RMS[ln(pred/obs)]``（波数 ``kmin..kmax``）。

    Returns:
        ``pd.Series``，index = 日期（``YYYYMMDD``），value = log-RMS×100。
    """
    pred = _numeric(pred)
    obs = _numeric(obs)
    band = (pred.index >= kmin) & (pred.index <= kmax)
    pred, obs = pred[band], obs[band]
    values: Dict[str, float] = {}
    for date in pred.columns:
        if date not in obs.columns:
            continue
        p, o = pred[date], obs[date]
        mask = (p > 0) & (o > 0)
        if not mask.any():
            continue
        ratio = (p[mask] / o[mask]).values
        values[str(date)] = 100.0 * float(np.sqrt(np.mean(np.log(ratio) ** 2)))
    return pd.Series(values, dtype=float)


def band_power_ratio(pred: pd.DataFrame, obs: pd.DataFrame,
                     kmin: int = 1, kmax: int = FULL_K_MAX) -> float:
    """波数 ``kmin..kmax``、全部日期上的带内积分能量比 ``mean(pred) / mean(obs)``。

    这里**不能**用 ``mean(pred/obs)``：高波数段观测功率接近零，逐波数比值的分母
    最先下溢，均值被极少数噪声点主导——小尺度段会算出 10^5 量级的数字，看着像
    「能量偏了二十万倍」，实际只是除零噪声，没有任何物理含义。先沿波数求和再相除
    是有界的，且正是「该尺度带内的能量偏了多少」这个问题的答案。
    """
    pred = _numeric(pred)
    obs = _numeric(obs)
    band = (pred.index >= kmin) & (pred.index <= kmax)
    pred, obs = pred[band], obs[band]
    powers = []
    for date in pred.columns:
        if date not in obs.columns:
            continue
        p, o = pred[date], obs[date]
        mask = (p > 0) & (o > 0)
        if mask.any():
            powers.append((p[mask].mean(), o[mask].mean()))
    if not powers:
        return float("nan")
    pred_mean = float(np.mean([p for p, _ in powers]))
    obs_mean = float(np.mean([o for _, o in powers]))
    if obs_mean == 0:
        return float("nan")
    return pred_mean / obs_mean


def _band_series(rows: pd.DataFrame, tag: str, date: str) -> pd.Series:
    """某个频带、某个起报日的值 → ``index = lead_h`` 的一列。"""
    own = rows[(rows["group"].astype(str) == tag) & (rows["init_date"].astype(str) == date)]
    return own.set_index("lead_h")["value"].sort_index()


def load_band_power(root, dates: Sequence[str], variable: str,
                    name: str = "") -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """``spherical_bands`` 的结果 → ``{频带标签: (pred, obs)}``。

    每个频带两张表，``index = lead_h``、``columns = 起报日``——两张都要，因为
    :func:`log_ratio_arrays` 按**逐样本**池化（骨架 §2.1 要的是总均方根，
    不是「先按日均值再对日均值取均方根」）。

    三个量在长表里各占一行（``spherical_band_power_forecast`` / ``_observation``
    / ``_ratio``），是哪个频带靠 ``group`` 列（``1_4`` / ``5_20`` …）。

    Raises:
        FileNotFoundError: 某个起报日没有该变量的球谐带结果。
        ValueError: 结果在，但 ``group`` 列缺失或全空，分不出频带。
    """
    forecast = metric_rows(root, "spherical_band_power_forecast")
    observed = metric_rows(root, "spherical_band_power_observation")
    forecast = forecast[forecast["variable"].astype(str) == str(variable)]
    observed = observed[observed["variable"].astype(str) == str(variable)]
    if forecast.empty:
        raise FileNotFoundError(f"{name or root}: 没有 {variable} 的球谐带功率结果")
    if "group" not in forecast.columns or forecast["group"].isna().all():
        raise ValueError(
            f"{name or root}: {variable} 的球谐带结果没有 group 列，认不出频带；"
            f"把几个频带混起来平均会得到一个不属于任何尺度的数，所以不做"
        )
    wanted = [str(date) for date in dates]
    have = set(forecast["init_date"].astype(str))
    missing = [date for date in wanted if date not in have]
    if missing:
        raise FileNotFoundError(
            f"{name or root}: {len(missing)} 个起报日没有 {variable} 的球谐带结果"
            f"（如 {missing[0]}）"
        )
    groups = set(forecast["group"].astype(str))
    table: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for low, high in SPHERICAL_BANDS:
        tag = _band_tag(low, high)
        if tag not in groups:
            continue
        pred_columns = {date: _band_series(forecast, tag, date) for date in wanted}
        obs_columns = {date: _band_series(observed, tag, date) for date in wanted}
        table[tag] = (pd.DataFrame(pred_columns), pd.DataFrame(obs_columns))
    return table


def probe_bands(root, dates: Sequence[str], variables: Sequence[str]) -> bool:
    """产物里有没有**可用**的球谐带功率结果。

    光有结果行不够——还得分得出频带（``group`` 列非空），否则 §5.1–5.3 只能记缺席。
    """
    if not dates or not variables:
        return False
    rows = metric_rows(root, "spherical_band_power_forecast")
    if rows.empty or "group" not in rows.columns or rows["group"].isna().all():
        return False
    return (
        {str(date) for date in dates} <= set(rows["init_date"].astype(str))
        and {str(v) for v in variables} <= set(rows["variable"].astype(str))
    )


def log_ratio_arrays(pred: pd.DataFrame, obs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """lead × 日期 的带功率表 → 全样本 ``(ln(pred/obs), pred/obs)``。

    骨架 §2.1 把球谐带 log-RMS 定义为「**所有**日期、变量、lead 的总均方根」，
    所以这里交回**逐样本**的数组而不是逐日的均值——先算 log 再平方平均，
    与「先按日均值、再对日均值取均方根」不是一回事。
    """
    pred = _numeric(pred)
    obs = _numeric(obs)
    shared = [d for d in pred.columns if d in obs.columns]
    if not shared:
        return np.empty(0), np.empty(0)
    p = pred[shared].to_numpy(dtype=float)
    o = obs[shared].to_numpy(dtype=float)
    mask = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0)
    if not mask.any():
        return np.empty(0), np.empty(0)
    ratio = p[mask] / o[mask]
    return np.log(ratio), ratio


def band_daily_logrms(pred: pd.DataFrame, obs: pd.DataFrame) -> pd.Series:
    """逐日 ``100 × RMS_lead[ln(pred/obs)]``（配对检验要的日度样本，§5.3 用）。"""
    pred = _numeric(pred)
    obs = _numeric(obs)
    values: Dict[str, float] = {}
    for date in pred.columns:
        if date not in obs.columns:
            continue
        p, o = pred[date].to_numpy(dtype=float), obs[date].to_numpy(dtype=float)
        mask = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0)
        if not mask.any():
            continue
        values[str(date)] = 100.0 * float(
            np.sqrt(np.mean(np.log(p[mask] / o[mask]) ** 2)))
    return pd.Series(values, dtype=float)


def _pooled_logrms(samples: Sequence[np.ndarray]) -> float:
    """若干 ``ln(pred/obs)`` 样本池 → ``100 ×`` 总均方根。"""
    stacked = [s for s in samples if s.size]
    if not stacked:
        return float("nan")
    values = np.concatenate(stacked)
    return 100.0 * float(np.sqrt(np.mean(values ** 2)))


def _mean_ratio(samples: Sequence[np.ndarray]) -> float:
    """若干 ``pred/obs`` 样本池 → 均值比（骨架 §5.1 那一列的「均值比」）。"""
    stacked = [s for s in samples if s.size]
    if not stacked:
        return float("nan")
    return float(np.mean(np.concatenate(stacked)))


def _bias_pct(samples: Sequence[np.ndarray]) -> float:
    """若干 ``pred/obs`` 样本池 → ``100 × mean|pred/obs − 1|``（骨架 §2.1 的比偏差）。"""
    stacked = [s for s in samples if s.size]
    if not stacked:
        return float("nan")
    return 100.0 * float(np.mean(np.abs(np.concatenate(stacked) - 1.0)))


def _first_lead_value(bundle: Bundle, variable: str) -> float:
    """各模型在该变量上首时效的值，取其中最大者。

    ``Bundle.acc_by_lead`` 在 :func:`det_report.analyze` 里已经乘过 ``BASELINE``，
    拿到的就是百分数，这里**不要再乘 100**。
    """
    values = []
    for name in bundle.names:
        frame = bundle.acc_by_lead.get(name)
        if frame is None or variable not in frame.columns or frame.empty:
            continue
        series = frame[variable].dropna()
        if not series.empty:
            values.append(float(series.iloc[0]))
    return max(values) if values else float("nan")


# ====================================================================== 分析


class WaveBundle(NamedTuple):
    """``_single`` 的 Bundle + 谱专项的一切。"""

    base: Bundle
    spectra: Dict[str, Dict[str, pd.DataFrame]]  # 模型 -> 变量 -> 波数 × {pred, obs}
    zonal_full: pd.Series       # 模型 -> 全波数 1–720 的 log-RMS×100
    zonal_matched: pd.Series    # 模型 -> 匹配尺度 1–128 的 log-RMS×100
    zonal_per_date: pd.DataFrame  # 日期 × 模型（全波数，多变量平均）
    band_logrms: pd.DataFrame   # 波数带 × 模型
    band_ratio: pd.DataFrame    # 波数带 × 模型
    variable_logrms: pd.DataFrame  # 变量 × 模型（全波数）
    variable_matched: pd.DataFrame  # 变量 × 模型（匹配尺度 1–128）
    zonal_tests: List[PairTest]   # 全波数口径的配对检验
    primary: str                # 主模型（FGVP_pinpu 之类），没有就是空串
    reference: str              # 主模型的对照模型（FGVP）
    sph_available: bool         # 产物里有没有可用的球谐带功率结果（含 group 列）
    sph_rms: pd.Series          # 模型 -> 球谐带 log-RMS×100（全频带/变量/lead/日期 总均方根）
    sph_bias: pd.Series         # 模型 -> 球谐带比偏差 (%)，mean|pred/obs−1|
    sph_band_logrms: pd.DataFrame   # 频带 × 模型
    sph_band_ratio: pd.DataFrame    # 频带 × 模型（均值比）
    sph_variable: pd.DataFrame      # 变量 × 模型（球谐带 log-RMS×100）
    sph_tests: Dict[str, List[PairTest]]  # 频带名 -> 该频带上的配对检验

    @property
    def names(self) -> List[str]:
        return self.base.names

    @property
    def dates(self) -> List[str]:
        return self.base.dates

    @property
    def spectrum_variables(self) -> List[str]:
        return self.base.spectrum_variables


def _pick_primary(names: Sequence[str]) -> Tuple[str, str]:
    """挑主模型与对照模型：``X_pinpu`` 是主，同前缀的 ``X`` 是参照。"""
    lowered = {str(name): str(name).lower() for name in names}
    for name, low in lowered.items():
        if low.endswith("_pinpu") or low.endswith("pinpu"):
            stem = low[: -len("_pinpu")] if low.endswith("_pinpu") else low[: -len("pinpu")]
            stem = stem.rstrip("_-")
            for other, other_low in lowered.items():
                if other != name and other_low == stem:
                    return name, other
            return name, ""
    return "", ""


def analyze_wave(archives: Sequence[Archive]) -> WaveBundle:
    """跑 ``_single`` 的全套统计，再叠上纬向谱专项。

    Raises:
        ValueError / FileNotFoundError: 见 :func:`det_report.analyze` 与
            :func:`det_report.load_spectrum`。
    """
    base = analyze(archives)
    names, dates = base.names, base.dates
    variables = base.spectrum_variables

    # ``spectra`` 存「波数 × 日期 平均后」的 pred/obs 均值谱（只够画图）；
    # ``raw`` 存逐日的 pred/obs 原表，log-RMS 必须逐日算，不能拿均值谱算。
    spectra: Dict[str, Dict[str, pd.DataFrame]] = {}
    raw: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = {}
    for archive in archives:
        means, originals = {}, {}
        for variable in variables:
            pred, obs = load_spectrum(archive.root, dates, variable, archive.name)
            originals[variable] = (pred, obs)
            means[variable] = pd.DataFrame({"pred": pred.mean(axis=1),
                                            "obs": obs.mean(axis=1)})
        spectra[archive.name] = means
        raw[archive.name] = originals

    # --- 逐变量 → 逐日 log-RMS，再对变量平均 ---
    zonal_per_date_frames = {}
    variable_logrms, variable_matched = {}, {}
    for name in names:
        per_date = []
        full_values, matched_values = {}, {}
        for variable in variables:
            pred, obs = raw[name][variable]
            full = log_rms_ln(pred, obs, 1, FULL_K_MAX)
            matched = log_rms_ln(pred, obs, 1, MATCHED_K_MAX)
            per_date.append(full)
            full_values[variable] = float(full.mean()) if not full.empty else float("nan")
            matched_values[variable] = float(matched.mean()) if not matched.empty else float("nan")
        variable_logrms[name] = pd.Series(full_values)
        variable_matched[name] = pd.Series(matched_values)
        zonal_per_date_frames[name] = pd.concat(per_date, axis=1).mean(axis=1) if per_date else pd.Series(dtype=float)

    zonal_per_date = pd.DataFrame(zonal_per_date_frames).reindex(index=dates)
    variable_logrms = pd.DataFrame(variable_logrms).reindex(index=variables)
    variable_matched = pd.DataFrame(variable_matched).reindex(index=variables)
    zonal_full = variable_logrms.mean(axis=0)
    zonal_matched = variable_matched.mean(axis=0)

    # --- 五个波数带 ---
    band_rows_logrms, band_rows_ratio = {}, {}
    for name in names:
        logrms_row, ratio_row = {}, {}
        for low, high, label in WAVE_BANDS:
            band_name = f"{label} {low}–{high}"
            logs, ratios = [], []
            for variable in variables:
                pred, obs = raw[name][variable]
                series = log_rms_ln(pred, obs, low, high)
                if not series.empty:
                    logs.append(float(series.mean()))
                ratio = band_power_ratio(pred, obs, low, high)
                if np.isfinite(ratio):
                    ratios.append(ratio)
            logrms_row[band_name] = float(np.mean(logs)) if logs else float("nan")
            ratio_row[band_name] = float(np.mean(ratios)) if ratios else float("nan")
        band_rows_logrms[name] = pd.Series(logrms_row)
        band_rows_ratio[name] = pd.Series(ratio_row)

    band_order = [f"{label} {low}–{high}" for low, high, label in WAVE_BANDS]
    band_logrms = pd.DataFrame(band_rows_logrms).reindex(index=band_order)
    band_ratio = pd.DataFrame(band_rows_ratio).reindex(index=band_order)

    zonal_tests = paired_tests(zonal_per_date)
    primary, reference = _pick_primary(names)

    # --- 球谐带功率（没有结果、或结果缺 group 列认不出频带时，整块留空、正文标注缺席）---
    band_names = [_band_label(lo, hi) for lo, hi in SPHERICAL_BANDS]
    sph_available = all(probe_bands(a.root, dates, variables) for a in archives)
    sph_tests: Dict[str, List[PairTest]] = {}
    sph_rms: Dict[str, float] = {}
    sph_bias: Dict[str, float] = {}
    sph_band_logrms = pd.DataFrame(index=band_names)
    sph_band_ratio = pd.DataFrame(index=band_names)
    sph_variable = pd.DataFrame(index=variables)
    if sph_available:
        logrms_rows, ratio_rows, variable_rows = {}, {}, {}
        daily: Dict[str, Dict[str, pd.Series]] = {}
        for archive in archives:
            # logs/ratios 是**逐样本**池（骨架 §2.1 要总均方根，不是日均值的均方根）；
            # per_variable 按变量分池，daily 按频带留日度样本供 §5.3 配对检验。
            logs: Dict[str, List[np.ndarray]] = {}
            ratios: Dict[str, List[np.ndarray]] = {}
            per_variable: Dict[str, List[np.ndarray]] = {}
            daily_parts: Dict[str, List[pd.Series]] = {b: [] for b in band_names}
            for variable in variables:
                table = load_band_power(archive.root, dates, variable, archive.name)
                for low, high in SPHERICAL_BANDS:
                    label = _band_label(low, high)
                    pair = table.get(_band_tag(low, high))
                    if pair is None:
                        continue
                    log_ratio, ratio = log_ratio_arrays(*pair)
                    logs.setdefault(label, []).append(log_ratio)
                    ratios.setdefault(label, []).append(ratio)
                    per_variable.setdefault(variable, []).append(log_ratio)
                    daily_parts[label].append(band_daily_logrms(*pair))
            logrms_rows[archive.name] = {
                b: _pooled_logrms(logs.get(b, [])) for b in band_names}
            ratio_rows[archive.name] = {
                b: _mean_ratio(ratios.get(b, [])) for b in band_names}
            variable_rows[archive.name] = {
                v: _pooled_logrms(s) for v, s in per_variable.items()}
            sph_rms[archive.name] = _pooled_logrms(
                [a for b in band_names for a in logs.get(b, [])])
            sph_bias[archive.name] = _bias_pct(
                [a for b in band_names for a in ratios.get(b, [])])
            daily[archive.name] = {
                b: (pd.concat(daily_parts[b], axis=1).mean(axis=1)
                    if daily_parts[b] else pd.Series(dtype=float))
                for b in band_names}
        for label in band_names:      # 同一频带内跨模型配对；指标即频带，故逐频带各做一轮
            frame = pd.DataFrame({n: daily[n][label] for n in names})
            sph_tests[label] = paired_tests(frame.reindex(index=dates))
        sph_band_logrms = pd.DataFrame(logrms_rows).reindex(index=band_names)
        sph_band_ratio = pd.DataFrame(ratio_rows).reindex(index=band_names)
        sph_variable = pd.DataFrame(variable_rows).reindex(index=variables)
    sph_rms = pd.Series({n: sph_rms.get(n, float("nan")) for n in names})
    sph_bias = pd.Series({n: sph_bias.get(n, float("nan")) for n in names})

    return WaveBundle(
        base=base, spectra=spectra,
        zonal_full=zonal_full, zonal_matched=zonal_matched,
        zonal_per_date=zonal_per_date,
        band_logrms=band_logrms, band_ratio=band_ratio,
        variable_logrms=variable_logrms, variable_matched=variable_matched,
        zonal_tests=zonal_tests, primary=primary, reference=reference,
        sph_available=sph_available, sph_rms=sph_rms, sph_bias=sph_bias,
        sph_band_logrms=sph_band_logrms, sph_band_ratio=sph_band_ratio,
        sph_variable=sph_variable, sph_tests=sph_tests,
    )


# ====================================================================== 出图


def _save_figure(figure, path: Path) -> Path:
    """落盘一张图。字体设置与保存约定跟 :mod:`visualization.det_plots` 对齐。"""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print(f"已保存: {path}")
    return path


def _bar_panels(out_dir: Path, filename: str, panels: Sequence[Tuple[str, pd.Series]],
                *, suptitle: str, ylabel: str = "") -> Path:
    """一排柱状子图：每个面板一个指标，柱子是各模型。"""
    setup_chinese_font()
    columns = len(panels)
    figure, axes = plt.subplots(1, columns, figsize=(3.2 * columns, 4.0), squeeze=False)
    axes = axes.ravel()
    for axis, (title, series) in zip(axes, panels):
        series = series.dropna()
        colors = model_colors(list(series.index))
        axis.bar([str(i) for i in series.index], series.values,
                 color=[colors[str(i)] for i in series.index], width=0.62)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(ylabel or title, fontsize=9)
        axis.grid(alpha=0.3, axis="y", linewidth=0.6)
        axis.tick_params(labelsize=8)
        for tick in axis.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")
    figure.suptitle(suptitle, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(figure, Path(out_dir) / filename)


def _grouped_bars(out_dir: Path, filename: str, frame: pd.DataFrame, *,
                  suptitle: str, ylabel: str, ref_line: Optional[float] = None) -> Path:
    """分组柱状图：index 是分组，columns 是模型。"""
    setup_chinese_font()
    groups = [str(i) for i in frame.index]
    models = [str(c) for c in frame.columns]
    colors = model_colors(models)
    width = 0.8 / max(1, len(models))
    positions = np.arange(len(groups))
    figure, axis = plt.subplots(figsize=(max(8.0, 1.7 * len(groups)), 5.2))
    for offset, model in enumerate(models):
        values = pd.to_numeric(frame[model], errors="coerce").values
        axis.bar(positions + offset * width - 0.4 + width / 2, values,
                 width=width, color=colors[model], label=model)
    if ref_line is not None:
        axis.axhline(ref_line, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(groups, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.grid(alpha=0.3, axis="y", linewidth=0.6)
    axis.legend(loc="best", fontsize=9)
    axis.set_title(suptitle, fontsize=12)
    figure.tight_layout()
    return _save_figure(figure, Path(out_dir) / filename)


def _plot_mean_spectrum(out_dir: Path, filename: str, spectra: Mapping[str, pd.DataFrame], *,
                        variable: str, suptitle: str) -> Path:
    """平均功率谱：双对数，各模型预测谱 + 一条共享观测谱。"""
    plotter = DetPlotter()
    path = Path(out_dir) / filename
    plotter.plot_model_spectrum(
        dict(spectra), variable, suptitle=suptitle, save_path=path,
    )
    plt.close("all")
    return path


def _plot_spectrum_ratio(out_dir: Path, filename: str, spectra: Mapping[str, pd.DataFrame], *,
                         variable: str, suptitle: str) -> Path:
    """pred/obs 比值曲线，纵轴限定 0.1–10。"""
    setup_chinese_font()
    colors = model_colors(list(spectra))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for name, frame in spectra.items():
        data = frame.copy()
        data.index = pd.to_numeric(data.index, errors="coerce")
        data = data[(data.index > 0)].sort_index()
        ratio = (data["pred"] / data["obs"]).replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio[ratio > 0]
        if ratio.empty:
            continue
        axis.plot(ratio.index, ratio.values, color=colors[name], linewidth=1.6, label=name)
    axis.axhline(1.0, color="#000000", linestyle="--", linewidth=1.2, label="理想值 = 1")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(0.1, 10.0)
    axis.set_xlabel("纬向波数 k", fontsize=10)
    axis.set_ylabel("pred / obs", fontsize=10)
    axis.grid(alpha=0.3, which="both", linewidth=0.6)
    axis.legend(loc="best", fontsize=9)
    axis.set_title(suptitle, fontsize=12)
    figure.tight_layout()
    return _save_figure(figure, Path(out_dir) / filename)


# ====================================================================== 渲染


def _section_conclusion(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    order = base.relative_composite.sort_values().index.tolist()
    best, worst = order[0], order[-1]
    zonal_best = bundle.zonal_full.idxmin()
    zonal_worst = bundle.zonal_full.idxmax()
    fa_best = base.fa_bias.idxmin()

    items = [
        f"{_cn(len(names))}个产物完整性检查通过，共同日期 {len(bundle.dates)} 天"
        f"（{_date_span(bundle.dates)}），全部模型 `n_ok = n_dates`，无失败起报点。",
        f"综合相对 RMSE 最好 {best}（{_fmt(base.relative_composite[best])}），"
        f"最差 {worst}（{_fmt(base.relative_composite[worst])}）；"
        f"ACC 最高 {base.acc_mean.idxmax()}（{_fmt(base.acc_mean.max())}%）。",
        f"幅度真实性（FA 偏差）最好 {fa_best}（{_fmt(base.fa_bias[fa_best], 4)} pp），"
        f"最差 {base.fa_bias.idxmax()}（{_fmt(base.fa_bias.max(), 4)} pp）。",
        f"纬向 FFT 谱（全波数 1–{FULL_K_MAX}）log-RMS×100 最小 {zonal_best}"
        f"（{_fmt(bundle.zonal_full[zonal_best], 2)}），最大 {zonal_worst}"
        f"（{_fmt(bundle.zonal_full[zonal_worst], 2)}）；"
        f"匹配尺度 1–{MATCHED_K_MAX} 口径下最小 {bundle.zonal_matched.idxmin()}"
        f"（{_fmt(bundle.zonal_matched.min(), 2)}）。",
    ]
    if bundle.sph_available:
        items.append(
            f"球谐带功率谱（{_cn(SPHERICAL_BAND_COUNT)}个球谐总阶数频带）"
            f"log-RMS×100 最小 {bundle.sph_rms.idxmin()}"
            f"（{_fmt(bundle.sph_rms.min(), 2)}），最大 {bundle.sph_rms.idxmax()}"
            f"（{_fmt(bundle.sph_rms.max(), 2)}）；比偏差最小 {bundle.sph_bias.idxmin()}"
            f"（{_fmt(bundle.sph_bias.min(), 2)}%），最大 {bundle.sph_bias.idxmax()}"
            f"（{_fmt(bundle.sph_bias.max(), 2)}%）。"
        )
    else:
        items.append(f"球谐带功率谱：{SPHERICAL_ABSENT}")
    if bundle.primary and bundle.reference:
        primary, reference = bundle.primary, bundle.reference
        delta = float(bundle.zonal_full[primary]) - float(bundle.zonal_full[reference])
        direction = "改善" if delta < 0 else "变差"
        items.append(
            f"主模型 {primary} 相对 {reference}：纬向谱 log-RMS×100 "
            f"{_fmt(bundle.zonal_full[reference], 2)} → {_fmt(bundle.zonal_full[primary], 2)}，"
            f"{direction} {_fmt(abs(delta), 2)}；综合相对 RMSE "
            f"{_fmt(base.relative_composite[reference])} → {_fmt(base.relative_composite[primary])}。"
        )
    variable = base.anomaly_variable
    items.append(
        f"异常变量检查：{base.anomaly_model} 的 {variable} RMSE 为其余模型中位数的 "
        f"{_fmt(base.anomaly_ratio, 2)} 倍"
        + ("，超出 3× 阈值，详见 6.1。" if base.anomaly_ratio >= 3.0 else "，未超 3× 阈值。")
    )
    items.append(
        f"选型取向：精度优先看综合相对 RMSE（{best}），幅度与谱真实性优先看 FA 偏差"
        f"（{fa_best}）与纬向谱（{zonal_best}），三者不一定指向同一模型。"
    )
    return ["# 1. 结论摘要", ""] + [f"• {item}" for item in items]


def _spectrum_inclusive_rank(base, bundle) -> pd.Series:
    """RMSE + ACC + FA + **纬向 FFT 谱** 的等权平均秩。§3.1 与 §7 共用。

    **不能**拿 ``base.rank_score`` 顶替：那是 ``det_report`` 的口径，第四项是
    单变量 log10 的 ``spectrum_rms``。本报告从 §2.1 到 §5 讲的都是纬向 FFT 谱
    （ln 底），两套数字摆在同一份报告里会互相打架——出现过 §3.1 写「加入纬向谱
    后平均秩最低的是 FuXi」，§7 的「多维度均衡」却写 AIFS 的情况。
    """
    return (
        _rank(base.relative_composite) + _rank(base.acc_mean, higher_is_better=True)
        + _rank(base.fa_bias) + _rank(bundle.zonal_full)
    ) / 4.0


def _section_scope(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    lines = ["# 2. 数据范围与检验口径", ""]

    rows = []
    nan_notes: List[str] = []
    for name in sorted(names):
        archive = base.archives[name]
        summary = archive.summary
        own_dates = sorted(set(summary["init_date"]))
        shared = len(set(own_dates) & set(bundle.dates))
        rmse_columns = [c for c in summary.columns if c.startswith("rmse_")]
        # 缺文件按 stem 实探，不看 summary 有没有行——两者可能对不上
        missing_dates = probe_dates(archive.root, bundle.dates, "rmse")
        missing = "无" if not missing_dates else f"{len(missing_dates)} 天：{missing_dates[0]} 等"
        nan_mask = summary[rmse_columns].isna().any(axis=1) if rmse_columns else None
        nan_all = summary.loc[nan_mask, "init_date"].tolist() if nan_mask is not None else []
        # 这一列统计的是**该模型自己的全部日期**，不是共同日期——不写清楚会被读成
        # 「共同 168 天里有 15 天是 NaN」，那是另一回事。
        nan_outside = [d for d in nan_all if d not in set(bundle.dates)]
        # 注意这里要的是**窗口之内**的天数。nan_outside 是窗口之外的那一批，
        # 拿它的长度去填「落在窗口之内」的句子会整体说反：全是窗口外的日期时
        # 反而报成「其中 15 天落在共同窗口之内」。
        nan_inside = len(nan_all) - len(nan_outside)
        rows.append([
            name, len(own_dates), _date_span(own_dates), shared,
            missing, f"{len(nan_all)}（全区间）" if nan_all else "无",
        ])
        if nan_all:
            nan_notes.append(
                f"{name} 的 {len(nan_all)} 天 RMSE 为 NaN"
                + (f"（{nan_all[0]}–{nan_all[-1]}）" if len(nan_all) > 1 else f"（{nan_all[0]}）")
                + ("全部落在共同窗口之外，不影响本报告任何统计"
                   if not nan_inside else
                   f"，其中 {nan_inside} 天落在共同窗口**之内**，需先查明原因")
                + "。"
            )
    lines += _table(
        ["**模型**", "**日期数**", "**日期范围**", "**共同日期**", "**核心文件缺失**",
         "**RMSE NaN（全区间）**"],
        rows,
    )
    if nan_notes:
        lines += [""] + nan_notes

    first, last, leads = _lead_span(base.rmse_by_lead[names[0]].index)
    variables = base.common_variables
    fa_variables = base.fa_variables
    spectrum_variables = bundle.spectrum_variables
    lines += [
        "",
        f"所有统计均使用{_cn(len(names))}个模型的共同{len(bundle.dates)}天和{leads}个 lead"
        f"（{first}–{last}h）。RMSE 使用{_cn(len(variables))}个共同变量；"
        f"FA 使用所有模型都提供的{_cn(len(fa_variables))}个变量："
        f"{'、'.join(fa_variables)}；功率谱使用{_cn(len(spectrum_variables))}个变量："
        f"{'、'.join(spectrum_variables)}。全部{_cn(len(names))}个模型在整个共同日期区间上都有完整结果，"
        f"无缺席模型。",
        "",
        "## 2.1 指标定义",
        "",
    ]
    lines += _table(
        ["**指标**", "**计算方式**", "**方向**"],
        [
            ["综合相对 RMSE",
             f"每个日期、每个变量先除以{_cn(len(names))}模型几何均值，再对{_cn(len(variables))}个变量取几何平均；100 为{_cn(len(names))}模型基线。",
             "越低越好"],
            ["ACC",
             f"{base.acc_variable} 每个日期{leads}个 lead 的平均 ACC，再对所有日期取平均。",
             "越高越好"],
            ["FA 偏差",
             f"{_cn(len(fa_variables))}个变量、全部 lead 的 \\|pred/obs−1\\|，再按日期和要素平均，单位 pp。",
             "越低越好"],
            ["纬向谱 log-RMS×100",
             f"对{_cn(len(spectrum_variables))}个变量的纬向 FFT 谱计算 RMS[ln(pred/obs)]×100；"
             f"报告同时给出全波数 1–{FULL_K_MAX} 和匹配尺度 1–{MATCHED_K_MAX}。",
             "越低越好"],
            ["球谐带 log-RMS×100",
             f"{_cn(SPHERICAL_BAND_COUNT)}个球谐总阶数频带上计算 RMS[ln(pred/obs)]×100，是所有日期、变量、lead 的总均方根。",
             "越低越好；球谐带补充谱指标" if bundle.sph_available else "越低越好；本批未出"],
            ["球谐带比偏差",
             "所有频带、变量、lead 与日期上的 mean \\|pred/obs−1\\|，单位 %。",
             "越低越好" if bundle.sph_available else "越低越好；本批未出"],
        ],
    )
    pairs = len(names) * (len(names) - 1) // 2
    lines += [
        "",
        f"配对比较采用{len(bundle.dates)}个日度样本的 Wilcoxon 符号秩检验，"
        f"并对同一指标族的{pairs}个模型对做 Holm 多重校正。统计显著不代表业务重要性。",
        "",
    ]
    return lines


def _section_overall(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    order = base.relative_composite.sort_values().index.tolist()
    lines = ["# 3. 总体结果", ""]
    rows = []
    for name in order:
        rows.append([
            name, _fmt(base.relative_composite[name]), _fmt(base.acc_mean[name]),
            _fmt(base.fa_bias[name], 4), _fmt(bundle.zonal_full[name], 2),
            _fmt(bundle.sph_rms[name], 2) if bundle.sph_available else "—",
            _fmt(bundle.sph_bias[name], 2) if bundle.sph_available else "—",
        ])
    lines += _table(
        ["**模型**", "**综合相对RMSE**", "**ACC (%)**", "**FA偏差 (pp)**",
         "**纬向谱 RMS×100**", "**球谐谱 RMS×100**", "**球谐比偏差 (%)**"],
        rows,
    )
    lines += ["", "相关图：", "", "![overall_summary](overall_summary.png)", "",
              "图 1：{模型数}模型的综合相对 RMSE、ACC、FA 偏差与纬向谱 log-RMS×100 对比"
              "（共同日期平均，柱高越低越好，ACC 除外）。".replace("{模型数}", _cn(len(bundle.names))), ""]

    # --- 3.1 排序随评价目标变化 ---
    rank_three = (
        _rank(base.relative_composite) + _rank(base.acc_mean, higher_is_better=True)
        + _rank(base.fa_bias)
    ) / 3.0
    rank_four = _spectrum_inclusive_rank(base, bundle)
    # 骨架 §3.1 第二行是「+ 球谐带功率谱」。球谐缺席时退回纬向 FFT 谱口径
    # （``rank_four``），并在解释里写清楚这一行统计的到底是哪套谱。
    if bundle.sph_available:
        rank_spectrum = (3.0 * rank_three + _rank(bundle.sph_rms)) / 4.0
        spectrum_name = "球谐带功率谱"
    else:
        rank_spectrum = rank_four
        spectrum_name = "纬向 FFT 谱"
    lines += ["## 3.1 排序随评价目标变化", ""]

    # 平均秩是几项指标的等权汇总，并列很常见（例如 FGVP 与 FGVP_pinpu 同为 7/3）。
    # 用 "<" 连两个显示相同的数字会把并列写成“优于”，所以按显示精度判并列、用 "=" 连。
    def describe(rank: pd.Series) -> str:
        chunks: List[str] = []
        previous: Optional[float] = None
        for name in rank.sort_values().index:
            shown = round(float(rank[name]), 2)
            separator = "" if previous is None else (" = " if shown == previous else " < ")
            chunks.append(f"{separator}{name} {_fmt(rank[name], 2)}")
            previous = shown
        return "".join(chunks)

    def tied_winners(rank: pd.Series) -> List[str]:
        best = round(float(rank.min()), 2)
        return [n for n in rank.index if round(float(rank[n]), 2) == best]

    def winners(rank: pd.Series) -> str:
        """并列第一时把并列者都列出来，不能只报 idxmin 挑中的那一个。"""
        tied = tied_winners(rank)
        suffix = f"（{_cn(len(tied))}者并列 {_fmt(rank.min(), 2)}）" if len(tied) > 1 else ""
        return "、".join(tied) + suffix

    lines += _table(
        ["**评价口径**", "**模型排序（平均秩，越低越好）**", "**第一**", "**解释**"],
        [
            ["RMSE + ACC + FA", describe(rank_three), winners(rank_three),
             "只看精度、相关与幅度，不看谱分布"],
            [f"RMSE + ACC + FA + {spectrum_name}",
             describe(rank_spectrum) if bundle.sph_available else "—",
             winners(rank_spectrum) if bundle.sph_available else "—",
             f"在上一口径基础上加入{spectrum_name}保真度" if bundle.sph_available
             else "球谐带功率谱本批未出，该口径无法计算；下表用纬向 FFT 谱代替"],
        ],
    )
    lines += [
        "",
        f"这里的“平均秩”只是{_cn(3)}项指标的等权汇总，不代替业务权重；"
        "写“=”表示两模型在该口径下平均秩并列，不代表指标完全相同。"
        f"加入{spectrum_name}后平均秩最低的是 {winners(rank_spectrum)}，"
        f"不含谱口径是 {winners(rank_three)}，"
        + ("两者是同一组模型，说明谱保真度没有改变精度主导的排序。"
           if set(tied_winners(rank_spectrum)) == set(tied_winners(rank_three))
           else "两者不是同一组，说明谱保真度会改变模型取舍。")
        + ("两套口径的头部都出现并列，说明这组模型的整体水平接近，"
           "排名对指标权重的敏感度高，选型时需明确以哪一类应用为准。"
           if len(tied_winners(rank_spectrum)) > 1 or len(tied_winners(rank_three)) > 1
           else "选型时需明确以哪一类应用为准。"),
        "",
    ]
    return lines


def _section_rmse(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    variables = base.common_variables
    names = bundle.names
    lines = ["# 4. RMSE 与时效表现", ""]

    mean_rmse = pd.DataFrame({
        name: base.rmse_by_date[name].mean(axis=0) for name in names
    }).reindex(index=variables)
    header = ["**变量**"] + [f"**{n}**" for n in names] + ["**最佳**"]
    rows = []
    for variable in sorted(variables):
        row = mean_rmse.loc[variable]
        rows.append([variable] + [_fmt(row[n], 4) for n in names] + [row.idxmin()])
    lines += _table(header, rows)

    best_overall = base.relative_composite.idxmin()
    per_variable_best = mean_rmse.idxmin(axis=1)
    tally = "、".join(
        f"{model} 在 {int(count)} 个变量上最优"
        for model, count in per_variable_best.value_counts().items()
    )
    wins = base.win_counts
    win_text = "；".join(
        f"{row['模型对'][0]} vs {row['模型对'][1]} 为 "
        f"{int(row['A 显著更优'])} : {int(row['B 显著更优'])}"
        f"（{int(row['无显著差异'])} 个变量无显著差异）"
        for _, row in wins.iterrows()
    )
    lines += [
        "",
        f"全局第一为 {best_overall}（综合相对 RMSE {_fmt(base.relative_composite[best_overall])}）；"
        f"逐变量看，{tally}。逐变量 Wilcoxon 配对检验（未做多重校正，只作方向性提示）：{win_text}。",
        "",
        "## 4.1 分时效综合相对 RMSE",
        "",
    ]

    lead_relative = base.lead_relative
    bands = ((0.0, 120.0), (120.0, 240.0), (240.0, 360.0))
    rows = []
    for low, high in bands:
        subset = lead_relative[(lead_relative.index > low) & (lead_relative.index <= high)]
        if subset.empty:
            continue
        means = subset.mean(axis=0)
        rows.append([f"{low:g}–{high:g}h"] + [_fmt(means[n]) for n in names] + [means.idxmin()])
    lines += _table(["**时效段**"] + [f"**{n}**" for n in names] + ["**最佳**"], rows)
    lines += [
        "",
        f"口径与第 3 节一致：逐 lead 先按变量做几何均值归一，再对 {_cn(len(variables))} 个共享变量"
        f"等权平均；未排除任何变量。",
        "",
        "分时效看，"
        + "；".join(
            f"{low:g}–{high:g}h 首选 "
            + f"{(lead_relative[(lead_relative.index > low) & (lead_relative.index <= high)].mean(axis=0)).idxmin()}"
            for low, high in bands
        )
        + "。短时效与中长时效的领先者未必是同一个模型，选型时应按实际预报时效长度取。",
        "",
        "![rmse_by_lead_range](rmse_by_lead_range.png)",
        "",
        f"图 2：{_cn(len(names))}模型分时效段（6–120h / 126–240h / 246–360h）的综合相对 RMSE 对比。",
        "",
    ]
    return lines


def _spherical_bands_section(bundle: WaveBundle) -> List[str]:
    """§5.1 球谐带分频结果：频带 × 模型，logRMS 与均值比各一组。"""
    names = bundle.names
    lines = ["## 5.1 球谐带分频结果", ""]
    if not bundle.sph_available:
        lines += [f"{SPHERICAL_ABSENT}因此本节无表、无图，保留节位以便将来补数后直接填入。", ""]
        return lines
    # 列序与 §5.4.1 一致：每个模型 logRMS / 均值比 两列相邻，「最佳」收尾
    header = ["**频带**"]
    for name in names:
        header += [f"**{name} logRMS**", f"**{name} 均值比**"]
    header += ["**最佳**"]
    rows = []
    for band in bundle.sph_band_logrms.index:
        logrms = bundle.sph_band_logrms.loc[band]
        ratios = bundle.sph_band_ratio.loc[band]
        rows.append([band] + [v for name in names
                              for v in (_fmt(logrms[name], 2), _fmt(ratios[name], 4))]
                    + [logrms.idxmin()])
    lines += _table(header, rows)
    band_list = list(bundle.sph_band_logrms.index)
    lines += [
        "",
        f"球谐带 log-RMS×100 最低的模型在"
        + "、".join(f"{band} 段是 {bundle.sph_band_logrms.loc[band].idxmin()}"
                    for band in band_list)
        + "；均值比是该频带内 pred/obs 的总平均，越接近 1 表示该尺度上的能量越不偏。"
        f"最高频带（{band_list[-1]}）的总阶数最高、观测功率最小，比值的对数对微小差异最敏感，"
        "该频带的 log-RMS 绝对值不宜与低频带直接比较——这是频谱类指标的固有特征，"
        "不是模型在该尺度上一定更差；均值比在同一频带仍有意义，因为它不依赖逐点的分母。"
        + f"按频带看，log-RMS 最低的模型分别是"
        + "、".join(
            f"{name} 拿下 {sum(1 for band in band_list if bundle.sph_band_logrms.loc[band].idxmin() == name)} 个频带"
            for name in names)
        + "。",
        "",
        "![spherical_bands_logrms](spherical_bands_logrms.png)",
        "",
        f"图 3：{_cn(len(names))}模型分球谐总阶数频带（{' / '.join(band_list)}）"
        f"的球谐带 log-RMS×100 对比。",
        "",
        "![spherical_bands_ratio](spherical_bands_ratio.png)",
        "",
        f"图 4：{_cn(len(names))}模型分球谐总阶数频带的 pred/obs 均值比对比（虚线为理想值 1）。",
        "",
    ]
    return lines


def _spherical_variables_section(bundle: WaveBundle) -> List[str]:
    """§5.2 球谐带分变量结果：变量 × 模型。"""
    names = bundle.names
    lines = ["## 5.2 球谐带分变量结果", ""]
    if not bundle.sph_available:
        lines += [f"{SPHERICAL_ABSENT}本节同 5.1，一并留空。", ""]
        return lines
    variables = sorted(bundle.sph_variable.index)
    rows = []
    for variable in variables:
        row = bundle.sph_variable.loc[variable]
        rows.append([variable] + [_fmt(row[n], 2) for n in names] + [row.idxmin()])
    lines += _table(["**变量**"] + [f"**{n}**" for n in names] + ["**最佳**"], rows)
    by_variable = bundle.sph_variable.mean(axis=0)
    by_band = bundle.sph_band_logrms.mean(axis=0)
    order_variable = by_variable.sort_values().index.tolist()
    order_band = by_band.sort_values().index.tolist()
    lines += [
        "",
        f"逐变量的球谐带 log-RMS×100 排序为 "
        + " < ".join(f"{n}（{_fmt(by_variable[n], 2)}）" for n in order_variable)
        + f"，按 §5.1 的频带口径汇总为 "
        + " < ".join(f"{n}（{_fmt(by_band[n], 2)}）" for n in order_band)
        + ("，两者一致。" if order_variable == order_band else
           "，**两者不一致**。")
        + "两者用的是同一批带功率样本，只是聚合维度不同（这里按变量汇总全部频带，"
        "§5.1 按频带汇总全部变量），排序不一致时说明模型间的优劣集中在个别频带上，"
        "选型时要指明关注哪一段尺度。",
        "",
    ]
    return lines


def _spherical_tests_section(bundle: WaveBundle) -> List[str]:
    """§5.3 球谐带配对检验：指标（频带）× 模型对。"""
    names = bundle.names
    lines = ["## 5.3 球谐带配对检验", ""]
    if not bundle.sph_available:
        lines += [f"{SPHERICAL_ABSENT}本节同 5.1，一并留空。", ""]
        return lines
    rows = []
    for band in bundle.sph_band_logrms.index:
        # 骨架要求「先按指标出现顺序、再按模型对字典序」，paired_tests 按列序出，
        # 这里补一次字典序排序
        for test in sorted(bundle.sph_tests.get(band, []), key=lambda t: (t.a, t.b)):
            rows.append([
                band, f"{test.a} vs {test.b}", _signed(test.delta_pct),
                _fmt(test.p_holm, 4), "是" if test.significant else "否", test.better,
            ])
    lines += _table(
        ["**指标**", "**模型对**", "**A相对B**", "**Holm p**", "**显著**", "**胜者**"],
        rows,
    )
    significant = [r for r in rows if r[4] == "是"]
    pair_count = len(rows) // max(1, len(bundle.sph_band_logrms.index))
    lines += [
        "",
        f"逐频带在{len(bundle.dates)}个日度样本上做 Wilcoxon 符号秩检验，"
        f"每个频带各自对{pair_count}个模型对做 Holm 校正（指标是频带本身，"
        "同一频带内的模型对构成一个检验族）。"
        + (f"全部 {len(rows)} 个「频带 × 模型对」组合里有 {len(significant)} 个达到显著，"
           "其余组合的差异被样本量或频带内的离散度淹没；"
           if significant else
           f"全部 {len(rows)} 个「频带 × 模型对」组合均未达显著，"
           "说明各模型在这些总阶数频带上的差异小于日度波动；")
        + "统计显著不代表业务重要性。",
        "",
    ]
    return lines


def _section_spectrum(bundle: WaveBundle, figures: Mapping[str, str]) -> List[str]:
    base = bundle.base
    names = bundle.names
    variables = bundle.spectrum_variables
    lines = [
        "# 5. 功率谱检验",
        "",
        f"产物保留两套功率谱定义：纬向 FFT 谱（`diagnostics/spectrum_by_init.csv`）"
        f"和球谐带功率（`scores.csv` 里 `spherical_bands` 的结果行）。"
        f"两者都只使用功率或功率谱，不包含相位；本报告分别给出完整结果，不用一套指标替代另一套。"
        + ("" if bundle.sph_available else SPHERICAL_ABSENT),
        "",
    ]
    lines += _spherical_bands_section(bundle)
    lines += _spherical_variables_section(bundle)
    lines += _spherical_tests_section(bundle)
    lines += [
        f"## 5.4 纬向 FFT 功率谱（完整）",
        "",
        f"纬向 FFT 对每个纬圈做一维 rFFT，经度方向波数 k=1..{FULL_K_MAX}，k=0 的纬向平均已置零。"
        f"纬向谱与球谐谱不是同一种变换：纬向谱保留完整 1–{FULL_K_MAX} 波数，"
        f"球谐带只汇总到总阶数 {MATCHED_K_MAX}。为避免把超高频近零观测功率的影响与"
        f"可训练频带混为一谈，本节同时给出全波数和 1–{MATCHED_K_MAX} 匹配尺度结果。",
        "",
    ]
    rows = [
        [f"全波数 1–{FULL_K_MAX}"] + [_fmt(bundle.zonal_full[n], 2) for n in names]
        + [bundle.zonal_full.idxmin(), "包含极高波数；受近零观测功率与数值端点影响较大"],
        [f"匹配尺度 1–{MATCHED_K_MAX}"] + [_fmt(bundle.zonal_matched[n], 2) for n in names]
        + [bundle.zonal_matched.idxmin(), "与球谐损失的最高阶数同范围，适合与球谐带对照"],
    ]
    lines += _table(
        ["**口径**"] + [f"**{n}**" for n in names] + ["**最佳**", "**解释**"], rows,
    )

    lines += ["", f"## 5.4.1 {_cn(len(WAVE_BANDS))}个波数带结果", ""]
    header = ["**波数带**"]
    for name in names:
        header += [f"**{name} logRMS**", f"**{name} 能量比**"]
    header += ["**最佳**"]
    rows = []
    for band in bundle.band_logrms.index:
        logrms = bundle.band_logrms.loc[band]
        ratios = bundle.band_ratio.loc[band]
        row = [band] + [v for name in names
                        for v in (_fmt(logrms[name], 2), _fmt(ratios[name], 4))]
        rows.append(row + [logrms.idxmin()])
    lines += _table(header, rows)
    lines += [
        "",
        f"波数带按行星尺度到小尺度划分。log-RMS 最低的模型在"
        + "、".join(
            f"{band.split()[0]}段是 {bundle.band_logrms.loc[band].idxmin()}"
            for band in bundle.band_logrms.index
        )
        + "；能量比为该波数带内预测谱与观测谱的平均功率之比（先沿波数取平均再相除），"
        "越接近 1 表示该尺度上的能量越不偏。"
        "高波数段（"
        + bundle.band_logrms.index[-1]
        + "）的 log-RMS 普遍高于低波数段，是纬向谱指标的固有特征："
        "高波数观测功率接近零，比值的对数发散，这一段的绝对值不宜与低波数段直接比较；"
        "能量比在同一段仍有意义，因为它不依赖逐波数的分母。",
        "",
        "![zonal_spectrum_bands](zonal_spectrum_bands.png)",
        "",
        f"图 5：{_cn(len(names))}模型分波数带（{' / '.join(b.split()[0] for b in bundle.band_logrms.index)}）"
        f"的纬向谱 log-RMS×100 对比。",
        "",
        "## 5.4.2 平均功率谱与比值曲线",
        "",
        f"下列曲线是{len(bundle.dates)}个日度功率谱的平均。黑线为 ERA5，"
        f"彩色线为{_cn(len(names))}个模型的预测谱；比值为 pred/obs，理想值为 1，"
        f"图中断面显示范围限定为 0.1–10。变量取 {base.spectrum_variable}。",
        "",
        "![mean_power_spectrum](mean_power_spectrum.png)",
        "",
        f"图 6：{base.spectrum_variable} 纬向功率谱（{len(bundle.dates)} 个共同日期平均，双对数坐标）。",
        "",
        "![spectrum_ratio](spectrum_ratio.png)",
        "",
        f"图 7：{base.spectrum_variable} 纬向谱 pred/obs 比值随波数的变化（纵轴限定 0.1–10）。",
        "",
        f"## 5.4.3 1–{MATCHED_K_MAX} 波数逐变量结果",
        "",
    ]
    rows = []
    for variable in sorted(variables):
        row = bundle.variable_matched.loc[variable]
        rows.append([variable] + [_fmt(row[n], 2) for n in names] + [row.idxmin()])
    lines += _table(["**变量**"] + [f"**{n}**" for n in names] + ["**最佳**"], rows)

    full_rank = bundle.variable_logrms.mean(axis=0).sort_values().index.tolist()
    matched_rank = bundle.variable_matched.mean(axis=0).sort_values().index.tolist()
    lines += [
        "",
        f"全波数 1–{FULL_K_MAX} 口径下的模型排序为 {' < '.join(full_rank)}，"
        f"1–{MATCHED_K_MAX} 匹配尺度口径下为 {' < '.join(matched_rank)}，"
        + ("两者一致，说明高波数段没有改变模型间格局。"
           if full_rank == matched_rank else
           "两者不一致，说明高波数段（近零观测功率主导）会改变排序，"
           "涉及业务可训练尺度时应以匹配尺度口径为准。"),
        "",
    ]

    # --- 5.5 主模型专项 ---
    if bundle.primary and bundle.reference:
        lines += _section_primary(bundle)

    lines += [""]
    lines += _heatmap_block(bundle, figures)
    lines += [""]
    lines += _spectrum_ratio_block(bundle)
    lines += [""]
    lines += _single_init_block(bundle, figures)
    return lines


def _section_primary(bundle: WaveBundle) -> List[str]:
    """FGVP_pinpu 这类「相对某模型做了谱优化」的主模型专项对比。"""
    primary, reference = bundle.primary, bundle.reference
    base = bundle.base
    names = bundle.names
    others = [n for n in names if n not in (primary, reference)]

    lines = [f"## 5.5 {primary} 相对 {reference} 的频谱专项对比", "",
             f"本节为{primary}（主模型）与{reference}（对照）的逐项对比，"
             f"用于判断频谱优化是否落地。", ""]

    rows = []
    for band in bundle.band_logrms.index:
        before = float(bundle.band_logrms.loc[band, reference])
        after = float(bundle.band_logrms.loc[band, primary])
        delta = after - before
        rows.append([
            band, _fmt(before, 2), _fmt(after, 2),
            _fmt(delta, 2),
            "改善" if delta < 0 else ("变差" if delta > 0 else "持平"),
        ])
    lines += _table(
        ["**波数带**", f"**{reference} logRMS**", f"**{primary} logRMS**",
         "**变化（负为好）**", "**判定**"], rows,
    )

    full_before = float(bundle.zonal_full[reference])
    full_after = float(bundle.zonal_full[primary])
    matched_before = float(bundle.zonal_matched[reference])
    matched_after = float(bundle.zonal_matched[primary])
    lines += [
        "",
        "**两种口径的总量**",
        "",
    ]
    lines += _table(
        ["**口径**", f"**{reference}**", f"**{primary}**", "**变化（负为好）**"],
        [
            [f"全波数 1–{FULL_K_MAX}", _fmt(full_before, 2), _fmt(full_after, 2),
             _fmt(full_after - full_before, 2)],
            [f"匹配尺度 1–{MATCHED_K_MAX}", _fmt(matched_before, 2), _fmt(matched_after, 2),
             _fmt(matched_after - matched_before, 2)],
            ["综合相对 RMSE", _fmt(base.relative_composite[reference]),
             _fmt(base.relative_composite[primary]),
             _fmt(float(base.relative_composite[primary]) - float(base.relative_composite[reference]), 2)],
            ["ACC (%)", _fmt(base.acc_mean[reference]), _fmt(base.acc_mean[primary]),
             _fmt(float(base.acc_mean[primary]) - float(base.acc_mean[reference]), 2)],
            ["FA 偏差 (pp)", _fmt(base.fa_bias[reference], 4), _fmt(base.fa_bias[primary], 4),
             _fmt(float(base.fa_bias[primary]) - float(base.fa_bias[reference]), 4)],
        ],
    )

    test = next((t for t in bundle.zonal_tests
                 if {t.a, t.b} == {primary, reference}), None)
    if test is not None:
        lines += [
            "",
            f"**配对检验（纬向谱 log-RMS，逐日）**：Wilcoxon 符号秩检验 "
            f"p = {_fmt(test.p_holm, 4)}（Holm 校正后），"
            + (f"显著优者为 {test.better}。" if test.significant else "未达显著。"),
        ]

    del others  # 只在上面挑对照模型时用过，这里不再需要
    lines += [
        "",
        f"在全部{_cn(len(names))}个模型中，{primary} 的纬向谱 log-RMS×100 排名第 "
        f"{int(bundle.zonal_full.rank().loc[primary])}，{reference} 排名第 "
        f"{int(bundle.zonal_full.rank().loc[reference])}。",
        "",
    ]
    return lines


def _heatmap_block(bundle: WaveBundle, figures: Mapping[str, str]) -> List[str]:
    """「§5.5 变量 × 时效 RMSE 热力图」（``图 8``）——给谱结论做 RMSE 侧对照。"""
    base = bundle.base
    if "rmse_heatmap" not in figures:
        return [
            "## 5.5 变量 × 时效 RMSE 热力图（谱结论的 RMSE 侧对照）",
            "",
            "**本批未出。** 只有单个 lead 时热力图没有可读的横向变化，"
            "渲染器不出这张图。",
        ]
    count = len(base.names)
    variables = len(base.common_variables)
    first, last, leads = _lead_span(base.rmse_by_lead[base.names[0]].index)
    return [
        "## 5.5 变量 × 时效 RMSE 热力图（谱结论的 RMSE 侧对照）",
        "",
        f"{count}幅子图，每个模型一幅：纵轴是{variables}个变量、横轴是全部{leads}个lead",
        f"（{first}–{last}h）。格子里的数**不是 RMSE 本身，而是该模型在该变量、该 lead 上",
        f"相对自己首时效（{first}h）的 RMSE 倍数**，除以首时效就把量纲与气候态差异约掉了。",
        "这一节是给前面几节的谱结论做**对照**用的，本身不是谱指标：谱上说某模型小尺度能量",
        "偏低，就该在这张图上看到对应的变量（通常是高层风场与位势高度）误差随 lead 加深更快；",
        "图上看不到对应信号，说明谱偏差还没传导到场误差，正文里要照实写，",
        "不能只报谱上的差距。",
        "",
        f"![multi_model_rmse_heatmap]({figures['rmse_heatmap']})",
        "",
        f"图 8：{variables}个变量 × {leads}个 lead 的 RMSE 倍数热力图"
        f"（各自除以本模型首时效 {first}h 的 RMSE；{_date_span(base.dates)} 共同日期平均）",
    ]


def _spectrum_ratio_block(bundle: WaveBundle) -> List[str]:
    """「§5.6 谱比随时效」（``图 9``）——产物给不出，见 :data:`SPECTRUM_RATIO_ABSENT`。"""
    base = bundle.base
    return [
        "## 5.6 谱比随时效",
        "",
        "横轴为波数（双对数），纵轴为 pred/obs，y=1 参考线画出；**每个 lead 一条曲线**，",
        "颜色由浅到深对应 lead 由短到长。§5.4.2 的 `spectrum_ratio` 是"
        f"{len(base.dates)}个日期、",
        "全部 lead 平均后的比值，看不出这条随时间的走向，两节要对着看。",
        "这一节回答的是「小尺度能量不足是随时间恶化，还是一开始就缺」：曲线整体贴着 1、",
        "随 lead 一起下移，是误差累积；曲线从最短时效就整体偏低、后续几乎不再下移，",
        "是模式本身的能量谱问题，**不是**预报时长带来的，调时效救不回来。",
        "",
        SPECTRUM_RATIO_ABSENT,
    ]


def _single_init_block(bundle: WaveBundle, figures: Mapping[str, str]) -> List[str]:
    """「§5.7 单起报谱曲线」（``图 10``）。"""
    base = bundle.base
    variable = base.spectrum_variable
    init = str(base.dates[0]) if base.dates else ""
    lines = [
        "## 5.7 单起报谱曲线",
        "",
        f"§5.4.2 是{len(base.dates)}个日期平均后的谱，平均会把个例差异抹平。这一节换成"
        f"**单个起报**（{init}）的谱，用来核对平均谱上的结论在个例上是否成立——平均谱上",
        "「小尺度偏低」如果只在少数个例出现，就不该写成模式的普遍特征。",
        "",
    ]
    if "spectrum_curve" in figures:
        lines += [
            f"![spectrum_curve]({figures['spectrum_curve']})",
            "",
            f"图 10：{init} 单起报的 {variable} 纬向功率谱"
            f"（双对数；黑色虚线为同时刻观测谱）",
        ]
    else:
        lines += [
            f"**本批未出。** 产物里没有 {variable} 的逐起报谱"
            f"（`diagnostics/spectrum_by_init.csv`），出不了这张图。",
        ]
    return lines


def _section_risks(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    variable = base.anomaly_variable
    model = base.anomaly_model
    # 骨架这一节原本是条件章节（异常检出），标题里的「尺度异常（高优先级）」只在
    # 真的超阈值时才成立。没检出的年份照抄标题，会出现「高优先级」下面跟着
    # 「不单独作为风险项」的自相矛盾，所以标题跟着结论走。
    detected = base.anomaly_ratio >= 3.0
    lines = ["# 6. 异常与数据质量风险", "",
             f"## 6.1 {model} {variable} "
             + ("尺度异常（高优先级）" if detected else "量级检查（未检出异常）"), ""]

    rows = []
    for lead in base.lead_relative.index:
        column = base.rmse_by_lead[model].get(variable)
        other_series = [base.rmse_by_lead[n].get(variable) for n in names if n != model]
        other_series = [s for s in other_series if s is not None and lead in s.index]
        if column is None or lead not in column.index or not other_series:
            continue
        median = float(np.median([float(s.loc[lead]) for s in other_series]))
        value = float(column.loc[lead])
        rows.append([f"{float(lead):g}"] + [_fmt(base.rmse_by_lead[n].get(variable, pd.Series()).get(lead), 4)
                                           for n in names]
                    + [_fmt(value / median, 3) if median else "—"])
    lines += _table(
        ["**Lead (h)**"] + [f"**{n}**" for n in names]
        + [f"**{model} / 其余{_cn(len(names) - 1)}模型中位数**"],
        rows,
    )
    ratio = base.anomaly_ratio
    lines += [
        "",
        f"{model} 的 {variable} RMSE 是其余模型中位数的 {_fmt(ratio, 2)} 倍"
        + ("，超出「同量级」的可接受范围（阈值 3×）。该差异在所有 lead 上一致存在，"
           "不是个别时效的抖动，更像是量纲/单位或变量定义层面的问题；"
           f"建议核对 {variable} 的预报与观测是否使用同一套单位与层次定义后，"
           f"再采信该变量的结论，并在修复前不要用 {variable} 参与模型排名。"
           if ratio >= 3.0 else
           "，未超过 3× 的异常阈值，属于正常的模型间水平差异，"
           "不单独作为风险项，但也说明该变量的模型间可比性弱于其他变量。"),
        "",
        "## 6.2 FA 幅度偏差",
        "",
        f"{_cn(len(base.fa_variables))}个 FA 变量的幅度偏差（100×mean\\|pred/obs−1\\|）为："
        + "、".join(f"{n} {_fmt(base.fa_bias[n], 4)} pp" for n in names)
        + f"。偏差最小（幅度最真实）的是 {base.fa_bias.idxmin()}，"
        f"偏差最大的是 {base.fa_bias.idxmax()}。"
        "FA 偏差是绝对值口径，丢掉了「偏强/偏弱」的方向，只能说明幅度失真多少，"
        "不能据此断言某个模型偏平滑或偏噪；要判方向需回到逐日 pred/obs 比值。",
        "",
        "## 6.3 原始场独立复算不足（中高优先级）",
        "",
        f"本报告的纬向谱、综合相对 RMSE、ACC 与 FA 全部读自各模型产物目录里的 "
        f"scores.csv 汇总行与 diagnostics/ 下的谱表，没有回到原始预报场与观测场做独立复算。"
        f"产物目录不含原始场，这一点无法在本报告内弥补。"
        f"因此评测环节（变量映射、插值、单位换算、谱变换）如果出错，"
        f"本报告会原样继承且无法自查。结论用于模型间横向比较是充分的；"
        f"用于绝对谱保真度认定前，建议抽一个日期回到原始场复算一次谱。",
        "",
        "## 6.4 ACC 定义风险（中高优先级）",
        "",
        f"{base.acc_variable} 的首时效 ACC 已达 "
        f"{_fmt(_first_lead_value(base, base.acc_variable), 3)}%"
        f"（最高模型 {base.acc_mean.idxmax()} 全程 {_fmt(base.acc_mean.max())}%）。"
        f"距平相关系数在近端接近饱和，对模型差异几乎不敏感，"
        f"用它排名会把不同模型压成几乎同分，不能单独作为选型依据；"
        f"建议与 RMSE、纬向谱三项同看。",
        "",
    ]
    return lines


def _section_acceptance(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    rmse_best = base.relative_composite.idxmin()
    acc_best = base.acc_mean.idxmax()
    fa_best = base.fa_bias.idxmin()
    zonal_best = bundle.zonal_full.idxmin()
    # 与 §3.1 的「RMSE + ACC + FA + 纬向 FFT 谱」口径保持一致，不用 base.rank_score。
    rank_four = _spectrum_inclusive_rank(base, bundle)
    rank_best = rank_four.idxmin()
    worst = base.relative_composite.idxmax()

    rows = [
        ["综合精度优先（默认）", rmse_best,
         f"综合相对 RMSE 最低（{_fmt(base.relative_composite[rmse_best])}）"],
        ["大尺度形势场", acc_best, f"ACC 最高（{_fmt(base.acc_mean[acc_best])}%）"],
        ["幅度与谱真实性优先", zonal_best,
         f"纬向谱 log-RMS×100 最低（{_fmt(bundle.zonal_full[zonal_best], 2)}）"],
        ["强度与活跃度", fa_best, f"FA 偏差最小（{_fmt(base.fa_bias[fa_best], 4)} pp）"],
        ["多维度均衡", rank_best, f"综合排名分最低（{_fmt(rank_four[rank_best], 2)}）"],
        ["暂缓采用", worst, f"综合相对 RMSE 最高（{_fmt(base.relative_composite[worst])}）"],
    ]
    if bundle.primary:
        rows.append([f"谱优化版本（{bundle.primary}）", bundle.primary,
                     f"纬向谱 log-RMS×100 {_fmt(bundle.zonal_full[bundle.primary], 2)}，"
                     f"综合相对 RMSE {_fmt(base.relative_composite[bundle.primary])}"])

    lines = ["# 7. 验收结论与选型建议", ""]
    lines += _table(["**目标**", "**推荐**", "**结论依据**"], rows)
    lines += [
        "",
        f"**总体判定：在 {len(bundle.dates)} 个共同日期（{_date_span(bundle.dates)}）上，"
        f"{rmse_best} 的综合相对 RMSE 最低、{acc_best} 的 ACC 最高、"
        f"{zonal_best} 的纬向谱保真度最好、{fa_best} 的幅度最真实；"
        f"四个口径没有指向同一个模型，验收时应按实际业务目标取权重，"
        f"不宜只凭单一指标定论。**",
        "",
    ]
    return lines


def _appendix(bundle: WaveBundle, figures: Mapping[str, str],
              archive: Optional[str], out_dir: Path) -> List[str]:
    base = bundle.base
    names = bundle.names
    first, last, leads = _lead_span(base.rmse_by_lead[names[0]].index)
    try:
        step = float(np.diff(sorted(set(base.rmse_by_lead[names[0]].index)))[0])
    except (IndexError, ValueError):
        step = float("nan")

    total_bytes = 0
    for name in names:
        for path in base.archives[name].root.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size

    lines = ["# 附录：数据与复算证据", ""]
    lines += _table(
        ["**项目**", "**值**"],
        [
            ["产物目录", archive or "、".join(names)],
            ["产物大小", f"{total_bytes / 1024 / 1024:.1f} MB"],
            ["SHA-256", "—（多目录产物，未逐文件计算）"],
            ["模型", "、".join(names)],
            ["共同日期", f"{len(bundle.dates)} 天，{_date_span(bundle.dates)}"],
            ["lead", f"{leads} 个，{first}–{last}h，间隔 {step:g}h"],
            ["分析结果目录", str(out_dir)],
        ],
    )
    lines += [
        "",
        f"**本补充报告不修订、不替代原评估报告；全部结论仅来自用户指定产物目录中的"
        f"{_cn(len(names))}个 single 输出"
        + ("，球谐带功率取自 scores.csv 里 `spherical_bands` 的结果行。**"
           if bundle.sph_available else
           "，产物里没有可用的球谐带功率结果，相关章节已标注「本批未出」。**"),
        "",
    ]
    # 骨架把可选块定在「附录那张表之后」，图件清单是渲染器自己加的一节，排在可选块后面。
    lines += _region_block(bundle, figures)
    lines += [""]
    lines += _season_block()
    lines += [
        "",
        "## 附：图件清单",
        "",
    ]
    for index, (key, filename) in enumerate(figures.items(), start=1):
        lines.append(f"{index}. `{filename}`（{key}）")
    lines.append("")
    return lines


def render_wave(bundle: WaveBundle, figures: Mapping[str, str], *,
                archive: Optional[str] = None, out_dir: Path = Path(".")) -> str:
    """把 :class:`WaveBundle` 渲染成骨架形状的完整 Markdown。"""
    names = bundle.names
    shown = sorted(names)
    lines: List[str] = []
    lines += [
        "# XMETAI single（确定性版本）评估检验补充报告",
        "",
        f"{'、'.join(shown)} {_cn(len(names))}模型输出与纬向 FFT、球谐谱补充分析",
        "",
        f"数据产物：{archive or '、'.join(names)}",
        "",
        f"报告日期：{bundle.dates[-1]}    模型：{'、'.join(shown)}    评估层级：输出级复核",
        "",
        f"**报告性质：补充分析。本报告不修订、不替代原输出级评估检验报告，"
        f"仅补充{_cn(len(names))}个 single 输出的统一重算、纬向 FFT 和球谐带功率谱结果。**",
        "",
    ]
    for section in (
        _section_conclusion(bundle),
        _section_scope(bundle),
        _section_overall(bundle),
        _section_rmse(bundle),
        _section_spectrum(bundle, figures),
        _section_risks(bundle),
        _section_acceptance(bundle),
        _appendix(bundle, figures, archive, out_dir),
    ):
        lines += section
        lines.append("")
    return "\n".join(lines)


# ====================================================================== 入口


def build_wave_report(archives: Sequence[Archive], out_dir: Path, *,
                      archive: Optional[str] = None) -> Dict[str, Path]:
    """出图 + 写 ``REPORT.md``，返回产物名 → 路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = analyze_wave(archives)
    names = bundle.names
    base = bundle.base

    figures = {
        "overall_summary": "overall_summary.png",
        "rmse_by_lead_range": "rmse_by_lead_range.png",
        "zonal_spectrum_bands": "zonal_spectrum_bands.png",
        "mean_power_spectrum": "mean_power_spectrum.png",
        "spectrum_ratio": "spectrum_ratio.png",
    }
    if bundle.sph_available:      # 图 3 / 图 4，插在分时效图与分波数带图之间
        figures = {
            "overall_summary": "overall_summary.png",
            "rmse_by_lead_range": "rmse_by_lead_range.png",
            "spherical_bands_logrms": "spherical_bands_logrms.png",
            "spherical_bands_ratio": "spherical_bands_ratio.png",
            "zonal_spectrum_bands": "zonal_spectrum_bands.png",
            "mean_power_spectrum": "mean_power_spectrum.png",
            "spectrum_ratio": "spectrum_ratio.png",
        }

    # 图 1：综合四指标柱状
    _bar_panels(
        out_dir, figures["overall_summary"],
        [("综合相对 RMSE", base.relative_composite),
         ("ACC (%)", base.acc_mean),
         ("FA 偏差 (pp)", base.fa_bias),
         ("纬向谱 RMS×100", bundle.zonal_full)],
        suptitle=f"{_cn(len(names))}模型总体指标对比",
    )

    # 图 2：分时效段综合相对 RMSE
    band_rows = {}
    for low, high in ((0.0, 120.0), (120.0, 240.0), (240.0, 360.0)):
        subset = base.lead_relative[(base.lead_relative.index > low) & (base.lead_relative.index <= high)]
        if not subset.empty:
            band_rows[f"{low:g}–{high:g}h"] = subset.mean(axis=0)
    _grouped_bars(
        out_dir, figures["rmse_by_lead_range"], pd.DataFrame(band_rows).T,
        suptitle=f"{_cn(len(names))}模型分时效段综合相对 RMSE（越低越好）",
        ylabel="综合相对 RMSE",
    )

    # 图 3 / 图 4：球谐带 log-RMS 与均值比
    if bundle.sph_available:
        _grouped_bars(
            out_dir, figures["spherical_bands_logrms"], bundle.sph_band_logrms.T,
            suptitle=f"{_cn(len(names))}模型分球谐总阶数频带 log-RMS×100（越低越好）",
            ylabel="log-RMS×100",
        )
        _grouped_bars(
            out_dir, figures["spherical_bands_ratio"], bundle.sph_band_ratio.T,
            suptitle=f"{_cn(len(names))}模型分球谐总阶数频带 pred/obs 均值比",
            ylabel="pred / obs", ref_line=1.0,
        )

    # 图 5：分波数带 log-RMS
    _grouped_bars(
        out_dir, figures["zonal_spectrum_bands"], bundle.band_logrms.T,
        suptitle=f"{_cn(len(names))}模型分波数带纬向谱 log-RMS×100（越低越好）",
        ylabel="log-RMS×100",
    )

    # 图 6 / 图 7：平均功率谱与比值曲线（取优先变量）
    variable = base.spectrum_variable
    spectra = {name: frame[variable] for name, frame in bundle.spectra.items()}
    _plot_mean_spectrum(
        out_dir, figures["mean_power_spectrum"], spectra, variable=variable,
        suptitle=f"{variable} 纬向功率谱（{len(bundle.dates)} 日期平均）",
    )
    _plot_spectrum_ratio(
        out_dir, figures["spectrum_ratio"], spectra, variable=variable,
        suptitle=f"{variable} 纬向谱 pred/obs 比值（理想值 1）",
    )

    # 图 8：变量 × 时效 RMSE 热力图（§5.5），给谱结论做 RMSE 侧对照。格子是
    # 「除以本模型首时效」的倍数——各变量单位不同，不归一化就没法共用一根色标。
    ratios: Dict[str, pd.DataFrame] = {}
    for name in names:
        frame = base.rmse_by_lead[name]
        if frame.empty:
            continue
        lead_values = sorted(frame.index)
        if len(lead_values) < 2:
            continue
        ratios[name] = frame.div(frame.loc[lead_values[0]].replace(0.0, np.nan))
    if ratios:
        figures["rmse_heatmap"] = "multi_model_rmse_heatmap.png"
        DetPlotter().plot_rmse_heatmap(
            ratios, base.common_variables,
            leads=sorted(base.rmse_by_lead[names[0]].index),
            suptitle=f"{_cn(len(names))}模型 RMSE 相对本模型首时效的倍数",
            save_path=out_dir / figures["rmse_heatmap"],
        )

    # 图 10：单起报谱曲线（§5.7）。取共同日期里最早的那个起报，其余日期仍进 §5.4.2
    # 的平均谱。产物没有逐起报谱就**不出图**，报告那边写「本批未出」。
    if base.dates:
        init = str(base.dates[0])
        single: Dict[str, pd.DataFrame] = {}
        for archive in archives:
            try:
                pred, obs = load_spectrum(archive.root, [init], base.spectrum_variable,
                                          archive.name)
            except (FileNotFoundError, ValueError):
                continue
            if init in pred.columns:
                single[archive.name] = pd.DataFrame({"pred": pred[init], "obs": obs[init]})
        if single:
            figures["spectrum_curve"] = f"spectrum_curve_{base.spectrum_variable}_{init}.png"
            _plot_mean_spectrum(
                out_dir, figures["spectrum_curve"], single,
                variable=base.spectrum_variable,
                suptitle=f"{base.spectrum_variable} 纬向功率谱（{init} 单起报）",
            )

    # 图 L1 / 图 L2：分纬度带（可选块）。没有共同的 region 行就**不出图**，也不往
    # figures 里塞不存在的文件名——报告那边会写「本批未出」，指一张没生成的图比不指更糟。
    if base.region_names:
        band_labels = [
            region_display(region, base.region_labels) for region in base.region_names
        ]
        figures["lat_band_lead"] = "lat_band_rmse_vs_lead.png"
        figures["lat_band_summary"] = "lat_band_summary.png"
        plotter = DetPlotter()
        plotter.plot_model_panels(
            {
                name: pd.DataFrame({
                    label: base.region_lead_relative[region][name]
                    for region, label in zip(base.region_names, band_labels)
                })
                for name in names
            },
            band_labels,
            ylabel="综合相对 RMSE（100 = 该带内几何均值）",
            ref_line=BASELINE,
            ncols=min(len(band_labels), 3),
            suptitle="分纬度带综合相对 RMSE 随预报时效的变化",
            save_path=out_dir / figures["lat_band_lead"],
        )
        summary = base.region_relative_composite.copy()
        summary.index = band_labels
        plotter.plot_group_bars(
            summary,
            ylabel="综合相对 RMSE（100 = 该带内几何均值）",
            ref_line=BASELINE,
            suptitle="分纬度带综合相对 RMSE",
            save_path=out_dir / figures["lat_band_summary"],
        )

    text = render_wave(bundle, figures, archive=archive, out_dir=out_dir)
    report_path = out_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"已保存: {report_path}")

    artifacts: Dict[str, Path] = {"report": report_path}
    for key, filename in figures.items():
        artifacts[key] = out_dir / filename
    return artifacts
