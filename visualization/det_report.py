#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性多模型评估报告：加载批次归档 → 统一口径统计 → 渲染 Markdown。

**输入是「批次归档目录」**，一个模型一个，目录形状由
``vfc/regr_ens.py`` / ``vfc/regr_summary.py`` 的写入代码决定::

    <model_root>/
      summary.csv                  # 每起报日一行：init_date, n_members, n_leads,
                                   #   variables, rmse_<v>_{24h,last,mean}
      batch_meta.json              # pred_root / target_zarr / n_dates / dates /
                                   #   n_ok / failures / outdir_root / options
      <YYYYMMDD>/
        rmse_<date>_det.csv        # index=lead_h, columns=<变量>
        acc_<date>_det.csv         # index=lead_h, columns=<变量>
        fa_<date>_det.csv          # index=lead_h, columns=<变量>_{pred,obs,bias,ratio}
        spectrum_<date>_<变量>.csv # index=波数 k, columns={det_pred, det_obs}

**统一口径**：正文与附录图都用 N 个模型共同的日期（原报告的正文 169 天 /
附录 349 天双口径不保留），所以**不读** ``det_summary_*.csv``——那批表是全日期口径，
混进来会让表与图各说各话。

**派生指标口径**（原报告没有生成脚本，这几个定义是反推的，写在这里备查）::

    相对 RMSE       逐变量 100 × v_m / geomean_模型(v)，再对共享变量等权平均
                    （几何均值基线 = 100；原报告 89.258/96.354/99.004/117.55 反推验证过）
    ACC (%)         共同日期 × 全 lead 的 ACC 变量平均
    FA 偏差 (pp)    100 × mean(|ratio − 1|)，对全部 FA 变量、全 lead 平均
    频谱对数 RMS×100 100 × mean_v sqrt(mean_k (log10(pred_k / obs_k))²)
    平均排名分      上述四个指标上名次的平均

**显著性**：逐日期配对 Wilcoxon 符号秩检验 + Holm 多重校正。
Holm 是手写的五行（``_holm``），不为它引入 statsmodels。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from visualization.det_plots import DetPlotter

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except ImportError:  # 推迟到真正做检验时报错，让"只出图"也能跑
    _scipy_wilcoxon = None


#: 几何均值基线：相对 RMSE 的 100 就是这个意思
BASELINE = 100.0

#: ACC / 谱图优先用的变量，缺了就退到字典序第一个
PREFERRED_VARIABLE = "z500"

#: ACC 在首时效高于这个百分数就算"饱和"，写进 6.3 的风险提示
SATURATED_ACC = 99.0

#: 分时效段的边界（左开右闭，单位 h）；空段不输出，段名按段内实际 lead 现写
LEAD_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 120.0),
    (120.0, 240.0),
    (240.0, 360.0),
)

_CN_NUM = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}

_FA_SUFFIXES = ("_pred", "_obs", "_bias", "_ratio")
_SPECTRUM_PRED = ("det_pred", "pred", "ensmean_pred")
_SPECTRUM_OBS = ("det_obs", "obs", "ensmean_obs")


# ====================================================================== 加载


class Archive(NamedTuple):
    """一个模型的批次归档。"""

    name: str
    root: Path
    summary: pd.DataFrame
    meta: Dict[str, object]
    dates: List[str]  # 磁盘上真实存在的 YYYYMMDD 子目录


def load_archive(name: str, root: Path) -> Archive:
    """读一个批次归档目录。

    Raises:
        FileNotFoundError: 目录、``summary.csv`` 或 ``batch_meta.json`` 不存在。
        ValueError: ``summary.csv`` 缺 ``init_date`` 列或没有数据行。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"归档目录不存在: {root}")

    summary_path = root / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{name}: 缺 summary.csv（{summary_path}）")
    summary = pd.read_csv(summary_path)
    if "init_date" not in summary.columns:
        raise ValueError(f"{name}: summary.csv 缺 init_date 列；现有列: {list(summary.columns)}")
    if summary.empty:
        raise ValueError(f"{name}: summary.csv 没有数据行")
    summary["init_date"] = summary["init_date"].astype(str).str.strip()
    for column in summary.columns:
        if column.startswith("rmse_"):
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    meta_path = root / "batch_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{name}: 缺 batch_meta.json（{meta_path}）")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    dates = sorted(
        entry.name for entry in root.iterdir()
        if entry.is_dir() and re.fullmatch(r"\d{8}", entry.name)
    )
    return Archive(name=str(name), root=root, summary=summary, meta=meta, dates=dates)


def common_dates(archives: Sequence[Archive]) -> List[str]:
    """所有模型**和它们的 summary.csv** 都有的日期（升序）。"""
    if not archives:
        raise ValueError("没有模型")
    shared: Optional[set] = None
    for archive in archives:
        own = set(archive.summary["init_date"])
        shared = own if shared is None else (shared & own)
    return sorted(shared or set())


def summary_variables(archive: Archive) -> List[str]:
    """``summary.csv`` 的 ``variables`` 列里声明过的变量（并集，字典序）。"""
    names: set = set()
    if "variables" in archive.summary.columns:
        for value in archive.summary["variables"].dropna():
            for item in str(value).split(","):
                item = item.strip()
                if item:
                    names.add(item)
    if not names:  # 没有 variables 列就退回从 rmse_<v>_mean 列名反推
        names = set(rmse_mean_columns(archive.summary))
    return sorted(names)


def rmse_mean_columns(summary: pd.DataFrame) -> List[str]:
    """``rmse_<v>_mean`` 列名里的变量（字典序）。"""
    found = []
    for column in summary.columns:
        match = re.fullmatch(r"rmse_(.+)_mean", str(column))
        if match:
            found.append(match.group(1))
    return sorted(found)


def _pick_metric_file(directory: Path, stem: str, date: str) -> Optional[Path]:
    """照 ``vfc/regr_summary.py:227`` 的容错顺序找一个逐日指标文件。"""
    for candidate in (f"{stem}_{date}_det.csv", f"{stem}_{date}.csv"):
        path = directory / candidate
        if path.is_file():
            return path
    for path in sorted(directory.glob(f"{stem}_*")):
        if path.name.endswith(("_ensmean.csv", "_ens.csv")):
            continue
        return path
    return None


def _read_lead_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_numeric(frame.index, errors="coerce")
    frame = frame[frame.index.notna()]
    frame.index.name = "lead_h"
    return frame.sort_index()


def probe_dates(root: Path, dates: Iterable[str], stem: str) -> List[str]:
    """这些日期里，**缺** ``stem`` 逐日文件的（升序）。用于第 2 节完整性检查。"""
    missing = []
    for date in dates:
        directory = Path(root) / date
        if not directory.is_dir() or _pick_metric_file(directory, stem, date) is None:
            missing.append(date)
    return sorted(missing)


def load_by_lead(root: Path, stem: str, dates: Sequence[str], name: str = "") -> pd.DataFrame:
    """逐日 ``<stem>_<date>_det.csv`` 按 lead 平均成一张 lead × 变量表。

    Raises:
        FileNotFoundError: 任何一个日期缺这个文件——按 SKILL.md 的硬性规则
            不静默跳过（跳过会让表和图的样本量对不上，且没人会发现）。
    """
    frames = []
    for date in dates:
        path = _pick_metric_file(Path(root) / date, stem, date)
        if path is None:
            raise FileNotFoundError(
                f"{name or root}: {date} 目录下没有 {stem}_*.csv；"
                f"该模型归档可能被裁剪过（只有 summary.csv 是算不出 ACC/FA/谱的）"
            )
        frames.append(_read_lead_frame(path))
    if not frames:
        raise ValueError(f"{name or root}: 没有可读的 {stem} 逐日文件")
    stacked = pd.concat(frames, axis=0)
    return stacked.groupby(level=0).mean().sort_index()


def load_fa_ratio(root: Path, dates: Sequence[str], name: str = "") -> pd.DataFrame:
    """逐日 ``fa_<date>_det.csv`` → lead × 变量 的 ratio 表。

    ``ratio`` 列缺失时由 ``pred / obs`` 现算（``vfc/regr_summary.py:297`` 同款兜底）。
    """
    frames = []
    for date in dates:
        path = _pick_metric_file(Path(root) / date, "fa", date)
        if path is None:
            raise FileNotFoundError(f"{name or root}: {date} 目录下没有 fa_*.csv")
        raw = _read_lead_frame(path)
        kinds: Dict[str, set] = {}
        for column in raw.columns:
            for suffix in _FA_SUFFIXES:
                if str(column).endswith(suffix):
                    # 存的是去掉下划线的种类名（pred/obs/bias/ratio），后面按它判断
                    kinds.setdefault(str(column)[: -len(suffix)], set()).add(suffix.lstrip("_"))
                    break
        columns: Dict[str, pd.Series] = {}
        for variable, suffixes in kinds.items():
            if "ratio" in suffixes:
                columns[variable] = raw[f"{variable}_ratio"]
            elif {"pred", "obs"} <= suffixes:
                columns[variable] = raw[f"{variable}_pred"] / raw[f"{variable}_obs"].replace(0.0, np.nan)
        if columns:
            frames.append(pd.DataFrame(columns))
    if not frames:
        raise ValueError(f"{name or root}: fa 逐日文件里没有任何 <变量>_ratio 列")
    stacked = pd.concat(frames, axis=0)
    return stacked.groupby(level=0).mean().sort_index()


def load_spectrum(root: Path, dates: Sequence[str], variable: str, name: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """逐日 ``spectrum_<date>_<变量>.csv`` → ``(pred, obs)`` 两张 波数 × 日期 表。

    Raises:
        FileNotFoundError: 某个日期没有该变量的谱文件。
    """
    preds: Dict[str, pd.Series] = {}
    observations: Dict[str, pd.Series] = {}
    for date in dates:
        path = Path(root) / date / f"spectrum_{date}_{variable}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"{name or root}: 缺 {path.name}（{date} 没算 {variable} 的纬向谱？）"
            )
        raw = _read_lead_frame(path)
        pred_column = next((c for c in _SPECTRUM_PRED if c in raw.columns), None)
        obs_column = next((c for c in _SPECTRUM_OBS if c in raw.columns), None)
        if pred_column is None or obs_column is None:
            raise ValueError(
                f"{path.name}: 找不到预报/观测谱列；现有列: {list(raw.columns)}"
            )
        preds[date] = raw[pred_column]
        observations[date] = raw[obs_column]
    if not preds:
        raise ValueError(f"{name or root}: 没有可读的 {variable} 谱文件")
    return pd.DataFrame(preds), pd.DataFrame(observations)


# ================================================================== 统计检验


def _wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """配对 Wilcoxon 符号秩检验的 p 值；样本不足或全部相等时返回 1.0。"""
    if _scipy_wilcoxon is None:
        raise ImportError(
            "做配对检验需要 scipy（scipy.stats.wilcoxon）；"
            "当前环境没有装，装上再跑，或不要调用报告生成"
        )
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 2:
        return float("nan")
    if np.allclose(left, right, equal_nan=True):
        return 1.0
    try:
        return float(_scipy_wilcoxon(left, right).pvalue)
    except ValueError:
        return 1.0


def _holm(pvalues: Sequence[float]) -> List[float]:
    """Holm 逐步下降法校正；p 里带 nan 的原样返回 nan。"""
    index = [i for i, p in enumerate(pvalues) if math.isfinite(p)]
    ordered = sorted(index, key=lambda i: pvalues[i])
    adjusted = [float("nan")] * len(pvalues)
    running = 0.0
    total = len(ordered)
    for rank, i in enumerate(ordered):
        running = max(running, (total - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


class PairTest(NamedTuple):
    """一对模型的检验结论。"""

    a: str
    b: str
    mean_a: float
    mean_b: float
    delta_pct: float  # (A − B) / B × 100
    p_raw: float
    p_holm: float
    significant: bool
    better: str  # 显著优者；不显著时是 "—"


def paired_tests(
    frame: pd.DataFrame,
    *,
    higher_is_better: bool = False,
) -> List[PairTest]:
    """``frame`` 是 日期 × 模型，两两配对做 Wilcoxon + Holm。

    Args:
        frame: index=配对样本（日期），columns=模型名。
        higher_is_better: ACC 这类"越大越好"的指标置 True。
    """
    names = [str(c) for c in frame.columns]
    comparisons = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if not comparisons:
        return []

    raw, means = [], {}
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce")
        means[name] = float(values.mean()) if values.notna().any() else float("nan")
    for a, b in comparisons:
        raw.append(_wilcoxon_p(frame[a], frame[b]))
    holm = _holm(raw)

    tests = []
    for (a, b), p_raw, p_holm in zip(comparisons, raw, holm):
        significant = math.isfinite(p_holm) and p_holm < 0.05
        if significant:
            a_wins = means[a] > means[b] if higher_is_better else means[a] < means[b]
            better = a if a_wins else b
        else:
            better = "—"
        base = means[b]
        delta = BASELINE * (means[a] - means[b]) / base if base else float("nan")
        tests.append(PairTest(a, b, means[a], means[b], delta, p_raw, p_holm, significant, better))
    return tests


def variable_win_counts(
    by_variable: Mapping[str, pd.DataFrame],
    names: Sequence[str],
) -> pd.DataFrame:
    """逐变量配对检验 → 每个模型对的「A 显著更优 / B 显著更优 / 无差异」计数。

    Args:
        by_variable: ``{变量: 日期 × 模型 表}``（RMSE 这类越小越好）。
    """
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wins_a = wins_b = ties = 0
            for frame in by_variable.values():
                if a not in frame.columns or b not in frame.columns:
                    continue
                p = _wilcoxon_p(frame[a], frame[b])
                if not (math.isfinite(p) and p < 0.05):
                    ties += 1
                elif float(frame[a].mean()) < float(frame[b].mean()):
                    wins_a += 1
                else:
                    wins_b += 1
            rows.append({
                "模型对": (a, b),
                "A 显著更优": wins_a,
                "B 显著更优": wins_b,
                "无显著差异": ties,
                "总变量数": wins_a + wins_b + ties,
            })
    return pd.DataFrame(rows)


# ================================================================== 派生指标


def _relative(table: pd.DataFrame) -> pd.DataFrame:
    """逐行（变量）做 100 × 值 / 几何均值，columns=模型、index=变量。"""
    geometric = np.exp(np.log(table.where(table > 0)).mean(axis=1))
    return BASELINE * table.div(geometric, axis=0)


def _composite(relative: pd.DataFrame) -> pd.Series:
    """对共享变量等权平均成每个模型一个数。"""
    return relative.mean(axis=0, skipna=True)


def _rank(series: pd.Series, *, higher_is_better: bool = False) -> pd.Series:
    """名次（1 最好）；nan 排最后。"""
    ordered = series.astype(float)
    return ordered.rank(ascending=not higher_is_better, method="min")


# ================================================================== 汇总分析


class Bundle(NamedTuple):
    """算完的一切。渲染只读它，不再碰文件。"""

    names: List[str]
    archives: Dict[str, Archive]
    dates: List[str]
    common_variables: List[str]
    acc_variable: str
    fa_variables: List[str]
    spectrum_variable: str
    spectrum_variables: List[str]
    rmse_by_date: Dict[str, pd.DataFrame]  # 模型 -> 日期 × 变量
    acc_by_lead: Dict[str, pd.DataFrame]
    fa_ratio_by_lead: Dict[str, pd.DataFrame]
    rmse_by_lead: Dict[str, pd.DataFrame]
    spectrum: Dict[str, pd.DataFrame]  # 模型 -> 波数 × {pred, obs}
    relative: pd.DataFrame  # 变量 × 模型
    per_date_relative: pd.DataFrame  # 日期 × 模型（综合相对 RMSE）
    relative_composite: pd.Series
    acc_mean: pd.Series
    fa_bias: pd.Series
    spectrum_rms: pd.Series
    rank_score: pd.Series
    per_date_acc: pd.DataFrame
    per_variable_rmse: Dict[str, pd.DataFrame]
    overall_tests: List[PairTest]
    acc_tests: List[PairTest]
    win_counts: pd.DataFrame
    anomaly_variable: str
    anomaly_model: str
    anomaly_ratio: float
    lead_relative: pd.DataFrame  # lead × 模型（逐 lead 综合相对 RMSE）


def _lead_relative(rmse_by_lead: Mapping[str, pd.DataFrame], names: Sequence[str],
                   variables: Sequence[str]) -> pd.DataFrame:
    """逐 lead 的综合相对 RMSE（lead × 模型）。"""
    columns = {}
    for name in names:
        frame = rmse_by_lead[name].reindex(columns=variables)
        columns[name] = frame
    stacked = pd.concat(columns, axis=1, keys=names)  # 列 = (模型, 变量)
    per_variable = {v: stacked.xs(v, level=1, axis=1) for v in variables}
    relative = [_relative(per_variable[v]) for v in variables]
    return sum(relative) / len(relative)


def analyze(
    archives: Sequence[Archive],
    *,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
) -> Bundle:
    """把 N 个归档算成一份 :class:`Bundle`。

    Raises:
        ValueError: 共同日期太少（< 2）或没有共享的 RMSE 变量。
        FileNotFoundError: 共同日期上缺逐日文件——不静默跳过。
    """
    if len(archives) < 2:
        raise ValueError("多模型报告至少要两个模型")
    names = [archive.name for archive in archives]
    by_name = {archive.name: archive for archive in archives}
    dates = common_dates(archives)
    if len(dates) < 2:
        raise ValueError(f"共同日期只有 {len(dates)} 天，做不了配对检验: {dates}")

    # --- 变量集：所有模型 summary 里都有、且真的算出了数的 RMSE 变量 ---
    shared = set(rmse_mean_columns(archives[0].summary))
    for archive in archives[1:]:
        shared &= set(rmse_mean_columns(archive.summary))
    variables = sorted(
        v for v in shared
        if all(archive.summary[f"rmse_{v}_mean"].notna().any() for archive in archives)
    )
    if not variables:
        raise ValueError("这些模型没有共享的 rmse_<变量>_mean 列，算不出分变量结果")

    rmse_by_date: Dict[str, pd.DataFrame] = {}
    for archive in archives:
        indexed = archive.summary.set_index("init_date")
        columns = {v: pd.to_numeric(indexed[f"rmse_{v}_mean"], errors="coerce") for v in variables}
        rmse_by_date[archive.name] = pd.DataFrame(columns).reindex(index=dates)

    # --- 逐日文件（缺就报错，不静默跳过）---
    # acc_<date>_det.csv 里存的是相关系数本身（-1..1），报告统一按百分数呈现，
    # 所以在入口乘 100，下游（表、图、检验）拿到的都是 %。
    acc_by_lead = {
        a.name: load_by_lead(a.root, "acc", dates, a.name) * BASELINE for a in archives
    }
    fa_ratio_by_lead = {a.name: load_fa_ratio(a.root, dates, a.name) for a in archives}
    rmse_by_lead = {a.name: load_by_lead(a.root, "rmse", dates, a.name) for a in archives}

    acc_columns = set(acc_by_lead[names[0]].columns)
    for name in names[1:]:
        acc_columns &= set(acc_by_lead[name].columns)
    acc_variable = _pick_variable(acc_columns, variables)

    fa_columns = set(fa_ratio_by_lead[names[0]].columns)
    for name in names[1:]:
        fa_columns &= set(fa_ratio_by_lead[name].columns)
    fa_variables = sorted(fa_columns)
    if not fa_variables:
        raise ValueError("这些模型没有共享的 <变量>_ratio 列，算不出 FA 偏差")

    spectrum_variables = None
    for archive in archives:
        # 谱文件在 <YYYYMMDD>/ 里，不在归档根下
        prefix = f"spectrum_{dates[0]}_"
        own = {
            path.name[len(prefix): -4]
            for path in (archive.root / dates[0]).glob(f"{prefix}*.csv")
        }
        spectrum_variables = own if spectrum_variables is None else (spectrum_variables & own)
    spectrum_variables = sorted(spectrum_variables or set())
    if not spectrum_variables:
        raise ValueError(f"这些模型在 {dates[0]} 都没有 spectrum_*.csv，算不出频谱指标")
    spectrum_variable = _pick_variable(set(spectrum_variables), variables)

    spectrum: Dict[str, pd.DataFrame] = {}
    spectrum_rms_values: Dict[str, float] = {}
    for archive in archives:
        pred, obs = load_spectrum(archive.root, dates, spectrum_variable, archive.name)
        spectrum[archive.name] = pd.DataFrame({"pred": pred.mean(axis=1), "obs": obs.mean(axis=1)})
        per_date = []
        for date in pred.columns:
            p = pred[date].where(lambda s: s > 0)
            o = obs[date].where(lambda s: s > 0)
            ratio = (p / o).dropna()
            ratio = ratio[ratio.index > 0]
            if not ratio.empty:
                per_date.append(float(np.sqrt(np.mean(np.log10(ratio.values) ** 2))))
        spectrum_rms_values[archive.name] = BASELINE * float(np.mean(per_date)) if per_date else float("nan")

    # --- 派生指标 ---
    mean_rmse = pd.DataFrame({name: rmse_by_date[name].mean(axis=0) for name in names}).reindex(variables)
    relative = _relative(mean_rmse)
    relative_composite = _composite(relative)

    per_variable_relative = {}
    for variable in variables:
        columns = {name: rmse_by_date[name][variable] for name in names}
        per_variable_relative[variable] = _relative(pd.DataFrame(columns).reindex(index=dates))
    per_date_relative = sum(per_variable_relative.values()) / len(variables)
    per_variable_rmse = {
        variable: pd.DataFrame({name: rmse_by_date[name][variable] for name in names}).reindex(index=dates)
        for variable in variables
    }

    # 逐日 ACC 直接读逐日文件，而不是从"先按 lead 平均、再按日期平均"的表里取——
    # 两者数值上等价，但配对检验要的是逐日样本，不能只有均值。
    per_date_acc = pd.DataFrame({
        name: _acc_by_date(by_name[name].root, dates, acc_variable, name) for name in names
    })
    acc_mean = per_date_acc.mean(axis=0)

    deviation = pd.DataFrame({
        name: (fa_ratio_by_lead[name][fa_variables] - 1.0).abs().mean(axis=1) for name in names
    })
    fa_bias = BASELINE * deviation.mean(axis=0)
    spectrum_rms = pd.Series(spectrum_rms_values)

    rank_score = (
        _rank(relative_composite)
        + _rank(acc_mean, higher_is_better=True)
        + _rank(fa_bias)
        + _rank(spectrum_rms)
    ) / 4.0

    overall_tests = paired_tests(per_date_relative)
    acc_tests = paired_tests(per_date_acc, higher_is_better=True)
    win_counts = variable_win_counts(per_variable_rmse, names)

    anomaly_variable, anomaly_model, anomaly_ratio = _detect_anomaly(mean_rmse, names)
    lead_relative = _lead_relative(rmse_by_lead, names, variables)

    return Bundle(
        names=names, archives=by_name, dates=dates, common_variables=variables,
        acc_variable=acc_variable, fa_variables=fa_variables,
        spectrum_variable=spectrum_variable, spectrum_variables=spectrum_variables,
        rmse_by_date=rmse_by_date, acc_by_lead=acc_by_lead, fa_ratio_by_lead=fa_ratio_by_lead,
        rmse_by_lead=rmse_by_lead, spectrum=spectrum,
        relative=relative, per_date_relative=per_date_relative,
        relative_composite=relative_composite, acc_mean=acc_mean, fa_bias=fa_bias,
        spectrum_rms=spectrum_rms, rank_score=rank_score, per_date_acc=per_date_acc,
        per_variable_rmse=per_variable_rmse,
        overall_tests=overall_tests, acc_tests=acc_tests, win_counts=win_counts,
        anomaly_variable=anomaly_variable, anomaly_model=anomaly_model,
        anomaly_ratio=anomaly_ratio, lead_relative=lead_relative,
    )


def _acc_by_date(root: Path, dates: Sequence[str], variable: str, name: str) -> pd.Series:
    """逐日 ACC 变量在全部 lead 上的平均（百分数），index=日期。"""
    values = {}
    for date in dates:
        path = _pick_metric_file(Path(root) / date, "acc", date)
        if path is None:
            raise FileNotFoundError(f"{name}: {date} 目录下没有 acc_*.csv")
        raw = _read_lead_frame(path)
        if variable not in raw.columns:
            raise ValueError(f"{name}: {path.name} 里没有 {variable} 列；现有列: {list(raw.columns)}")
        values[date] = BASELINE * float(pd.to_numeric(raw[variable], errors="coerce").mean())
    return pd.Series(values)


def _pick_variable(candidates: Iterable[str], fallback: Sequence[str]) -> str:
    """优先 z500，其次候选里字典序最小的，最后退回 fallback 的第一个。"""
    options = sorted(str(c) for c in candidates)
    if PREFERRED_VARIABLE in options:
        return PREFERRED_VARIABLE
    if options:
        return options[0]
    return str(fallback[0])


def _detect_anomaly(mean_rmse: pd.DataFrame, names: Sequence[str]) -> Tuple[str, str, float]:
    """挑「跨模型量级最离谱」的变量：max / median 比值最大的那个。"""
    ratios = {}
    for variable in mean_rmse.index:
        row = mean_rmse.loc[variable].dropna()
        median = float(row.median())
        ratios[variable] = float(row.max()) / median if median > 0 else float("nan")
    finite = {v: r for v, r in ratios.items() if math.isfinite(r)}
    if not finite:
        variable = str(mean_rmse.index[0])
    else:
        variable = max(finite, key=lambda v: finite[v])
    row = mean_rmse.loc[variable].dropna()
    model = str(row.idxmax())
    return variable, model, float(row.max()) / float(row.median()) if float(row.median()) > 0 else float("nan")


# ==================================================================== 渲染


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value, digits: int = 3) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _signed(value, digits: int = 2) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:+.{digits}f}"


def _model_count_word(count: int) -> str:
    return _CN_NUM.get(count, str(count))


def _date_span(dates: Sequence[str]) -> str:
    if not dates:
        return "—"
    return f"{dates[0]}–{dates[-1]}" if len(dates) > 1 else dates[0]


def _lead_span(leads: Sequence[float]) -> Tuple[str, str, int]:
    finite = [float(x) for x in leads if math.isfinite(float(x))]
    if not finite:
        return "—", "—", 0
    return f"{min(finite):g}", f"{max(finite):g}", len(finite)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _bullets(items: Sequence[str], empty: str = "• 无。") -> List[str]:
    return [f"• {item}" for item in items] if items else [empty]


# ------------------------------------------------------------------ 各章节


def _section_samples(bundle: Bundle, declared: Optional[Mapping[str, Sequence[str]]]) -> List[str]:
    lines = ["# 2. 样本与完整性", "", "**样本与完整性**", ""]
    rows = []
    for name in bundle.names:
        archive = bundle.archives[name]
        own = set(archive.summary["init_date"])
        common = own & set(bundle.dates)
        rows.append([
            name, str(len(own)), _date_span(sorted(own)),
            str(len(common)), str(len(own) - len(common)),
        ])
    lines += _table(["**模型**", "**有效日期数**", "**日期范围**", "**共同日期**", "**非共同日期**"], rows)
    lines += ["", "**完整性检查**", ""]

    check_rows = []
    for name in bundle.names:
        archive = bundle.archives[name]
        meta = archive.meta
        listed = [str(d) for d in (meta.get("dates") or [])]
        missing = probe_dates(archive.root, sorted(set(archive.summary["init_date"])), "rmse")
        extra = sorted(set(archive.dates) - set(archive.summary["init_date"]))
        nan_variables = [
            v for v in rmse_mean_columns(archive.summary)
            if archive.summary[f"rmse_{v}_mean"].isna().all()
        ]
        n_dates = meta.get("n_dates")
        n_ok = meta.get("n_ok")
        check_rows.append([
            name,
            f"{n_ok}/{n_dates}" if n_ok is not None and n_dates is not None else "—",
            str(len(archive.summary)),
            f"{len(set(archive.summary['init_date']))} 天" + (f"（meta 列 {len(listed)}）" if listed else ""),
            str(len(missing)) if missing else "无",
            "无" if not nan_variables else "、".join(nan_variables[:4]),
            str(len(extra)) if extra else "无",
        ])
    lines += _table(
        ["**模型**", "**n_ok/n_dates**", "**summary 行数**", "**日期集合**",
         "**缺失文件**", "**NaN RMSE**", "**额外目录**"],
        check_rows,
    )
    lines.append("")

    failures = {name: bundle.archives[name].meta.get("failures") or [] for name in bundle.names}
    broken = {name: items for name, items in failures.items() if items}
    if broken:
        detail = "；".join(f"{name} 有 {len(items)} 个起报点失败" for name, items in broken.items())
    else:
        detail = "所有模型的 batch_meta 都记录 n_ok = n_dates，没有失败起报点"
    lines.append(
        f"{detail}。正文与附录统一使用 {len(bundle.dates)} 个共同日期"
        f"（{_date_span(bundle.dates)}），非共同日期不参与任何统计与曲线，"
        f"因此各表的样本量一致、可直接横向比较。"
    )
    lines.append("")

    declared = declared or {}
    unknown = [name for name in bundle.names if name not in declared]
    if not unknown:
        lines.append("各模型的声明变量清单由 `--declared` 提供，与归档实际变量逐一对照见第 6.2 节。")
    elif len(unknown) == len(bundle.names):
        lines.append(
            "本次未提供各模型的声明变量清单（`--declared`），"
            "第 6.2 节的「声明变量数」一律记 —，只比较归档里实际算出来的变量。"
        )
    else:
        lines.append(
            f"本次只为 {'、'.join(n for n in bundle.names if n not in unknown)} 提供了声明变量清单"
            f"（`--declared`），{'、'.join(unknown)} 的「声明变量数」记 —，"
            f"变量覆盖是否完整只能看归档里实际算出来的变量。"
        )

    lines += ["", "## 指标口径说明（FA）", ""]
    first, last, count = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)
    lines.append(
        f"本报告中的FA仅指Forecast Activity。为避免与降水二分类指标False Alarm"
        f"（同样缩写为FA）混淆，FA统一采用全程{first}–{last}h、共{count}个lead的统计结果，"
        f"不做短、中、长时效分段；文中的分时效比较仅针对RMSE，"
        f"并明确排除{bundle.anomaly_variable}。"
    )
    return lines


def _section_overall(bundle: Bundle) -> List[str]:
    lines = ["# 3. 总体模型表现", "", f"**{len(bundle.names)}模型总体指标**", ""]
    order = bundle.relative_composite.sort_values().index.tolist()
    rank = _rank(bundle.relative_composite)
    rows = []
    for name in order:
        rows.append([
            f"{int(rank[name])}", name,
            _fmt(bundle.relative_composite[name]),
            _fmt(bundle.acc_mean[name]),
            _fmt(bundle.fa_bias[name], 4),
            _fmt(bundle.spectrum_rms[name], 2),
            _fmt(bundle.rank_score[name], 1),
        ])
    lines += _table(
        ["**排名**", "**模型**", "**相对 RMSE**", "**ACC (%)**",
         "**FA 偏差(全程, pp)**", "**频谱 RMS×100**", "**平均排名分**"],
        rows,
    )
    lines += ["", "**综合相对 RMSE 配对 Wilcoxon + Holm 检验**", ""]
    test_rows = []
    for test in bundle.overall_tests:
        test_rows.append([
            f"{test.a} vs {test.b}", _signed(test.delta_pct),
            "是" if test.significant else "否", test.better,
            _fmt(test.p_holm, 4),
        ])
    lines += _table(
        ["**比较**", "**相对变化 A vs B (%)**", "**Holm 显著**", "**显著优者**", "**Holm p**"],
        test_rows,
    )
    lines.append("")
    best = order[0]
    worst = order[-1]
    lines.append(
        f"相对RMSE为100时表示{len(bundle.names)}模型几何均值基线；越低越好。"
        f"{best} 的综合相对 RMSE 最低（{_fmt(bundle.relative_composite[best])}），"
        f"{worst} 最高（{_fmt(bundle.relative_composite[worst])}）；"
        f"表内检验全部基于 {len(bundle.dates)} 个共同日期的逐日配对，"
        f"Holm p < 0.05 才算显著。"
    )
    return lines


def _section_rmse(bundle: Bundle) -> List[str]:
    lines = ["# 4. RMSE 分变量结果", "", "**分变量 RMSE（共同日期平均）**", ""]
    rows = []
    for variable in bundle.common_variables:
        values = pd.Series({name: bundle.rmse_by_date[name][variable].mean() for name in bundle.names})
        best = str(values.idxmin())
        row = [variable] + [_fmt(values[name], 4) for name in bundle.names] + [best]
        rows.append(row)
    lines += _table(["**变量**"] + list(bundle.names) + ["**Best**"], rows)

    lines += ["", "**分变量配对检验显著性汇总**", ""]
    test_rows = []
    for _, record in bundle.win_counts.iterrows():
        a, b = record["模型对"]
        test_rows.append([
            f"{a} vs {b}", str(int(record["A 显著更优"])), str(int(record["B 显著更优"])),
            str(int(record["无显著差异"])), str(int(record["总变量数"])),
        ])
    lines += _table(
        ["**模型对**", "**A 显著更优变量数**", "**B 显著更优变量数**", "**无显著差异**", "**总变量数**"],
        test_rows,
    )
    lines.append("")
    lines.append(
        "表内模型对的次序与第 3 节一致（A、B 即「模型对」列里写明的两个模型，前者为 A）。"
        "逐变量检验用共同日期上的逐日 RMSE 做配对 Wilcoxon，未做多重校正，"
        "因此「显著」只作为方向性提示；要看校正后的结论请看第 3 节的综合检验。"
    )
    return lines


def _section_acc_fa_spectrum(bundle: Bundle) -> List[str]:
    first, last, _ = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)
    lines = [
        f"# 5. ACC、FA（Forecast Activity，全程{first}–{last}h）与频谱",
        "",
        f"**{len(bundle.names)}模型 ACC、FA 与频谱**",
        "",
    ]
    rows = []
    for name in bundle.names:
        rows.append([
            name, _fmt(bundle.acc_mean[name]),
            _fmt(bundle.fa_bias[name], 4), _fmt(bundle.spectrum_rms[name], 2),
        ])
    lines += _table(["**模型**", "**ACC (%)**", "**FA 偏差(全程, pp)**", "**频谱对数 RMS×100**"], rows)

    lines += ["", "**ACC 配对检验**", ""]
    test_rows = []
    for test in bundle.acc_tests:
        test_rows.append([
            f"{test.a} vs {test.b}", _signed(test.mean_a - test.mean_b, 3),
            "是" if test.significant else "否", test.better, _fmt(test.p_holm, 4),
        ])
    lines += _table(
        ["**比较**", "**ACC 差 A-B (pp)**", "**Holm 显著**", "**显著优者**", "**Holm p**"],
        test_rows,
    )
    lines.append("")

    acc_order = bundle.acc_mean.sort_values(ascending=False)
    spread = float(acc_order.iloc[0] - acc_order.iloc[-1])
    near = []
    for name in bundle.names:
        frame = bundle.acc_by_lead[name]
        if bundle.acc_variable in frame.columns:
            first_lead = float(frame[bundle.acc_variable].iloc[0])
            near.append((name, first_lead))
    near.sort(key=lambda item: -item[1])
    acc_note = (
        f"{bundle.acc_variable} ACC 的极差为 {_fmt(spread)} pp（{acc_order.index[0]} 最高 "
        f"{_fmt(acc_order.iloc[0])}，{acc_order.index[-1]} 最低 {_fmt(acc_order.iloc[-1])}）。"
    )
    if near and near[0][1] > SATURATED_ACC:
        acc_note += (
            f"需要注意的是 {near[0][0]} 在首时效的 {bundle.acc_variable} ACC 达到 "
            f"{_fmt(near[0][1])}%，已逼近饱和，"
            f"存在「ACC 定义过宽、近端几乎不可分辨」的风险（见 6.3）。"
        )
    lines.append(acc_note)
    return lines


def _section_risks(bundle: Bundle, declared: Optional[Mapping[str, Sequence[str]]]) -> List[str]:
    variable = bundle.anomaly_variable
    lines = ["# 6. 异常与风险", "", f"## 6.1 {variable}量级异常", ""]

    values = pd.Series({name: bundle.rmse_by_date[name][variable].mean() for name in bundle.names})
    values = values.sort_values(ascending=False)
    rows = [[name, _fmt(values[name], 4)] for name in values.index]
    lines += _table(["**模型**", f"**共同日期平均 {variable} RMSE**"], rows)

    lines += ["", f"**逐时效倍数（{bundle.anomaly_model} / 其他{len(bundle.names) - 1}模型中位数）**", ""]
    lead_rows = []
    for lead in bundle.lead_relative.index:
        column = bundle.rmse_by_lead[bundle.anomaly_model].get(variable)
        others = [
            bundle.rmse_by_lead[name].get(variable) for name in bundle.names
            if name != bundle.anomaly_model
        ]
        others = [s for s in others if s is not None and lead in s.index]
        if column is None or lead not in column.index or not others:
            continue
        median = float(np.median([float(s.loc[lead]) for s in others]))
        value = float(column.loc[lead])
        lead_rows.append([f"{float(lead):g}", _fmt(value / median, 3) if median else "—"])
    lines += _table(["**Lead (h)**", f"**{bundle.anomaly_model} / 其他{len(bundle.names) - 1}模型中位数**"], lead_rows)
    lines.append("")

    threshold = 3.0
    if math.isfinite(bundle.anomaly_ratio) and bundle.anomaly_ratio >= threshold:
        lines.append(
            f"{bundle.anomaly_model} 的 {variable} RMSE 是其余模型中位数的 "
            f"{_fmt(bundle.anomaly_ratio, 2)} 倍，超出「同量级」的可接受范围（阈值 {threshold:g}×）。"
            f"该差异在所有 lead 上一致存在，不是个别时效的抖动，"
            f"更像是量纲/单位或变量定义层面的问题，而不是预报质量问题——"
            f"建议核对 {variable} 的预报与观测是否使用了同一套单位与层次定义后再采信该变量的结论。"
        )
    else:
        lines.append(
            f"{bundle.anomaly_model} 的 {variable} RMSE 是其余模型中位数的 "
            f"{_fmt(bundle.anomaly_ratio, 2)} 倍，未超过 {threshold:g}× 的异常阈值，"
            f"属于正常的模型间水平差异，不单独作为风险项。"
        )

    lines += ["", "## 6.2 变量覆盖不完整", "", "**声明变量数与汇总变量数**", ""]
    declared = declared or {}
    rows = []
    for name in bundle.names:
        actual = summary_variables(bundle.archives[name])
        claimed = declared.get(name)
        if claimed is None:
            rows.append([name, "—", str(len(actual)), "—"])
        else:
            claimed_list = [str(v).strip() for v in claimed if str(v).strip()]
            gap = sorted(set(claimed_list) - set(actual))
            rows.append([name, str(len(claimed_list)), str(len(actual)), "、".join(gap) if gap else "无"])
    lines += _table(["**模型**", "**声明变量数**", "**汇总变量数**", "**汇总缺失变量**"], rows)
    lines.append("")

    counts = {name: len(summary_variables(bundle.archives[name])) for name in bundle.names}
    if len(set(counts.values())) == 1:
        lines.append(
            f"各模型的 summary.csv 都覆盖 {list(counts.values())[0]} 个变量，变量覆盖一致；"
            f"本报告只对这 {len(bundle.common_variables)} 个共享 RMSE 变量做横向比较。"
        )
    else:
        detail = "、".join(f"{name} {count} 个" for name, count in sorted(counts.items()))
        lines.append(
            f"各模型的汇总变量数不一致（{detail}），说明有模型没算全变量。"
            f"本报告只对 {len(bundle.common_variables)} 个共享变量做横向比较，"
            f"缺变量的模型在这些变量上的结论不完整，选型时需要一并考虑。"
        )

    lines += ["", "## 6.3 ACC 定义风险", ""]
    first_lead = None
    frame = bundle.acc_by_lead[bundle.names[0]]
    if bundle.acc_variable in frame.columns and not frame.empty:
        first_lead = float(frame[bundle.acc_variable].iloc[0])
    if first_lead is not None and first_lead > SATURATED_ACC:
        lines.append(
            f"首时效 {bundle.acc_variable} ACC 已到 {_fmt(first_lead)}%，接近饱和（阈值 "
            f"{SATURATED_ACC:g}%）。这说明所用 ACC 定义（距平相关）在近端对模型差异几乎不敏感，"
            f"用它排名会把不同模型压成几乎同分，不能单独作为选型依据；"
            f"建议同时看 RMSE 与频谱。"
        )
    else:
        lines.append(
            f"首时效 {bundle.acc_variable} ACC 为 "
            f"{_fmt(first_lead) + '%' if first_lead is not None else '—'}，"
            f"没有出现接近饱和（{SATURATED_ACC:g}%）的现象，"
            f"ACC 在本批次上对模型差异仍有区分度；但 ACC 的口径（距平基准、区域权重）"
            f"由评测侧定义，跨批次比较时需确认口径未变。"
        )

    lines += ["", "## 6.4 原始场独立复算不足", ""]
    lines.append(
        "本报告的 RMSE / ACC / FA / 频谱全部读自各模型归档目录里的逐日结果文件，"
        "没有回到原始预报场与观测场做独立复算；"
        "因此归档生成环节（变量映射、插值、单位换算）如果出错，本报告会原样继承，无法自查。"
        "归档目录里也确实不含原始场（只有 summary.csv、batch_meta.json 与逐日结果 CSV），"
        "这一点无法在本报告内弥补。结论用于模型间横向比较是充分的，"
        "用于绝对精度认定前建议抽一个日期回到原始场复算一次。"
    )
    return lines


def _section_acceptance(bundle: Bundle) -> List[str]:
    lines = ["# 7. 验收建议", ""]
    items = []
    best = bundle.relative_composite.idxmin()
    items.append(
        f"以共同日期上的综合相对 RMSE 为准，{best} 最低"
        f"（{_fmt(bundle.relative_composite[best])}），可作为默认首选模型。"
    )
    acc_best = bundle.acc_mean.idxmax()
    items.append(
        f"大尺度形势场优先看 ACC，本批次 {acc_best} 最高（{_fmt(bundle.acc_mean[acc_best])}）。"
    )
    fa_best = bundle.fa_bias.idxmin()
    items.append(
        f"强度/活跃度类应用优先看 FA 偏差，{fa_best} 偏离 1 最少"
        f"（{_fmt(bundle.fa_bias[fa_best], 4)} pp）。"
    )
    worst = bundle.relative_composite.idxmax()
    items.append(
        f"{worst} 的综合相对 RMSE 最高（{_fmt(bundle.relative_composite[worst])}），"
        f"在与 {best} 的配对检验中若被判定显著更差（见第 3 节），暂不建议作为主用模型。"
    )
    items.append(
        f"全部结论都建立在 {len(bundle.dates)} 个共同日期（{_date_span(bundle.dates)}）上，"
        f"样本期外的季节与区域不外推。"
    )
    lines += _bullets(items)
    return lines


def _section_comparison(bundle: Bundle) -> List[str]:
    count = len(bundle.names)
    lines = [
        f"# 8. {count}模型具体对比说明",
        "",
        f"本节在{len(bundle.dates)}个共同日期、{len(bundle.common_variables)}个共享RMSE变量、"
        f"{bundle.acc_variable} ACC、{len(bundle.fa_variables)}个FA ratio变量和"
        f"{len(bundle.spectrum_variables)}个频谱变量的统一口径下，对{count}个模型进行逐项比较。"
        f"所有配对检验均使用Wilcoxon符号秩检验并进行Holm多重校正；"
        f"统计显著不代表业务差异一定具有实际重要性。",
        "",
        f"**{count}模型综合对比**",
        "",
    ]
    order = bundle.relative_composite.sort_values().index.tolist()
    ranks = {
        "rel": _rank(bundle.relative_composite),
        "acc": _rank(bundle.acc_mean, higher_is_better=True),
        "fa": _rank(bundle.fa_bias),
        "spec": _rank(bundle.spectrum_rms),
    }
    rows = []
    for name in order:
        row = [
            name,
            f"{_fmt(bundle.relative_composite[name])}（第{int(ranks['rel'][name])}）",
            f"{_fmt(bundle.acc_mean[name])}（第{int(ranks['acc'][name])}）",
            f"{_fmt(bundle.fa_bias[name], 4)}（第{int(ranks['fa'][name])}）",
            f"{_fmt(bundle.spectrum_rms[name], 2)}（第{int(ranks['spec'][name])}）",
        ]
        strengths, weaknesses = _pros_cons(bundle, name, ranks, count)
        rows.append(row + [strengths, weaknesses])
    lines += _table(
        ["**模型**", "**综合相对RMSE**", "**ACC**", "**FA偏差(pp)**", "**频谱对数RMS×100**",
         "**主要优势**", "**主要短板**"],
        rows,
    )

    lines += ["", "## 8.1 逐模型结论", ""]
    items = []
    for name in order:
        strengths, weaknesses = _pros_cons(bundle, name, ranks, count)
        items.append(
            f"{name}：综合相对RMSE {_fmt(bundle.relative_composite[name])}"
            f"（第{int(ranks['rel'][name])}）、ACC {_fmt(bundle.acc_mean[name])}%"
            f"（第{int(ranks['acc'][name])}）、FA偏差 {_fmt(bundle.fa_bias[name], 4)} pp"
            f"（第{int(ranks['fa'][name])}）、频谱对数RMS×100 {_fmt(bundle.spectrum_rms[name], 2)}"
            f"（第{int(ranks['spec'][name])}）；优势在{strengths}，短板在{weaknesses}。"
        )
    lines += _bullets(items)

    lines += ["", "## 8.2 关键配对差异", "", "**关键配对差异**", ""]
    rows = []
    for test in bundle.overall_tests:
        record = bundle.win_counts[
            bundle.win_counts["模型对"].apply(lambda pair: set(pair) == {test.a, test.b})
        ]
        if record.empty:
            wins = "—"
        else:
            wins_a = int(record.iloc[0]["A 显著更优"])
            wins_b = int(record.iloc[0]["B 显著更优"])
            wins = f"{test.a} {wins_a} : {wins_b} {test.b}"
        if test.significant:
            summary = f"综合相对RMSE 差异 {_signed(test.delta_pct)}%，Holm p = {_fmt(test.p_holm, 4)}"
            explain = f"{test.better} 在综合口径上显著更优"
        else:
            summary = f"综合相对RMSE 差异 {_signed(test.delta_pct)}%，Holm p = {_fmt(test.p_holm, 4)}"
            explain = "未通过 Holm 校正，两者的总体差距不足以判定优劣"
        rows.append([f"{test.a} vs {test.b}", summary, wins, explain])
    lines += _table(["**模型对**", "**综合差异**", "**显著更优变量数**", "**解释**"], rows)
    return lines


def _pros_cons(bundle: Bundle, name: str, ranks: Mapping[str, pd.Series], count: int) -> Tuple[str, str]:
    labels = {
        "rel": "综合 RMSE", "acc": "ACC",
        "fa": "FA 偏差", "spec": "频谱保真",
    }
    ordered = sorted(ranks, key=lambda key: int(ranks[key][name]))
    best = ordered[0]
    worst = ordered[-1]
    if int(ranks[best][name]) == 1:
        strength = f"{labels[best]}（{count}模型第一）"
    elif int(ranks[best][name]) <= max(1, count // 2):
        strength = f"{labels[best]}（第{int(ranks[best][name])}）"
    else:
        strength = f"各维度均无领先项（最好的一项{labels[best]}也只是第{int(ranks[best][name])}）"
    weakness = f"{labels[worst]}（第{int(ranks[worst][name])}）"
    return strength, weakness


def _section_sensitivity(bundle: Bundle) -> List[str]:
    variable = bundle.anomaly_variable
    lines = [f"# 9. {variable}敏感性分析与分时效对比", ""]
    lines.append(
        f"{variable} 在模型间存在量级差异（见 6.1），为确认它是否主导了总体排序，"
        f"下表给出「含 {variable}」与「不含 {variable}」两种口径下的综合相对 RMSE。"
    )
    lines += ["", f"**含/不含{variable}两种口径**", ""]

    without = [v for v in bundle.common_variables if v != variable]
    if without:
        recomputed = _composite(_relative(
            pd.DataFrame({
                name: bundle.rmse_by_date[name][without].mean(axis=0) for name in bundle.names
            }).reindex(without)
        ))
        without_order = recomputed.sort_values().index.tolist()
        without_note = " < ".join(f"{n} {_fmt(recomputed[n])}" for n in without_order)
        without_cells = [_fmt(recomputed[name]) for name in bundle.names]
    else:
        without_order = None
        without_note = "没有其他共享变量可比"
        without_cells = ["—"] * len(bundle.names)

    with_order = bundle.relative_composite.sort_values().index.tolist()
    with_note = " < ".join(f"{n} {_fmt(bundle.relative_composite[n])}" for n in with_order)
    lines += _table(
        ["**口径**"] + list(bundle.names) + ["**结论**"],
        [
            [f"含{variable}"] + [_fmt(bundle.relative_composite[n]) for n in bundle.names] + [with_note],
            [f"不含{variable}"] + without_cells + [without_note],
        ],
    )
    lines.append("")
    if without_order is None or with_order == without_order:
        lines.append(
            f"两种口径下排序完全一致，说明 {variable} 的量级差异虽然明显，"
            f"但没有改变模型间的相对格局，总体结论对该变量不敏感。"
        )
    else:
        lines.append(
            f"两种口径下排序发生变化，说明 {variable} 主导了总体排序；"
            f"在 {variable} 的单位与定义核对清楚之前，不应据此对相关模型下最终结论。"
        )

    lines += ["", "**分时效段相对综合RMSE**", ""]
    rows = []
    for low, high in LEAD_BANDS:
        leads = [lead for lead in bundle.lead_relative.index if low < float(lead) <= high]
        if not leads:
            continue
        band = bundle.lead_relative.loc[leads].mean(axis=0)
        order = band.sort_values()
        span = f"{float(min(leads)):g}–{float(max(leads)):g}h"
        rows.append([
            span, str(order.index[0]),
            " < ".join(f"{n} {_fmt(band[n])}" for n in order.index),
            f"{len(leads)} 个 lead",
        ])
    lines += _table(
        ["**时效段**", "**最佳模型**", f"**{len(bundle.names)}模型相对综合RMSE排序（越低越好）**", "**说明**"],
        rows,
    )
    lines.append("")
    lines.append(
        "分时效口径与第 3 节一致：逐 lead 先按变量做几何均值归一，再对共享变量等权平均；"
        "因此这里的数值与第 3 节的全程值同量纲，但不能直接与第 3 节的表相加。"
    )
    return lines


def _section_conclusion(bundle: Bundle) -> List[str]:
    lines = ["# 10. 总述结论", ""]
    order = bundle.relative_composite.sort_values().index.tolist()
    best = order[0]
    items = [
        f"在 {len(bundle.dates)} 个共同日期（{_date_span(bundle.dates)}）上，"
        f"{best} 的综合相对 RMSE 最低（{_fmt(bundle.relative_composite[best])}），"
        f"较几何均值基线低 {_fmt(BASELINE - bundle.relative_composite[best])}。",
        f"ACC 最高的是 {bundle.acc_mean.idxmax()}（{_fmt(bundle.acc_mean.max())}%），"
        f"最低的是 {bundle.acc_mean.idxmin()}（{_fmt(bundle.acc_mean.min())}%），"
        f"极差 {_fmt(bundle.acc_mean.max() - bundle.acc_mean.min())} pp。",
        f"FA 偏差最小的是 {bundle.fa_bias.idxmin()}（{_fmt(bundle.fa_bias.min(), 4)} pp），"
        f"最大的是 {bundle.fa_bias.idxmax()}（{_fmt(bundle.fa_bias.max(), 4)} pp）。",
        f"频谱对数 RMS×100 最小的是 {bundle.spectrum_rms.idxmin()}"
        f"（{_fmt(bundle.spectrum_rms.min(), 2)}），说明其小尺度能量分布最接近观测。",
        f"平均排名分最低（最好）的是 {bundle.rank_score.idxmin()}"
        f"（{_fmt(bundle.rank_score.min(), 1)}），最高的是 {bundle.rank_score.idxmax()}"
        f"（{_fmt(bundle.rank_score.max(), 1)}）。",
        f"{bundle.anomaly_variable} 的量级差异是本批次最需要先核实的一项"
        f"（{bundle.anomaly_model} 为其余模型中位数的 {_fmt(bundle.anomaly_ratio, 2)} 倍），"
        f"在单位与定义确认前，该变量相关结论只作参考。",
    ]
    lines += _bullets(items)
    return lines


def _section_selection(bundle: Bundle) -> List[str]:
    lines = ["# 11. 最终选型建议", "", "**分场景选型建议**", ""]
    rows = [
        ["综合精度优先（默认）", str(bundle.relative_composite.idxmin()),
         f"综合相对 RMSE 最低（{_fmt(bundle.relative_composite.min())}）"],
        ["大尺度形势场", str(bundle.acc_mean.idxmax()),
         f"ACC 最高（{_fmt(bundle.acc_mean.max())}%）"],
        ["强度与活跃度", str(bundle.fa_bias.idxmin()),
         f"FA 偏差最小（{_fmt(bundle.fa_bias.min(), 4)} pp）"],
        ["小尺度结构与谱保真", str(bundle.spectrum_rms.idxmin()),
         f"频谱对数 RMS×100 最小（{_fmt(bundle.spectrum_rms.min(), 2)}）"],
    ]
    if bundle.rank_score.notna().any():
        rows.append([
            "多维度均衡", str(bundle.rank_score.idxmin()),
            f"平均排名分最低（{_fmt(bundle.rank_score.min(), 1)}）",
        ])
    rows.append([
        "暂缓采用", str(bundle.relative_composite.idxmax()),
        f"综合相对 RMSE 最高（{_fmt(bundle.relative_composite.max())}）",
    ])
    lines += _table(["**应用目标**", "**推荐模型**", "**理由**"], rows)
    return lines


def _appendix(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    count = len(bundle.names)
    listing = "、".join(bundle.names)
    first, last, leads = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)
    lines = [
        f"# 附录：{count}模型曲线图",
        "",
        f"本附录使用{count}个模型共同的{len(bundle.dates)}个日期"
        f"（{bundle.dates[0]} 至 {bundle.dates[-1]}）。"
        f"以下每张图均包含 {listing} {count} 条模型曲线。"
        f"正文统计表与图中曲线使用同一批共同日期。",
        "",
        f"## 附 1 RMSE：{len(bundle.common_variables)}个变量{count}模型曲线",
        "",
        f"每幅子图包含 {listing} {count} 条曲线。",
        "",
        f"![multi_model_rmse_vs_lead]({figures['rmse_plot']})",
        "",
        f"图 1：{len(bundle.common_variables)}个变量 RMSE 随预报时效的变化（{_date_span(bundle.dates)} 共同日期平均）",
        "",
        f"## 附 2 {bundle.acc_variable} ACC：{count}模型曲线",
        "",
        f"{count}条曲线对应{count}个模型，纵轴越高越好。",
        "",
        f"![multi_model_acc_vs_lead]({figures['acc_plot']})",
        "",
        f"图 2：{bundle.acc_variable} ACC 随预报时效的变化（{_date_span(bundle.dates)} 共同日期平均）",
        "",
        f"## 附 3 FA（Forecast Activity）：{len(bundle.fa_variables)}个变量、全程{first}–{last}h {count}模型曲线",
        "",
        f"每个子图均包含{count}条模型曲线，并覆盖全部{leads}个lead（{first}–{last}h）；"
        f"ratio越接近1越好。",
        "",
        f"![multi_model_fa_ratio_vs_lead]({figures['fa_plot']})",
        "",
        f"图 3：{len(bundle.fa_variables)}个变量 FA ratio 随预报时效的变化（{_date_span(bundle.dates)} 共同日期平均）",
        "",
        f"## 附 4 {bundle.spectrum_variable} 功率谱：{count}模型曲线",
        "",
        f"双对数坐标；{count}条颜色曲线对应{count}个模型的预测谱。观测谱为黑色虚线。",
        "",
        f"![multi_model_spectrum]({figures['spectrum_plot']})",
        "",
        f"图 4：{bundle.spectrum_variable} 纬向功率谱（{_date_span(bundle.dates)} 共同日期平均）",
    ]
    return lines


# ------------------------------------------------------------------ 装配


def render(
    bundle: Bundle,
    *,
    title: Optional[str] = None,
    change: Optional[str] = None,
    archive: Optional[str] = None,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
    figures: Optional[Mapping[str, str]] = None,
) -> str:
    """把 :class:`Bundle` 渲染成完整 Markdown。"""
    count = len(bundle.names)
    figures = figures or {
        "rmse_plot": "multi_model_rmse_vs_lead.png",
        "acc_plot": "multi_model_acc_vs_lead.png",
        "fa_plot": "multi_model_fa_ratio_vs_lead.png",
        "spectrum_plot": f"multi_model_spectrum_{bundle.spectrum_variable}.png",
    }
    first, last, leads = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)

    lines: List[str] = []
    lines.append(title or f"# XMETAI 确定性{_model_count_word(count)}模型评估检验报告（单成员）")
    lines += ["", f"**归档：{archive or '、'.join(bundle.names)}**", ""]
    lines += [
        f"报告日期：{bundle.dates[-1]}    评估层级：输出级复核",
        "",
        "## 报告定位与衔接",
        "",
        f"报告类型：确定性单成员预报（single/det）。本报告覆盖 {count} 个模型的批次归档目录，"
        f"正文与附录统一使用 {len(bundle.dates)} 个共同日期（{_date_span(bundle.dates)}），"
        f"逐日结果按 lead 平均后比较；非共同日期不参与任何统计。"
        + (f"本次相对上一版的改动：{change}" if change else ""),
        "",
        f"本报告中的FA指Forecast Activity，统一使用全程{first}–{last}h、{leads}个lead，"
        f"不对FA做短中长时效分段；分时效分析仅用于RMSE。",
        "",
    ]
    lines += _conclusion_summary(bundle, change)
    lines += [""]
    lines += _section_samples(bundle, declared)
    lines += [""]
    lines += _section_overall(bundle)
    lines += [""]
    lines += _section_rmse(bundle)
    lines += [""]
    lines += _section_acc_fa_spectrum(bundle)
    lines += [""]
    lines += _section_risks(bundle, declared)
    lines += [""]
    lines += _section_acceptance(bundle)
    lines += [""]
    lines += _section_comparison(bundle)
    lines += [""]
    lines += _section_sensitivity(bundle)
    lines += [""]
    lines += _section_conclusion(bundle)
    lines += [""]
    lines += _section_selection(bundle)
    lines += [""]
    lines += _appendix(bundle, figures)
    lines.append("")
    return "\n".join(lines)


def _conclusion_summary(bundle: Bundle, change: Optional[str]) -> List[str]:
    order = bundle.relative_composite.sort_values().index.tolist()
    best, worst = order[0], order[-1]
    items = [
        f"共同日期 {len(bundle.dates)} 天（{_date_span(bundle.dates)}）；"
        f"综合相对 RMSE 最好 {best}（{_fmt(bundle.relative_composite[best])}），"
        f"最差 {worst}（{_fmt(bundle.relative_composite[worst])}）。",
        f"ACC 最好 {bundle.acc_mean.idxmax()}（{_fmt(bundle.acc_mean.max())}%），"
        f"FA 偏差最小 {bundle.fa_bias.idxmin()}（{_fmt(bundle.fa_bias.min(), 4)} pp），"
        f"频谱对数 RMS×100 最小 {bundle.spectrum_rms.idxmin()}（{_fmt(bundle.spectrum_rms.min(), 2)}）。",
        f"平均排名分："
        + "、".join(f"{name} {_fmt(bundle.rank_score[name], 1)}" for name in bundle.rank_score.sort_values().index)
        + "。",
    ]
    significant = [t for t in bundle.overall_tests if t.significant]
    if significant:
        items.append(
            "综合相对 RMSE 上通过 Holm 校正的显著差异："
            + "；".join(f"{t.a} 优于 {t.b}（p={_fmt(t.p_holm, 4)}）" for t in significant)
            + "。"
        )
    else:
        items.append(
            "综合相对 RMSE 上没有任何模型对通过 Holm 校正（p < 0.05），"
            "模型间的总体差距在统计上不稳健，选型应更依赖分维度表现。"
        )
    items.append(
        f"{bundle.anomaly_variable} 存在模型间量级差异"
        f"（{bundle.anomaly_model} 为其余模型中位数的 {_fmt(bundle.anomaly_ratio, 2)} 倍），"
        f"已在 6.1 与第 9 节单独分析。"
    )
    if change:
        items.append(f"本次改动：{change}")
    return ["# 1. 结论摘要", ""] + _bullets(items)


# ------------------------------------------------------------------ 入口


def build_det_report(
    archives: Sequence[Archive],
    out_dir: Path,
    *,
    title: Optional[str] = None,
    change: Optional[str] = None,
    archive: Optional[str] = None,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
    ncols: int = 4,
) -> Dict[str, Path]:
    """出图 + 写 ``REPORT.md``，返回产物名 → 路径。

    Raises:
        ValueError / FileNotFoundError: 见 :func:`analyze`。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = analyze(archives, declared=declared)
    plotter = DetPlotter()

    figures = {
        "rmse_plot": "multi_model_rmse_vs_lead.png",
        "acc_plot": "multi_model_acc_vs_lead.png",
        "fa_plot": "multi_model_fa_ratio_vs_lead.png",
        "spectrum_plot": f"multi_model_spectrum_{bundle.spectrum_variable}.png",
    }

    rmse_series = {name: bundle.rmse_by_lead[name] for name in bundle.names}
    plotter.plot_model_panels(
        rmse_series, bundle.common_variables, ylabel="RMSE",
        ncols=ncols, suptitle=f"{len(bundle.names)}模型 RMSE 随预报时效的变化",
        save_path=out_dir / figures["rmse_plot"],
    )

    acc_series = {
        name: bundle.acc_by_lead[name].reindex(columns=[bundle.acc_variable])
        for name in bundle.names
    }
    plotter.plot_model_panels(
        acc_series, [bundle.acc_variable], ylabel="ACC", ncols=1,
        suptitle=f"{bundle.acc_variable} ACC 随预报时效的变化（越高越好）",
        save_path=out_dir / figures["acc_plot"],
    )

    plotter.plot_model_panels(
        {name: bundle.fa_ratio_by_lead[name] for name in bundle.names},
        bundle.fa_variables, ylabel="ratio（预报/观测）", ref_line=1.0, ncols=ncols,
        suptitle="FA（Forecast Activity）ratio 随预报时效的变化（越接近 1 越好）",
        save_path=out_dir / figures["fa_plot"],
    )

    plotter.plot_model_spectrum(
        bundle.spectrum, bundle.spectrum_variable,
        save_path=out_dir / figures["spectrum_plot"],
    )

    text = render(
        bundle, title=title, change=change, archive=archive,
        declared=declared, figures=figures,
    )
    report_path = out_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"已保存: {report_path}")

    artifacts: Dict[str, Path] = {"report": report_path}
    for key, filename in figures.items():
        artifacts[key] = out_dir / filename
    return artifacts
