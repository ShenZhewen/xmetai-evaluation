#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两套评测结果对拍：scores.csv × scores.csv 或 scores.csv × WB2 nc。

用法：
    python tools/compare_wb2_vs_v2.py <A目录> <B目录>
    python tools/compare_wb2_vs_v2.py <A目录> <B目录> --log-file tools/compare.log
    python tools/compare_wb2_vs_v2.py <A目录> <B目录> --name-a ts --name-b datakit
    python tools/compare_wb2_vs_v2.py <A目录> <B目录> --log-file tools/compare.log \\
            --out tools/compare_A_vs_B.md

A/B 支持两种格式（自动识别）：
  - 含 scores.csv 的目录：evaluation 长表，一行一个 (变量, 指标, 起报, 时效)
  - 含 *_deterministic.nc 的 WB2 目录（可选 wb2_2022_extra.csv）

支持组合：
  - scores.csv × scores.csv
  - scores.csv × WB2 nc（WB2 放 A 侧会自动交换）

流程：
  1. 识别输入格式并读入
  2. 一致性检查：是否同一模型 / 同一 run / 同一覆盖
  3. 列出可对表范围：共同变量、共同指标、只在一侧存在的指标
  4. 逐指标对比：逐时效数值、比值/倍率、趋势
  5. 附加检查：T5 增长曲线、T6 聚合口径（仅 WB2 模式）、T7 起报散布
  6. 输出总表

输出：日志默认打印到控制台，给 --log-file 时同时写入该文件（UTF-8）；
      Markdown 报告默认写在本脚本旁边 compare_<A>_vs_<B>.md，可用 --out 改。
方向约定：A 为第一参数，B 为第二参数；Δ = A − B，Δ% = (A − B) / |B|。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
# 可调参数：直接改这里（不用环境变量）
# ══════════════════════════════════════════════════════════════════════
# 相对差判定阈值（RMSE / MAE / 活跃度等有量纲指标）
TOL_PASS_REL = 0.005     # A 一致 ≤0.5%
TOL_CLOSE_REL = 0.010    # B 接近 ≤1%
TOL_MINOR_REL = 0.030    # C 小差 ≤3%
TOL_MAJOR_REL = 0.100    # D 大差 ≤10%，再大看趋势

# Bias 是近零量，按自身相对差判会爆掉；改用 |Δbias|/RMSE
BIAS_PASS = 0.02
BIAS_CLOSE = 0.05
BIAS_MINOR = 0.10
BIAS_MAJOR = 0.20

# ACC 是无量纲、数值贴着 1，用绝对差判定
ACC_ABS_PASS = 2e-3
ACC_ABS_CLOSE = 5e-3
ACC_ABS_MINOR = 1.5e-2
ACC_ABS_MAJOR = 3e-2

VERDICT_CN = {
    "PASS": "一致", "CLOSE": "接近", "MINOR": "小差", "MAJOR": "大差",
    "SHAPE": "大差·形状", "MISMATCH": "不一致",
    "SCALE": "倍率", "UNIT": "单位",
}
VERDICT_BADGE = {
    "PASS": "✅ PASS", "CLOSE": "🟢 CLOSE", "MINOR": "🟡 MINOR",
    "MAJOR": "🟠 MAJOR", "SHAPE": "🟣 SHAPE", "SCALE": "🔵 SCALE",
    "UNIT": "🔵 UNIT", "MISMATCH": "❌ MISMATCH",
}
VERDICT_RANK = {
    "PASS": 0, "CLOSE": 1, "MINOR": 2, "MAJOR": 3,
    "SHAPE": 4, "SCALE": 5, "UNIT": 5, "MISMATCH": 6,
}

OUTLIER_FACTOR = 5.0     # 单时效 |相对差| 超过中位数的 5 倍算异常时效
CSV_CHUNK = 1_000_000    # scores.csv 分块行数

# 参与标量对表的指标（其余指标如谱/球谐带只做覆盖统计）
V2_METRICS = {
    "rmse", "bias", "mae", "acc",
    "activity_forecast", "activity_observation",
    "activity_ratio", "activity_bias",
}
COMPARE_ORDER = (
    "rmse", "acc", "bias", "mae",
    "activity_forecast", "activity_observation",
    "activity_ratio", "activity_bias",
)
METRIC_LABEL = {
    "rmse": "RMSE", "acc": "ACC", "bias": "Bias", "mae": "MAE",
    "activity_forecast": "Activity(fcst)",
    "activity_observation": "Activity(obs)",
    "activity_ratio": "Activity(ratio)",
    "activity_bias": "Activity(bias)",
}

# v2 变量 -> WB2 deterministic nc 里 level 坐标的取值（None=地面量）
LEVELS = {
    "z500": 500, "q700": 700, "t700": 700, "t850": 850,
    "u850": 850, "v850": 850,
}

# nc 文件名 token -> 变量名
TOKEN2VAR = {
    "z500": "z500", "q700": "q700", "t700": "t700", "t850": "t850",
    "t2m": "t2m", "msl": "msl", "u850": "u850", "v850": "v850",
    "u10": "u10m", "v10": "v10m", "tp": "tp",
}

# extra.csv 的 (WB2 变量名, level) -> 变量名
EXTRA_KEY2VAR = {
    ("geopotential", 500): "z500",
    ("specific_humidity", 700): "q700",
    ("temperature", 700): "t700",
    ("temperature", 850): "t850",
    ("u_component_of_wind", 850): "u850",
    ("v_component_of_wind", 850): "v850",
    ("10m_u_component_of_wind", -1): "u10m",
    ("10m_v_component_of_wind", -1): "v10m",
    ("2m_temperature", -1): "t2m",
    ("mean_sea_level_pressure", -1): "msl",
    ("total_precipitation_6hr", -1): "tp",
}

SAMPLE_LEADS = (6, 24, 48, 72, 96, 120, 168, 240, 360)

log = logging.getLogger("compare")


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Agg:
    """一个 (变量, 指标, 时效) 上跨起报的累计器。"""
    n: int = 0
    s1: float = 0.0
    s2: float = 0.0

    def add(self, n: int, s1: float, s2: float) -> None:
        self.n += n
        self.s1 += s1
        self.s2 += s2

    @property
    def mean(self) -> float:
        return self.s1 / self.n if self.n else float("nan")

    @property
    def rms(self) -> float:
        return math.sqrt(self.s2 / self.n) if self.n else float("nan")

    @property
    def std(self) -> float:
        """逐起报值的标准差（批间散布）。"""
        if not self.n:
            return float("nan")
        m = self.mean
        return math.sqrt(max(self.s2 / self.n - m * m, 0.0))


@dataclass
class PairResult:
    var: str
    label: str
    metric_a: str
    metric_b: str
    n: int
    leads: List[int]
    x: np.ndarray
    y: np.ndarray
    scale: float
    scale_kind: str
    ratio_median: float
    ratio_cv: float
    rel_max: float
    rel_median: float
    abs_max: float
    pearson: float
    spearman: float
    slope: float
    intercept: float
    verdict: str
    note: str
    outliers: List[int] = field(default_factory=list)
    init_scatter_ratio: Optional[float] = None
    x_mean: Optional[np.ndarray] = None
    bias_rel_max: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════
# 输入读取
# ══════════════════════════════════════════════════════════════════════
def load_scores(path: Path) -> Tuple[Dict, Dict]:
    """流式读 scores.csv，按 (变量, 指标, 时效) 聚合逐起报值。"""
    p = path / "scores.csv" if path.is_dir() else path
    if not p.is_file():
        raise SystemExit(f"找不到 scores.csv: {p}")

    usecols = ["run_id", "model_id", "dataset_id", "variable", "metric",
               "lead_h", "init_time", "region", "value", "unit", "status"]
    data: Dict[str, Dict[str, Dict[float, Agg]]] = {}
    all_keys: Set[Tuple[str, str]] = set()
    units: Dict[Tuple[str, str], Set[str]] = {}
    run_ids: Set[str] = set()
    models: Set[str] = set()
    datasets: Set[str] = set()
    inits: Set[str] = set()
    leads: Set[float] = set()
    variables: Set[str] = set()
    regions: Set[str] = set()
    dropped_regions: Set[str] = set()
    globalish_seen = False
    n_rows = 0

    reader = pd.read_csv(p, usecols=usecols, chunksize=CSV_CHUNK, low_memory=False)
    for chunk in reader:
        n_rows += len(chunk)
        run_ids.update(chunk["run_id"].dropna().astype(str).unique())
        models.update(chunk["model_id"].dropna().astype(str).unique())
        datasets.update(chunk["dataset_id"].dropna().astype(str).unique())
        inits.update(chunk["init_time"].dropna().astype(str).unique())
        variables.update(chunk["variable"].dropna().astype(str).unique())
        leads.update(pd.to_numeric(chunk["lead_h"], errors="coerce").dropna().unique().tolist())

        # 全球档案的 region 通常为空；少数档案混入 tropics 等分区行，不能一起平均
        region = chunk["region"].fillna("").astype(str)
        regions.update(x for x in region.unique() if x)
        globalish = region.isin(["", "global", "all", "Global", "GLOBAL"])
        if globalish.any():
            globalish_seen = True
        if globalish_seen and not globalish.all():
            dropped_regions.update(region[~globalish].unique())
            chunk = chunk[globalish]

        status = chunk["status"].fillna("success").astype(str)
        chunk = chunk[status.isin(["success", ""])]
        if chunk.empty:
            continue
        all_keys.update(zip(chunk["variable"].astype(str),
                            chunk["metric"].astype(str)))

        sub = chunk[chunk["metric"].isin(V2_METRICS)].copy()
        if sub.empty:
            continue
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub["lead_h"] = pd.to_numeric(sub["lead_h"], errors="coerce")
        sub = sub[sub["value"].notna() & sub["lead_h"].notna()]
        if sub.empty:
            continue

        for (var, met), g in sub.groupby(["variable", "metric"], sort=False):
            var, met = str(var), str(met)
            store = data.setdefault(var, {}).setdefault(met, {})
            us = [u for u in g["unit"].dropna().astype(str).unique() if u]
            if us:
                units.setdefault((var, met), set()).update(us)
            for lead, gg in g.groupby("lead_h", sort=False):
                vals = gg["value"].to_numpy(dtype=float)
                key = float(lead)
                acc = store.get(key)
                if acc is None:
                    store[key] = Agg(len(vals), float(vals.sum()),
                                     float(np.square(vals).sum()))
                else:
                    acc.add(len(vals), float(vals.sum()),
                            float(np.square(vals).sum()))

    meta = {
        "path": str(p),
        "size_mb": p.stat().st_size / 1e6,
        "rows": n_rows,
        "run_ids": sorted(run_ids),
        "models": sorted(models),
        "datasets": sorted(datasets),
        "init_times": inits,
        "leads": leads,
        "variables": variables,
        "all_keys": all_keys,
        "units": units,
        "regions": regions,
        "dropped_regions": dropped_regions,
    }
    if dropped_regions:
        log.warning("%s：已剔除非全球 region 行 %s（只保留空/global/all）",
                    p, _fmt_set(dropped_regions, 8))
    return data, meta


def _lead_hours(values) -> np.ndarray:
    vals = np.asarray(values)
    if np.issubdtype(vals.dtype, np.timedelta64):
        return vals.astype("timedelta64[s]").astype(np.int64) / 3600.0
    return vals.astype(float)


def load_wb2_deterministic(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[int, float]], List[str], Dict[str, str]]:
    """读 wb2_*_deterministic.nc，返回 {(变量, 指标): {lead_h: 值}} 和文件属性。"""
    if not path.is_dir():
        raise SystemExit(f"WB2 结果路径不是目录: {path}")
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("读取 .nc 需要 xarray：pip install xarray netCDF4") from exc

    out: Dict[Tuple[str, str], Dict[int, float]] = {}
    files_used: List[str] = []
    attrs: Dict[str, str] = {}
    files = sorted(path.glob("*_deterministic.nc"))
    if not files:
        raise SystemExit(f"{path} 下没有 *_deterministic.nc")

    for f in files:
        m = re.match(r"^wb2_\d{4}_(?P<tok>.+)_deterministic$", f.stem)
        if not m:
            log.warning("跳过文件名不识别: %s", f.name)
            continue
        var = TOKEN2VAR.get(m.group("tok"))
        if var is None:
            log.warning("跳过未知变量 token: %s", m.group("tok"))
            continue

        # 新版 xarray 不接受 lead_time 上残留的 dtype 属性（WB2 文件里带着
        # dtype=timedelta64[s]）；关掉时间解码，raw 值是 int64 小时数，够用。
        ds = xr.open_dataset(f, decode_times=False, decode_timedelta=False)
        try:
            if not attrs:
                attrs = {str(k): str(v) for k, v in ds.attrs.items()}
            names = list(ds.data_vars)
            if not names:
                continue
            arr = ds[names[0]]

            if "region" in arr.dims:
                regions = [str(x) for x in np.asarray(arr["region"].values)]
                arr = arr.sel(region="global" if "global" in regions else regions[0],
                              drop=True)

            if "level" in arr.dims:
                levels = np.asarray(arr["level"].values, dtype=float)
                want = LEVELS.get(var)
                idx = 0
                if want is not None:
                    hit = np.where(np.isclose(levels, want))[0]
                    if hit.size:
                        idx = int(hit[0])
                arr = arr.isel(level=idx, drop=True)

            if "metric" not in arr.dims or "lead_time" not in arr.coords:
                log.warning("跳过 %s：缺少 metric/lead_time", f.name)
                continue
            lead_h = _lead_hours(arr["lead_time"].values)

            for met in [str(x) for x in np.asarray(arr["metric"].values)]:
                vals = np.asarray(arr.sel(metric=met).values, dtype=float).ravel()
                series = {int(round(h)): float(v)
                          for h, v in zip(lead_h, vals) if math.isfinite(v)}
                out.setdefault((var, met), {}).update(series)
                if met == "mse":
                    # WB2 的 rmse = sqrt(时间平均后的 mse)
                    out.setdefault((var, "rmse"), {}).update(
                        {k: math.sqrt(v) if v >= 0 else float("nan")
                         for k, v in series.items()})
        finally:
            ds.close()
        files_used.append(f.name)

    return out, files_used, attrs


def load_wb2_extra(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[int, float]], Set[Tuple[str, str, str]], Dict]:
    """读 wb2_2022_extra.csv（prediction_activity / csi / fss …）。"""
    f = path / "wb2_2022_extra.csv" if path.is_dir() else path
    if not f.is_file():
        return {}, set(), {"rows": 0, "models": set(), "truths": set(),
                           "years": set(), "resolutions": set()}
    df = pd.read_csv(f)
    need = {"metric", "var", "level", "lead_h", "value"}
    if not need.issubset(df.columns):
        raise SystemExit(f"extra.csv 缺少列: {sorted(need - set(df.columns))}")

    out: Dict[Tuple[str, str], Dict[int, float]] = {}
    unmatched: Set[Tuple[str, str, str]] = set()
    for row in df.itertuples(index=False):
        wb2_var = str(row.var)
        level = int(round(float(row.level)))
        var = EXTRA_KEY2VAR.get((wb2_var, level))
        met = str(row.metric)
        lead = int(round(float(row.lead_h)))
        val = float(row.value)
        if var is None:
            unmatched.add((wb2_var, str(level), met))
            continue
        if math.isfinite(val):
            out.setdefault((var, met), {})[lead] = val

    def _uniq(col: str) -> Set[str]:
        if col not in df.columns:
            return set()
        return {str(x) for x in df[col].dropna().unique()}

    meta = {
        "rows": len(df),
        "models": _uniq("model"),
        "truths": _uniq("truth"),
        "years": _uniq("year"),
        "resolutions": _uniq("resolution"),
    }
    return out, unmatched, meta


def detect_format(path: Path) -> str:
    """返回 'scores' 或 'wb2'。"""
    if path.is_file():
        if path.name == "scores.csv":
            return "scores"
        raise SystemExit(f"不支持的输入文件: {path}（只接受 scores.csv 或目录）")
    if not path.is_dir():
        raise SystemExit(f"输入路径不存在: {path}")
    if (path / "scores.csv").is_file():
        return "scores"
    if any(path.glob("*_deterministic.nc")):
        return "wb2"
    raise SystemExit(f"{path} 下既没有 scores.csv，也没有 *_deterministic.nc")


def _manifest_info(path: Path) -> Dict:
    """读目录下 manifest.json（如果有内容）。"""
    f = path / "manifest.json" if path.is_dir() else path.parent / "manifest.json"
    try:
        if f.is_file() and f.stat().st_size:
            with f.open(encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return {}


# ══════════════════════════════════════════════════════════════════════
# 一致性检查
# ══════════════════════════════════════════════════════════════════════
def _same_set(a: Set, b: Set) -> bool:
    return a == b


def _fmt_set(values: Set, limit: int = 8) -> str:
    vals = sorted(values)
    if not vals:
        return "（无）"
    shown = ", ".join(str(v) for v in vals[:limit])
    return shown + (" …" if len(vals) > limit else "")


def check_scores_pair(meta_a: Dict, meta_b: Dict, name_a: str, name_b: str) -> Dict:
    """判断两份 scores.csv 是否同一模型/同一 run/同一覆盖。"""
    log.info("")
    log.info("=== 2. 一致性检查（是否同一结果的评测）===")
    log.info("[A] %s：%d 行，%.1f MB", name_a, meta_a["rows"], meta_a["size_mb"])
    log.info("    run_id=%s model_id=%s dataset_id=%s",
             ", ".join(meta_a["run_ids"]) or "—",
             ", ".join(meta_a["models"]) or "—",
             ", ".join(meta_a["datasets"]) or "—")
    log.info("    起报 %d 个，时效 %d 个，变量 %d 个",
             len(meta_a["init_times"]), len(meta_a["leads"]),
             len(meta_a["variables"]))
    log.info("[B] %s：%d 行，%.1f MB", name_b, meta_b["rows"], meta_b["size_mb"])
    log.info("    run_id=%s model_id=%s dataset_id=%s",
             ", ".join(meta_b["run_ids"]) or "—",
             ", ".join(meta_b["models"]) or "—",
             ", ".join(meta_b["datasets"]) or "—")
    log.info("    起报 %d 个，时效 %d 个，变量 %d 个",
             len(meta_b["init_times"]), len(meta_b["leads"]),
             len(meta_b["variables"]))

    same_model = _same_set(set(meta_a["models"]), set(meta_b["models"]))
    same_run = _same_set(set(meta_a["run_ids"]), set(meta_b["run_ids"]))
    same_data = _same_set(set(meta_a["datasets"]), set(meta_b["datasets"]))
    same_inits = _same_set(meta_a["init_times"], meta_b["init_times"])
    same_leads = _same_set(meta_a["leads"], meta_b["leads"])
    same_vars = _same_set(meta_a["variables"], meta_b["variables"])

    def mark(ok: bool) -> str:
        return "一致" if ok else "不同"

    log.info("[检查] 模型 %s；run %s；数据集实现 %s",
             mark(same_model), mark(same_run), mark(same_data))
    log.info("[检查] 起报 %s；时效 %s；变量 %s",
             mark(same_inits), mark(same_leads), mark(same_vars))
    if not same_data:
        log.info("       数据集实现不同：A=%s，B=%s（属于评测实现/观测读取差异）",
                 ", ".join(meta_a["datasets"]) or "—",
                 ", ".join(meta_b["datasets"]) or "—")
    if not same_inits:
        common = meta_a["init_times"] & meta_b["init_times"]
        log.warning("       起报集合不同：共同 %d 个（A %d / B %d），"
                    "只比较共同起报上的聚合值，均值口径可能受影响",
                    len(common), len(meta_a["init_times"]), len(meta_b["init_times"]))
    if not same_leads:
        common = meta_a["leads"] & meta_b["leads"]
        log.warning("       时效集合不同：共同 %d 个（A %d / B %d）",
                    len(common), len(meta_a["leads"]), len(meta_b["leads"]))

    if same_model and same_run and same_inits and same_leads:
        verdict = ("同一模型、同一 run、同一覆盖：属于同一个预报结果的两次评测，"
                   "差异只可能来自评测实现/观测读取")
    elif same_model and same_run:
        verdict = "同一模型、同一 run，但覆盖不同：可比较，但需注意共同样本范围"
    elif same_model:
        verdict = "同一模型、不同 run：数值差包含初值/批次差异，不只是评测口径"
    else:
        verdict = "不同模型：数值差主要来自模型本身，不是评测口径对拍"
    log.info("[判定] %s", verdict)

    # manifest 里的补充信息
    ma, mb = _manifest_info(Path(meta_a["path"])), _manifest_info(Path(meta_b["path"]))
    for tag, m in ((name_a, ma), (name_b, mb)):
        if m:
            log.info("[manifest] %s：created_at=%s，n_results=%s",
                     tag, m.get("created_at", "—"), m.get("n_results", "—"))

    return {
        "same_model": same_model,
        "same_run": same_run,
        "same_data": same_data,
        "same_inits": same_inits,
        "same_leads": same_leads,
        "same_vars": same_vars,
        "verdict": verdict,
    }


def check_scores_wb2(meta_a: Dict, meta_b: Dict, name_a: str, name_b: str) -> Dict:
    """scores × WB2：检查变量/时效交集；WB2 侧通常没有 model/run 元数据。"""
    log.info("")
    log.info("=== 2. 一致性检查（是否同一结果的评测）===")
    log.info("[A] %s（scores.csv）：%d 行，%.1f MB",
             name_a, meta_a["rows"], meta_a["size_mb"])
    log.info("    run_id=%s model_id=%s dataset_id=%s",
             ", ".join(meta_a["run_ids"]) or "—",
             ", ".join(meta_a["models"]) or "—",
             ", ".join(meta_a["datasets"]) or "—")
    log.info("    起报 %d 个，时效 %d 个，变量 %d 个",
             len(meta_a["init_times"]), len(meta_a["leads"]),
             len(meta_a["variables"]))
    log.info("[B] %s（WB2）：%d 个 deterministic nc，extra.csv %d 行",
             name_b, len(meta_b["files"]), meta_b["extra_rows"])
    log.info("    变量 %d 个，时效 %d 个",
             len(meta_b["variables"]), len(meta_b["leads"]))
    log.info("    变量：%s", _fmt_set(set(meta_b["variables"]), 12))

    em = meta_b.get("extra_meta") or {}
    if em.get("models") or em.get("truths"):
        log.info("[B] extra.csv 元数据：model=%s truth=%s year=%s resolution=%s",
                 _fmt_set(em.get("models", set()), 4),
                 _fmt_set(em.get("truths", set()), 4),
                 _fmt_set(em.get("years", set()), 4),
                 _fmt_set(em.get("resolutions", set()), 4))
        wb2_models = set(em.get("models", set()))
        score_models = set(meta_a["models"])
        if wb2_models and score_models and not (wb2_models & score_models):
            log.warning("[注意] WB2 标注 model=%s，scores 侧 model_id=%s，两者不一致；"
                        "对表前请确认是不是同一模型的评测",
                        _fmt_set(wb2_models, 4), _fmt_set(score_models, 4))
    if meta_b["attrs"]:
        picked = {k: v for k, v in meta_b["attrs"].items()
                  if k.lower() in {"source", "model", "model_id", "run_id",
                                   "dataset", "title", "history"}}
        if picked:
            log.info("    nc 属性：%s", picked)
        else:
            log.info("    nc 属性：%s …", _fmt_set(set(meta_b["attrs"]), 8))
    else:
        log.info("    nc 没有全局属性")

    common_vars = set(meta_a["variables"]) & set(meta_b["variables"])
    common_leads = meta_a["leads"] & meta_b["leads"]
    log.info("[交集] 变量 %d 个（A %d / B %d）：%s",
             len(common_vars), len(meta_a["variables"]),
             len(meta_b["variables"]), _fmt_set(common_vars, 12))
    log.info("[交集] 时效 %d 个（A %d / B %d）",
             len(common_leads), len(meta_a["leads"]), len(meta_b["leads"]))
    if not common_vars:
        raise SystemExit("A/B 没有共同变量，无法对表")
    if len(common_leads) < 2:
        raise SystemExit("A/B 共同时效少于 2 个，无法对表")

    wb2_models = set(em.get("models", set()))
    score_models = set(meta_a["models"])
    if wb2_models and score_models and not (wb2_models & score_models):
        verdict = (f"元数据标注的模型不同（scores={_fmt_set(score_models, 3)}，"
                   f"WB2={_fmt_set(wb2_models, 3)}），无法确认是同一结果的评测；"
                   "以下对表只能当口径参照")
    elif wb2_models and score_models:
        verdict = ("模型标注一致，但 WB2 侧没有 run/起报信息，无法完全确认是同一结果；"
                   "以下按共同变量/时效对表")
    else:
        verdict = ("WB2 侧没有 model/run/起报元数据，无法直接确认是否同一预报结果；"
                   "以下按共同变量/时效对表，结果代表评测口径一致性")
    log.info("[判定] %s", verdict)
    return {
        "common_vars": common_vars,
        "common_leads": common_leads,
        "verdict": verdict,
    }


def log_coverage_scores(data_a: Dict, data_b: Dict,
                        all_keys_a: Set[Tuple[str, str]],
                        all_keys_b: Set[Tuple[str, str]]) -> Dict:
    """scores × scores 的可对表指标与缺口。"""
    pairs = {(var, met)
             for var in set(data_a) & set(data_b)
             for met in set(data_a[var]) & set(data_b[var])
             if met in V2_METRICS}
    only_a = {(var, met)
              for var in set(data_a) & set(data_b)
              for met in set(data_a[var]) - set(data_b[var])
              if met in V2_METRICS}
    only_b = {(var, met)
              for var in set(data_a) & set(data_b)
              for met in set(data_b[var]) - set(data_a[var])
              if met in V2_METRICS}
    only_vars_a = set(data_a) - set(data_b)
    only_vars_b = set(data_b) - set(data_a)

    log.info("")
    log.info("=== 3. 可对表范围 ===")
    log.info("[可对表] %d 个 (变量, 指标) 组合，涉及指标：%s",
             len(pairs),
             ", ".join(sorted({m for _, m in pairs},
                              key=lambda m: COMPARE_ORDER.index(m)
                              if m in COMPARE_ORDER else 99)))
    log.info("[仅 A 有] %s", _fmt_set({f"{v}/{m}" for v, m in only_a}, 12))
    log.info("[仅 B 有] %s", _fmt_set({f"{v}/{m}" for v, m in only_b}, 12))
    if only_vars_a:
        log.info("[仅 A 有的变量] %s", _fmt_set(only_vars_a, 12))
    if only_vars_b:
        log.info("[仅 B 有的变量] %s", _fmt_set(only_vars_b, 12))
    other = ({m for _, m in all_keys_a} | {m for _, m in all_keys_b}) - V2_METRICS
    if other:
        log.info("[存在但不参与标量对表] %s（谱/球谐带/分类指标等）",
                 ", ".join(sorted(other)))
    return {
        "pairs": pairs, "only_a": only_a, "only_b": only_b,
        "only_vars_a": only_vars_a, "only_vars_b": only_vars_b,
        "other_metrics": other,
    }


def log_unit_diff(meta_a: Dict, meta_b: Dict, name_a: str, name_b: str) -> List[Tuple[Tuple[str, str], Set[str], Set[str]]]:
    """单位标签差异（例如 q700 的 kg/kg vs g/kg）。"""
    diffs: List[Tuple[Tuple[str, str], Set[str], Set[str]]] = []
    for key in sorted(set(meta_a["units"]) & set(meta_b["units"])):
        ua = meta_a["units"][key]
        ub = meta_b["units"][key]
        if ua != ub:
            diffs.append((key, set(ua), set(ub)))
    if diffs:
        log.info("[单位] 检测到 %d 个单位标签差异（数值对表不受影响，但报告引用时要注意）：",
                 len(diffs))
        for key, ua, ub in diffs:
            log.info("       %s/%s: A=%s B=%s", key[0], key[1],
                     sorted(ua), sorted(ub))
    return diffs


def log_sample_counts(data_a: Dict, data_b: Dict) -> Tuple[int, int, List[str]]:
    """逐 (变量, 指标, 时效) 比较两侧的逐起报样本数。"""
    total = bad = 0
    examples: List[str] = []
    for var in sorted(set(data_a) & set(data_b)):
        for met in COMPARE_ORDER:
            if met not in data_a[var] or met not in data_b[var]:
                continue
            for lead in set(data_a[var][met]) & set(data_b[var][met]):
                total += 1
                na = data_a[var][met][lead].n
                nb = data_b[var][met][lead].n
                if na != nb:
                    bad += 1
                    if len(examples) < 6:
                        examples.append(f"{var}/{met}/{int(lead)}h: {na} ≠ {nb}")
    if total:
        log.info("[样本数] 共同 (变量,指标,时效) 键 %d 个：样本数一致 %d 个，不一致 %d 个",
                 total, total - bad, bad)
        for e in examples:
            log.info("         %s", e)
    return total, bad, examples


# ══════════════════════════════════════════════════════════════════════
# 比对
# ══════════════════════════════════════════════════════════════════════
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return _pearson(ra, rb)


def _verdict(abs_metric: bool, rel_max: float, abs_max: float,
             pearson: float, scale_kind: str) -> str:
    if scale_kind == "unit":
        return "UNIT"
    if scale_kind == "approx":
        return "SCALE"
    if abs_metric:
        limits = (ACC_ABS_PASS, ACC_ABS_CLOSE, ACC_ABS_MINOR, ACC_ABS_MAJOR)
        value = abs_max
    else:
        limits = (TOL_PASS_REL, TOL_CLOSE_REL, TOL_MINOR_REL, TOL_MAJOR_REL)
        value = rel_max
    for limit, tier in zip(limits, ("PASS", "CLOSE", "MINOR", "MAJOR")):
        if value <= limit:
            return tier
    if math.isfinite(pearson) and pearson >= 0.999:
        return "SHAPE"
    return "MISMATCH"


def _verdict_bias(value: float, pearson: float) -> str:
    """Bias 专用：按 |Δbias|/RMSE 分档。"""
    limits = (BIAS_PASS, BIAS_CLOSE, BIAS_MINOR, BIAS_MAJOR)
    for limit, tier in zip(limits, ("PASS", "CLOSE", "MINOR", "MAJOR")):
        if value <= limit:
            return tier
    if math.isfinite(pearson) and pearson >= 0.999:
        return "SHAPE"
    return "MISMATCH"


def compare_pair(var: str, label: str,
                 metric_a: str, metric_b: str,
                 x_map: Dict[int, float], y_map: Dict[int, float],
                 *, abs_metric: bool, note: str,
                 init_std: Optional[Dict[int, float]] = None,
                 x_mean_map: Optional[Dict[int, float]] = None,
                 normalize_by: Optional[Dict[int, float]] = None,
                 scale_override: Optional[float] = None) -> Optional[PairResult]:
    leads = sorted(set(x_map) & set(y_map))
    if len(leads) < 2:
        return None
    x = np.array([x_map[l] for l in leads], dtype=float)
    y = np.array([y_map[l] for l in leads], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    leads = [l for l, k in zip(leads, ok) if k]
    x, y = x[ok], y[ok]
    if x.size < 2:
        return None

    pearson = _pearson(x, y)
    spearman = _spearman(x, y)

    # 倍率推断：ratio = A/B；scale 是“把 A 换算到 B 口径”的因子，取 1/ratio。
    ratio = np.full(x.size, np.nan)
    nz = np.abs(y) > 1e-12
    ratio[nz] = x[nz] / y[nz]
    ratio_median = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan")
    ratio_cv = (float(np.nanstd(ratio) / abs(ratio_median))
                if math.isfinite(ratio_median) and ratio_median != 0 else float("nan"))
    scale, scale_kind = 1.0, "none"
    if (not abs_metric and x.size >= 3 and math.isfinite(ratio_median)
            and ratio_median != 0):
        if ratio_cv <= 1e-9 and abs(ratio_median - 1.0) > 1e-6:
            scale, scale_kind = 1.0 / ratio_median, "unit"
        elif ratio_cv <= 0.01 and abs(ratio_median - 1.0) > 0.01:
            scale, scale_kind = 1.0 / ratio_median, "approx"
    if scale_override is not None:
        scale = float(scale_override)

    abs_diff = x * scale - y
    rel = np.full(x.size, np.nan)
    rel[nz] = (x[nz] * scale - y[nz]) / np.abs(y[nz])
    fine = np.isfinite(rel)
    rel_max = float(np.max(np.abs(rel[fine]))) if fine.any() else float("nan")
    rel_median = float(np.median(np.abs(rel[fine]))) if fine.any() else float("nan")
    abs_max = float(np.max(np.abs(abs_diff))) if abs_diff.size else float("nan")

    bias_rel_max = None
    if normalize_by is not None:
        denom = np.array([normalize_by.get(l, np.nan) for l in leads], dtype=float)
        valid = np.isfinite(denom) & (denom > 0)
        if valid.any():
            bias_rel_max = float(
                np.max(np.abs(x[valid] * scale - y[valid]) / denom[valid])
            )

    outliers: List[int] = []
    if fine.any() and math.isfinite(rel_median) and rel_median > 0:
        thr = max(OUTLIER_FACTOR * rel_median, TOL_CLOSE_REL)
        outliers = [l for l, r in zip(leads, rel)
                    if math.isfinite(r) and abs(r) > thr]

    slope, intercept = float("nan"), float("nan")
    if np.std(x) > 0:
        try:
            slope, intercept = np.polyfit(x, y, 1)
            slope, intercept = float(slope), float(intercept)
        except (np.linalg.LinAlgError, ValueError):
            pass

    scatter_ratio = None
    if init_std:
        stds = [init_std.get(l, float("nan")) * scale for l in leads]
        stds = np.array([s for s in stds if math.isfinite(s)], dtype=float)
        deltas = np.abs(abs_diff)
        if stds.size and np.median(stds) > 0:
            scatter_ratio = float(np.median(deltas) / np.median(stds))

    if bias_rel_max is not None:
        verdict = _verdict_bias(bias_rel_max, pearson)
    else:
        verdict = _verdict(abs_metric, rel_max, abs_max, pearson, scale_kind)

    note_parts = [note]
    if bias_rel_max is not None:
        note_parts.append(f"max|Δbias|/RMSE={bias_rel_max:.4f}（判定依据）")
    if scale_kind in {"unit", "approx"}:
        tag = "精确" if scale_kind == "unit" else "近似"
        note_parts.append(f"{tag}倍率 {ratio_median:.6g}×（A/B）")
    if outliers:
        shown = ", ".join(str(l) for l in outliers[:8])
        more = "…" if len(outliers) > 8 else ""
        note_parts.append(f"异常时效 {len(outliers)} 个: {shown}{more}")
    if scatter_ratio is not None:
        tag = "小于" if scatter_ratio < 1 else "大于"
        note_parts.append(f"中位|Δ| {tag}起报间标准差 ({scatter_ratio:.2f}×)")

    x_mean = None
    if x_mean_map is not None:
        x_mean = np.array([x_mean_map.get(l, np.nan) for l in leads], dtype=float)

    return PairResult(
        var=var, label=label, metric_a=metric_a, metric_b=metric_b,
        n=x.size, leads=leads, x=x, y=y, scale=scale, scale_kind=scale_kind,
        ratio_median=ratio_median, ratio_cv=ratio_cv,
        rel_max=rel_max, rel_median=rel_median, abs_max=abs_max,
        pearson=pearson, spearman=spearman,
        slope=slope, intercept=intercept,
        verdict=verdict, note="；".join(note_parts),
        outliers=outliers, init_scatter_ratio=scatter_ratio, x_mean=x_mean,
        bias_rel_max=bias_rel_max,
    )


def build_pairs_scores(data_a: Dict, data_b: Dict) -> List[PairResult]:
    """scores × scores：两侧都按逐起报值的 mean 聚合后逐指标比较。"""
    results: List[PairResult] = []
    for var in sorted(set(data_a) & set(data_b)):
        mets_a, mets_b = data_a[var], data_b[var]
        rmse_pair: Optional[PairResult] = None

        for met in COMPARE_ORDER:
            if met not in mets_a or met not in mets_b:
                continue
            x_map = {l: agg.mean for l, agg in mets_a[met].items()}
            y_map = {l: agg.mean for l, agg in mets_b[met].items()}
            init_std = {l: agg.std for l, agg in mets_a[met].items()}
            abs_metric = met == "acc"

            note = {
                "rmse": "两侧均按逐起报 RMSE 的 mean 聚合",
                "acc": "两侧均为距平相关；用绝对差判定",
                "bias": "逐格平均偏差；判定用 |Δbias|/RMSE_B",
                "mae": "两侧均按逐起报 MAE 的 mean 聚合",
            }.get(met, "同名指标直接比较")

            normalize_by = None
            scale_override = None
            if met == "bias":
                normalize_by = {l: agg.mean for l, agg in mets_b["rmse"].items()} \
                    if "rmse" in mets_b else None
                scale_override = rmse_pair.scale if rmse_pair else None

            r = compare_pair(
                var, METRIC_LABEL.get(met, met), met, met,
                x_map, y_map,
                abs_metric=abs_metric, note=note,
                init_std=init_std,
                normalize_by=normalize_by,
                scale_override=scale_override,
            )
            if r:
                results.append(r)
                if met == "rmse":
                    rmse_pair = r
    return results


def build_pairs_wb2(data_a: Dict, det: Dict, extra: Dict) -> List[PairResult]:
    """scores × WB2：保持原 T1–T7 的配对逻辑。"""
    results: List[PairResult] = []

    for var in sorted(data_a):
        mets = data_a[var]

        if "rmse" in mets and (var, "rmse") in det:
            x_map = {l: a.rms for l, a in mets["rmse"].items()}
            x_mean_map = {l: a.mean for l, a in mets["rmse"].items()}
            init_std = {l: a.std for l, a in mets["rmse"].items()}
            r = compare_pair(
                var, "RMSE", "rmse", "rmse", x_map, det[(var, "rmse")],
                abs_metric=False, init_std=init_std, x_mean_map=x_mean_map,
                note="scores 侧用逐起报 RMSE 的 RMS 聚合；WB2 为 sqrt(mse)",
            )
            if r:
                results.append(r)

        if "acc" in mets and (var, "acc") in det:
            x_map = {l: a.mean for l, a in mets["acc"].items()}
            r = compare_pair(
                var, "ACC", "acc", "acc", x_map, det[(var, "acc")],
                abs_metric=True,
                note="均为距平相关（未去域均）；气候态来源不同",
            )
            if r:
                results.append(r)

        if "bias" in mets and (var, "bias") in det:
            x_map = {l: a.mean for l, a in mets["bias"].items()}
            rmse_pair = next(
                (p for p in results if p.var == var and p.label == "RMSE"), None)
            r = compare_pair(var, "Bias", "bias", "bias", x_map, det[(var, "bias")],
                             abs_metric=False, note="逐格平均偏差；判定用 |Δbias|/RMSE",
                             normalize_by=det.get((var, "rmse")),
                             scale_override=rmse_pair.scale if rmse_pair else None)
            if r:
                results.append(r)

        if "mae" in mets and (var, "mae") in det:
            x_map = {l: a.mean for l, a in mets["mae"].items()}
            r = compare_pair(var, "MAE", "mae", "mae", x_map, det[(var, "mae")],
                             abs_metric=False, note="平均绝对误差")
            if r:
                results.append(r)

        if "activity_forecast" in mets and (var, "prediction_activity") in extra:
            x_map = {l: a.mean for l, a in mets["activity_forecast"].items()}
            r = compare_pair(
                var, "Activity(fcst)", "activity_forecast", "prediction_activity",
                x_map, extra[(var, "prediction_activity")],
                abs_metric=False,
                note="口径：scores=距平标准差，WB2=距平 RMS；气候态也不同",
            )
            if r:
                results.append(r)

    return results


# ══════════════════════════════════════════════════════════════════════
# 附加检查
# ══════════════════════════════════════════════════════════════════════
def aggregation_sensitivity(results: Sequence[PairResult]) -> List[Tuple[str, float, float, float, str]]:
    """T6：RMSE 的 mean vs RMS 聚合，哪个更贴 WB2。"""
    rows = []
    for r in results:
        if r.label != "RMSE" or r.x_mean is None:
            continue
        nz = np.abs(r.y) > 1e-12
        rel_rms = np.abs((r.x * r.scale - r.y)[nz] / r.y[nz])
        rel_mean = np.abs((r.x_mean * r.scale - r.y)[nz] / r.y[nz])
        mr = float(np.max(rel_rms)) if rel_rms.size else float("nan")
        mm = float(np.max(rel_mean)) if rel_mean.size else float("nan")
        rows.append((r.var, mr, mm, r.ratio_median,
                     "RMS" if mr <= mm else "mean"))
    return rows


def growth_curve_check(results: Sequence[PairResult]) -> Dict:
    """T5：按首时效归一的增长曲线形状 + 跨变量秩相关。"""
    curves = {}
    for r in results:
        if r.label != "RMSE" or r.n < 2 or r.y[0] == 0 or r.x[0] == 0:
            continue
        curves[r.var] = (r.x / r.x[0], r.y / r.y[0], r.leads)
    if not curves:
        return {"per_var": [], "rank_spearman_mean": float("nan"), "rank_leads": []}

    per_var = []
    for var, (xn, yn, _) in curves.items():
        rel = np.abs(xn - yn) / np.maximum(np.abs(yn), 1e-12)
        per_var.append((var, _pearson(xn, yn), float(np.max(rel))))

    rank_leads, rank_vals = [], []
    lead_sets = [set(c[2]) for c in curves.values()]
    common = sorted(set.intersection(*lead_sets)) if lead_sets else []
    for lead in common:
        xs, ys = [], []
        for xn, yn, leads in curves.values():
            i = leads.index(lead)
            xs.append(xn[i])
            ys.append(yn[i])
        if len(xs) >= 3:
            rho = _spearman(np.array(xs), np.array(ys))
            if math.isfinite(rho):
                rank_leads.append(lead)
                rank_vals.append(rho)

    return {
        "per_var": per_var,
        "rank_spearman_mean": float(np.mean(rank_vals)) if rank_vals else float("nan"),
        "rank_leads": list(zip(rank_leads, rank_vals)),
    }


# ══════════════════════════════════════════════════════════════════════
# 日志输出
# ══════════════════════════════════════════════════════════════════════
def _f(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}g}" if abs(v) < 1e3 and v != 0 else f"{v:.{nd}f}"


def _pct(v) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v * 100:.3f}%"


def _sample_indices(leads: Sequence[int]) -> List[int]:
    idx = []
    for s in SAMPLE_LEADS:
        if s in leads:
            idx.append(leads.index(s))
    if not idx:
        idx = list(range(min(6, len(leads))))
    if leads and (len(leads) - 1) not in idx:
        idx.append(len(leads) - 1)
    return sorted(set(idx))


def log_pair(r: PairResult) -> None:
    """逐指标对比：一行摘要 + 备注 + RMSE 抽样时效表。"""
    if r.bias_rel_max is not None:
        judge = f"|Δbias|/RMSE={r.bias_rel_max:.4f}"
    elif r.label == "ACC":
        judge = f"max|Δ|={_f(r.abs_max, 5)}"
    else:
        judge = f"max|Δ%|={_pct(r.rel_max)}"
    log.info("  %-6s %-16s n=%2d  比值中位=%s  %s  Pearson=%.6f  => %s",
             r.var, r.label, r.n, _f(r.ratio_median), judge,
             r.pearson, r.verdict)
    log.info("        %s", r.note)
    if r.label == "RMSE":
        idx = _sample_indices(r.leads)
        rows = []
        for i in idx:
            rel = ((r.x[i] * r.scale - r.y[i]) / abs(r.y[i])
                   if r.y[i] else float("nan"))
            rows.append({
                "lead": r.leads[i],
                "B": _f(r.y[i], 6),
                "A": _f(r.x[i], 6),
                "delta%": _pct(rel),
                "ratio": _f(r.x[i] / r.y[i], 6) if r.y[i] else "—",
            })
        log.info("\n%s", pd.DataFrame(rows).to_string(index=False))


def log_details(results: Sequence[PairResult], name_a: str, name_b: str) -> None:
    log.info("")
    log.info("=== 4. 逐指标对比（A=%s, B=%s；Δ=A−B，Δ%%=(A−B)/|B|）===",
             name_a, name_b)
    if not results:
        log.warning("没有可对表的 (变量, 指标) 组合")
        return
    for r in results:
        log_pair(r)


def log_extra_checks(results: Sequence[PairResult], *, wb2_mode: bool) -> None:
    """T5 / T6 / T7。"""
    log.info("")
    log.info("=== 5. 附加检查 ===")

    growth = growth_curve_check(results)
    log.info("[T5] 归一增长曲线（各自除以首个时效）")
    if growth["per_var"]:
        for var, corr, dev in growth["per_var"]:
            log.info("      %-8s Pearson=%s  最大归一偏差=%s",
                     var, _f(corr, 6), _pct(dev))
        if math.isfinite(growth["rank_spearman_mean"]):
            log.info("[T5] 跨变量排序一致性：Spearman 均值 %.4f（接近 1 说明没有变量串名/错位）",
                     growth["rank_spearman_mean"])
    else:
        log.info("      （没有可比较的 RMSE 曲线）")

    if wb2_mode:
        rows = aggregation_sensitivity(results)
        log.info("[T6] RMSE 聚合口径：scores 侧逐起报 RMSE 的 RMS vs mean")
        if rows:
            for var, mr, mm, ratio, winner in rows:
                log.info("      %-8s max|Δ|(RMS)=%s  max|Δ|(mean)=%s  比值中位=%s  -> %s 更贴 WB2",
                         var, _pct(mr), _pct(mm), _f(ratio), winner)
            wins = sum(1 for row in rows if row[4] == "RMS")
            log.info("      结论：%d/%d 个变量 RMS 聚合更贴 WB2", wins, len(rows))
        else:
            log.info("      （没有 RMSE 对表，跳过）")

    log.info("[T7] 起报间散布：RMSE 的中位|Δ| / A 侧逐起报标准差")
    any_scatter = False
    for r in results:
        if r.label != "RMSE" or r.init_scatter_ratio is None:
            continue
        any_scatter = True
        tag = "小于" if r.init_scatter_ratio < 1 else "大于"
        log.info("      %-8s %-16s %.3f（%s起报间标准差）",
                 r.var, r.label, r.init_scatter_ratio, tag)
    if not any_scatter:
        log.info("      （没有起报级散布数据）")


def log_summary(results: Sequence[PairResult]) -> None:
    """最后一张总表（纯 ASCII 列，便于对齐）。"""
    log.info("")
    log.info("=== 6. 总表 ===")
    if not results:
        log.warning("没有可对表的指标")
        return

    order = {m: i for i, m in enumerate(COMPARE_ORDER)}
    rows = []
    for r in sorted(results, key=lambda z: (z.var, order.get(z.metric_a, 99))):
        rows.append({
            "var": r.var,
            "metric": r.label,
            "n": r.n,
            "ratio_med": _f(r.ratio_median),
            "max_rel": ("" if (r.bias_rel_max is not None or r.metric_a == "acc")
                        else _pct(r.rel_max)),
            "max_abs": _f(r.abs_max, 4),
            "bias_rel": (_f(r.bias_rel_max, 4)
                         if r.bias_rel_max is not None else ""),
            "pearson": _f(r.pearson, 6),
            "spearman": _f(r.spearman, 5),
            "verdict": r.verdict,
        })
    log.info("\n%s", pd.DataFrame(rows).to_string(index=False))

    counts: Dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    log.info("判定分布：%s",
             "，".join(f"{k}={v}" for k, v in sorted(counts.items())))
    log.info("判定说明：PASS≤%.1f%%，CLOSE≤%.1f%%，MINOR≤%.1f%%，MAJOR≤%.1f%%，"
             "SHAPE=幅度有差但趋势一致，UNIT/SCALE=恒定倍率，MISMATCH=不一致",
             TOL_PASS_REL * 100, TOL_CLOSE_REL * 100,
             TOL_MINOR_REL * 100, TOL_MAJOR_REL * 100)
    log.info("Bias 专用判定：bias_rel 列 = max|Δbias|/RMSE，"
             "PASS≤%.2f，CLOSE≤%.2f，MINOR≤%.2f，MAJOR≤%.2f",
             BIAS_PASS, BIAS_CLOSE, BIAS_MINOR, BIAS_MAJOR)


# ══════════════════════════════════════════════════════════════════════
# Markdown 报告
# ══════════════════════════════════════════════════════════════════════
def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> List[str]:
    out = ["| " + " | ".join(_md_cell(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
    return out


def _metric_rank(met: str) -> int:
    return COMPARE_ORDER.index(met) if met in COMPARE_ORDER else 99


def _dev_text(r: PairResult) -> str:
    if r.bias_rel_max is not None:
        return f"|Δbias|/RMSE = {r.bias_rel_max:.4f}"
    if r.metric_a == "acc":
        return f"max|ΔACC| = {_f(r.abs_max, 5)}"
    return f"max|Δ%| = {_pct(r.rel_max)}"


def _worst_result(results: Sequence[PairResult]) -> Optional[PairResult]:
    if not results:
        return None

    def dev(r: PairResult) -> float:
        if r.bias_rel_max is not None:
            return r.bias_rel_max
        if r.metric_a == "acc":
            return r.abs_max
        return r.rel_max

    return max(results, key=lambda r: (VERDICT_RANK.get(r.verdict, -1), dev(r)))


def write_markdown(path: Path, ctx: Dict) -> None:
    """把本次对拍结果写成 Markdown 报告。"""
    name_a, name_b = ctx["name_a"], ctx["name_b"]
    meta_a, meta_b = ctx["meta_a"], ctx["meta_b"]
    consistency = ctx.get("consistency") or {}
    coverage = ctx.get("coverage") or {}
    results: List[PairResult] = ctx.get("results") or []
    wb2_mode = bool(ctx.get("wb2_mode"))

    L: List[str] = [f"# 评测结果对拍：{name_a} vs {name_b}", ""]
    L += [f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　　"
          f"A = `{ctx['path_a']}`　　B = `{ctx['path_b']}`", ""]

    # 0. 概览
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    L += ["## 0. 结论速览", ""]
    if results:
        if set(counts) == {"PASS"}:
            headline = f"全部 {len(results)} 个可对表项在输出精度内一致（PASS）"
        else:
            worst = _worst_result(results)
            headline = (f"最大偏差项：{worst.var}/{worst.label}，"
                        f"{_dev_text(worst)}（{worst.verdict}）")
        dist = "，".join(
            f"{VERDICT_BADGE.get(k, k)} × {v}"
            for k, v in sorted(counts.items(),
                               key=lambda kv: VERDICT_RANK.get(kv[0], 9)))
        L += [
            f"- **一致性**：{consistency.get('verdict', '—')}",
            f"- **可对表**：{len(results)} 个 (变量, 指标) 组合",
            f"- **结果分布**：{dist}",
            f"- **一句话**：{headline}",
        ]
    else:
        L += ["- **可对表**：0 个组合（没有共同变量或指标）"]
    L += ["", "---", ""]

    # 1. 输入信息
    L += ["## 1. 输入信息", ""]
    if not wb2_mode:
        ma = _manifest_info(Path(meta_a["path"]))
        mb = _manifest_info(Path(meta_b["path"]))
        rows = [
            ["路径", f"`{meta_a['path']}`", f"`{meta_b['path']}`"],
            ["格式", "scores.csv", "scores.csv"],
            ["行数", f"{meta_a['rows']:,}", f"{meta_b['rows']:,}"],
            ["run_id", ", ".join(meta_a["run_ids"]) or "—",
             ", ".join(meta_b["run_ids"]) or "—"],
            ["model_id", ", ".join(meta_a["models"]) or "—",
             ", ".join(meta_b["models"]) or "—"],
            ["dataset_id", ", ".join(meta_a["datasets"]) or "—",
             ", ".join(meta_b["datasets"]) or "—"],
            ["起报 / 时效 / 变量",
             f"{len(meta_a['init_times'])} / {len(meta_a['leads'])} / {len(meta_a['variables'])}",
             f"{len(meta_b['init_times'])} / {len(meta_b['leads'])} / {len(meta_b['variables'])}"],
        ]
        if ma or mb:
            rows.append(["manifest",
                         f"created_at={ma.get('created_at', '—')}<br>n_results={ma.get('n_results', '—')}",
                         f"created_at={mb.get('created_at', '—')}<br>n_results={mb.get('n_results', '—')}"])
    else:
        em = meta_b.get("extra_meta") or {}
        rows = [
            ["路径", f"`{meta_a['path']}`", f"`{ctx['path_b']}`"],
            ["格式", "scores.csv",
             f"WB2 deterministic nc（{len(meta_b['files'])} 个）+ extra.csv"],
            ["规模", f"{meta_a['rows']:,} 行",
             f"nc {len(meta_b['files'])} 个；extra {meta_b['extra_rows']:,} 行"],
            ["run_id", ", ".join(meta_a["run_ids"]) or "—", "—"],
            ["model_id", ", ".join(meta_a["models"]) or "—",
             _fmt_set(em.get("models", set()), 4)],
            ["dataset / truth", ", ".join(meta_a["datasets"]) or "—",
             f"truth={_fmt_set(em.get('truths', set()), 4)}，"
             f"year={_fmt_set(em.get('years', set()), 4)}"],
            ["起报 / 时效 / 变量",
             f"{len(meta_a['init_times'])} / {len(meta_a['leads'])} / {len(meta_a['variables'])}",
             f"— / {len(meta_b['leads'])} / {len(meta_b['variables'])}"],
        ]
    L += _md_table(["项", f"A：{name_a}", f"B：{name_b}"], rows)
    L += [""]

    # 2. 一致性
    L += ["## 2. 一致性检查", ""]
    if not wb2_mode:
        checks = [
            ("模型", consistency.get("same_model")),
            ("run", consistency.get("same_run")),
            ("数据集实现", consistency.get("same_data")),
            ("起报集合", consistency.get("same_inits")),
            ("时效集合", consistency.get("same_leads")),
            ("变量集合", consistency.get("same_vars")),
        ]
        for label, ok in checks:
            L.append(f"- {'✅' if ok else '❌'} {label}：{'一致' if ok else '不同'}")
        if not consistency.get("same_data"):
            L.append(f"  - 数据集实现：A = `{', '.join(meta_a['datasets']) or '—'}`，"
                     f"B = `{', '.join(meta_b['datasets']) or '—'}`"
                     "（评测实现/观测读取差异，不是预报差异）")
        sc = ctx.get("sample_counts")
        if sc:
            total, bad, examples = sc
            L.append(f"- {'✅' if bad == 0 else '⚠️'} 逐键样本数：{total} 个键，"
                     f"{bad} 个不一致")
            if examples:
                L.append("  - " + "；".join(examples))
    else:
        em = meta_b.get("extra_meta") or {}
        if em.get("models") or em.get("truths"):
            L.append(f"- WB2 extra 元数据：model = `{_fmt_set(em.get('models', set()), 4)}`，"
                     f"truth = `{_fmt_set(em.get('truths', set()), 4)}`，"
                     f"year = `{_fmt_set(em.get('years', set()), 4)}`")
        L.append(f"- 共同变量 {len(consistency.get('common_vars', set()))} 个；"
                 f"共同时效 {len(consistency.get('common_leads', set()))} 个")
    L += ["", f"> **判定**：{consistency.get('verdict', '—')}", ""]

    ud = ctx.get("unit_diffs") or []
    L += ["### 单位标签", ""]
    if ud:
        L += _md_table(["变量/指标", f"A：{name_a}", f"B：{name_b}"],
                       [[f"{key[0]}/{key[1]}", ", ".join(sorted(ua)),
                         ", ".join(sorted(ub))] for key, ua, ub in ud])
        L += ["", "> 单位标签不同不代表数值不同；以数值对比为准。", ""]
    else:
        L += ["无差异。", ""]

    # 3. 可对表范围
    L += ["## 3. 可对表范围", ""]
    if not wb2_mode:
        pairs = coverage.get("pairs", set())
        metrics = sorted({m for _, m in pairs}, key=_metric_rank)
        L.append(f"- **可对表**：{len(pairs)} 个 (变量, 指标) 组合；"
                 f"指标：{', '.join(metrics) or '—'}")
        only_a = coverage.get("only_a", set())
        only_b = coverage.get("only_b", set())
        if only_a:
            L.append("- **仅 A 有**：" + ", ".join(sorted(f"{v}/{m}" for v, m in only_a)))
        if only_b:
            L.append("- **仅 B 有**：" + ", ".join(sorted(f"{v}/{m}" for v, m in only_b)))
        other = coverage.get("other_metrics", set())
        if other:
            L.append("- **存在但不参与标量对表**：" + ", ".join(sorted(other)))
    else:
        L.append("- **WB2 deterministic 指标**："
                 + (", ".join(ctx.get("det_metrics") or []) or "—"))
        L.append("- **WB2 extra 指标**："
                 + (", ".join(ctx.get("extra_metrics") or []) or "—"))
        if ctx.get("extra_unmatched"):
            L.append(f"- **extra 未映射**：{len(ctx['extra_unmatched'])} 条")
        L.append(f"- **实际对表**：{len(results)} 个组合")
    L += ["", "---", ""]

    # 4. 逐指标对比
    L += ["## 4. 逐指标对比", "", "### 4.1 总表", ""]
    rows = []
    for r in sorted(results, key=lambda z: (z.var, _metric_rank(z.metric_a))):
        rows.append([
            r.var, r.label, r.n, _f(r.ratio_median),
            "" if (r.bias_rel_max is not None or r.metric_a == "acc")
            else _pct(r.rel_max),
            _f(r.abs_max, 4),
            "" if r.bias_rel_max is None else _f(r.bias_rel_max, 4),
            _f(r.pearson, 6), _f(r.spearman, 5),
            VERDICT_BADGE.get(r.verdict, r.verdict),
        ])
    if rows:
        L += _md_table(["变量", "指标", "时效数", "比值中位", "最大相对差",
                        "最大绝对差", "|Δbias|/RMSE", "Pearson", "Spearman", "判定"],
                       rows)
        L += ["", "> Δ = A − B，Δ% = (A − B) / |B|；bias_rel = max|Δbias|/RMSE。", ""]
    else:
        L += ["没有可对表的指标。", ""]

    rmse_results = sorted([r for r in results if r.label == "RMSE"],
                          key=lambda x: x.var)
    if rmse_results:
        L += ["### 4.2 RMSE 逐时效明细", ""]
        for r in rmse_results:
            L.append(f"<details><summary><b>{r.var}</b> — n={r.n}，"
                     f"max|Δ%|={_pct(r.rel_max)}，"
                     f"{VERDICT_BADGE.get(r.verdict, r.verdict)}</summary>")
            L.append("")
            rows = []
            for i, lead in enumerate(r.leads):
                y = r.y[i]
                rel = ((r.x[i] * r.scale - y) / abs(y)) if y else float("nan")
                rows.append([int(lead), _f(r.x[i], 6), _f(y, 6), _pct(rel),
                             _f(r.x[i] / y, 6) if y else "—"])
            L += _md_table(["时效(h)", f"A：{name_a}", f"B：{name_b}",
                            "Δ%", "比值 A/B"], rows)
            L += ["", "</details>", ""]
    L += ["---", ""]

    # 5. 附加检查
    growth = growth_curve_check(results)
    L += ["## 5. 附加检查", "", "### T5 归一增长曲线", ""]
    if growth["per_var"]:
        L += _md_table(["变量", "Pearson", "最大归一偏差"],
                       [[var, _f(corr, 6), _pct(dev)]
                        for var, corr, dev in growth["per_var"]])
        if math.isfinite(growth["rank_spearman_mean"]):
            L += ["", f"- 跨变量排序一致性：Spearman 均值 "
                      f"**{growth['rank_spearman_mean']:.4f}**"]
    else:
        L.append("无可用 RMSE 曲线。")

    if wb2_mode:
        L += ["", "### T6 RMSE 聚合口径", ""]
        rows6 = aggregation_sensitivity(results)
        if rows6:
            L += _md_table(["变量", "max|Δ|（RMS 聚合）", "max|Δ|（mean 聚合）",
                            "比值中位", "更贴 WB2"],
                           [[var, _pct(mr), _pct(mm), _f(ratio), winner]
                            for var, mr, mm, ratio, winner in rows6])
            wins = sum(1 for row in rows6 if row[4] == "RMS")
            L += ["", f"- {wins}/{len(rows6)} 个变量的 RMS 聚合更贴 WB2。"]
        else:
            L.append("无 RMSE 对表，跳过。")

    L += ["", "### T7 起报间散布", ""]
    scat = [(r.var, r.init_scatter_ratio) for r in results
            if r.label == "RMSE" and r.init_scatter_ratio is not None]
    if scat:
        L += _md_table(["变量", "中位|Δ| / A 起报间标准差"],
                       [[var, f"{val:.3f}"] for var, val in scat])
    else:
        L.append("无起报级散布数据。")
    L += ["", "---", ""]

    # 6. 口径说明
    L += ["## 6. 口径与判定说明", "",
          "- 方向：Δ = A − B，Δ% = (A − B) / |B|。",
          "- scores × scores：两侧都按 (变量, 指标, 时效) 的逐起报值均值聚合。",
          "- scores × WB2：scores 侧 RMSE 用逐起报 RMSE 的 RMS 聚合，"
          "WB2 侧 RMSE = sqrt(mse)。",
          f"- 相对差判定：PASS ≤ {TOL_PASS_REL:.2%}，CLOSE ≤ {TOL_CLOSE_REL:.2%}，"
          f"MINOR ≤ {TOL_MINOR_REL:.2%}，MAJOR ≤ {TOL_MAJOR_REL:.2%}；"
          "再大则看 Pearson，≥0.999 记为 SHAPE，否则 MISMATCH。",
          f"- ACC 用绝对差：PASS ≤ {ACC_ABS_PASS:g}，CLOSE ≤ {ACC_ABS_CLOSE:g}，"
          f"MINOR ≤ {ACC_ABS_MINOR:g}，MAJOR ≤ {ACC_ABS_MAJOR:g}。",
          f"- Bias 用 |Δbias|/RMSE：PASS ≤ {BIAS_PASS}，CLOSE ≤ {BIAS_CLOSE}，"
          f"MINOR ≤ {BIAS_MINOR}，MAJOR ≤ {BIAS_MAJOR}。",
          "- 谱/球谐带/分类指标不参与标量对表。",
          "- scores.csv 若混入非全球 region 行，会被剔除并记录在日志中。",
          ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L).rstrip() + "\n", encoding="utf-8")
    log.info("Markdown 报告写入：%s", path)


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════
def setup_logging(log_file: Optional[str]) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        log.info("日志同时写入：%s", p)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="两套评测结果对拍：scores.csv × scores.csv 或 scores.csv × WB2 nc",
    )
    ap.add_argument("a", metavar="A", help="第一个结果目录（scores.csv 或 WB2 nc）")
    ap.add_argument("b", metavar="B", help="第二个结果目录（scores.csv 或 WB2 nc）")
    ap.add_argument("--log-file", default=None,
                    help="同时把日志写入该文件（默认只输出到控制台）")
    ap.add_argument("--name-a", default=None, help="A 侧显示名（默认用目录名）")
    ap.add_argument("--name-b", default=None, help="B 侧显示名（默认用目录名）")
    ap.add_argument("--out", default=None,
                    help="Markdown 报告路径（默认写在本脚本旁边 compare_<A>_vs_<B>.md）")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file)

    path_a, path_b = Path(args.a), Path(args.b)
    fmt_a, fmt_b = detect_format(path_a), detect_format(path_b)
    name_a = args.name_a or path_a.name
    name_b = args.name_b or path_b.name

    # WB2 放 A 侧时交换，让 scores 始终在 A 侧
    if fmt_a == "wb2" and fmt_b == "scores":
        log.info("检测到 A 是 WB2、B 是 scores，自动交换两侧")
        path_a, path_b = path_b, path_a
        fmt_a, fmt_b = fmt_b, fmt_a
        name_a, name_b = name_b, name_a

    if fmt_a == "wb2" and fmt_b == "wb2":
        raise SystemExit("不支持 WB2 × WB2：两边都是 *_deterministic.nc")

    log.info("=== 1. 输入识别 ===")
    log.info("A = %s（%s）", name_a, "scores.csv" if fmt_a == "scores" else "WB2 nc")
    log.info("B = %s（%s）", name_b, "scores.csv" if fmt_b == "scores" else "WB2 nc")

    log.info("")
    log.info("[读取] %s", path_a)
    data_a, meta_a = load_scores(path_a)
    log.info("       %d 行，指标 key %d 个", meta_a["rows"], len(meta_a["all_keys"]))

    if fmt_b == "scores":
        log.info("[读取] %s", path_b)
        data_b, meta_b = load_scores(path_b)
        log.info("       %d 行，指标 key %d 个", meta_b["rows"], len(meta_b["all_keys"]))
        if not (set(data_a) & set(data_b)):
            raise SystemExit("A/B 没有共同变量，无法对表")
        consistency = check_scores_pair(meta_a, meta_b, name_a, name_b)
        unit_diffs = log_unit_diff(meta_a, meta_b, name_a, name_b)
        sample_counts = log_sample_counts(data_a, data_b)
        coverage = log_coverage_scores(data_a, data_b,
                                       meta_a["all_keys"], meta_b["all_keys"])
        results = build_pairs_scores(data_a, data_b)
        log.info("[可对表] 实际构建 %d 个 (变量, 指标) 对", len(results))
        wb2_mode = False
        det_metrics = extra_metrics = extra_unmatched = None
    else:
        log.info("[读取] %s", path_b)
        det, files_used, attrs = load_wb2_deterministic(path_b)
        extra, extra_unmatched, extra_meta = load_wb2_extra(path_b)
        meta_b = {
            "files": files_used,
            "attrs": attrs,
            "extra_rows": extra_meta["rows"],
            "extra_meta": extra_meta,
            "variables": {v for v, _ in det},
            "leads": {float(l) for d in det.values() for l in d},
        }
        log.info("       nc %d 个，指标 key %d 个，extra %d 行",
                 len(files_used), len(det), extra_meta["rows"])
        consistency = check_scores_wb2(meta_a, meta_b, name_a, name_b)
        log.info("")
        log.info("=== 3. 可对表范围 ===")
        det_metrics = sorted({m for _, m in det})
        log.info("[WB2 指标] %s", ", ".join(det_metrics))
        extra_metrics = sorted({m for _, m in extra})
        if extra:
            log.info("[WB2 extra 指标] %s", ", ".join(extra_metrics))
        if extra_unmatched:
            log.info("[extra 未映射] %d 条（%s）", len(extra_unmatched),
                     ", ".join(sorted({m for _, _, m in extra_unmatched})))
        extra_unmatched = sorted(extra_unmatched)
        results = build_pairs_wb2(data_a, det, extra)
        log.info("[可对表] 实际构建 %d 个 (变量, 指标) 对", len(results))
        wb2_mode = True
        unit_diffs = None
        sample_counts = None
        coverage = None

    log_details(results, name_a, name_b)
    log_extra_checks(results, wb2_mode=wb2_mode)
    log_summary(results)

    out_path = (Path(args.out) if args.out
                else Path(__file__).resolve().parent / f"compare_{name_a}_vs_{name_b}.md")
    write_markdown(out_path, {
        "name_a": name_a, "name_b": name_b,
        "path_a": path_a.resolve(), "path_b": path_b.resolve(),
        "meta_a": meta_a, "meta_b": meta_b,
        "consistency": consistency,
        "unit_diffs": unit_diffs,
        "sample_counts": sample_counts,
        "coverage": coverage,
        "results": results,
        "wb2_mode": wb2_mode,
        "det_metrics": det_metrics,
        "extra_metrics": extra_metrics,
        "extra_unmatched": extra_unmatched,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
