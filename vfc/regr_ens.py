# -*- coding: utf-8 -*-
# [连续量/regression] 编排层（ensemble/盘古 run_ensemble/ens_main）——属主：同 regr_pair
"""连续量验证（single + ensemble）共享实现。

由原集合入口整合过来：CLI 统一在 run_rmse.py，实现体放在 vfc/ 以避免多人同时改顶层 run_*.py 冲突。
实现体放在 vfc/ 以避免多人同时改顶层 run_*.py 冲突。"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

from vfc.ensemble_io import EnsembleForecast
from vfc.pangu_io import PanguForecast, scan_pangu_dates
from vfc.aifs_io import AifsForecast, scan_aifs_dates
from vfc.zarr_target import ZarrTarget
from vfc.io_nc import DataFileError
from vfc.metrics import lat_weights, rmse_by_lead, crps_by_lead, \
    brier_by_lead, AnomalyCorrelationAccumulator, ActivityAccumulator
from vfc.metrics.spectrum import zonal_spectrum, wavenumber_axis,\
    wavelength_km
from vfc.specs import get_spec, VAR_SPECS

_METRICS = ("rmse", "crps", "brier", "acc", "fa", "spectrum")
_EPOCH = _dt.datetime(1970, 1, 1)

# 比湿类要素（vfc/specs 中目标单位为 kg kg-1）。部分模式输出按 g/kg 存储
# （如盘古/FuXi/AIFS 的 q700/q2m），而 ERA5 zarr 真值为 kg/kg → 用
# --target-q-scale / --pred-q-scale 统一单位后再算指标（默认 1.0 = 不缩放）。
_HUMIDITY_VARS = tuple(v for v, s in VAR_SPECS.items() if s.unit == "kg kg-1")


def _ts():
    """当前时间标签 [YYYY-MM-DD HH:MM:SS]。"""
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_init(s, fallback_date=None):
    """解析起报时刻：'2025-01-06' / '2025-01-06T00:00' / '20250106' 等。"""
    if s is None:
        s = fallback_date
    if s is None:
        return None
    s = str(s).strip().replace("/", "-")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2})(?::(\d{2}))?)?", s)
    if not m:
        m = re.search(r"(\d{4})(\d{2})(\d{2})", s)
    if not m:
        raise ValueError("无法解析起报时刻 %r（如 2025-01-06T00:00）" % s)
    if len(m.groups()) == 3:
        y, mo, d = (int(x) for x in m.groups())
        return _dt.datetime(y, mo, d)
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh = int(m.group(4) or 0)
    mm = int(m.group(5) or 0)
    return _dt.datetime(y, mo, d, hh, mm)


def _target_indices(target, init, lead_hours):
    """目标 time 轴上与 init+lead 对应的位置。"""
    need = [init + _dt.timedelta(hours=float(h)) for h in lead_hours]
    th = target.time_hours
    nh = np.array([(d - _EPOCH).total_seconds() / 3600.0 for d in need])
    idx = np.searchsorted(th, nh)
    ok = (idx < th.size) & np.isclose(th[np.clip(idx, 0, th.size - 1)], nh,
                                       rtol=0, atol=1e-9)
    if not ok.all():
        bad = [need[k] for k in np.where(~ok)[0][:5]]
        raise DataFileError(
            "目标 %s 缺少验证时次 %s（目标 time 范围 %s .. %s）"
            % (getattr(target, "path", "?"), bad, target.time[0],
               target.time[-1]))
    return idx


def _align_info(p_lat, p_lon, t_lat, t_lon):
    """预报为基准：返回 (lat_flip, lon_roll)。lon 按 mod 360 环形匹配。"""
    if np.array_equal(p_lat, t_lat):
        lat_flip = False
    elif np.array_equal(p_lat, t_lat[::-1]):
        lat_flip = True
    else:
        raise DataFileError(
            "预报与目标 lat 网格不一致：\n  预报 %s..%s/%d\n  目标 %s..%s/%d"
            % (p_lat[0], p_lat[-1], p_lat.size, t_lat[0], t_lat[-1], t_lat.size))
    if np.array_equal(p_lon, t_lon):
        lon_roll = 0
    else:
        pl = np.mod(p_lon, 360.0)
        tl = np.mod(t_lon, 360.0)
        lon_roll = None
        for k in range(t_lon.size):
            if np.allclose(np.roll(tl, k), pl, rtol=0, atol=1e-6):
                lon_roll = k
                break
        if lon_roll is None:
            raise DataFileError(
                "预报与目标 lon 网格不一致（找不到环形平移）：\n  预报 %s..%s/%d\n"
                "  目标 %s..%s/%d"
                % (p_lon[0], p_lon[-1], p_lon.size, t_lon[0], t_lon[-1],
                   t_lon.size))
    return lat_flip, lon_roll


def _apply_target_align(a, lat_flip, lon_roll):
    if lat_flip:
        a = a[:, ::-1, :]
    if lon_roll:
        a = np.roll(a, lon_roll, axis=-1)
    return a


def _bbox_indices(lat, lon, bbox):
    la0, la1, lo0, lo1 = bbox
    lon = np.mod(np.asarray(lon), 360.0)
    ilat = np.where((lat >= min(la0, la1)) & (lat <= max(la0, la1)))[0]
    ilon = np.where((lon >= min(lo0, lo1)) & (lon <= max(lo0, lo1)))[0]
    if ilat.size == 0 or ilon.size == 0:
        raise ValueError("bbox %s 内没有任何格点" % (bbox,))
    return ilat, ilon


def _run_accum(acc, n_lead, p, o, block):
    for i0 in range(0, n_lead, block):
        acc.update(i0, p[i0:i0 + block], o[i0:i0 + block])
    return acc.finalize()


def _spread_by_lead(p, lat, weighted):
    """集合离散度：逐格点 member 间 std → 纬度加权平均，随时效。"""
    w = lat_weights(lat, weighted)[:, None]
    s = p.std(axis=0)                              # (T, lat, lon)
    ok = np.isfinite(s)
    sw = (ok.astype("f8") * w).sum(axis=(1, 2))
    return np.sqrt(np.where(ok, s * s, 0.0).sum(axis=(1, 2)) / np.maximum(sw, 1e-12))


class CompositeTarget(object):
    """多个 ZarrTarget 合并视图：按要素名自动归属；通道名支持下划线别名。

    别名规则：目标通道名里的 '_' 剥离后参与匹配（如 z_500 ↔ z500、u_200 ↔
    u200），精确名优先。matchable 给出预报侧可匹配的要素名。
    """

    def __init__(self, targets):
        self.targets = list(targets)
        if not self.targets:
            raise ValueError("至少需要一个目标 zarr")
        self._owner = {}          # 精确通道名 -> 目标
        self._alias = {}          # 剥离下划线名 -> (目标, 实际通道名)
        for t in self.targets:
            for c in t.levels:
                self._owner.setdefault(c, t)
                self._alias.setdefault(c.replace("_", ""), (t, c))
        self.levels = sorted(self._owner)          # 实际通道名
        self.matchable = sorted(self._alias)       # 预报侧可匹配名
        self.lat = self.targets[0].lat
        self.lon = self.targets[0].lon
        self.time = self.targets[0].time
        self.time_hours = self.targets[0].time_hours
        self.lon_fixed = False
        self.engine = "+".join(t.engine for t in self.targets)

    def owner(self, varname):
        """返回 (目标, 实际通道名)；找不到抛错。"""
        v = varname.lower()
        if v in self._owner:
            return self._owner[v], v
        norm = v.replace("_", "")
        if norm in self._alias:
            return self._alias[norm]
        raise DataFileError("目标 zarr 中没有要素 %r（可匹配: %s）"
                            % (varname, self.matchable))

    def read(self, varname, time_idx=None):
        t, c = self.owner(varname)
        return t.read(c, time_idx=time_idx)

    def read_multi(self, names, time_idx=None):
        """一次读多个要素：同属一个目标且支持 read_multi 时一次取。"""
        names = [str(n).lower() for n in names]
        groups = {}
        for n in names:
            t, c = self.owner(n)
            groups.setdefault(id(t), []).append((n, c, t))
        out = {}
        for g in groups.values():
            t = g[0][2]
            acts = [x[1] for x in g]
            if hasattr(t, "read_multi"):
                mm = t.read_multi(acts, time_idx=time_idx)
                for n, c, _ in g:
                    out[n] = mm[c]
            else:
                for n, c, _ in g:
                    out[n] = t.read(c, time_idx=time_idx)
        return out
# ---------------------------------------------------------------- 主流程

def run_ensemble(pred_dir, target_zarr, init=None, members=None,
                 first_lead=6.0, lead_step=6.0, variables=None,
                 metrics=("rmse",), var_metrics=None,
                 outdir="results/ens", name=None,
                 engine="auto", lat_weighted=True, bbox=None, block=16,
                 plot=True, target_denorm=False, target_tp_exp=False,
                 target_tp_scale=1000.0, pred_tp_scale=1.0,
                 target_q_scale=1.0, pred_q_scale=1.0,
                 target_time_units=None, tp_grid=False,
                 brier_thresholds=None,
                 climo_path=None, climo_obj=None,
                 climo_source="未标注", climo_window=15,
                 acc_centered=False, spec_lead_range=None, verbose=True):
    """主流程：返回 (results, meta)。target_zarr 可为路径列表或目标对象列表。"""
    metrics = [m.lower() for m in metrics]
    bad = [m for m in metrics if m not in _METRICS]
    if bad:
        raise ValueError("未知指标 %s，可选 %s" % (bad, _METRICS))
    if var_metrics:
        var_metrics = {str(k).lower(): {str(x).lower() for x in vals}
                       for k, vals in var_metrics.items()}
        for _v0, _ms0 in var_metrics.items():
            _bad0 = [m for m in _ms0 if m not in _METRICS]
            if _bad0:
                raise ValueError("--var-metrics %s 含未知指标 %s"
                                 % (_v0, sorted(_bad0)))

    def _vmet(v):
        """该要素实际要算的指标 = --metrics 与 --var-metrics 交集。"""
        if not var_metrics or v not in var_metrics:
            return tuple(metrics)
        return tuple(m for m in metrics if m in var_metrics[v])

    need_climo = bool({"acc", "fa"} & set(metrics))
    brier_thresholds = dict(brier_thresholds or {})
    if "brier" in metrics and not brier_thresholds:
        raise ValueError(
            "brier 需要事件阈值：请用 --brier-thresholds 指定，如 "
            "'z500:100' 'tp:0.1,10,25' 'msl:1000'（事件 = 场值 ≥ 阈值）")

    if hasattr(pred_dir, "read"):                 # 已构造好的预报对象（如 PanguForecast）
        pred = pred_dir
    else:
        pred = EnsembleForecast(pred_dir, members=members,
                                first_lead=first_lead, lead_step=lead_step,
                                engine=engine)
    pred_disp = getattr(pred, "root", str(pred_dir))

    if isinstance(target_zarr, (list, tuple)):
        paths = list(target_zarr)
    else:
        paths = [target_zarr]
    tgts = []
    for p in paths:
        if hasattr(p, "read"):
            tgts.append(p)
        else:
            tgts.append(ZarrTarget(p, denorm=target_denorm,
                                   tp_exp=target_tp_exp,
                                   time_units=target_time_units))
    target = CompositeTarget(tgts)

    if init is None and getattr(pred, "init_date", None) is not None:
        init = pred.init_date
    else:
        init = _parse_init(init, fallback_date=os.path.basename(
            os.path.normpath(pred_disp)))
    if init is None:
        raise ValueError("无法确定起报时刻，请用 --init 指定（如 2025-01-06T00:00）")

    leads = pred.lead_hours
    n_lead = pred.n_leads

    ilat = ilon = None
    if bbox is not None:
        ilat, ilon = _bbox_indices(pred.lat, pred.lon, bbox)
        lat_sub = pred.lat[ilat]
    else:
        lat_sub = pred.lat

    # ---- 要素交集（预报 ∩ 全部目标；含下划线别名） ----
    common = [v for v in pred.levels if v in target.matchable]
    _explicit_vars = variables is not None
    variables = list(variables) if _explicit_vars else common
    def _derivable(v):
        sp = VAR_SPECS.get(v)
        return bool(sp and sp.derived
                    and all(c in common for c in sp.derived))
    missing = [v for v in variables
               if v not in common and not _derivable(v)]
    if missing and verbose:
        print("[%s][vfc-ens] 提示: 以下要素在目标 zarr 中不存在，跳过: %s" % (_ts(), missing))
    variables = [v for v in variables if v in common or _derivable(v)]
    if not variables:
        raise ValueError(
            "预报与目标没有公共要素（预报 %s vs 目标 %s）——请检查 --vars / zarr 通道"
            % (pred.levels, target.levels))
    # 派生要素（风速 ws = sqrt(u^2+v^2)）：u/v 分量都在公共要素里时补进清单
    for _dv, _s in VAR_SPECS.items():
        if not _s.derived:
            continue
        if all(c in common for c in _s.derived):
            if (not _explicit_vars) or (_dv in variables):
                if _dv not in variables:
                    variables.append(_dv)
    # tp 的正式检验为站点 TS/BIAS（run_ts.py，24h 累积）；默认不进网格指标
    if "tp" in variables and not tp_grid:
        variables.remove("tp")
        if not variables:
            raise ValueError("唯一可评要素是 tp，而 tp 默认走站点 TS/BIAS"
                             "（run_ts.py）；如需网格过渡指标请加 --tp-grid")
        if verbose:
            print("[%s][vfc-ens] 提示: tp 已从网格指标排除（正式评分为站点 TS/BIAS，"
                  "24h 累积；过渡性网格 RMSE 可用 --tp-grid）" % _ts())

    # ---- 气候态（acc/fa） ----
    climo = None
    climo_owned = False
    if need_climo:
        from vfc.climo import DailyClimatology
        if climo_obj is not None:
            climo = climo_obj              # 批量共享：只读一次，避免每点重读 6GB
        else:
            if not climo_path:
                raise ValueError(
                    "acc/fa 需要逐日气候态：请用 --climo 指定（CRA40/ERA5 逐日 0.25° "
                    "网格，格式见 vfc/climo.py）。当前 zarr 只有 18 个月，当不了气候态")
            climo = DailyClimatology(climo_path, engine=engine,
                                     window=climo_window, fix_lon=True,
                                     source=climo_source)
            climo_owned = True
        # 气候态网格允许与预报升降序相反 / lon 差半圈（自动对齐）
        cl_lat_flip, cl_lon_roll = _align_info(pred.lat, pred.lon,
                                               climo.lat, climo.lon)
        def _climo_ok(v):
            if climo.has(v):
                return True
            cc = get_spec(v).derived
            return bool(cc and all(climo.has(x) for x in cc))
        lack = [v for v in variables if not _climo_ok(v)]
        if lack:
            if len(lack) == len(variables):
                raise ValueError("气候态 %s 一个要素都没有（需要 %s，现有 %s）"
                                 % (climo_path, variables, climo.levels))
            if verbose:
                print("[%s][vfc-ens] 提示: 气候态无要素 %s，这些要素跳过 acc/fa"
                      "（气候态有 %s）" % (_ts(), lack, climo.levels))

    name = name or os.path.basename(os.path.normpath(pred_disp))
    os.makedirs(outdir, exist_ok=True)

    member_names = [os.path.basename(m) for m in pred.members]
    _is_ensemble = len(member_names) > 1        # 单确定性(det)：不输出 ensmean/spread
    # VFC_ENS_ENSMEAN_ONLY=1：ensemble 只算/只写 ensmean 的 rmse/acc/fa/spectrum
    # （CRPS/spread/spread÷RMSE 本身需要全部成员，照常）；逐 member 表不再输出。
    _ensonly = (_is_ensemble
                and os.environ.get("VFC_ENS_ENSMEAN_ONLY", "").strip().lower()
                in ("1", "true", "yes", "on"))
    rmse_tables = {m: {} for m in member_names}
    rmse_tables["ensmean"] = {}
    crps_tables = {}
    brier_tables = {}
    spread_tables = {}
    acc_tables = {m: {} for m in member_names}
    acc_tables["ensmean"] = {}
    fa_tables = {m: {} for m in member_names}
    fa_tables["ensmean"] = {}
    spectra = {}
    align_meta = {}

    _use_block = (len(member_names) > 1 and getattr(pred, "read_block", None)
                  is not None and not ({"acc", "fa", "spectrum"} & set(metrics))
                  and not any(get_spec(v).derived for v in variables)
                  and os.environ.get("VFC_BLOCK_MULTIVAR", "").strip().lower()
                  in ("1", "true", "yes", "on"))
    if _use_block:
        _vars = list(variables)
        _blk = max(1, int(os.environ.get("VFC_ENS_BLOCK", "1")))   # 多变量同时驻留，块小一点更省内存
        _lidx = pd.Index(leads, name="lead_h")
        _ofull = {}
        for v in _vars:
            owner, actual = target.owner(v)
            lat_flip, lon_roll = _align_info(pred.lat, pred.lon, owner.lat,
                                             owner.lon)
            align_meta[getattr(owner, "path", "memory")] = {
                "lat_flip": bool(lat_flip), "lon_roll": int(lon_roll)}
            t_idx = _target_indices(owner, init, leads)
            o = _apply_target_align(owner.read(actual, time_idx=t_idx),
                                    lat_flip, lon_roll)
            if v == "tp" and target_tp_scale != 1.0:
                o = o * float(target_tp_scale)
            if v in _HUMIDITY_VARS and target_q_scale != 1.0:
                o = o * float(target_q_scale)
            if ilat is not None:
                o = o[:, ilat][:, :, ilon]
            _ofull[v] = o
            if verbose:
                print("\n[%s][vfc-ens] 要素 %s: 块内多变量读取(%d member × %d 步)…"
                      % (_ts(), v, len(member_names), n_lead))
        _rm = {v: {m: np.empty(n_lead) for m in member_names} for v in _vars}
        _rmE = {v: np.empty(n_lead) for v in _vars}
        _sp = {v: np.empty(n_lead) for v in _vars}
        _cr = {v: (np.empty(n_lead) if "crps" in metrics else None)
               for v in _vars}
        _br = {}
        if "brier" in metrics:
            for v in _vars:
                for thr in brier_thresholds.get(v, []):
                    _br["%s@%g" % (v, thr)] = np.empty(n_lead)
        for i0 in range(0, n_lead, _blk):
            j1 = min(i0 + _blk, n_lead)
            pv = pred.read_block(_vars, list(range(i0, j1)))
            for v in _vars:
                pc = pv[v]
                if v == "tp" and pred_tp_scale != 1.0:
                    pc = pc * float(pred_tp_scale)
                if v in _HUMIDITY_VARS and pred_q_scale != 1.0:
                    pc = pc * float(pred_q_scale)
                if ilat is not None:
                    pc = pc[:, :, ilat][:, :, :, ilon]
                oc = _ofull[v][i0:j1]
                if "rmse" in metrics:
                    if not _ensonly:
                        for _mi, _m in enumerate(member_names):
                            _rm[v][_m][i0:j1] = rmse_by_lead(
                                pc[_mi], oc, lat_sub, lat_weighted)
                    _rmE[v][i0:j1] = rmse_by_lead(
                        pc.mean(axis=0), oc, lat_sub, lat_weighted)
                _sp[v][i0:j1] = _spread_by_lead(pc, lat_sub, lat_weighted)
                if _cr[v] is not None:
                    _cr[v][i0:j1] = crps_by_lead(pc, oc, lat_sub, lat_weighted)
                if "brier" in metrics:
                    for thr in brier_thresholds.get(v, []):
                        key = "%s@%g" % (v, thr)
                        _br[key][i0:j1] = brier_by_lead(
                            pc, oc, lat_sub, float(thr), lat_weighted)
        for v in _vars:
            if "rmse" in metrics:
                if not _ensonly:
                    for _m in member_names:
                        rmse_tables[_m][v] = pd.Series(_rm[v][_m], index=_lidx,
                                                       name=v)
                rmse_tables["ensmean"][v] = pd.Series(_rmE[v], index=_lidx,
                                                      name=v)
            spread_tables[v] = pd.Series(_sp[v], index=_lidx, name=v)
            if _cr[v] is not None:
                crps_tables[v] = pd.Series(_cr[v], index=_lidx, name=v)
        for _key, _arr in _br.items():
            brier_tables[_key] = pd.Series(_arr, index=_lidx, name=_key)
        variables = []          # skip generic per-variable loop below
    # 合并派生风速组：wsX 与其 u/v 分量同评、u/v 只要 rmse 时，u/v 在 wsX 流式读取里一次读
    # （VFC_ENS_WIND_GROUP=1 默认开；0 关闭=各自独立 pass）
    _wind_skip = set()
    _wind_uv = None
    if (len(member_names) > 1
            and getattr(pred, "read_block", None) is not None
            and os.environ.get("VFC_ENS_STREAM_DERIVED", "1").strip().lower()
            in ("1", "true", "yes", "on")
            and os.environ.get("VFC_ENS_WIND_GROUP", "1").strip().lower()
            in ("1", "true", "yes", "on")):
        _vset = set(variables)
        for _dv in variables:
            _sp2 = get_spec(_dv)
            if not (_sp2.derived and len(_sp2.derived) == 2):
                continue
            _a2, _b2 = _sp2.derived
            if _a2 not in _vset or _b2 not in _vset:
                continue
            _gm2 = {x: set(_vmet(x)) for x in (_a2, _b2)}
            if (_gm2[_a2] - {"rmse"}) or (_gm2[_b2] - {"rmse"}):
                continue
            _wind_uv = (_dv, _a2, _b2)
            _wind_skip = {_a2, _b2}
            break
    for v in variables:
        if v in _wind_skip:
            continue
        comps = get_spec(v).derived
        if comps:
            if verbose:
                print("\n[%s][vfc-ens] 要素 %s（派生风速 = √(u²+v²)，%s）："
                      "读取预报(%d member × %d 步)…"
                      % (_ts(), v, "+".join(comps), len(member_names), n_lead))
        else:
            owner, actual = target.owner(v)
            lat_flip, lon_roll = _align_info(pred.lat, pred.lon, owner.lat,
                                             owner.lon)
            align_meta[getattr(owner, "path", "memory")] = {
                "lat_flip": bool(lat_flip), "lon_roll": int(lon_roll)}
            t_idx = _target_indices(owner, init, leads)
            if verbose:
                print("\n[%s][vfc-ens] 要素 %s（目标通道 %s，目标 %s）："
                      "读取预报(%d member × %d 步)…"
                      % (_ts(), v, actual, getattr(owner, "path", "memory"),
                         len(member_names), n_lead))
        _stream_derived = (not comps) or os.environ.get(
            "VFC_ENS_STREAM_DERIVED", "1").strip().lower() in ("1", "true", "yes", "on")
        # single(det) 默认也走块流式（省内存；VFC_SINGLE_STREAM=0 关闭=旧整读）
        _single_stream = (len(member_names) > 1
                          or os.environ.get("VFC_SINGLE_STREAM", "1").strip().lower()
                          in ("1", "true", "yes", "on"))
        if _single_stream and getattr(pred, "read_leads", None) is not None \
                and _stream_derived:
            # ---- multi-member streaming branch: read pred by lead-blocks ----
            #      （派生要素 ws200 等也按 lead 分块读 u/v，避免整集合全量驻留内存）
            a = b = None
            _obs_block = os.environ.get("VFC_OBS_BLOCK", "1").strip().lower() \
                in ("1", "true", "yes", "on")
            _uo = _vo = o = None
            _ua = _va = None
            if comps:
                a, b = comps
                _own0, _act0 = target.owner(a)
                _fl0, _ro0 = _align_info(pred.lat, pred.lon,
                                         _own0.lat, _own0.lon)
                _ti0 = _target_indices(_own0, init, leads)
                align_meta.setdefault(
                    getattr(_own0, "path", "memory"),
                    {"lat_flip": bool(_fl0), "lon_roll": int(_ro0)})
                if not _obs_block:
                    if hasattr(target, "read_multi"):
                        _mm = target.read_multi([a, b], time_idx=_ti0)
                        _uo = _apply_target_align(_mm[a], _fl0, _ro0)
                        _vo = _apply_target_align(_mm[b], _fl0, _ro0)
                    else:
                        _uo = _apply_target_align(
                            _own0.read(_act0, time_idx=_ti0), _fl0, _ro0)
                        _vo = _apply_target_align(
                            target.read(b, time_idx=_ti0), _fl0, _ro0)
                    o = np.sqrt(_uo * _uo + _vo * _vo)      # (T, lat, lon)
                    if ilat is not None:
                        o = o[:, ilat][:, :, ilon]
                    if _wind_uv is not None and v == _wind_uv[0]:
                        _ua = _uo[:, ilat][:, :, ilon] if ilat is not None else _uo
                        _va = _vo[:, ilat][:, :, ilon] if ilat is not None else _vo
            else:
                if not _obs_block:
                    o = _apply_target_align(owner.read(actual, time_idx=t_idx),
                                            lat_flip, lon_roll)  # (T, lat, lon)
                    if v == "tp" and target_tp_scale != 1.0:
                        o = o * float(target_tp_scale)          # target m -> mm
                    if v in _HUMIDITY_VARS and target_q_scale != 1.0:
                        o = o * float(target_q_scale)
                    if ilat is not None:
                        o = o[:, ilat][:, :, ilon]
            blk = max(1, int(os.environ.get("VFC_ENS_BLOCK", "4")))   # lead block size (memory knob)
            _lidx = pd.Index(leads, name="lead_h")
            _rm = {m: np.empty(n_lead) for m in member_names}
            _rm["ensmean"] = np.empty(n_lead)
            _sp = np.empty(n_lead)
            _cr = np.empty(n_lead) if "crps" in _vmet(v) else None
            _merge_uv = _wind_uv is not None and v == _wind_uv[0]
            _rmA = _rmB = _spA = _spB = None
            if _merge_uv:
                _rmA = {m: np.empty(n_lead) for m in member_names}
                _rmA["ensmean"] = np.empty(n_lead)
                _rmB = {m: np.empty(n_lead) for m in member_names}
                _rmB["ensmean"] = np.empty(n_lead)
                _spA = np.empty(n_lead)
                _spB = np.empty(n_lead)
            _br = {}
            if "brier" in _vmet(v) and v in brier_thresholds:
                for _thr in brier_thresholds[v]:
                    _br["%s@%g" % (v, _thr)] = np.empty(n_lead)
            _need_acc = "acc" in _vmet(v)
            _need_fa = "fa" in _vmet(v)
            # 气候态按 lead 块读取（acc/fa）：避免每变量整段 60 步驻留
            _accfa = ((_need_acc or _need_fa) and climo is not None
                      and (climo.has(v)
                           or (comps and all(climo.has(x) for x in comps))))
            _acc = {}
            _fa = {}
            if _need_acc:
                if not _ensonly:
                    for _m in member_names:
                        _acc[_m] = AnomalyCorrelationAccumulator(
                            n_lead, lat_sub, weighted=lat_weighted,
                            centered=acc_centered)
                _acc["ensmean"] = AnomalyCorrelationAccumulator(
                    n_lead, lat_sub, weighted=lat_weighted,
                    centered=acc_centered)
            if _need_fa:
                if not _ensonly:
                    for _m in member_names:
                        _fa[_m] = ActivityAccumulator(n_lead, lat_sub,
                                                      weighted=lat_weighted)
                _fa["ensmean"] = ActivityAccumulator(n_lead, lat_sub,
                                                     weighted=lat_weighted)
            _winsum = {}
            _wincnt = 0
            if "spectrum" in _vmet(v) and v != "tp":
                _mwin = np.ones(n_lead, dtype=bool)
                if spec_lead_range:
                    _mwin &= ((leads >= spec_lead_range[0])
                              & (leads <= spec_lead_range[1]))
                if not _mwin.any():
                    raise ValueError("lead-window %s empty for spectrum"
                                     % (spec_lead_range,))
                _kax = wavenumber_axis(pred.lon.size)
                if not _ensonly:
                    for _m in member_names:
                        _winsum[_m + "_pred"] = np.zeros(_kax.size)
                _winsum["ensmean_pred"] = np.zeros(_kax.size)
                _winsum["obs"] = np.zeros(_kax.size)
            for i0 in range(0, n_lead, blk):
                j1 = min(i0 + blk, n_lead)
                _idx = list(range(i0, j1))
                if comps:
                    _rb = getattr(pred, "read_block", None)
                    if _rb is not None:
                        _mv = _rb([a, b], _idx)
                        _pa = _mv[a]
                        _pb = _mv[b]
                        pc = np.sqrt(_pa * _pa + _pb * _pb)
                    else:
                        _pa = pred.read_leads(a, _idx)
                        _pb = pred.read_leads(b, _idx)
                        pc = np.sqrt(_pa * _pa + _pb * _pb)
                else:
                    pc = pred.read_leads(v, _idx)   # (n, b, lat, lon)
                if v == "tp" and pred_tp_scale != 1.0:
                    pc = pc * float(pred_tp_scale)
                if v in _HUMIDITY_VARS and pred_q_scale != 1.0:
                    pc = pc * float(pred_q_scale)
                if ilat is not None:
                    pc = pc[:, :, ilat][:, :, :, ilon]
                _oa = _ob = None
                if _obs_block:
                    _ti = (_ti0[i0:j1] if comps else t_idx[i0:j1])
                    if comps:
                        if hasattr(target, "read_multi"):
                            _mm = target.read_multi([a, b], time_idx=_ti)
                            _uob = _apply_target_align(_mm[a], _fl0, _ro0)
                            _vob = _apply_target_align(_mm[b], _fl0, _ro0)
                        else:
                            _uob = _apply_target_align(
                                _own0.read(_act0, time_idx=_ti), _fl0, _ro0)
                            _vob = _apply_target_align(
                                target.read(b, time_idx=_ti), _fl0, _ro0)
                        if ilat is not None:
                            _uob = _uob[:, ilat][:, :, ilon]
                            _vob = _vob[:, ilat][:, :, ilon]
                        oc = np.sqrt(_uob * _uob + _vob * _vob)
                        if _merge_uv:
                            _oa = _uob
                            _ob = _vob
                    else:
                        oc = _apply_target_align(
                            owner.read(actual, time_idx=_ti),
                            lat_flip, lon_roll)
                        if v == "tp" and target_tp_scale != 1.0:
                            oc = oc * float(target_tp_scale)
                        if v in _HUMIDITY_VARS and target_q_scale != 1.0:
                            oc = oc * float(target_q_scale)
                        if ilat is not None:
                            oc = oc[:, ilat][:, :, ilon]
                else:
                    oc = o[i0:j1]
                    if _merge_uv:
                        _oa = _ua[i0:j1]
                        _ob = _va[i0:j1]
                if _merge_uv:
                    if ilat is not None:
                        _pa = _pa[:, :, ilat][:, :, :, ilon]
                        _pb = _pb[:, :, ilat][:, :, :, ilon]
                    if "rmse" in _vmet(_wind_uv[1]):
                        if not _ensonly:
                            for _i, _m in enumerate(member_names):
                                _rmA[_m][i0:j1] = rmse_by_lead(
                                    _pa[_i], _oa, lat_sub, lat_weighted)
                                _rmB[_m][i0:j1] = rmse_by_lead(
                                    _pb[_i], _ob, lat_sub, lat_weighted)
                        _rmA["ensmean"][i0:j1] = rmse_by_lead(
                            _pa.mean(axis=0), _oa, lat_sub, lat_weighted)
                        _rmB["ensmean"][i0:j1] = rmse_by_lead(
                            _pb.mean(axis=0), _ob, lat_sub, lat_weighted)
                    _spA[i0:j1] = _spread_by_lead(_pa, lat_sub, lat_weighted)
                    _spB[i0:j1] = _spread_by_lead(_pb, lat_sub, lat_weighted)
                if "rmse" in metrics:
                    if not _ensonly:
                        for _i, _m in enumerate(member_names):
                            _rm[_m][i0:j1] = rmse_by_lead(pc[_i], oc, lat_sub,
                                                          lat_weighted)
                    _rm["ensmean"][i0:j1] = rmse_by_lead(
                        pc.mean(axis=0), oc, lat_sub, lat_weighted)
                _sp[i0:j1] = _spread_by_lead(pc, lat_sub, lat_weighted)
                if _cr is not None:
                    _cr[i0:j1] = crps_by_lead(pc, oc, lat_sub, lat_weighted)
                for _key in _br:
                    _thr = float(_key.split("@")[1])
                    _br[_key][i0:j1] = brier_by_lead(
                        pc, oc, lat_sub, _thr, lat_weighted)
                if _accfa:
                    _lb = [float(h) for h in leads[i0:j1]]
                    if climo.has(v):
                        _cfb = _apply_target_align(
                            climo.field_for(v, _lb, init),
                            cl_lat_flip, cl_lon_roll)
                    else:
                        _cfa = None
                        for _x in comps:
                            _xi = _apply_target_align(
                                climo.field_for(_x, _lb, init),
                                cl_lat_flip, cl_lon_roll)
                            _cfa = _xi if _cfa is None else \
                                np.sqrt(_cfa * _cfa + _xi * _xi)
                        _cfb = _cfa
                    _oa = oc - _cfb
                    if _need_acc:
                        if not _ensonly:
                            for _i, _m in enumerate(member_names):
                                _acc[_m].update(i0, pc[_i] - _cfb, _oa)
                        _acc["ensmean"].update(i0, pc.mean(axis=0) - _cfb,
                                               _oa)
                    if _need_fa:
                        if not _ensonly:
                            for _i, _m in enumerate(member_names):
                                _fa[_m].update(i0, pc[_i] - _cfb, _oa)
                        _fa["ensmean"].update(i0, pc.mean(axis=0) - _cfb,
                                              _oa)
                if _winsum:
                    _seg = _mwin[i0:j1]
                    if _seg.any():
                        pcw = pc[:, _seg]
                        if not _ensonly:
                            for _i, _m in enumerate(member_names):
                                _winsum[_m + "_pred"] += zonal_spectrum(
                                    pcw[_i], lat_sub,
                                    weighted=lat_weighted).sum(axis=0)
                        _winsum["ensmean_pred"] += zonal_spectrum(
                            pcw.mean(axis=0), lat_sub,
                            weighted=lat_weighted).sum(axis=0)
                        _winsum["obs"] += zonal_spectrum(
                            oc[_seg], lat_sub,
                            weighted=lat_weighted).sum(axis=0)
                        _wincnt += int(_seg.sum())
            if _merge_uv:
                _an, _bn = _wind_uv[1], _wind_uv[2]
                if "rmse" in _vmet(_an):
                    if not _ensonly:
                        for _m in member_names:
                            rmse_tables[_m][_an] = pd.Series(
                                _rmA[_m], index=_lidx, name=_an)
                            rmse_tables[_m][_bn] = pd.Series(
                                _rmB[_m], index=_lidx, name=_bn)
                    rmse_tables["ensmean"][_an] = pd.Series(
                        _rmA["ensmean"], index=_lidx, name=_an)
                    rmse_tables["ensmean"][_bn] = pd.Series(
                        _rmB["ensmean"], index=_lidx, name=_bn)
                spread_tables[_an] = pd.Series(_spA, index=_lidx, name=_an)
                spread_tables[_bn] = pd.Series(_spB, index=_lidx, name=_bn)
            if "rmse" in metrics:
                if not _ensonly:
                    for _m in member_names:
                        rmse_tables[_m][v] = pd.Series(_rm[_m], index=_lidx,
                                                       name=v)
                rmse_tables["ensmean"][v] = pd.Series(_rm["ensmean"],
                                                      index=_lidx, name=v)
            spread_tables[v] = pd.Series(_sp, index=_lidx, name=v)
            if _cr is not None:
                crps_tables[v] = pd.Series(_cr, index=_lidx, name=v)
            for _key, _arr in _br.items():
                brier_tables[_key] = pd.Series(_arr, index=_lidx, name=_key)
            if _accfa:
                if _need_acc:
                    if not _ensonly:
                        for _m in member_names:
                            acc_tables[_m][v] = pd.Series(
                                _acc[_m].finalize(), index=_lidx, name=v)
                    acc_tables["ensmean"][v] = pd.Series(
                        _acc["ensmean"].finalize(), index=_lidx, name=v)
                if _need_fa:
                    if not _ensonly:
                        for _m in member_names:
                            _pf, _of = _fa[_m].finalize()
                            fa_tables[_m][v + "_pred"] = pd.Series(_pf,
                                                                   index=leads)
                            fa_tables[_m][v + "_obs"] = pd.Series(_of,
                                                                  index=leads)
                            fa_tables[_m][v + "_bias"] = pd.Series(_pf - _of,
                                                                   index=leads)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                fa_tables[_m][v + "_ratio"] = pd.Series(
                                    _pf / _of, index=leads)
                    _pe, _oe = _fa["ensmean"].finalize()
                    fa_tables["ensmean"][v + "_pred"] = pd.Series(_pe,
                                                                  index=leads)
                    fa_tables["ensmean"][v + "_obs"] = pd.Series(_oe,
                                                                 index=leads)
                    fa_tables["ensmean"][v + "_bias"] = pd.Series(
                        _pe - _oe, index=leads)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fa_tables["ensmean"][v + "_ratio"] = pd.Series(
                            _pe / _oe, index=leads)
            elif _need_acc or _need_fa:
                print("[%s][vfc-ens] hint: no climo for %s, skip acc/fa"
                      % (_ts(), v))
            if _winsum:
                _cols = {}
                if not _ensonly:
                    for _m in member_names:
                        _cols[_m + "_pred"] = _winsum[_m + "_pred"] / _wincnt
                        _cols[_m + "_obs"] = _winsum["obs"] / _wincnt
                _cols["ensmean_pred"] = _winsum["ensmean_pred"] / _wincnt
                _cols["ensmean_obs"] = _winsum["obs"] / _wincnt
                _d = pd.DataFrame(_cols, index=pd.Index(_kax,
                                                        name="wavenumber"))
                _d.insert(0, "wavelength_km", wavelength_km(_kax))
                spectra[v] = _d.iloc[1:]           # k=0 already zeroed
        else:
            if comps:
                a, b = comps
                # 真值：同目标一次取 u/v（read_multi），否则逐个读
                _own0, _act0 = target.owner(a)
                _fl0, _ro0 = _align_info(pred.lat, pred.lon,
                                         _own0.lat, _own0.lon)
                _ti0 = _target_indices(_own0, init, leads)
                align_meta.setdefault(
                    getattr(_own0, "path", "memory"),
                    {"lat_flip": bool(_fl0), "lon_roll": int(_ro0)})
                if hasattr(target, "read_multi"):
                    _mm = target.read_multi([a, b], time_idx=_ti0)
                    _uo = _apply_target_align(_mm[a], _fl0, _ro0)
                    _vo = _apply_target_align(_mm[b], _fl0, _ro0)
                else:
                    _uo = _apply_target_align(
                        _own0.read(_act0, time_idx=_ti0), _fl0, _ro0)
                    _vo = _apply_target_align(
                        target.read(b, time_idx=_ti0), _fl0, _ro0)
                o = np.sqrt(_uo * _uo + _vo * _vo)      # (T, lat, lon)
                # 预报：read_vars 一次取 u/v，否则逐个读
                if hasattr(pred, "read_vars"):
                    _mv = pred.read_vars([a, b])
                    _up = _mv[a]                       # (n_member, T, lat, lon)
                    _vp = _mv[b]
                else:
                    _up = pred.read(a)
                    _vp = pred.read(b)
                p = np.sqrt(_up * _up + _vp * _vp)
            else:
                p = pred.read(v)                     # (n_member, T, lat, lon)
                o = _apply_target_align(owner.read(actual, time_idx=t_idx),
                                        lat_flip, lon_roll)  # (T, lat, lon)
            if v == "tp" and target_tp_scale != 1.0:
                o = o * float(target_tp_scale)     # 真值 m → mm
            if v == "tp" and pred_tp_scale != 1.0:
                p = p * float(pred_tp_scale)     # 预报 m → mm（AIFS 输出）
            if v in _HUMIDITY_VARS and target_q_scale != 1.0:
                o = o * float(target_q_scale)     # 真值 kg/kg → g/kg（视统一方向）
            if v in _HUMIDITY_VARS and pred_q_scale != 1.0:
                p = p * float(pred_q_scale)     # 预报 g/kg → kg/kg（若×0.001）
            if ilat is not None:
                p = p[:, :, ilat][:, :, :, ilon]
                o = o[:, ilat][:, :, ilon]

            gmet = _vmet(v)
            # ---- RMSE + 集合离散度 ----
            if "rmse" in gmet:
                if not _ensonly:
                    for m_name, pa in zip(member_names, p):
                        rmse_tables[m_name][v] = pd.Series(
                            rmse_by_lead(pa, o, lat_sub, lat_weighted),
                            index=pd.Index(leads, name="lead_h"), name=v)
                rmse_tables["ensmean"][v] = pd.Series(
                    rmse_by_lead(p.mean(axis=0), o, lat_sub, lat_weighted),
                    index=pd.Index(leads, name="lead_h"), name=v)
            spread_tables[v] = pd.Series(
                _spread_by_lead(p, lat_sub, lat_weighted),
                index=pd.Index(leads, name="lead_h"), name=v)

            # ---- CRPS（集合整体评分；m=1 退化为 MAE） ----
            if "crps" in gmet:
                crps_tables[v] = pd.Series(
                    crps_by_lead(p, o, lat_sub, lat_weighted),
                    index=pd.Index(leads, name="lead_h"), name=v)

            # ---- Brier（事件 = 场值 ≥ 阈值；成员比例做概率） ----
            if "brier" in gmet and v in brier_thresholds:
                for thr in brier_thresholds[v]:
                    key = "%s@%g" % (v, thr)
                    brier_tables[key] = pd.Series(
                        brier_by_lead(p, o, lat_sub, float(thr), lat_weighted),
                        index=pd.Index(leads, name="lead_h"), name=key)

            # ---- ACC / FA（需气候态；ws850/ws10m/ws200 用 u/v 分量气候态合成） ----
            if "acc" in gmet or "fa" in gmet:
                _cf_good = False
                cf = None
                if climo is not None:
                    if climo.has(v):
                        cf = _apply_target_align(
                            climo.field_for(v, leads, init),
                            cl_lat_flip, cl_lon_roll)
                        _cf_good = True
                    else:
                        _cc = get_spec(v).derived
                        if _cc and all(climo.has(x) for x in _cc):
                            for _x in _cc:
                                _xi = _apply_target_align(
                                    climo.field_for(_x, leads, init),
                                    cl_lat_flip, cl_lon_roll)
                                cf = _xi if cf is None else \
                                    np.sqrt(cf * cf + _xi * _xi)
                            _cf_good = True
                if _cf_good:
                    o_an = o - cf
                    p_an = p - cf
                    if "acc" in gmet:
                        if not _ensonly:
                            for m_name, pa in zip(member_names, p_an):
                                acc = AnomalyCorrelationAccumulator(
                                    n_lead, lat_sub, weighted=lat_weighted,
                                    centered=acc_centered)
                                acc_tables[m_name][v] = pd.Series(
                                    _run_accum(acc, n_lead, pa, o_an, block),
                                    index=pd.Index(leads, name="lead_h"), name=v)
                        acc = AnomalyCorrelationAccumulator(
                            n_lead, lat_sub, weighted=lat_weighted,
                            centered=acc_centered)
                        acc_tables["ensmean"][v] = pd.Series(
                            _run_accum(acc, n_lead, p_an.mean(axis=0), o_an, block),
                            index=pd.Index(leads, name="lead_h"), name=v)
                    if "fa" in gmet:
                        if not _ensonly:
                            for m_name, pa in zip(member_names, p_an):
                                fa = ActivityAccumulator(n_lead, lat_sub,
                                                         weighted=lat_weighted)
                                f_f, f_o = _run_accum(fa, n_lead, pa, o_an,
                                                      block)
                                fa_tables[m_name][v + "_pred"] = pd.Series(
                                    f_f, index=leads)
                                fa_tables[m_name][v + "_obs"] = pd.Series(
                                    f_o, index=leads)
                                fa_tables[m_name][v + "_bias"] = pd.Series(
                                    f_f - f_o, index=leads)
                                with np.errstate(divide="ignore",
                                                 invalid="ignore"):
                                    fa_tables[m_name][v + "_ratio"] = pd.Series(
                                        f_f / f_o, index=leads)
                        fa = ActivityAccumulator(n_lead, lat_sub,
                                                 weighted=lat_weighted)
                        f_f, f_o = _run_accum(fa, n_lead, p_an.mean(axis=0), o_an,
                                              block)
                        fa_tables["ensmean"][v + "_pred"] = pd.Series(
                            f_f, index=leads)
                        fa_tables["ensmean"][v + "_obs"] = pd.Series(
                            f_o, index=leads)
                        fa_tables["ensmean"][v + "_bias"] = pd.Series(
                            f_f - f_o, index=leads)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            fa_tables["ensmean"][v + "_ratio"] = pd.Series(
                                f_f / f_o, index=leads)
                elif verbose:
                    print("[%s][vfc-ens] 提示: 气候态无 %s，跳过 acc/fa" % (_ts(), v))

            # ---- 功率谱（tp 跳过；时效窗平均） ----
            if "spectrum" in gmet and v != "tp":
                m = np.ones(n_lead, dtype=bool)
                if spec_lead_range:
                    m &= ((leads >= spec_lead_range[0])
                          & (leads <= spec_lead_range[1]))
                if not m.any():
                    raise ValueError("时效窗 %s 内没有谱计算时效" % (spec_lead_range,))
                k = wavenumber_axis(pred.lon.size)
                cols = {}
                if not _ensonly:
                    for m_name, pa in zip(member_names, p):
                        cols[m_name + "_pred"] = zonal_spectrum(
                            pa[m], lat_sub, weighted=lat_weighted).mean(axis=0)
                        cols[m_name + "_obs"] = zonal_spectrum(
                            o[m], lat_sub, weighted=lat_weighted).mean(axis=0)
                cols["ensmean_pred"] = zonal_spectrum(
                    p.mean(axis=0)[m], lat_sub, weighted=lat_weighted).mean(axis=0)
                cols["ensmean_obs"] = zonal_spectrum(
                    o[m], lat_sub, weighted=lat_weighted).mean(axis=0)
                d = pd.DataFrame(cols, index=pd.Index(k, name="wavenumber"))
                d.insert(0, "wavelength_km", wavelength_km(k))
                spectra[v] = d.iloc[1:]            # k=0 已置零，不输出

    if _use_block:
        variables = _vars

    # q700/q2m 统一按 g/kg 报告（内部 kg/kg；rmse/crps/spread/fa ×1000，谱 ×1e6）
    _QREP = 1000.0
    for _m in list(rmse_tables):
        for _v in list(rmse_tables[_m]):
            if _v in _HUMIDITY_VARS:
                rmse_tables[_m][_v] = rmse_tables[_m][_v] * _QREP
    for _v in list(crps_tables):
        if _v in _HUMIDITY_VARS:
            crps_tables[_v] = crps_tables[_v] * _QREP
    for _v in list(spread_tables):
        if _v in _HUMIDITY_VARS:
            spread_tables[_v] = spread_tables[_v] * _QREP
    for _m in list(fa_tables):
        for _k in list(fa_tables[_m]):
            _base = _k[:-5] if _k.endswith("_bias") else (
                _k[:-5] if _k.endswith("_pred") else (
                _k[:-4] if _k.endswith("_obs") else None))
            if _base in _HUMIDITY_VARS:
                fa_tables[_m][_k] = fa_tables[_m][_k] * _QREP
    for _v in list(spectra):
        if _v in _HUMIDITY_VARS:
            for _col in list(spectra[_v].columns):
                if _col != "wavelength_km":
                    spectra[_v][_col] = spectra[_v][_col] * 1e6

    if not _is_ensemble:
        spread_tables.clear()                  # 单确定性：无“集合离散度”

    # ---- 落盘 ----
    files, results = [], {}

    def _write(df, tag, suffix=""):
        csv = os.path.join(outdir, "%s_%s%s.csv" % (tag, name, suffix))
        df.to_csv(csv, float_format="%.6g")
        files.append(csv)
        return csv

    for m_name in member_names + (["ensmean"] if _is_ensemble else []):
        if rmse_tables[m_name]:
            df = pd.DataFrame(rmse_tables[m_name])
            df.index.name = "lead_h"
            _write(df, "rmse", "_" + m_name)
        if acc_tables[m_name]:
            df = pd.DataFrame(acc_tables[m_name])
            df.index.name = "lead_h"
            _write(df, "acc", "_" + m_name)
        if fa_tables[m_name]:
            df = pd.DataFrame(fa_tables[m_name])
            df.index.name = "lead_h"
            _write(df, "fa", "_" + m_name)
    if spread_tables:
        df = pd.DataFrame(spread_tables)
        df.index.name = "lead_h"
        _write(df, "spread", "")
    if spread_tables and rmse_tables.get("ensmean") and len(member_names) > 1:
        # 集合离散度-误差比：spread ÷ ensmean RMSE（逐 lead 逐变量，理想≈1）
        _spr = pd.DataFrame(spread_tables)
        _rms = pd.DataFrame(rmse_tables["ensmean"])
        with np.errstate(divide="ignore", invalid="ignore"):
            _ratio = _spr / _rms
        _ratio.index.name = "lead_h"
        _write(_ratio, "spread_rmse_ratio", "")
    if crps_tables:
        df = pd.DataFrame(crps_tables)
        df.index.name = "lead_h"
        _write(df, "crps", "")
    if brier_tables:
        df = pd.DataFrame(brier_tables)
        df.index.name = "lead_h"
        _write(df, "brier", "")
    if spectra:
        for v, d in spectra.items():
            _write(d, "spectrum", "_" + v)
    results["rmse"] = {k: pd.DataFrame(v) for k, v in rmse_tables.items() if v}
    results["acc"] = {k: pd.DataFrame(v) for k, v in acc_tables.items() if v}
    results["fa"] = {k: pd.DataFrame(v) for k, v in fa_tables.items() if v}
    results["spread"] = pd.DataFrame(spread_tables) if spread_tables else None
    results["crps"] = pd.DataFrame(crps_tables) if crps_tables else None
    results["brier"] = pd.DataFrame(brier_tables) if brier_tables else None
    results["spectrum"] = spectra

    if plot:
        _plot_rmse(results.get("rmse"), outdir, name, files)
        _plot_spectra(spectra, outdir, name, files)

    target_paths = ([getattr(t, "path", "memory") for t in
                     (target.targets if isinstance(target, CompositeTarget)
                      else [target])])
    meta = {
        "name": name,
        "pred_dir": pred_disp, "target_zarr": target_paths,
        "init": init.isoformat(),
        "members": member_names, "n_members": len(member_names),
        "n_leads": int(n_lead),
        "lead_hours": [float(x) for x in leads],
        "first_lead": float(first_lead), "lead_step": float(lead_step),
        "variables": variables,
        "metrics": metrics,
        "matched_vars": variables,
        "skipped_vars": missing,
        "grid": {"n_lat": int(pred.lat.size), "n_lon": int(pred.lon.size),
                 "lat": [float(pred.lat[0]), float(pred.lat[-1])],
                 "lon": [float(pred.lon[0]), float(pred.lon[-1])]},
        "target_alignment": align_meta,
        "tp": {"target_scale_to_mm": float(target_tp_scale),
               "pred_scale_to_mm": float(pred_tp_scale),
               "target_mode": "per-6h 时段量", "pred_mode": "per-6h 时段量",
               "differenced": False},
        "q": {"unit": "kg kg-1", "vars": list(_HUMIDITY_VARS),
              "target_scale_to_kgkg": float(target_q_scale),
              "pred_scale_to_kgkg": float(pred_q_scale)},
        "options": {"lat_weighted": bool(lat_weighted),
                    "bbox": list(bbox) if bbox else None,
                    "block": int(block),
                    "target_denorm": bool(target_denorm),
                    "target_tp_exp": bool(target_tp_exp),
                    "tp_grid": bool(tp_grid),
                    "brier_thresholds": {k: [float(x) for x in v]
                                         for k, v in brier_thresholds.items()},
                    "acc_centered": bool(acc_centered),
                    "spec_lead_range": list(spec_lead_range)
                    if spec_lead_range else None},
        "climatology": ({"path": climo_path, "source": climo_source,
                         "window_days": climo_window,
                         "lat_flip": bool(cl_lat_flip),
                         "lon_roll": int(cl_lon_roll)} if climo_path else None),
    }
    jp = os.path.join(outdir, "%s_meta.json" % name)
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    files.append(jp)

    if verbose:
        if "rmse" in results and results["rmse"]:
            print("\n[%s] === RMSE（集合平均场，%s）===" % (_ts(),
                  ("cos(lat) 加权" if lat_weighted else "不加权")))
            print(results["rmse"]["ensmean"].round(4).to_string())
            print("\n[%s] === 集合离散度（member 间场加权 std）===" % _ts())
            if results["spread"] is not None:
                print(results["spread"].round(4).to_string())
        if results.get("crps") is not None:
            print("\n[%s] === CRPS（集合整体，%s）===" % (_ts(),
                  ("cos(lat) 加权" if lat_weighted else "不加权")))
            print(results["crps"].round(4).to_string())
        if results.get("brier") is not None:
            print("\n[%s] === Brier（事件=场值≥阈值，%s）===" % (_ts(),
                  ("cos(lat) 加权" if lat_weighted else "不加权")))
            print(results["brier"].round(4).to_string())
        print("\n[%s] 已写出:" % _ts())
        for fp in files:
            print("        " + fp)
    if climo is not None and climo_owned:
        climo.close()
    return results, meta


# ---------------------------------------------------------------- 画图

def _unit_display(unit):
    """单位 → 绘图显示（mathtext 上标，如 m2 s-2 → m² s⁻²）。"""
    if not unit:
        return ""
    u = str(unit)
    table = [("m2 s-2", r"$\mathrm{m^2\,s^{-2}}$"),
             ("m s-1", r"$\mathrm{m\,s^{-1}}$"),
             ("kg kg-1", r"$\mathrm{kg\,kg^{-1}}$"),
             ("m/s", r"$\mathrm{m\,s^{-1}}$")]
    for k, v in table:
        if u.startswith(k):
            return v
    if u.startswith("mm"):
        return "mm"
    return u


def _plot_rmse(rmse_tables, outdir, name, files):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ensmean = rmse_tables.get("ensmean")
    if ensmean is None or ensmean.empty:
        return
    cols = [c for c in ensmean.columns if c != "tp"]   # tp 不画图（站点 TS/BIAS）
    if not cols:
        return
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.4), squeeze=False)
    for k, c in enumerate(cols):
        ax = axes[0][k]
        for m_name, df in rmse_tables.items():
            if m_name == "ensmean":
                ax.plot(df.index / 24.0, df[c].values, lw=2.2, color="k",
                        label="ensmean")
            else:
                ax.plot(df.index / 24.0, df[c].values, lw=1.0, alpha=0.6)
        unit = get_spec(c).unit
        ud = _unit_display(unit)
        ax.set_title(("%s (%s)" % (c, ud)) if ud else c, fontsize=11)
        ax.set_xlabel("lead (day)", fontsize=9)
        ax.set_ylabel(("RMSE (%s)" % ud) if ud else "RMSE", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("RMSE  %s" % name, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(outdir, "rmse_%s.png" % name)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    files.append(png)


def _plot_spectra(spectra, outdir, name, files):
    if not spectra:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    vars_ = [v for v in spectra if v != "tp"]          # tp 不画图
    if not vars_:
        return
    n = len(vars_)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.6), squeeze=False)
    for k, v in enumerate(vars_):
        ax = axes[0][k]
        d = spectra[v]
        ax.loglog(d.index, d["ensmean_pred"], lw=1.8, color="#1f77b4",
                  label="ensmean pred")
        ax.loglog(d.index, d["ensmean_obs"], lw=1.8, color="#d62728",
                  label="obs")
        for m_name in [c for c in d.columns
                       if c.endswith("_pred") and c != "ensmean_pred"]:
            ax.loglog(d.index, d[m_name], lw=0.8, alpha=0.4,
                      color="#1f77b4")
        unit = get_spec(v).unit
        ud = _unit_display(unit)
        ax.set_title(("%s (%s)" % (v, ud)) if ud else v, fontsize=11)
        ax.set_xlabel("zonal wavenumber k", fontsize=9)
        ax.set_ylabel("power", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("power spectrum  %s" % name, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(outdir, "spectrum_%s.png" % name)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    files.append(png)
# ---------------------------------------------------------------- 批量起报点

def _has_members(d):
    """目录下是否含预报数据：member_xxx/NN.nc（集合）或 直接 NN.nc（单确定性）。"""
    if not os.path.isdir(d):
        return False
    if any(re.search(r"\d+\.nc$", fn) for fn in os.listdir(d)):
        return True                                   # 单确定性：根下直接 NN.nc
    for sub in os.listdir(d):
        p = os.path.join(d, sub)
        if os.path.isdir(p) and any(re.search(r"\d+\.nc$", fn)
                                    for fn in os.listdir(p)):
            return True
    return False


def _resume_ok(outdir_date, date, metrics):
    """该日期是否已完整（meta + 各请求指标产物都在）。"""
    if not os.path.exists(os.path.join(outdir_date,
                                        date + "_meta.json")):
        return False
    names = [f for f in os.listdir(outdir_date)]
    pre = lambda tag: any(f.startswith(tag + "_" + date)
                           for f in names)
    for m in (metrics or ("rmse",)):
        if m in ("rmse", "acc", "fa", "brier", "spectrum") \
                and not pre(m):
            return False
        if m == "crps" and not os.path.exists(
                os.path.join(outdir_date, "crps_%s.csv" % date)):
            return False
    return True


def _resume_load(outdir_date, date, metrics):
    """从已写出的 CSV/meta 重建 (res, meta)。"""
    mp = os.path.join(outdir_date, date + "_meta.json")
    with open(mp, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    def _tables(tag):
        outd = {}
        pre = tag + "_" + date + "_"
        for fn in sorted(os.listdir(outdir_date)):
            if fn.startswith(pre) and fn.endswith(".csv"):
                member = fn[len(pre):-4]
                outd[member] = pd.read_csv(
                    os.path.join(outdir_date, fn), index_col=0)
        return outd

    res = {}
    for tag in ("rmse", "acc", "fa"):
        tbl = _tables(tag)
        if "ensmean" not in tbl and "det" in tbl:
            tbl["ensmean"] = tbl["det"].copy()
        if tbl:
            res[tag] = tbl
    for tag, fn in (("crps", "crps_%s.csv" % date),
                    ("spread", "spread_%s.csv" % date)):
        fp = os.path.join(outdir_date, fn)
        if os.path.exists(fp):
            res[tag] = pd.read_csv(fp, index_col=0)
    bfs = sorted(f for f in os.listdir(outdir_date)
                 if f.startswith("brier_" + date)
                 and f.endswith(".csv"))
    if bfs:
        res["brier"] = pd.concat(
            [pd.read_csv(os.path.join(outdir_date, f), index_col=0)
             for f in bfs], axis=1)
    spec = {}
    pre = "spectrum_" + date + "_"
    for fn in sorted(os.listdir(outdir_date)):
        if fn.startswith(pre) and fn.endswith(".csv"):
            spec[fn[len(pre):-4]] = pd.read_csv(
                os.path.join(outdir_date, fn), index_col=0)
    if spec:
        res["spectrum"] = spec
    meta.setdefault("n_members",
                    len(meta.get("members")
                        or meta.get("member_names") or ["det"]))
    return res, meta


def _batch_row(date, res, meta):
    """把单个起报点的结果压成 summary 一行。"""
    row = {"init_date": date, "n_members": meta["n_members"],
           "n_leads": meta["n_leads"],
           "variables": ",".join(meta.get("matched_vars", []))}
    rmse = res.get("rmse")
    if rmse and "ensmean" in rmse:
        df = rmse["ensmean"]
        for v in df.columns:
            s = df[v]
            i24 = int(np.argmin(np.abs(s.index.values - 24.0)))
            row["rmse_%s_24h" % v] = float(s.iloc[i24])
            row["rmse_%s_last" % v] = float(s.iloc[-1])
            row["rmse_%s_mean" % v] = float(s.mean())
    crps = res.get("crps")
    if crps is not None:
        for v in crps.columns:
            row["crps_%s_mean" % v] = float(crps[v].mean())
    brier = res.get("brier")
    if brier is not None:
        for c in brier.columns:
            row["brier_%s_mean" % c] = float(brier[c].mean())
    return row


_LEAD_METRICS = ("rmse", "acc", "fa", "crps", "brier", "spread")


def _collect_agg(res, agg, sp):
    """把单个起报点结果收集到聚合器（rmse/acc/fa 取 ensmean）。"""
    for t in ("rmse", "acc", "fa"):
        d = res.get(t)
        if d and d.get("ensmean") is not None and not d["ensmean"].empty:
            agg[t].append(d["ensmean"])
    for t in ("crps", "brier", "spread"):
        d = res.get(t)
        if d is not None and not d.empty:
            agg[t].append(d)
    s = res.get("spectrum")
    if s:
        for v, df in s.items():
            sp.setdefault(v, []).append(df)


def _agg_mean(items):
    """[DataFrame...]（行=lead，列=要素）→ 跨起报点按行平均。"""
    keep = [d for d in items
            if d is not None and not (hasattr(d, "empty") and d.empty)]
    if not keep:
        return None
    cols = list(keep[0].columns)
    df = pd.concat([d.reindex(columns=cols) for d in keep], axis=0)
    return df.groupby(level=0).mean()


def _write_means(outdir_root, agg, sp):
    """写 mean_*.csv（行=lead，列=要素，值=所有起报点平均）。"""
    written = []
    means = {}
    for t, items in agg.items():
        m = _agg_mean(items)
        means[t] = m
        if m is not None:
            fp = os.path.join(outdir_root, "mean_%s.csv" % t)
            m.to_csv(fp, float_format="%.6g")
            written.append(fp)
    # spread/rmse 的 mean 上面刚算过（_agg_mean 是纯函数，重算结果相同），
    # 直接复用，省两次全量 concat 的内存和 CPU。
    spr, rms = means.get("spread"), means.get("rmse")
    if spr is not None and rms is not None:
            common = [c for c in spr.columns if c in rms.columns]
            if common:
                ratio = spr[common].div(rms[common].replace(0.0, np.nan))
                ratio.index.name = "lead_h"
                fp = os.path.join(outdir_root,
                                  "mean_spread_rmse_ratio.csv")
                ratio.to_csv(fp, float_format="%.6g")
                written.append(fp)
    for v, items in sp.items():
        m = _agg_mean(items)
        if m is not None:
            fp = os.path.join(outdir_root, "mean_spectrum_%s.csv" % v)
            m.to_csv(fp, float_format="%.6g")
            written.append(fp)
    return written

def run_batch(pred_root, target_zarr, outdir_root, dates=None, verbose=True,
              resume=False, **kw):
    """对 pred_root 下多个 YYYYMMDD 起报点逐个跑 run_ensemble。

    输出 <outdir_root>/<YYYYMMDD>/ 各点结果 + <outdir_root>/summary.csv +
    batch_meta.json。单点失败不中断，记入 failures。
    resume 时若存在 .parts（并行崩溃残留），先把其中已完成的日期回收回根目录。
    """
    pred_root = str(pred_root)
    if resume and os.path.isdir(os.path.join(outdir_root, ".parts")):
        import shutil as _sh
        _parts = os.path.join(outdir_root, ".parts")
        for _part in sorted(os.listdir(_parts)):
            _pd = os.path.join(_parts, _part)
            if not os.path.isdir(_pd):
                continue
            for _d in sorted(os.listdir(_pd)):
                _src = os.path.join(_pd, _d)
                if not (os.path.isdir(_src)
                        and os.path.exists(os.path.join(
                            _src, _d + "_meta.json"))):
                    continue
                _dst = os.path.join(outdir_root, _d)
                if os.path.isdir(_dst):
                    _sh.rmtree(_dst)
                _sh.move(_src, _dst)
                if verbose:
                    print("[%s][batch] 回收 .parts 已完成日期: %s"
                          % (_ts(), _d))
    if not os.path.isdir(pred_root):
        raise ValueError("--pred-root 目录不存在: %s" % pred_root)
    _D = r"(\d{8}|\d{4}-\d{2}-\d{2})"
    candidates = sorted(d for d in os.listdir(pred_root)
                        if re.fullmatch(_D, d)
                        and _has_members(os.path.join(pred_root, d)))
    if dates:
        dates = [str(d) for d in dates
                 if re.fullmatch(r"(\d{8}|\d{4}-\d{2}-\d{2})", str(d))]
        bad = [d for d in dates if d not in candidates]
        if bad:
            raise ValueError("以下起报点没有预报数据: %s" % bad)
        candidates = [d for d in candidates if d in dates]
    if not candidates:
        raise ValueError("%s 下没有 YYYYMMDD 起报点目录" % pred_root)

    if isinstance(target_zarr, (list, tuple)):
        tpaths = [getattr(t, "path", "memory") for t in target_zarr]
    else:
        tpaths = [getattr(target_zarr, "path", "memory")]

    os.makedirs(outdir_root, exist_ok=True)
    climo_obj = None
    _mets = kw.get("metrics") or ("rmse",)
    if {"acc", "fa"} & set(_mets) and isinstance(kw.get("climo_path"), str):
        from vfc.climo import DailyClimatology
        climo_obj = DailyClimatology(
            kw["climo_path"], engine=kw.get("engine", "auto"),
            window=kw.get("climo_window", 15),
            source=kw.get("climo_source", "未标注"))
    rows, metas, failures = [], [], []
    agg = {t: [] for t in _LEAD_METRICS}
    sp = {}
    for date in candidates:
        out = os.path.join(outdir_root, date)
        if resume and _resume_ok(out, date, kw.get("metrics")):
            if verbose:
                print("[%s][batch] %s 已存在，跳过（读回结果）" % (_ts(), date))
            try:
                r2, m2 = _resume_load(out, date, kw.get("metrics"))
                rows.append(_batch_row(date, r2, m2))
                metas.append(m2)
                _collect_agg(r2, agg, sp)
                gc.collect()
                continue
            except Exception as e2:
                if verbose:
                    print("[%s][batch] %s 读回失败(%s)，改为重算" % (_ts(), date, e2))
        if verbose:
            print("\n[%s] ############ 起报点 %s ############" % (_ts(), date))
        try:
            res, meta = run_ensemble(
                os.path.join(pred_root, date), target_zarr, init=None,
                outdir=out, name=None, climo_obj=climo_obj,
                verbose=verbose, **kw)
            rows.append(_batch_row(date, res, meta))
            metas.append(meta)
            _collect_agg(res, agg, sp)
            gc.collect()
        except Exception as e:
            failures.append((date, repr(e)))
            if verbose:
                print("[%s][batch] %s 失败: %s" % (_ts(), date, e))
    df = pd.DataFrame(rows)
    if df.size:
        df.to_csv(os.path.join(outdir_root, "summary.csv"),
                  index=False, float_format="%.6g")
    mean_files = _write_means(outdir_root, agg, sp)
    if verbose and mean_files:
        print("[%s] 均值表（跨 %d 个起报点平均，按 lead）: %s"
              % (_ts(), len(rows), ", ".join(os.path.basename(f)
                                             for f in mean_files)))
    bmeta = {
        "pred_root": pred_root, "target_zarr": tpaths,
        "n_dates": len(candidates), "dates": candidates,
        "n_ok": len(rows), "failures": failures,
        "outdir_root": str(outdir_root), "options": kw,
        "mean_files": mean_files,
    }
    with open(os.path.join(outdir_root, "batch_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(bmeta, f, ensure_ascii=False, indent=2)
    if verbose:
        print("\n[%s] ===== 批量汇总（%d 成功 / %d 失败）====="
              % (_ts(), len(rows), len(failures)))
        if df.size:
            print(df.round(4).to_string(index=False))
        if failures:
            for d, e in failures:
                print("  [%s] 失败 %s: %s" % (_ts(), d, e))
    if climo_obj is not None:
        climo_obj.close()
    return df, bmeta

def _run_batch_part(args):
    """并行 worker：pred_root/target_zarr/part_root/chunk/kw → (part, n_ok, fails)。"""
    pred_root, target_zarr, part_root, chunk, kw = args
    # 崩溃定位：记下"哪个 pid 在跑哪个日期"。worker 被 OOM killer / 段错误杀掉时，
    # 进程池只报一个笼统的 BrokenProcessPool——回看日志，最后一条"start 了却没有
    # 对应 done"的日期就是嫌疑日期；pid 还能和 dmesg 的 oom-kill 行对上。
    # 每块只多两行输出，量可以忽略。
    print(f"[{_ts()}][worker {os.getpid()}] start {chunk[0]}..{chunk[-1]}", flush=True)
    # 静默跑：逐日期/逐要素日志全部关闭，进度由父进程汇总打印
    _, meta = run_batch(pred_root, target_zarr, part_root, dates=chunk,
                        verbose=False, **kw)
    n_ok = meta.get("n_ok", 0)
    fails = meta.get("failures", [])
    print(f"[{_ts()}][worker {os.getpid()}] done  {chunk[0]}..{chunk[-1]} ok={n_ok}", flush=True)
    if fails:
        print(f"[{_ts()}][worker] Chunk {chunk[0]}..{chunk[-1]}: "
              f"{len(fails)}/{len(chunk)} failed: "
              + "; ".join(f"{d}: {e}" for d, e in fails), flush=True)
    return part_root, n_ok, fails


def run_batch_parallel(pred_root, target_zarr, outdir_root, dates=None,
                       n_workers=4, verbose=True, resume=False, **kw):
    """起报点并行：分 n_workers 份各用一个进程跑 run_batch，
    最后把各份的 summary 与 mean_*.csv 合并到 outdir_root。

    注意：每个进程独立缓存气候态（acc/fa 时内存 ≈ 进程数 × 气候态占用），
    不开 acc/fa（不加 --climo/acc fa）则几乎无额外内存。
    """
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    pred_root = str(pred_root)
    _D = r"(\d{8}|\d{4}-\d{2}-\d{2})"
    candidates = sorted(d for d in os.listdir(pred_root)
                        if re.fullmatch(_D, d)
                        and _has_members(os.path.join(pred_root, d)))
    if dates:
        dates = [str(d) for d in dates
                 if re.fullmatch(r"(\d{8}|\d{4}-\d{2}-\d{2})", str(d))]
        bad = [d for d in dates if d not in candidates]
        if bad:
            raise ValueError("以下起报点没有预报数据: %s" % bad)
        candidates = [d for d in candidates if d in dates]
    if not candidates:
        raise ValueError("%s 下没有预报起报点目录" % pred_root)
    _all_candidates = list(candidates)
    if resume:
        candidates = [d for d in candidates
                      if not os.path.exists(os.path.join(
                          outdir_root, d, d + "_meta.json"))]
    if isinstance(target_zarr, (list, tuple)):
        tpaths = [getattr(t, "path", "memory") for t in target_zarr]
    else:
        tpaths = [getattr(target_zarr, "path", "memory")]

    os.makedirs(outdir_root, exist_ok=True)
    parts_root = os.path.join(outdir_root, ".parts")
    if os.path.isdir(parts_root) and not resume:
        shutil.rmtree(parts_root)
    os.makedirs(parts_root, exist_ok=True)
    # 短命 worker：每进程只处理 VFC_DATES_PER_CHILD 个日期就重启，
    # 内存不跨日期累积 → 高并发也较难 OOM（配合 --resume 可断点续跑）
    _dpc = max(1, int(os.environ.get("VFC_DATES_PER_CHILD", "2")))
    chunks = [candidates[i:i + _dpc]
              for i in range(0, len(candidates), _dpc)]
    n_workers = max(1, min(int(n_workers), len(chunks)))
    task_kw = dict(kw)
    task_kw["resume"] = bool(resume)
    tasks = [(pred_root, target_zarr,
              os.path.join(parts_root, "part_%03d" % i), chunk,
              task_kw)
             for i, chunk in enumerate(chunks)]

    print(f"[{_ts()}][parallel] Starting {n_workers} workers for {len(chunks)} chunks", flush=True)
    print(f"[{_ts()}][parallel] Total dates: {len(candidates)}, dates per chunk: {_dpc}", flush=True)

    results = []
    t_par = _dt.datetime.now()
    try:
        _ex = ProcessPoolExecutor(max_workers=n_workers,
                                 max_tasks_per_child=1)
    except TypeError:
        _ex = ProcessPoolExecutor(max_workers=n_workers)
    with _ex as ex:
        done = 0
        # 用 as_completed 而不是 ex.map：ex.map 按**提交顺序** yield，第一个块没跑完
        # 一行都不打，之后还被慢块堵着——外面看起来就跟卡死一样。
        # results 的顺序变成完成顺序，但下面合并只按 part 目录搬文件，与顺序无关。
        futures = [ex.submit(_run_batch_part, task) for task in tasks]
        # future -> 该块的首日期。池崩时用它报出"没跑完的日期"——
        # worker 是被杀的（OOM/段错误）还是正常失败，答案就在这批日期里。
        _chunk_of = {f: t[3] for f, t in zip(futures, tasks)}
        _done_futures = set()
        try:
            for future in as_completed(futures):
                results.append(future.result())
                _done_futures.add(future)
                done += 1
                if done == 1 or done % 10 == 0 or done == len(tasks):
                    elapsed = (_dt.datetime.now() - t_par).total_seconds()
                    eta = elapsed / done * (len(tasks) - done)
                    print(f"[{_ts()}][parallel] Progress: {done}/{len(tasks)} chunks "
                          f"({100*done/len(tasks):.1f}%, ETA {eta/60:.1f}m)", flush=True)
        except BrokenProcessPool as exc:
            _left = sorted({c[0] for f, c in _chunk_of.items()
                            if f not in _done_futures})
            print(f"[{_ts()}][parallel] 进程池崩了（{exc}）：有 worker 被强行终止"
                  f"（OOM killer / 段错误 / kill -9 都长这样）。", flush=True)
            print(f"[{_ts()}][parallel] 没跑完的日期（坏数据/OOM 嫌疑就在这些里，"
                  f"配合上面最后一条只有 start 没有 done 的行定位）：", flush=True)
            print(f"[{_ts()}][parallel]   {', '.join(_left)}", flush=True)
            raise

    print(f"[{_ts()}][parallel] All workers completed, merging results...", flush=True)
    # ---- 合并：把各 part 的日期目录移到根目录 ----
    all_fails = []
    merged_count = 0
    for part, n_ok, fails in results:
        all_fails.extend(fails)
        for d in sorted(os.listdir(part)):
            pp = os.path.join(part, d)
            if not os.path.isdir(pp):
                continue
            dst = os.path.join(outdir_root, d)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(pp, dst)
            merged_count += 1
    print(f"[{_ts()}][parallel] Merged {merged_count} date directories from parts to {outdir_root}", flush=True)

    if os.path.isdir(parts_root):
        shutil.rmtree(parts_root, ignore_errors=True)
        print(f"[{_ts()}][parallel] Cleaned up parts directory", flush=True)

    # ---- 汇总：扫描根目录里所有”已完整”的日期（读回结果），
    #      串行跑过的、并行跑过的、断点残留的都能统一进 summary/mean ----
    print(f"[{_ts()}][parallel] Aggregating results from {len(_all_candidates)} dates...", flush=True)
    rows, metas, agg = [], [], {}
    for tag in _LEAD_METRICS:
        agg[tag] = []
    sp = {}
    done_dates = [d for d in _all_candidates
                  if os.path.exists(os.path.join(outdir_root, d,
                                                 d + "_meta.json"))]
    print(f"[{_ts()}][parallel] Found {len(done_dates)} completed dates with metadata", flush=True)

    for d in done_dates:
        try:
            r2, m2 = _resume_load(os.path.join(outdir_root, d), d,
                                  kw.get("metrics"))
            rows.append(_batch_row(d, r2, m2))
            metas.append(m2)
            _collect_agg(r2, agg, sp)
        except Exception as e:
            all_fails.append((d, repr(e)))
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.size:
        summary_path = os.path.join(outdir_root, "summary.csv")
        df.to_csv(summary_path, index=False, float_format="%.6g")
        print(f"[{_ts()}][parallel] Wrote summary: {summary_path}", flush=True)
    mean_files = _write_means(outdir_root, agg, sp)
    if mean_files:
        print(f"[{_ts()}][parallel] Wrote {len(mean_files)} mean files:", flush=True)
        for mf in mean_files:
            print(f"[{_ts()}][parallel]   {mf}", flush=True)

    bmeta = {
        "pred_root": pred_root, "target_zarr": tpaths,
        "n_dates": len(_all_candidates), "dates": _all_candidates,
        "n_ok": len(done_dates), "n_workers": n_workers,
        "failures": all_fails, "outdir_root": str(outdir_root),
        "options": kw, "mean_files": mean_files,
    }
    with open(os.path.join(outdir_root, "batch_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(bmeta, f, ensure_ascii=False, indent=2)
    print(f"[{_ts()}][parallel] Wrote batch metadata: {os.path.join(outdir_root, 'batch_meta.json')}", flush=True)

    if verbose:
        print(f"\n[{_ts()}][parallel] ===== Batch Summary =====", flush=True)
        print(f"[{_ts()}][parallel] Total dates: {len(_all_candidates)}", flush=True)
        print(f"[{_ts()}][parallel] Completed: {len(done_dates)}", flush=True)
        print(f"[{_ts()}][parallel] Failed: {len(all_fails)}", flush=True)
        if df.size:
            print(df.round(4).to_string(index=False), flush=True)
        if all_fails:
            for d, e in all_fails[:10]:
                print(f"  [{_ts()}][parallel] Failed {d}: {e}", flush=True)
            if len(all_fails) > 10:
                print(f"  [{_ts()}][parallel] ... and {len(all_fails)-10} more failures", flush=True)
    return df, bmeta


def run_pangu_batch(pangu_root, target_zarr, outdir_root, dates=None,
                    verbose=True, **kw):
    """对盘古 DF 根目录下多个 YYYYMMDD 起报点逐个跑 run_ensemble。

    输出 <outdir_root>/<YYYYMMDD>/ + summary.csv + batch_meta.json。
    单点失败不中断，记入 failures。
    """
    pangu_root = str(pangu_root)
    candidates = scan_pangu_dates(pangu_root)
    if dates:
        dates = [str(d) for d in dates]
        bad = [d for d in dates if d not in candidates]
        if bad:
            raise ValueError("以下起报点没有盘古数据: %s" % bad)
        candidates = [d for d in candidates if d in dates]
    if not candidates:
        raise ValueError("%s 下没有盘古起报点目录" % pangu_root)
    if isinstance(target_zarr, (list, tuple)):
        tpaths = [getattr(t, "path", "memory") for t in target_zarr]
    else:
        tpaths = [getattr(target_zarr, "path", "memory")]
    os.makedirs(outdir_root, exist_ok=True)
    climo_obj = None
    _mets = kw.get("metrics") or ("rmse",)
    if {"acc", "fa"} & set(_mets) and isinstance(kw.get("climo_path"), str):
        from vfc.climo import DailyClimatology
        climo_obj = DailyClimatology(
            kw["climo_path"], engine=kw.get("engine", "auto"),
            window=kw.get("climo_window", 15),
            source=kw.get("climo_source", "未标注"))
    rows, metas, failures = [], [], []
    agg = {t: [] for t in _LEAD_METRICS}
    sp = {}
    engine = kw.get("engine", "auto")
    for date in candidates:
        out = os.path.join(outdir_root, date)
        if verbose:
            print("\n[%s] ############ 盘古起报点 %s ############" % (_ts(), date))
        try:
            pf = PanguForecast(os.path.join(pangu_root, date), engine=engine)
            res, meta = run_ensemble(pf, target_zarr, init=None,
                                     outdir=out, name=None, climo_obj=climo_obj, **kw)
            rows.append(_batch_row(date, res, meta))
            metas.append(meta)
            _collect_agg(res, agg, sp)
            gc.collect()
        except Exception as e:
            failures.append((date, repr(e)))
            if verbose:
                print("[%s][pangu-batch] %s 失败: %s" % (_ts(), date, e))
    df = pd.DataFrame(rows)
    if df.size:
        df.to_csv(os.path.join(outdir_root, "summary.csv"),
                  index=False, float_format="%.6g")
    mean_files = _write_means(outdir_root, agg, sp)
    if verbose and mean_files:
        print("[%s] 均值表（跨 %d 个起报点平均，按 lead）: %s"
              % (_ts(), len(rows), ", ".join(os.path.basename(f)
                                             for f in mean_files)))
    bmeta = {
        "pangu_root": pangu_root, "target_zarr": tpaths,
        "n_dates": len(candidates), "dates": candidates,
        "n_ok": len(rows), "failures": failures,
        "outdir_root": str(outdir_root), "options": kw,
        "mean_files": mean_files,
    }
    with open(os.path.join(outdir_root, "batch_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(bmeta, f, ensure_ascii=False, indent=2)
    if verbose:
        print("\n[%s] ===== 盘古批量汇总（%d 成功 / %d 失败）====="
              % (_ts(), len(rows), len(failures)))
        if df.size:
            print(df.round(4).to_string(index=False))
        for d, e in failures:
            print("  [%s] 失败 %s: %s" % (_ts(), d, e))
    if climo_obj is not None:
        climo_obj.close()
    return df, bmeta

def run_aifs_batch(pred_root, target_root, outdir_root, dates=None,
                   verbose=True, resume=False, **kw):
    """对 AIFS 预测/真值根目录下多个 YYYY-MM-DD 起报点逐个跑 run_ensemble。

    输出 <outdir_root>/<日期>/ + summary.csv + batch_meta.json。单点失败不中断。
    resume 时若存在 .parts（并行崩溃残留），先把其中已完成的日期回收回根目录；
    已有 <日期>_meta.json 且产物齐全的日期直接跳过（读回结果进 summary/mean）。
    """
    pred_root = str(pred_root)
    if resume and os.path.isdir(os.path.join(outdir_root, ".parts")):
        import shutil as _sh
        _parts = os.path.join(outdir_root, ".parts")
        for _part in sorted(os.listdir(_parts)):
            _pd = os.path.join(_parts, _part)
            if not os.path.isdir(_pd):
                continue
            for _d in sorted(os.listdir(_pd)):
                _src = os.path.join(_pd, _d)
                if not (os.path.isdir(_src)
                        and os.path.exists(os.path.join(
                            _src, _d + "_meta.json"))):
                    continue
                _dst = os.path.join(outdir_root, _d)
                if os.path.isdir(_dst):
                    _sh.rmtree(_dst)
                _sh.move(_src, _dst)
                if verbose:
                    print("[%s][aifs-batch] 回收 .parts 已完成日期: %s"
                          % (_ts(), _d))
    candidates = scan_aifs_dates(pred_root)
    if dates:
        dates = [str(d) for d in dates]
        bad = [d for d in dates if d not in candidates]
        if bad:
            raise ValueError("以下起报点没有 AIFS 预测数据: %s" % bad)
        candidates = [d for d in candidates if d in dates]
    if not candidates:
        raise ValueError("%s 下没有 AIFS 起报点目录" % pred_root)
    os.makedirs(outdir_root, exist_ok=True)
    climo_obj = None
    _mets = kw.get("metrics") or ("rmse",)
    if {"acc", "fa"} & set(_mets) and isinstance(kw.get("climo_path"), str):
        from vfc.climo import DailyClimatology
        climo_obj = DailyClimatology(
            kw["climo_path"], engine=kw.get("engine", "auto"),
            window=kw.get("climo_window", 15),
            source=kw.get("climo_source", "未标注"))
    rows, failures = [], []
    agg = {t: [] for t in _LEAD_METRICS}
    sp = {}
    engine = kw.get("engine", "auto")
    for date in candidates:
        out = os.path.join(outdir_root, date)
        if resume and _resume_ok(out, date, kw.get("metrics")):
            if verbose:
                print("[%s][aifs-batch] %s 已存在，跳过（读回结果）"
                      % (_ts(), date))
            try:
                r2, m2 = _resume_load(out, date, kw.get("metrics"))
                rows.append(_batch_row(date, r2, m2))
                _collect_agg(r2, agg, sp)
                gc.collect()
                continue
            except Exception as e2:
                if verbose:
                    print("[%s][aifs-batch] %s 读回失败(%s)，改为重算"
                          % (_ts(), date, e2))
        if verbose:
            print("\n[%s] ############ AIFS 起报点 %s ############" % (_ts(), date))
        try:
            pf = AifsForecast(os.path.join(pred_root, date), var_name="output",
                              engine=engine)
            tf = AifsForecast(os.path.join(target_root, date), var_name="target",
                              engine=engine, as_target=True)
            res, meta = run_ensemble(pf, [tf], init=None, outdir=out,
                                     name=None, climo_obj=climo_obj, **kw)
            rows.append(_batch_row(date, res, meta))
            _collect_agg(res, agg, sp)
            gc.collect()
        except Exception as e:
            failures.append((date, repr(e)))
            if verbose:
                print("[%s][aifs-batch] %s 失败: %s" % (_ts(), date, e))
    df = pd.DataFrame(rows)
    if df.size:
        df.to_csv(os.path.join(outdir_root, "summary.csv"),
                  index=False, float_format="%.6g")
    mean_files = _write_means(outdir_root, agg, sp)
    if verbose and mean_files:
        print("[%s] 均值表（跨 %d 个起报点平均，按 lead）: %s"
              % (_ts(), len(rows), ", ".join(os.path.basename(f)
                                             for f in mean_files)))
    bmeta = {
        "pred_root": pred_root, "target_root": str(target_root),
        "n_dates": len(candidates), "dates": candidates,
        "n_ok": len(rows), "failures": failures,
        "outdir_root": str(outdir_root), "options": kw,
        "mean_files": mean_files,
    }
    with open(os.path.join(outdir_root, "batch_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(bmeta, f, ensure_ascii=False, indent=2)
    if verbose:
        print("\n[%s] ===== AIFS 批量汇总（%d 成功 / %d 失败）====="
              % (_ts(), len(rows), len(failures)))
        if df.size:
            print(df.round(4).to_string(index=False))
        for d, e in failures:
            print("  [%s] 失败 %s: %s" % (_ts(), d, e))
    if climo_obj is not None:
        climo_obj.close()
    return df, bmeta



def _run_aifs_batch_part(args):
    """并行 worker：pred_root/target_root/part_root/chunk/kw → (part, n_ok, fails)。"""
    pred_root, target_root, part_root, chunk, kw = args
    _, meta = run_aifs_batch(pred_root, target_root, part_root, dates=chunk,
                             verbose=False, **kw)
    return part_root, meta.get("n_ok", 0), meta.get("failures", [])


def run_aifs_batch_parallel(pred_root, target_root, outdir_root, dates=None,
                            n_workers=4, verbose=True, resume=False, **kw):
    """AIFS 起报点并行：与 FuXi run_batch_parallel 同款机制。

    - 日期按 VFC_DATES_PER_CHILD（默认 2）分块，每个进程只跑一块（短命 worker，
      内存不跨日期累积 → 高并发也较难 OOM）；
    - 各块写 <outdir_root>/.parts/part_XXX/，全部结束后统一合并回根目录；
    - --resume：根目录已有 <日期>_meta.json 的日期直接跳过（n_workers 换值也能续），
      崩溃残留 .parts 由串行 run_aifs_batch / 下次并行合并时回收；
    - 最终 summary.csv / mean_*.csv / batch_meta.json 扫描根目录所有完整日期统一生成。
    """
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    pred_root = str(pred_root)
    target_root = str(target_root)
    if not os.path.isdir(pred_root):
        raise ValueError("--aifs-pred-root 目录不存在: %s" % pred_root)
    candidates = scan_aifs_dates(pred_root)
    if dates:
        dates = [str(d) for d in dates]
        bad = [d for d in dates if d not in candidates]
        if bad:
            raise ValueError("以下起报点没有 AIFS 预测数据: %s" % bad)
        candidates = [d for d in candidates if d in dates]
    if not candidates:
        raise ValueError("%s 下没有 AIFS 起报点目录" % pred_root)
    _all_candidates = list(candidates)
    if resume:
        candidates = [d for d in candidates
                      if not os.path.exists(os.path.join(
                          outdir_root, d, d + "_meta.json"))]

    os.makedirs(outdir_root, exist_ok=True)
    parts_root = os.path.join(outdir_root, ".parts")
    if os.path.isdir(parts_root) and not resume:
        shutil.rmtree(parts_root)
    os.makedirs(parts_root, exist_ok=True)
    _dpc = max(1, int(os.environ.get("VFC_DATES_PER_CHILD", "2")))
    chunks = [candidates[i:i + _dpc]
              for i in range(0, len(candidates), _dpc)]
    n_workers = max(1, min(int(n_workers), len(chunks)))
    task_kw = dict(kw)
    task_kw["resume"] = bool(resume)
    tasks = [(pred_root, target_root,
              os.path.join(parts_root, "part_%03d" % i), chunk,
              task_kw)
             for i, chunk in enumerate(chunks)]

    results = []
    try:
        _ex = ProcessPoolExecutor(max_workers=n_workers,
                                 max_tasks_per_child=1)
    except TypeError:
        _ex = ProcessPoolExecutor(max_workers=n_workers)
    with _ex as ex:
        for r in ex.map(_run_aifs_batch_part, tasks):
            results.append(r)

    # ---- 合并：把各 part 的日期目录移到根目录 ----
    all_fails = []
    for part, _n_ok, fails in results:
        all_fails.extend(fails)
        for d in sorted(os.listdir(part)):
            pp = os.path.join(part, d)
            if not os.path.isdir(pp):
                continue
            dst = os.path.join(outdir_root, d)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(pp, dst)
    if os.path.isdir(parts_root):
        shutil.rmtree(parts_root, ignore_errors=True)

    # ---- 汇总：扫描根目录里所有“已完整”的日期（读回结果），
    #      串行跑过的、并行跑过的、断点残留的都能统一进 summary/mean ----
    rows, _metas, agg = [], [], {}
    for tag in _LEAD_METRICS:
        agg[tag] = []
    sp = {}
    done_dates = [d for d in _all_candidates
                  if os.path.exists(os.path.join(outdir_root, d,
                                                 d + "_meta.json"))]
    for d in done_dates:
        try:
            r2, m2 = _resume_load(os.path.join(outdir_root, d), d,
                                  kw.get("metrics"))
            rows.append(_batch_row(d, r2, m2))
            _metas.append(m2)
            _collect_agg(r2, agg, sp)
        except Exception as e:
            all_fails.append((d, repr(e)))
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.size:
        df.to_csv(os.path.join(outdir_root, "summary.csv"),
                  index=False, float_format="%.6g")
    mean_files = _write_means(outdir_root, agg, sp)

    bmeta = {
        "pred_root": pred_root, "target_root": target_root,
        "n_dates": len(_all_candidates), "dates": _all_candidates,
        "n_ok": len(done_dates), "n_workers": n_workers,
        "failures": all_fails, "outdir_root": str(outdir_root),
        "options": kw, "mean_files": mean_files,
    }
    with open(os.path.join(outdir_root, "batch_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(bmeta, f, ensure_ascii=False, indent=2)
    if verbose:
        print("\n[%s] ===== AIFS 并行批量汇总（%d 成功 / %d 失败，%d 进程）====="
              % (_ts(), len(done_dates), len(all_fails), n_workers))
        if df.size:
            print(df.round(4).to_string(index=False))
        for d, e in all_fails:
            print("  [%s] 失败 %s: %s" % (_ts(), d, e))
    return df, bmeta


# ---------------------------------------------------------------- 自检

def _write_step_nc(path, fields, lat, lon):
    """写单时步 nc（netCDF4 优先 / h5py 回退）。fields: {大写名: (lat,lon)}。"""
    try:
        import netCDF4 as nc
        with nc.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("lat", lat.size)
            ds.createDimension("lon", lon.size)
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            for k, v in fields.items():
                ds.createVariable(k, "f4", ("lat", "lon"))[:] = v
        return
    except ImportError:
        import h5py
        with h5py.File(path, "w") as h:
            h["lat"] = lat
            h["lon"] = lon
            for k, v in fields.items():
                h[k] = v.astype("f4")


class _SimpleTarget(object):
    """内存目标（自检用），接口与 ZarrTarget 一致。"""

    def __init__(self, fields, time, lat, lon, path="memory"):
        # fields: {小写名: (T, lat, lon)}
        self._fields = {k: np.asarray(v, dtype="f4") for k, v in fields.items()}
        self.levels = sorted(self._fields)
        self.lat = np.asarray(lat, dtype="f8")
        self.lon = np.asarray(lon, dtype="f8")
        self.time = list(time)
        self.time_hours = np.array(
            [(d - _EPOCH).total_seconds() / 3600.0 for d in self.time])
        self.lon_fixed = False
        self.engine = "memory"
        self.path = path

    def read(self, varname, time_idx=None):
        a = self._fields[varname.lower()]
        if time_idx is not None:
            a = a[np.asarray(time_idx, dtype="i8")]
        return a


def _self_test() -> int:
    ok = [0, 0]

    def check(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))

    tmp = tempfile.mkdtemp(prefix="vfc_ens_selftest_")
    try:
        import shutil
        lat = np.arange(90.0, -90.01, -30.0)          # 7 点
        lon = np.arange(0.0, 360.0, 30.0)             # 12 点
        nlat, nlon = lat.size, lon.size
        T = 5
        leads = 6.0 * np.arange(1, T + 1)             # 6..30h
        A = 5.0
        tgt_lon = np.arange(-180.0, 180.0, 30.0)      # 目标 lon（与预报差半圈）

        # 同一物理场分别在两个 lon 网格上取值（滚动对齐后应逐点相等）
        def field_on(grid, amp, kx, phase=0.0):
            w = np.sin(kx * 2 * np.pi * grid / 360.0 + phase)
            return amp * w[None, :] + np.zeros((nlat, 1))

        wave_p = field_on(lon, A, 3)
        wave_t = field_on(tgt_lon, A, 3)
        u_p = field_on(lon, 15.0, 1, 0.5) + 10.0
        u_t = field_on(tgt_lon, 15.0, 1, 0.5) + 10.0
        v_p = field_on(lon, 8.0, 2, 1.0) + 3.0
        v_t = field_on(tgt_lon, 8.0, 2, 1.0) + 3.0
        z_obs_p = np.broadcast_to(wave_p, (T, nlat, nlon)).copy() \
            + np.arange(T)[:, None, None] * 0.1
        z_obs_t = np.broadcast_to(wave_t, (T, nlat, nlon)).copy() \
            + np.arange(T)[:, None, None] * 0.1
        msl_obs = 100000.0 + 0.0 * z_obs_p
        tp_obs = np.full((T, nlat, nlon), 0.5)        # 逐 6h 时段量（mm）
        u_obs_p = np.broadcast_to(u_p, (T, nlat, nlon)).copy()
        u_obs_t = np.broadcast_to(u_t, (T, nlat, nlon)).copy()
        v_obs_p = np.broadcast_to(v_p, (T, nlat, nlon)).copy()
        v_obs_t = np.broadcast_to(v_t, (T, nlat, nlon)).copy()

        pred_root = os.path.join(tmp, "members")
        os.makedirs(pred_root)
        # 预报 = 实况 + 常数偏移 → RMSE 精确 = |偏移|
        offs = {"member_000": 0.7, "member_001": -0.5}
        for m, off in offs.items():
            d = os.path.join(pred_root, m)
            os.makedirs(d)
            for k in range(T):
                fields = {
                    "Z500": z_obs_p[k] + off,
                    "MSL": msl_obs[k] + 100.0 * off,
                    "TP": tp_obs[k] + off,
                    "U200": u_obs_p[k] + off,
                    "V200": v_obs_p[k] + off,
                }
                _write_step_nc(os.path.join(d, "%03d.nc" % (k + 1)), fields,
                               lat, lon)

        init = _dt.datetime(2025, 1, 1, 0, 0)
        tgt_times = [init + _dt.timedelta(hours=h) for h in leads]
        sfc_t = _SimpleTarget({"msl": msl_obs, "tp": tp_obs}, tgt_times, lat,
                              tgt_lon, path="sfc")
        # pl 目标用下划线通道名（真实 pl zarr 是 z_500/u_200/v_200），
        # 检验 run_ensemble 的下划线别名匹配
        pl_t = _SimpleTarget({"z_500": z_obs_t, "u_200": u_obs_t,
                              "v_200": v_obs_t}, tgt_times, lat, tgt_lon,
                             path="pl")

        res, meta = run_ensemble(
            pred_root, [sfc_t, pl_t], init=init,
            metrics=["rmse", "crps", "brier", "spectrum"],
            outdir=os.path.join(tmp, "out"),
            target_tp_scale=1.0, tp_grid=True,
            brier_thresholds={"msl": [100000.0], "tp": [0.5]},
            plot=False, verbose=False)
        res_def, _ = run_ensemble(
            pred_root, [sfc_t, pl_t], init=init,
            metrics=["rmse"], outdir=os.path.join(tmp, "out_def"),
            target_tp_scale=1.0, plot=False, verbose=False)
        check("默认排除 tp（--tp-grid 关），其余要素照常",
              "tp" not in res_def["rmse"]["ensmean"].columns
              and "z500" in res_def["rmse"]["ensmean"].columns
              and "msl" in res_def["rmse"]["ensmean"].columns)

        all_vars = {"z500", "msl", "tp", "u200", "v200"}
        check("要素交集 = 6 个要素（双目标合并 + 派生 ws200）",
              set(res["rmse"]["ensmean"].columns) == (all_vars | {"ws200"}))
        for m, off in offs.items():
            for v in all_vars:
                exp = abs(100.0 * off) if v == "msl" else abs(off)
                check("RMSE %s %s == |%.1f|" % (m, v, exp),
                      np.allclose(res["rmse"][m][v].values, exp,
                                  rtol=0, atol=1e-4))
        check("RMSE 集合平均 == 0.1（z500）",
              np.allclose(res["rmse"]["ensmean"]["z500"].values, 0.1,
                          rtol=0, atol=1e-4))
        check("tp 逐时段量直配（差分不触发）member_000 == 0.7",
              np.allclose(res["rmse"]["member_000"]["tp"].values, 0.7,
                          rtol=0, atol=1e-4))
        # CRPS 解析解：两成员相对实况偏移 {0.7,-0.5}
        # E|F-o|=(0.7+0.5)/2=0.6；pair=(1/m²)Σ_{i<j}|Δ|=(1/4)*1.2=0.3 → CRPS=0.3
        for v in ("z500", "tp", "u200", "v200"):
            check("CRPS %s == 0.3" % v,
                  np.allclose(res["crps"][v].values, 0.3,
                              rtol=0, atol=1e-4))
        check("CRPS msl == 30（偏移 70/-50）",
              np.allclose(res["crps"]["msl"].values, 30.0,
                          rtol=0, atol=1e-4))
        # Brier 解析解：msl 成员 {100070, 99950}、obs 100000、阈值 100000
        # p=1/2，o=1 → BS=(0.5-1)^2=0.25；tp 成员 {1.2, 0.0}、obs 0.5、阈值 0.5 同理
        check("Brier msl@100000 == 0.25",
              np.allclose(res["brier"]["msl@100000"].values, 0.25,
                          rtol=0, atol=1e-4))
        check("Brier tp@0.5 == 0.25",
              np.allclose(res["brier"]["tp@0.5"].values, 0.25,
                          rtol=0, atol=1e-4))
        check("集合离散度 shape/数值>0（含派生 ws200）",
              res["spread"] is not None and res["spread"].shape == (T, 6)
              and res["spread"].values.min() > 0)
        check("谱 z500 列齐全",
              set(res["spectrum"]["z500"].columns)
              >= {"ensmean_pred", "ensmean_obs", "wavelength_km"})
        # ws200 流式（默认 VFC_ENS_STREAM_DERIVED=1）与全量读取（=0）结果一致
        _old_sd = os.environ.get("VFC_ENS_STREAM_DERIVED")
        os.environ["VFC_ENS_STREAM_DERIVED"] = "0"
        try:
            res_leg, _ = run_ensemble(
                pred_root, [sfc_t, pl_t], init=init,
                metrics=["rmse", "crps", "spectrum"],
                outdir=os.path.join(tmp, "out_leg"),
                target_tp_scale=1.0, plot=False, verbose=False)
        finally:
            if _old_sd is None:
                os.environ.pop("VFC_ENS_STREAM_DERIVED", None)
            else:
                os.environ["VFC_ENS_STREAM_DERIVED"] = _old_sd
        check("ws200 流式与全量读取一致（rmse/crps/spread/谱）",
              "ws200" in res_leg["rmse"]["ensmean"].columns
              and np.allclose(res["rmse"]["ensmean"]["ws200"].values,
                              res_leg["rmse"]["ensmean"]["ws200"].values,
                              atol=1e-5)
              and np.allclose(res["crps"]["ws200"].values,
                              res_leg["crps"]["ws200"].values, atol=1e-5)
              and np.allclose(res["spread"]["ws200"].values,
                              res_leg["spread"]["ws200"].values, atol=1e-5)
              and "ensmean_pred" in res_leg["spectrum"]["ws200"].columns
              and np.allclose(res["spectrum"]["ws200"]["ensmean_pred"].values,
                              res_leg["spectrum"]["ws200"]["ensmean_pred"].values,
                              atol=1e-4))
        # ensmean-only（VFC_ENS_ENSMEAN_ONLY=1）：只算 ensmean 的 rmse/谱，
        # 逐 member 表不输出；CRPS / spread / ensmean RMSE 与全算一致
        _old_eo = os.environ.get("VFC_ENS_ENSMEAN_ONLY")
        os.environ["VFC_ENS_ENSMEAN_ONLY"] = "1"
        try:
            res_eo, _meta_eo = run_ensemble(
                pred_root, [sfc_t, pl_t], init=init,
                metrics=["rmse", "crps", "spectrum"],
                outdir=os.path.join(tmp, "out_eo"),
                target_tp_scale=1.0, plot=False, verbose=False)
        finally:
            if _old_eo is None:
                os.environ.pop("VFC_ENS_ENSMEAN_ONLY", None)
            else:
                os.environ["VFC_ENS_ENSMEAN_ONLY"] = _old_eo
        check("ensmean-only：无逐 member rmse/谱，CRPS/spread/ensmean RMSE 不变",
              set(res_eo["rmse"].keys()) == {"ensmean"}
              and "member_000" not in res_eo["rmse"]
              and set(res_eo["spectrum"]["z500"].columns)
              <= {"ensmean_pred", "ensmean_obs", "wavelength_km"}
              and np.allclose(res_eo["crps"]["z500"].values,
                              res["crps"]["z500"].values, atol=1e-6)
              and np.allclose(res_eo["spread"]["z500"].values,
                              res["spread"]["z500"].values, atol=1e-6)
              and np.allclose(res_eo["rmse"]["ensmean"]["z500"].values,
                              res["rmse"]["ensmean"]["z500"].values,
                              rtol=0, atol=1e-4))
        # wind-group：u/v/ws200 一次读 vs 各自分开读，结果一致；谱按 var-metrics 限定
        def _run_wg(flag):
            _oldg = os.environ.get("VFC_ENS_WIND_GROUP")
            os.environ["VFC_ENS_WIND_GROUP"] = flag
            try:
                return run_ensemble(
                    pred_root, [sfc_t, pl_t], init=init,
                    metrics=["rmse", "spectrum"],
                    variables=["z500", "msl", "u200", "v200", "ws200"],
                    var_metrics={"z500": {"rmse", "spectrum"},
                                 "msl": {"rmse"},
                                 "u200": {"rmse"}, "v200": {"rmse"},
                                 "ws200": {"rmse", "spectrum"}},
                    outdir=os.path.join(tmp, "out_wg_" + flag),
                    target_tp_scale=1.0, plot=False, verbose=False)[0]
            finally:
                if _oldg is None:
                    os.environ.pop("VFC_ENS_WIND_GROUP", None)
                else:
                    os.environ["VFC_ENS_WIND_GROUP"] = _oldg
        rw1 = _run_wg("1")
        rw0 = _run_wg("0")
        check("wind-group：u/v/ws200 一次读与分开读结果一致，谱仅 z500/ws200",
              set(rw0["spectrum"].keys()) == {"z500", "ws200"}
              and set(rw1["spectrum"].keys()) == {"z500", "ws200"}
              and np.allclose(rw1["rmse"]["ensmean"]["u200"].values,
                              rw0["rmse"]["ensmean"]["u200"].values,
                              rtol=0, atol=1e-6)
              and np.allclose(rw1["rmse"]["ensmean"]["ws200"].values,
                              rw0["rmse"]["ensmean"]["ws200"].values,
                              rtol=0, atol=1e-6)
              and np.allclose(rw1["spread"]["ws200"].values,
                              rw0["spread"]["ws200"].values,
                              rtol=0, atol=1e-6)
              and np.allclose(rw1["spectrum"]["ws200"]["ensmean_pred"].values,
                              rw0["spectrum"]["ws200"]["ensmean_pred"].values,
                              rtol=0, atol=1e-9))
        # obs-block：观测逐块读 vs 整读，结果一致
        def _run_obsblk(flag):
            _old = os.environ.get("VFC_OBS_BLOCK")
            os.environ["VFC_OBS_BLOCK"] = flag
            try:
                return run_ensemble(
                    pred_root, [sfc_t, pl_t], init=init,
                    metrics=["rmse", "crps", "spectrum"],
                    variables=["z500", "msl", "u200", "v200", "ws200"],
                    var_metrics={"z500": {"rmse", "crps", "spectrum"},
                                 "msl": {"rmse"}, "u200": {"rmse"},
                                 "v200": {"rmse"}, "ws200": {"rmse", "spectrum"}},
                    outdir=os.path.join(tmp, "out_obs_" + flag),
                    target_tp_scale=1.0, plot=False, verbose=False)[0]
            finally:
                if _old is None:
                    os.environ.pop("VFC_OBS_BLOCK", None)
                else:
                    os.environ["VFC_OBS_BLOCK"] = _old
        ob1 = _run_obsblk("1")
        ob0 = _run_obsblk("0")
        check("obs 块读与整读一致（rmse/crps/谱）",
              np.allclose(ob1["rmse"]["ensmean"]["z500"].values,
                          ob0["rmse"]["ensmean"]["z500"].values,
                          rtol=0, atol=1e-6)
              and np.allclose(ob1["crps"]["z500"].values,
                              ob0["crps"]["z500"].values,
                              rtol=0, atol=1e-6)
              and np.allclose(ob1["spectrum"]["ws200"]["ensmean_pred"].values,
                              ob0["spectrum"]["ws200"]["ensmean_pred"].values,
                              rtol=0, atol=1e-9))
        check("meta target_alignment 双目标 lon_roll=半圈",
              all(v["lon_roll"] == nlon // 2
                  for v in meta["target_alignment"].values()))

        # ---- ZarrTarget 端到端（zarr 可用时） ----
        try:
            import zarr
        except ImportError:
            print("  [--] zarr 未安装，跳过 zarr 端到端（其余自检不受影响）")
        else:
            def _write_zarr(zdir, arr, names):
                zg = zarr.open(zdir, mode="w")
                zg["data"] = arr.astype("f4")
                zg["channel"] = np.array(names, dtype="<U6")
                zg["lat"] = lat
                zg["lon"] = tgt_lon
                zg["time"] = np.array([int((t - _dt.datetime(2025, 1, 1))
                                           .total_seconds() // 3600)
                                       for t in tgt_times])
                zg["time"].attrs["units"] = "hours since 2025-01-01 00:00:00"
                with open(os.path.join(zdir, "channel_names.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(names, f)

            zsfc = os.path.join(tmp, "sfc.zarr")
            zpl = os.path.join(tmp, "pl.zarr")
            _write_zarr(zsfc, np.stack([msl_obs, tp_obs], axis=1),
                        ["msl", "tp"])
            _write_zarr(zpl, np.stack([z_obs_t, u_obs_t, v_obs_t], axis=1),
                        ["z_500", "u_200", "v_200"])
            zt = ZarrTarget(zpl)
            check("ZarrTarget 读 z_500 形状 (T,lat,lon)",
                  zt.read("z_500").shape == (T, nlat, nlon))
            check("ZarrTarget time 解析正确",
                  zt.time[0] == tgt_times[0] and zt.time[-1] == tgt_times[-1])
            res2, meta2 = run_ensemble(
                pred_root, [zsfc, zpl], init=init,
                metrics=["rmse"], outdir=os.path.join(tmp, "out2"),
                target_tp_scale=1.0, tp_grid=True, plot=False, verbose=False)
            check("zarr 端到端：集合平均 RMSE(z500) == 0.1",
                  np.allclose(res2["rmse"]["ensmean"]["z500"].values, 0.1,
                              rtol=0, atol=1e-4))
            check("zarr 端到端：tp 直配 RMSE(member_000) == 0.7",
                  np.allclose(res2["rmse"]["member_000"]["tp"].values, 0.7,
                              rtol=0, atol=1e-4))

        # 6) 批量：2 个起报点 × 2 member × 3 步
        batch_root = os.path.join(tmp, "batch_root")
        os.makedirs(batch_root)
        tgt_times_b, z_tgt_b, msl_b = [], np.empty((0, nlat, nlon)), \
            np.empty((0, nlat, nlon))
        for di in (1, 2):
            base = _dt.datetime(2025, 1, di)
            for h in (6.0, 12.0, 18.0):
                tgt_times_b.append(base + _dt.timedelta(hours=h))
                z_tgt_b = np.concatenate([z_tgt_b, wave_t[None]], axis=0)
                msl_b = np.concatenate([msl_b, msl_obs[:1]], axis=0)
        sfc_b = _SimpleTarget({"msl": msl_b}, tgt_times_b, lat, tgt_lon,
                              path="sfc_b")
        pl_b = _SimpleTarget({"z500": z_tgt_b}, tgt_times_b, lat, tgt_lon,
                             path="pl_b")
        for di, dd in enumerate(("20250101", "20250102")):
            date_dir = os.path.join(batch_root, dd)
            os.makedirs(date_dir)
            for mi, off in offs.items():
                md = os.path.join(date_dir, mi)
                os.makedirs(md)
                for k in range(3):
                    _write_step_nc(
                        os.path.join(md, "%03d.nc" % (k + 1)),
                        {"Z500": wave_p + off,
                         "MSL": msl_obs[0] + 100.0 * off}, lat, lon)
        bdf, bmeta = run_batch(
            batch_root, [sfc_b, pl_b], os.path.join(tmp, "batch_out"),
            dates=["20250101", "20250102"],
            metrics=["rmse", "crps"], target_tp_scale=1.0,
            plot=False, verbose=False)
        check("批量：2 个起报点均成功",
              len(bdf) == 2 and bmeta["n_ok"] == 2 and not bmeta["failures"])
        check("批量：各点 ensmean RMSE(z500)_mean == 0.1",
              np.allclose(bdf["rmse_z500_mean"].values, 0.1,
                          rtol=0, atol=1e-4))
        check("批量：各点 CRPS(z500)_mean == 0.3",
              np.allclose(bdf["crps_z500_mean"].values, 0.3,
                          rtol=0, atol=1e-4))
        check("批量：summary 含 24h/末时效 RMSE 列",
              {"rmse_z500_24h", "rmse_z500_last"} <= set(bdf.columns))
        mfp = os.path.join(tmp, "batch_out", "mean_rmse.csv")
        check("批量平均：mean_rmse.csv 生成且 z500 == 0.1",
              os.path.exists(mfp)
              and np.allclose(pd.read_csv(mfp, index_col=0)["z500"].values,
                              0.1, rtol=0, atol=1e-6))
        # 批量共享气候态（acc/fa）：跨起报点只开一次，mean_acc/fa 正常
        climo_nc2 = os.path.join(tmp, "climo_shared.nc")
        import h5py as _h5
        with _h5.File(climo_nc2, "w") as h:
            h["lat"] = lat
            h["lon"] = tgt_lon
            h["time"] = np.arange(365, dtype="f8")
            h["time"].attrs["units"] = "days since 2000-01-01"
            h["z500"] = np.zeros((365, nlat, nlon), dtype="f4")
            h["msl"] = np.zeros((365, nlat, nlon), dtype="f4")
        cbdf, cbmeta = run_batch(
            batch_root, [sfc_b, pl_b], os.path.join(tmp, "batch_out2"),
            dates=["20250101", "20250102"],
            metrics=["rmse", "acc", "fa"], climo_path=climo_nc2,
            climo_window=1, target_tp_scale=1.0, plot=False, verbose=False)
        check("批量共享气候态：mean_acc/mean_fa 生成且 2 点成功",
              os.path.exists(os.path.join(tmp, "batch_out2", "mean_acc.csv"))
              and os.path.exists(os.path.join(tmp, "batch_out2",
                                             "mean_fa.csv"))
              and cbmeta["n_ok"] == 2)
        # 并行批量：2 进程跑 2 点 → summary/mean 合并正确
        pdf, pmeta = run_batch_parallel(
            batch_root, [sfc_b, pl_b], os.path.join(tmp, "batch_par"),
            dates=["20250101", "20250102"], n_workers=2,
            metrics=["rmse", "crps"], target_tp_scale=1.0,
            plot=False, verbose=False)
        check("并行批量：2 进程合并后 2 点成功 + mean_rmse z500==0.1",
              len(pdf) == 2 and pmeta["n_ok"] == 2
              and os.path.exists(os.path.join(tmp, "batch_par",
                                              "mean_rmse.csv"))
              and np.allclose(pd.read_csv(
                  os.path.join(tmp, "batch_par", "mean_rmse.csv"),
                  index_col=0)["z500"].values, 0.1, rtol=0, atol=1e-6))

        # 2b) AIFS 布局：日期目录为 YYYY-MM-DD，批量可识别
        dash_root = os.path.join(tmp, "dash_root")
        os.makedirs(dash_root)
        dd = "2025-01-03"
        dd_dir = os.path.join(dash_root, dd)
        os.makedirs(dd_dir)
        for mi, off in offs.items():
            md = os.path.join(dd_dir, mi)
            os.makedirs(md)
            for k in range(3):
                _write_step_nc(os.path.join(md, "%03d.nc" % (k + 1)),
                               {"Z500": wave_p + off,
                                "MSL": msl_obs[0] + 100.0 * off}, lat, lon)
        dash_times = [_dt.datetime(2025, 1, 3) + _dt.timedelta(hours=h)
                      for h in (6, 12, 18)]
        dash_sfc = _SimpleTarget(
            {"msl": np.broadcast_to(msl_obs[:1], (3, nlat, nlon)).copy()},
            dash_times, lat, tgt_lon, path="dash_sfc")
        dash_pl = _SimpleTarget(
            {"z500": np.broadcast_to(wave_t, (3, nlat, nlon)).copy()},
            dash_times, lat, tgt_lon, path="dash_pl")
        ddf, _ = run_batch(dash_root, [dash_sfc, dash_pl],
                           os.path.join(tmp, "dash_out"), dates=[dd],
                           metrics=["rmse"], target_tp_scale=1.0,
                           plot=False, verbose=False)
        check("批量支持 YYYY-MM-DD（AIFS 布局）：1 点且 RMSE(z500)_mean==0.1",
              len(ddf) == 1 and ddf.iloc[0]["init_date"] == "2025-01-03"
              and np.allclose(ddf["rmse_z500_mean"].values, 0.1,
                              rtol=0, atol=1e-4))

        # 7) 盘古（Pangu-Weather）确定性预报：升序 lat 对齐 + 变量映射
        from vfc.pangu_io import PanguForecast, scan_pangu_dates
        pg_root = os.path.join(tmp, "pangu")
        os.makedirs(pg_root)
        p_lat = np.arange(-90.0, 90.01, 30.0)          # 升序（盘古）
        t_lat_desc = p_lat[::-1]                       # 目标 lat 降序
        A = 5.0

        def pfield(lat2d, lon2d):
            lon360 = np.mod(lon2d, 360.0)
            return A * np.sin(3 * 2 * np.pi * lon360 / 360.0) \
                * (1 + 0.5 * np.cos(2 * np.pi * lat2d / 180.0))

        def _write_pangu_step(path, fields, glat, glon):
            import h5py
            with h5py.File(path, "w") as h:
                h["lat"] = glat
                h["lon"] = glon
                for k, v in fields.items():
                    h[k] = np.asarray(v, dtype="f4")[None, None]

        def _pangu_name(kind, date, lead):
            return ("Pangu_GLB_%s_OP25_6HOR_FCST_%s00_%03d.nc"
                    % (kind, date, lead))

        def _mk_pangu_date(date_dir, date, z_obs2, msl_obs2, tp_obs2):
            os.makedirs(date_dir, exist_ok=True)
            for lead in (6, 12):
                _write_pangu_step(
                    os.path.join(date_dir, _pangu_name("SURFACE", date, lead)),
                    {"MSL": msl_obs2 + 70.0,
                     "T2M": np.full((7, 12), 280.7),
                     "TP": tp_obs2 + 0.7}, p_lat, np.arange(0, 360, 30))
                _write_pangu_step(
                    os.path.join(date_dir, _pangu_name("PLEVELS", date, lead)),
                    {"Z500": z_obs2 + 0.7}, p_lat, np.arange(0, 360, 30))

        LONP, LATP = np.meshgrid(np.arange(0, 360, 30), p_lat)
        z_pg = pfield(LATP, LONP)                       # (7,12)
        LONT, LATT = np.meshgrid(np.arange(-180, 180, 30), t_lat_desc)
        z_tg = pfield(LATT, LONT)
        pg_times = [_dt.datetime(2026, 8, 24, 0) + _dt.timedelta(hours=h)
                    for h in (6, 12)]
        pg_sfc = _SimpleTarget(
            {"msl": np.broadcast_to(np.full((7, 12), 100000.0),
                                    (2, 7, 12)).copy(),
             "t2m": np.broadcast_to(np.full((7, 12), 280.0),
                                    (2, 7, 12)).copy(),
             "tp": np.full((2, 7, 12), 0.5)},
            pg_times, t_lat_desc, np.arange(-180, 180, 30), path="pg_sfc")
        pg_pl = _SimpleTarget({"z500": np.broadcast_to(z_tg, (2, 7, 12)).copy()},
                              pg_times, t_lat_desc,
                              np.arange(-180, 180, 30), path="pg_pl")
        _mk_pangu_date(os.path.join(pg_root, "20260824"), "20260824",
                       z_pg, np.full((7, 12), 100000.0), np.full((7, 12), 0.5))
        pf = PanguForecast(os.path.join(pg_root, "20260824"))
        pg_res, pg_meta = run_ensemble(
            pf, [pg_sfc, pg_pl], metrics=["rmse", "crps"],
            outdir=os.path.join(tmp, "pg_out"), target_tp_scale=1.0,
            tp_grid=True, plot=False, verbose=False)
        check("盘古：要素交集 msl/t2m/tp/z500（u10/v10→u10m/v10m 不在测试集）",
              set(pg_res["rmse"]["det"].columns) == {"msl", "t2m", "tp", "z500"})
        check("盘古：起报时刻从文件名解析 == 2026-08-24T00:00",
              pg_meta["init"] == "2026-08-24T00:00:00")
        check("盘古：lead_hours = [6,12]",
              np.allclose(pg_meta["lead_hours"], [6.0, 12.0]))
        check("盘古：lat 升序→目标降序，lat_flip=True",
              all(v["lat_flip"] for v in pg_meta["target_alignment"].values()))
        check("盘古：rmse z500 == 0.7 / msl == 70 / tp == 0.7 / t2m == 0.7",
              np.allclose(pg_res["rmse"]["det"]["z500"].values, 0.7, atol=1e-4)
              and np.allclose(pg_res["rmse"]["det"]["msl"].values, 70.0,
                              atol=1e-4)
              and np.allclose(pg_res["rmse"]["det"]["tp"].values, 0.7,
                              atol=1e-4)
              and np.allclose(pg_res["rmse"]["det"]["t2m"].values, 0.7,
                              atol=1e-4))
        check("盘古：CRPS(z500) == 0.7（单成员→MAE）",
              np.allclose(pg_res["crps"]["z500"].values, 0.7, atol=1e-4))
        pg2, _ = run_ensemble(
            pf, [pg_sfc, pg_pl], metrics=["rmse"],
            outdir=os.path.join(tmp, "pg_out2"), target_tp_scale=1.0,
            pred_tp_scale=2.0, tp_grid=True, plot=False, verbose=False)
        check("pred-tp-scale：tp 预报×2（1.2→2.4）→ RMSE == 1.9",
              np.allclose(pg2["rmse"]["det"]["tp"].values, 1.9,
                          rtol=0, atol=1e-4))

        # 8) 盘古批量：2 个起报点
        _mk_pangu_date(os.path.join(pg_root, "20260825"), "20260825",
                       z_pg, np.full((7, 12), 100000.0), np.full((7, 12), 0.5))
        pg_times4 = [(_dt.datetime(2026, 8, 24 + d, 0)
                      + _dt.timedelta(hours=h))
                     for d in (0, 1) for h in (6, 12)]
        pg_sfc4 = _SimpleTarget(
            {"msl": np.broadcast_to(np.full((7, 12), 100000.0),
                                    (4, 7, 12)).copy(),
             "tp": np.full((4, 7, 12), 0.5)},
            pg_times4, t_lat_desc, np.arange(-180, 180, 30), path="pg_sfc4")
        pg_pl4 = _SimpleTarget(
            {"z500": np.broadcast_to(z_tg, (4, 7, 12)).copy()},
            pg_times4, t_lat_desc, np.arange(-180, 180, 30), path="pg_pl4")
        pg_bdf, pg_bmeta = run_pangu_batch(
            pg_root, [pg_sfc4, pg_pl4], os.path.join(tmp, "pg_batch"),
            dates=["20260824", "20260825"],
            metrics=["rmse", "crps"], target_tp_scale=1.0, tp_grid=True,
            plot=False, verbose=False)
        check("盘古批量：2 点成功，scan_pangu_dates 识别 2 点",
              len(pg_bdf) == 2 and pg_bmeta["n_ok"] == 2
              and scan_pangu_dates(pg_root) == ["20260824", "20260825"])
        check("盘古批量：rmse_z500_mean == 0.7",
              np.allclose(pg_bdf["rmse_z500_mean"].values, 0.7,
                          rtol=0, atol=1e-4))

        # 9) 嵌套布局：根/20250106/member_xxx 自动识别（对应数据被套一层）
        from vfc.ensemble_io import EnsembleForecast as _EF
        nested_root = os.path.join(tmp, "nested", "20250106")
        os.makedirs(nested_root)
        for mi, off in offs.items():
            md = os.path.join(nested_root, mi)
            os.makedirs(md)
            for k in range(3):
                _write_step_nc(os.path.join(md, "%03d.nc" % (k + 1)),
                               {"Z500": wave_p + off}, lat, lon)
        nef = _EF(os.path.join(tmp, "nested"))
        check("嵌套布局自动识别（member 相对路径 + 读取形状）",
              nef.members == ["20250106/member_000", "20250106/member_001"]
              and nef.read("z500").shape == (2, 3, nlat, nlon))
        nres, nmeta = run_ensemble(
            os.path.join(tmp, "nested"), [sfc_b, pl_b], init="2025-01-01T00:00",
            metrics=["rmse"], outdir=os.path.join(tmp, "nested_out"),
            target_tp_scale=1.0, plot=False, verbose=False)
        check("嵌套布局端到端：ensmean RMSE(z500) == 0.1",
              np.allclose(nres["rmse"]["ensmean"]["z500"].values, 0.1,
                          rtol=0, atol=1e-4))

        # 10) 新格式气候态（lat/lon/time + 独立变量）读取 + acc/fa 端到端
        climo_nc = os.path.join(tmp, "climo_newfmt.nc")
        import h5py
        with h5py.File(climo_nc, "w") as h:
            h["lat"] = p_lat[::-1]                 # 降序（测自动翻转对齐）
            h["lon"] = np.arange(0, 360, 30)
            h["time"] = np.arange(365, dtype="f8")
            h["time"].attrs["units"] = "days since 2000-01-01"
            h["z500"] = np.zeros((365, nlat, nlon), dtype="f4")
            h["msl"] = np.zeros((365, nlat, nlon), dtype="f4")
        from vfc.climo import DailyClimatology as _DC
        _dc = _DC(climo_nc, window=1)
        check("新格式气候态：levels/步长/field_for 正确",
              _dc.levels == ["msl", "z500"] and _dc.steps_per_day == 1
              and np.allclose(_dc.field_for("z500", [6.0, 12.0],
                                            _dt.datetime(2025, 1, 1)), 0.0))
        _dc.close()
        cres, cmeta = run_ensemble(
            pf, [pg_sfc, pg_pl], metrics=["acc", "fa"],
            climo_path=climo_nc, climo_window=1,
            outdir=os.path.join(tmp, "climo_out"), target_tp_scale=1.0,
            tp_grid=True, plot=False, verbose=False)
        check("新格式气候态 acc/fa 端到端（列齐全 + 气候态翻转对齐）",
              "z500" in cres["acc"]["det"].columns
              and "z500_pred" in cres["fa"]["det"].columns
              and "z500_bias" in cres["fa"]["det"].columns
              and "z500_ratio" in cres["fa"]["det"].columns
              and np.allclose(cres["fa"]["det"]["z500_bias"].values,
                              cres["fa"]["det"]["z500_pred"].values
                              - cres["fa"]["det"]["z500_obs"].values,
                              atol=1e-6)
              and cmeta["climatology"]["lat_flip"] is True)

        # 11) 366 天（含闰日）气候态：非闰年 3/1 跳过 2/29 槽、闰年 2/29 命中
        climo366 = os.path.join(tmp, "climo366.nc")
        with h5py.File(climo366, "w") as h:
            h["lat"] = p_lat
            h["lon"] = np.arange(0, 360, 30)
            h["time"] = np.arange(366 * 4, dtype="f8") * 6.0
            h["time"].attrs["units"] = "hours since 2000-01-01"
            vals = (np.arange(366 * 4) // 4).astype("f4")   # 值=日序号
            h["z500"] = np.broadcast_to(
                vals[:, None, None], (366 * 4, nlat, nlon)).copy()
        _dc2 = _DC(climo366, window=1)
        f60 = _dc2.field_for("z500", [6.0], _dt.datetime(2025, 3, 1, 0))
        f59 = _dc2.field_for("z500", [0.0], _dt.datetime(2024, 2, 29, 0))
        check("366 天气候态：非闰年 3/1 → 日序号 60；闰年 2/29 → 59",
              _dc2.steps_per_day == 4 and _dc2.n_days == 366
              and np.allclose(f60[0, 0, 0], 60.0)
              and np.allclose(f59[0, 0, 0], 59.0))
        _dc2.close()

        # 12) 单确定性输出：root 直接放 NN.nc（fuxi_single_output/<日期>/001.nc…）
        single_root = os.path.join(tmp, "single")
        os.makedirs(single_root)
        for k in range(3):
            _write_step_nc(os.path.join(single_root, "%03d.nc" % (k + 1)),
                           {"Z500": wave_p + 0.7, "MSL": msl_obs[0] + 70.0},
                           lat, lon)
        sef = _EF(single_root)
        check("单确定性：root 直接 NN.nc → 单成员 det + 读取形状",
              sef.members == ["det"]
              and sef.read("z500").shape == (1, 3, nlat, nlon))
        sres, smeta = run_ensemble(
            single_root, [sfc_b, pl_b], init="2025-01-01T00:00",
            metrics=["rmse", "crps"], outdir=os.path.join(tmp, "single_out"),
            target_tp_scale=1.0, plot=False, verbose=False)
        check("单确定性端到端：det RMSE(z500) == 0.7 / CRPS == 0.7",
              np.allclose(sres["rmse"]["det"]["z500"].values, 0.7,
                          rtol=0, atol=1e-4)
              and np.allclose(sres["crps"]["z500"].values, 0.7,
                              rtol=0, atol=1e-4))
        # 12a) single 块流式（VFC_SINGLE_STREAM 默认开）与旧整读结果一致
        def _run_single_st(flag):
            _old = os.environ.get("VFC_SINGLE_STREAM")
            os.environ["VFC_SINGLE_STREAM"] = flag
            try:
                return run_ensemble(
                    single_root, [sfc_b, pl_b], init="2025-01-01T00:00",
                    metrics=["rmse", "spectrum"],
                    outdir=os.path.join(tmp, "single_st_" + flag),
                    target_tp_scale=1.0, plot=False, verbose=False)[0]
            finally:
                if _old is None:
                    os.environ.pop("VFC_SINGLE_STREAM", None)
                else:
                    os.environ["VFC_SINGLE_STREAM"] = _old
        ss1 = _run_single_st("1")
        ss0 = _run_single_st("0")
        check("single 块流式与整读一致（rmse/谱，det）",
              np.allclose(ss1["rmse"]["det"]["z500"].values,
                          ss0["rmse"]["det"]["z500"].values, rtol=0, atol=1e-6)
              and np.allclose(ss1["rmse"]["det"]["msl"].values,
                              ss0["rmse"]["det"]["msl"].values, rtol=0, atol=1e-6)
              and np.allclose(ss1["spectrum"]["z500"]["det_pred"].values,
                              ss0["spectrum"]["z500"]["det_pred"].values,
                              rtol=0, atol=1e-9))

        # 12b) q700 单位缩放：预报按 g/kg 存、真值 ERA5 为 kg/kg → pred_q_scale=0.001
        q_obs2 = 0.004 + 0.0002 * wave_t / A       # kg/kg（~4 g/kg，非均匀，真值在 tgt_lon）
        q_pred_kgkg = 0.004 + 0.0002 * field_on(lon, A, 3) / A   # 预报场在自身 lon 网格
        q_pred_gkg = q_pred_kgkg * 1000.0 + 0.7    # 预报文件按 g/kg（误差 0.7 g/kg）
        q_root = os.path.join(tmp, "single_q")
        os.makedirs(q_root)
        q_times = [_dt.datetime(2025, 1, 1) + _dt.timedelta(hours=h)
                   for h in (6.0, 12.0, 18.0)]
        for k in range(3):
            _write_step_nc(os.path.join(q_root, "%03d.nc" % (k + 1)),
                           {"Q700": q_pred_gkg}, lat, lon)
        q_pl = _SimpleTarget(
            {"q700": np.broadcast_to(q_obs2, (3, nlat, nlon)).copy()},
            q_times, lat, tgt_lon, path="q_pl")
        q_raw, _ = run_ensemble(
            q_root, [q_pl], init="2025-01-01T00:00", metrics=["rmse"],
            outdir=os.path.join(tmp, "single_q_out"),
            plot=False, verbose=False)
        check("q700 默认不缩放（g/kg vs kg/kg）→ RMSE 量级≈g/kg 值（>1）",
              float(q_raw["rmse"]["det"]["q700"].values[0]) > 1.0)
        q_ok, q_ok_meta = run_ensemble(
            q_root, [q_pl], init="2025-01-01T00:00", metrics=["rmse"],
            outdir=os.path.join(tmp, "single_q_out2"),
            pred_q_scale=0.001, plot=False, verbose=False)
        check("pred-q-scale：g/kg→kg/kg 换算后按 g/kg 报告 → RMSE == 0.7",
              np.allclose(q_ok["rmse"]["det"]["q700"].values, 0.7,
                          rtol=0, atol=1e-6))
        check("q 缩放记入 meta.q（pred_scale_to_kgkg == 0.001）",
              q_ok_meta["q"]["pred_scale_to_kgkg"] == 0.001
              and q_ok_meta["q"]["target_scale_to_kgkg"] == 1.0)
        # 13) AIFS（output/target 同构，channel 轴格式）
        from vfc.aifs_io import AifsForecast, scan_aifs_dates

        def _write_aifs(path, var_name, arr, chans, glat, glon):
            import h5py as _h5
            with _h5.File(path, "w") as h:
                h["lat"] = glat
                h["lon"] = glon
                h["time"] = np.asarray([0], dtype="i8")
                h["time"].attrs["units"] = "days since 2000-01-01"
                h["step"] = np.asarray([1], dtype="i8")
                h["channel"] = np.array(chans, dtype=_h5.string_dtype())
                h[var_name] = np.asarray(arr, dtype="f4")[None, None]

        a_lat = np.arange(90.0, -90.01, -30.0)        # 降序（同真实 AIFS）
        a_lon = np.arange(0, 360, 30)
        wave_a = A * np.sin(3 * 2 * np.pi * a_lon / 360.0) + np.zeros((7, 1))
        wave_a = np.broadcast_to(wave_a, (7, 12)).copy()
        aifs_root = os.path.join(tmp, "aifs")
        for di, dd in enumerate(("2025-01-02", "2025-01-03")):
            pdir = os.path.join(aifs_root, "pred", dd)
            tdir = os.path.join(aifs_root, "target", dd)
            os.makedirs(pdir)
            os.makedirs(tdir)
            for k in range(3):
                tgt = np.stack([wave_a + di * 0.01,
                                np.full((7, 12), 100000.0),
                                np.full((7, 12), 0.5)], axis=0)
                out = np.stack([wave_a + di * 0.01 + 0.7,
                                np.full((7, 12), 100070.0),
                                np.full((7, 12), 1.2)], axis=0)
                _write_aifs(os.path.join(pdir, "%03d.nc" % (k + 1)),
                            "output", out, ["z500", "msl", "tp"], a_lat,
                            a_lon)
                _write_aifs(os.path.join(tdir, "%03d.nc" % (k + 1)),
                            "target", tgt, ["z500", "msl", "tp"], a_lat,
                            a_lon)
        apf = AifsForecast(os.path.join(aifs_root, "pred", "2025-01-02"),
                           var_name="output")
        atf = AifsForecast(os.path.join(aifs_root, "target", "2025-01-02"),
                           var_name="target", as_target=True)
        check("AIFS 读取：levels/lead/网格/起报",
              apf.levels == ["msl", "tp", "z500"]
              and np.allclose(apf.lead_hours, [6.0, 12.0, 18.0])
              and apf.lat[0] == 90.0 and apf.lat[-1] == -90.0
              and apf.lon[0] == 0.0
              and apf.init_date.strftime("%Y-%m-%d") == "2025-01-02")
        ares, ameta = run_ensemble(
            apf, [atf], metrics=["rmse", "crps"], target_tp_scale=1.0,
            pred_tp_scale=1.0, tp_grid=True, plot=False, verbose=False,
            outdir=os.path.join(tmp, "aifs_out"))
        check("AIFS 端到端：det RMSE(z500)=0.7 / msl=70 / tp=0.7 / CRPS=0.7",
              np.allclose(ares["rmse"]["det"]["z500"].values, 0.7, atol=1e-4)
              and np.allclose(ares["rmse"]["det"]["msl"].values, 70.0,
                              atol=1e-4)
              and np.allclose(ares["rmse"]["det"]["tp"].values, 0.7,
                              atol=1e-4)
              and np.allclose(ares["crps"]["z500"].values, 0.7, atol=1e-4))
        abdf, abmeta = run_aifs_batch(
            os.path.join(aifs_root, "pred"), os.path.join(aifs_root, "target"),
            os.path.join(tmp, "aifs_batch"),
            dates=["2025-01-02", "2025-01-03"],
            metrics=["rmse"], target_tp_scale=1.0, plot=False, verbose=False)
        check("AIFS 批量：2 点成功且 rmse_z500_mean==0.7",
              len(abdf) == 2 and abmeta["n_ok"] == 2
              and np.allclose(abdf["rmse_z500_mean"].values, 0.7,
                              rtol=0, atol=1e-4))
        # 13b) AIFS 并行批量 + resume（与 FuXi 同款 .parts 机制）
        _abp = os.path.join(tmp, "aifs_batch_p")
        _saved_dpc = os.environ.get("VFC_DATES_PER_CHILD")
        os.environ["VFC_DATES_PER_CHILD"] = "1"   # 2 日期 → 2 块，n=2 真起 2 进程
        _abp1, _abm1 = run_aifs_batch_parallel(
            os.path.join(aifs_root, "pred"), os.path.join(aifs_root, "target"),
            _abp, n_workers=2, dates=["2025-01-02", "2025-01-03"],
            metrics=["rmse"], target_tp_scale=1.0, plot=False, verbose=False)
        if _saved_dpc is None:
            os.environ.pop("VFC_DATES_PER_CHILD", None)
        else:
            os.environ["VFC_DATES_PER_CHILD"] = _saved_dpc
        check("AIFS 并行批量：n=2 两日期成功且 rmse_z500_mean==0.7",
              len(_abp1) == 2 and _abm1["n_ok"] == 2
              and _abm1["n_workers"] == 2
              and np.allclose(_abp1["rmse_z500_mean"].values, 0.7,
                              rtol=0, atol=1e-4))
        # resume：同一 outdir-root 换 n=1 再跑，已完成日期应跳过、结果不变
        _abp2, _abm2 = run_aifs_batch_parallel(
            os.path.join(aifs_root, "pred"), os.path.join(aifs_root, "target"),
            _abp, n_workers=1, resume=True, dates=["2025-01-02", "2025-01-03"],
            metrics=["rmse"], target_tp_scale=1.0, plot=False, verbose=False)
        check("AIFS resume：换 n=1 续跑仍 2 点成功且 .parts 已清空",
              len(_abp2) == 2 and _abm2["n_ok"] == 2
              and np.allclose(_abp2["rmse_z500_mean"].values, 0.7,
                              rtol=0, atol=1e-4)
              and os.path.isdir(_abp)
              and not os.path.isdir(os.path.join(_abp, ".parts")))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n自检结果: %d/%d 通过" % (ok[0], ok[1]))
    return 0 if ok[0] == ok[1] else 1


def ens_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="集合预报 vs ERA5 zarr 检验（RMSE/ACC/FA/功率谱；可多 zarr）")
    ap.add_argument("--pred-dir",
                    help="预报目录：member_xxx/NN.nc（FuXi 集合）或根下直接 NN.nc（单确定性 det，如 fuxi_single_output）")
    ap.add_argument("--pred-root", help="多个起报点的父目录（含多个 YYYYMMDD，"
                    "每个下面有 member_xxx/NN.nc）——批量模式")
    ap.add_argument("--dates", nargs="*", default=None,
                    help="批量模式只跑这些起报点 YYYYMMDD（默认全部）")
    ap.add_argument("--outdir-root", default=None,
                    help="批量模式输出根目录（默认同 --outdir；每点写 <根>/<YYYYMMDD>/）")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="起报点并行进程数（>1 时把日期分 N 份并行跑，最后合并 summary/mean）")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：跳过已完整(有 <日期>_meta.json)的日期，读回结果进 summary/mean")
    ap.add_argument("--pangu-dir", default=None,
                    help="盘古单个起报点目录（如 .../DF/20260824，确定性单成员）")
    ap.add_argument("--pangu-root", default=None,
                    help="盘古 DF 根目录（含多个 YYYYMMDD）——盘古批量模式")
    ap.add_argument("--aifs-pred-dir", default=None,
                    help="AIFS 预测单个起报点目录（如 .../2025-01-02）")
    ap.add_argument("--aifs-target-dir", default=None,
                    help="AIFS 真值单个起报点目录（与预测日期对应）")
    ap.add_argument("--aifs-pred-root", default=None,
                    help="AIFS 预测根目录（含多个 YYYY-MM-DD）——批量模式")
    ap.add_argument("--aifs-target-root", default=None,
                    help="AIFS 真值根目录（与预测日期对应）")
    ap.add_argument("--target-zarr", nargs="+", default=None,
                    help="ERA5 zarr 路径（可多个：sfc + pl，按要素自动归属；AIFS 模式不需要）")
    ap.add_argument("--init", default=None,
                    help="起报时刻，如 2025-01-06T00:00；缺省从目录名（如 20250106）解析")
    ap.add_argument("--members", nargs="*", default=None,
                    help="只评这些 member（默认全部）")
    ap.add_argument("--first-lead", type=float, default=6.0,
                    help="首时效小时（默认 6，即 001.nc=+6h）")
    ap.add_argument("--lead-step", type=float, default=6.0,
                    help="时效步长小时（默认 6）")
    ap.add_argument("-v", "--vars", nargs="*", default=None,
                    help="待评要素（缺省=预报∩目标）")
    ap.add_argument("--metrics", nargs="+", default=["rmse"],
                    choices=list(_METRICS), help="指标（默认仅 rmse）")
    ap.add_argument("--var-metrics", action="append", default=None,
                    metavar="VAR:M1,M2",
                    help="逐要素指标白名单（single/确定性 det 用），可重复，如 "
                         "'z500:acc,fa,rmse,spectrum'（POD=spectrum 功率谱）"
                         " 't850:rmse'；未列出的要素仍按 --metrics 全算")
    ap.add_argument("--outdir", default="results/ens")
    ap.add_argument("--name", default=None)
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "netcdf4", "h5py"])
    ap.add_argument("--bbox", nargs=4, type=float,
                    metavar=("LAT0", "LAT1", "LON0", "LON1"), default=None)
    ap.add_argument("--no-lat-weight", action="store_true")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--target-denorm", action="store_true",
                    help="zarr data 若为归一化值则按 mean/std 反算（本批实测已是物理量，默认关）")
    ap.add_argument("--target-tp-exp", action="store_true",
                    help="zarr tp 若为 log1p 存储则 expm1 还原（默认关）")
    ap.add_argument("--target-tp-scale", type=float, default=1000.0,
                    help="真值 tp 换算到 mm 的系数（zarr 为 m → 默认 1000）")
    ap.add_argument("--pred-tp-scale", type=float, default=1.0,
                    help="预报 tp 换算到 mm 的系数（AIFS 输出为 m → 用 1000）")
    ap.add_argument("--target-q-scale", type=float, default=1.0,
                    help="真值比湿（q700/q2m）换算到 kg/kg 的系数（ERA5 已为 kg/kg → 默认 1.0）")
    ap.add_argument("--pred-q-scale", type=float, default=1.0,
                    help="预报比湿（q700/q2m）换算到 kg/kg 的系数（若预报按 g/kg 输出 → 用 0.001）")
    ap.add_argument("--target-time-units", default=None,
                    help="覆盖 zarr time:units（默认取数组 attrs）")
    ap.add_argument("--tp-grid", action="store_true",
                    help="把 tp 也纳入网格指标（默认排除：tp 正式评分为站点 TS/BIAS）")
    ap.add_argument("--brier-thresholds", nargs="+", default=None,
                    metavar="VAR:V1[,V2,...]",
                    help="Brier 事件阈值，如 'z500:100' 'tp:0.1,10,25' "
                         "'msl:1000'（事件=场值≥阈值；需 --metrics brier）")
    ap.add_argument("--climo", default=None,
                    help="逐日气候态 nc（acc/fa 必需）")
    ap.add_argument("--climo-source", default="未标注")
    ap.add_argument("--climo-window", type=int, default=15)
    ap.add_argument("--acc-centered", action="store_true")
    ap.add_argument("--spec-leads", nargs=2, type=float,
                    metavar=("H0", "H1"), default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.target_zarr and not (args.aifs_pred_dir or args.aifs_pred_root):
        ap.error("需要 --target-zarr（可多个；AIFS 模式真值走 --aifs-target-*，不需要 zarr）")
    if not (args.pred_dir or args.pred_root or args.pangu_dir
            or args.pangu_root or args.aifs_pred_dir or args.aifs_pred_root):
        ap.error("需要 --pred-dir / --pred-root / --pangu-dir / --pangu-root /"
                 " --aifs-pred-dir / --aifs-pred-root（或 --self-test）")

    brier_thresholds = {}
    if args.brier_thresholds:
        for item in args.brier_thresholds:
            sep = ":" if ":" in item else ("=" if "=" in item else None)
            if sep is None:
                ap.error("--brier-thresholds 项 %r 需为 var:thr,thr... 格式" % item)
            var, vals = item.split(sep, 1)
            var = var.strip().lower()
            brier_thresholds[var] = [float(x) for x in vals.split(",") if x.strip()]

    var_metrics = {}
    if args.var_metrics:
        for item in args.var_metrics:
            if ":" not in item:
                ap.error("--var-metrics 项 %r 需为 var:rmse,acc,fa,spectrum 格式" % item)
            var0, ms = item.split(":", 1)
            var_metrics[var0.strip().lower()] = sorted(
                {x.strip().lower() for x in ms.split(",") if x.strip()})

    kw = dict(
        members=args.members, first_lead=args.first_lead,
        lead_step=args.lead_step, variables=args.vars,
        metrics=args.metrics, engine=args.engine,
        lat_weighted=not args.no_lat_weight,
        bbox=tuple(args.bbox) if args.bbox else None,
        block=args.block, plot=not args.no_plot,
        target_denorm=args.target_denorm,
        target_tp_exp=args.target_tp_exp,
        target_tp_scale=args.target_tp_scale,
        pred_tp_scale=args.pred_tp_scale,
        target_q_scale=args.target_q_scale,
        pred_q_scale=args.pred_q_scale,
        target_time_units=args.target_time_units,
        tp_grid=args.tp_grid,
        brier_thresholds=brier_thresholds or None,
        var_metrics=var_metrics or None,
        climo_path=args.climo, climo_source=args.climo_source,
        climo_window=args.climo_window,
        acc_centered=args.acc_centered,
        spec_lead_range=tuple(args.spec_leads) if args.spec_leads else None,
    )

    if args.aifs_pred_root:
        if not args.aifs_target_root:
            ap.error("--aifs-pred-root 需要 --aifs-target-root")
        if args.n_workers and args.n_workers > 1:
            run_aifs_batch_parallel(args.aifs_pred_root,
                                    args.aifs_target_root,
                                    args.outdir_root or args.outdir,
                                    dates=args.dates,
                                    n_workers=args.n_workers,
                                    resume=bool(args.resume), **kw)
        else:
            run_aifs_batch(args.aifs_pred_root, args.aifs_target_root,
                           args.outdir_root or args.outdir, dates=args.dates,
                           resume=bool(args.resume), **kw)
    elif args.aifs_pred_dir:
        if not args.aifs_target_dir:
            ap.error("--aifs-pred-dir 需要 --aifs-target-dir")
        run_ensemble(
            AifsForecast(args.aifs_pred_dir, var_name="output",
                         engine=args.engine),
            [AifsForecast(args.aifs_target_dir, var_name="target",
                          engine=args.engine, as_target=True)],
            init=args.init, outdir=args.outdir, name=args.name, **kw)
    elif args.pangu_root:
        run_pangu_batch(args.pangu_root, args.target_zarr,
                        args.outdir_root or args.outdir, dates=args.dates,
                        **kw)
    elif args.pangu_dir:
        run_ensemble(PanguForecast(args.pangu_dir, engine=args.engine),
                     args.target_zarr, init=args.init, outdir=args.outdir,
                     name=args.name, **kw)
    elif args.pred_root:
        if args.n_workers and args.n_workers > 1:
            run_batch_parallel(args.pred_root, args.target_zarr,
                               args.outdir_root or args.outdir,
                               dates=args.dates, n_workers=args.n_workers,
                               resume=bool(args.resume), **kw)
        else:
            run_batch(args.pred_root, args.target_zarr,
                      args.outdir_root or args.outdir, dates=args.dates,
                      resume=bool(args.resume), **kw)
    else:
        run_ensemble(args.pred_dir, args.target_zarr, init=args.init,
                     outdir=args.outdir, name=args.name, **kw)
    return 0

