# -*- coding: utf-8 -*-
"""盘古（Pangu-Weather）确定性预报读取层。

目录布局（DF/<YYYYMMDD>/）：
  Pangu_GLB_SURFACE_OP25_6HOR_FCST_<YYYYMMDD>00_<lead3>.nc    面要素
  Pangu_GLB_PLEVELS_OP25_6HOR_FCST_<YYYYMMDD>00_<lead3>.nc    气压层要素

每文件 = 一个时效步；变量为 2D 场（shape (1,1,lat,lon)）：
  面：MSL(Pa) / T2M(K) / TP(mm，逐 6h 时段量，非自起报累积) / U10 / V10 / Q2M
  气压层：Z500 / T850 / U850 / V850 / Q700（变量名内嵌层次）
网格：lat -90→90（升序，南→北），lon 0→359.75，0.25°；
time 单位 "days since 2025-01-01"（起报时刻），dtime = 时效小时（与文件名一致）。

对外接口与 EnsembleForecast 一致（levels / lat / lon / lead_hours /
members / read），可被 run_ensemble 直接消费（确定性单成员 → members=["det"]）。
u10/v10 映射为 u10m/v10m 以匹配 ERA5 面通道名。
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import sys

import numpy as np

try:
    import netCDF4 as _nc
except Exception:                                    # pragma: no cover
    _nc = None
try:
    import h5py as _h5
except Exception:                                    # pragma: no cover
    _h5 = None

from .io_nc import DataFileError
from .ensemble_io import _pick_engine

# 盘古变量名 → 规范小写名（u10/v10 → u10m/v10m 对齐 ERA5 面通道）
_VAR_MAP = {
    "MSL": "msl", "T2M": "t2m", "TP": "tp", "U10": "u10m", "V10": "v10m",
    "Q2M": "q2m", "Z500": "z500", "T850": "t850", "U850": "u850",
    "V850": "v850", "Q700": "q700",
}
_REV = {v: k for k, v in _VAR_MAP.items()}

_FILE_RE = re.compile(
    r"Pangu_GLB_(SURFACE|PLEVELS)_OP25_6HOR_FCST_(\d{8})00_(\d{3})\.nc$")


def _list_files(d, kind):
    """{时效小时: 路径}。kind = "SURFACE" | "PLEVELS"；跳过 0 字节坏文件。"""
    out = {}
    for fn in os.listdir(d):
        m = _FILE_RE.match(fn)
        if not (m and m.group(1) == kind):
            continue
        p = os.path.join(d, fn)
        try:
            if os.path.getsize(p) == 0:
                print("[pangu] 警告: 跳过 0 字节文件 %s" % p, file=sys.stderr)
                continue
        except OSError:
            continue
        out[int(m.group(3))] = p
    return out


def _read_grid(path, engine):
    eng = _pick_engine(engine)
    if eng == "netcdf4":
        import netCDF4
        ds = netCDF4.Dataset(path)
        try:
            lat = np.asarray(ds.variables["lat"][:], dtype="f8")
            lon = np.asarray(ds.variables["lon"][:], dtype="f8")
        finally:
            ds.close()
    else:
        import h5py
        with h5py.File(path, "r") as h:
            lat = np.asarray(h["lat"][:], dtype="f8")
            lon = np.asarray(h["lon"][:], dtype="f8")
    return lat, lon


def _read_var(path, pangu_name, engine):
    """读一个 2D 场 (lat, lon) float32。"""
    eng = _pick_engine(engine)
    if eng == "netcdf4":
        import netCDF4
        ds = netCDF4.Dataset(path)
        try:
            v = ds.variables[pangu_name]
            a = np.ma.filled(v[0, 0], np.nan)
        finally:
            ds.close()
    else:
        import h5py
        with h5py.File(path, "r") as h:
            a = np.asarray(h[pangu_name][0, 0])
    return np.ascontiguousarray(a, dtype="f4")


def _read_var_names(path, engine):
    eng = _pick_engine(engine)
    if eng == "netcdf4":
        import netCDF4
        ds = netCDF4.Dataset(path)
        try:
            return [n for n in ds.variables
                    if n not in ("lat", "lon", "time", "dtime")]
        finally:
            ds.close()
    else:
        import h5py
        with h5py.File(path, "r") as h:
            return [n for n in h
                    if n not in ("lat", "lon", "time", "dtime")]


def scan_pangu_dates(root):
    """root 下所有含 SURFACE 文件的 YYYYMMDD 起报点（升序）。"""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and re.fullmatch(r"\d{8}", d) \
                and _list_files(p, "SURFACE"):
            out.append(d)
    return out


class PanguForecast(object):
    """一个盘古确定性预报（DF/<日期>/）。

    lead_hours 取 SURFACE 与 PLEVELS 的公共时效；若某族缺失/时效不一致给出警告。
    read(varname) → (1, T, lat, lon) float32（单成员 "det"）。
    """

    def __init__(self, date_dir, engine="auto"):
        self.root = str(date_dir)
        self.engine = engine
        if not os.path.isdir(self.root):
            raise DataFileError("盘古预报目录不存在: %s" % self.root)
        self._sfc = _list_files(self.root, "SURFACE")
        self._plv = _list_files(self.root, "PLEVELS")
        if not self._sfc:
            raise DataFileError("%s 下没有 Pangu_GLB_SURFACE_* 文件" % self.root)
        sfc_leads = sorted(self._sfc)
        plv_leads = sorted(self._plv)
        common = (sorted(set(sfc_leads) & set(plv_leads)) if plv_leads
                  else sfc_leads)
        if not common:
            raise DataFileError("%s: SURFACE 与 PLEVELS 时效无交集" % self.root)
        if plv_leads and common != sfc_leads:
            print("[pangu] 警告: %s 的 PLEVELS 时效(%d..%d, %d个) 与 SURFACE"
                  "(%d..%d, %d个) 不一致，统一取公共时效 %d..%d（%d个）"
                  % (self.root, plv_leads[0], plv_leads[-1], len(plv_leads),
                     sfc_leads[0], sfc_leads[-1], len(sfc_leads),
                     common[0], common[-1], len(common)), file=sys.stderr)
        self.lead_hours = np.asarray(common, dtype="f8")
        self.n_leads = int(self.lead_hours.size)
        self.lat, self.lon = _read_grid(self._sfc[common[0]], engine)
        self.lon_fixed = False
        self.members = ["det"]
        self.init_date = _init_from_name(self._sfc[common[0]])
        # 变量注册：canonical -> family（某族探测失败则跳过该族并告警）
        self._var_family = {}
        for fam, files in (("surface", self._sfc), ("plevels", self._plv)):
            if common[0] not in files:
                continue
            try:
                names = _read_var_names(files[common[0]], engine)
            except Exception as e:
                print("[pangu] 警告: 读取 %s 变量名失败(%s)，跳过 %s 族"
                      % (files[common[0]], e, fam), file=sys.stderr)
                continue
            for n in names:
                can = _VAR_MAP.get(n)
                if can:
                    self._var_family[can] = fam
        self.levels = sorted(self._var_family)

    def read(self, varname):
        v = str(varname).lower()
        if v not in self._var_family:
            raise DataFileError("%s 中没有要素 %r，现有: %s"
                                % (self.root, varname, self.levels))
        fam = self._var_family[v]
        files = self._sfc if fam == "surface" else self._plv
        pname = _REV[v]
        arrs = [_read_var(files[lead], pname, self.engine)
                for lead in self.lead_hours]
        return np.stack(arrs, axis=0)[None]          # (1, T, lat, lon) f4


def _init_from_name(path):
    m = re.search(r"_(\d{8})00_", os.path.basename(str(path)))
    if not m:
        return None
    s = m.group(1)
    return _dt.datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))