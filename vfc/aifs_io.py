# -*- coding: utf-8 -*-
"""AIFS 确定性预报/真值读取层（自动兼容两种文件格式）。

格式 A（老 AIFS single-v9，channel 版）：
  dims: time(1), step(1), channel(12), lat(721), lon(1440)
  主变量：output / target，shape (1,1,12,721,1440)；channel 轴给要素名。
格式 B（per-var 2D 版，本机 aifs-single-output/target 实际格式）：
  一文件一时效 NN.nc，文件里直接是多个 2D 变量场：
  Z500/Q700/T700/T850/U850/V850/U10M/V10M/T2M/D2M/MSL/TP（lat,lon）。
两种都按 NN.nc 文件名定时效（001=+6h）、lat 90→-90、lon 0→359.75(0.25°)。
目录：<root>/<YYYY-MM-DD> 或 <root>/<YYYYMMDD>/001.nc…060.nc。

对外接口与 EnsembleForecast 一致（levels/lat/lon/lead_hours/members/read），
可被 run_ensemble 直接消费（单成员 "det"）。
"""
from __future__ import annotations

import datetime as _dt
import os
import re

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

_EPOCH = _dt.datetime(1970, 1, 1)
_SKIP = {"lat", "lon", "time", "channel", "step", "initial_time",
         "valid_time", "level", "isobaricInhPa"}


def _list_files(d):
    """{时效小时: 路径}，按 NN.nc 文件名解析。"""
    pat = re.compile(r"(\d+)\.nc$")
    out = {}
    for fn in os.listdir(d):
        m = pat.search(fn)
        if m:
            out[int(m.group(1))] = os.path.join(d, fn)
    return out


def _open_vars(path, engine):
    """返回变量名字典视图（netcdf4 / h5py 统一）。"""
    eng = _pick_engine(engine)
    if eng == "netcdf4":
        ds = _nc.Dataset(path)
        return ds, ds.variables
    h = _h5.File(path, "r")
    return h, h


def _read_meta(path, engine):
    """返回 (lat, lon, channels)。

    channel 版取 channel 轴；per-var 2D 版取顶层 2D 变量名（排除坐标）。
    """
    eng = _pick_engine(engine)
    if eng == "netcdf4":
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            lat = np.asarray(ds.variables["lat"][:], dtype="f8")
            lon = np.asarray(ds.variables["lon"][:], dtype="f8")
            if "channel" in ds.variables:
                ch = [str(x) for x in ds.variables["channel"][:]]
            else:
                ch = [n for n in ds.variables if n not in _SKIP]
    else:
        import h5py
        with h5py.File(path, "r") as h:
            lat = np.asarray(h["lat"][:], dtype="f8")
            lon = np.asarray(h["lon"][:], dtype="f8")
            if "channel" in h:
                raw = np.asarray(h["channel"][:])
                ch = [x.decode("utf-8", "ignore") if isinstance(x, bytes)
                      else str(x) for x in raw.ravel()]
            else:
                ch = [n for n in h if n not in _SKIP]
    return lat, lon, ch


def _has_channel_axis(path, engine):
    """True=老 channel 版；False=per-var 2D 版。"""
    eng = _pick_engine(engine)
    if eng == "netcdf4":
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            return "channel" in ds.variables
    import h5py
    with h5py.File(path, "r") as h:
        return "channel" in h


def _read_var(path, var_name, ch, engine):
    """channel 版：读主变量第 ch 通道 2D 场 (lat, lon)。"""
    eng = _pick_engine(engine)
    if eng == "netcdf4":
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            a = np.ma.filled(ds.variables[var_name][0, 0, ch], np.nan)
    else:
        import h5py
        with h5py.File(path, "r") as h:
            a = np.asarray(h[var_name][0, 0, ch])
    return np.ascontiguousarray(a, dtype="f4")


def _read_field2d(path, name, engine):
    """per-var 2D 版：读顶层 2D 变量（大小写不敏感匹配）。"""
    eng = _pick_engine(engine)
    if eng == "netcdf4":
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            key = _find_key(ds.variables, name)
            if key is None:
                raise DataFileError("%s 中无变量 %r" % (path, name))
            a = np.ma.filled(ds.variables[key][:], np.nan)
    else:
        import h5py
        with h5py.File(path, "r") as h:
            key = _find_key(h, name)
            if key is None:
                raise DataFileError("%s 中无变量 %r" % (path, name))
            a = np.asarray(h[key][:])
    return np.ascontiguousarray(a, dtype="f4")


def _find_key(mapping, name):
    """在 mapping 里找大小写不敏感的名字（优先精确）。"""
    low = str(name).lower()
    if low in mapping:
        return low
    for k in mapping:
        if str(k).lower() == low:
            return k
    return None


def _init_from_dirname(root):
    base = os.path.basename(str(root))
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})$", base)
    if m:
        return _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})(\d{2})(\d{2})$", base)
    if m:
        return _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


class AifsForecast(object):
    """一个 AIFS 确定性起报点（<root>/<日期>/001.nc…），自动兼容两种格式。

    var_name: "output"/"target"（仅 channel 版用）；per-var 2D 版忽略。
    as_target=True 时 read() 返回 (T,lat,lon)，否则 (1,T,lat,lon)。
    """

    def __init__(self, date_dir, var_name="output", engine="auto",
                 as_target=False):
        self.root = str(date_dir)
        self.var_name = str(var_name)
        self.engine = engine
        self.as_target = bool(as_target)
        if not os.path.isdir(self.root):
            raise DataFileError("AIFS 目录不存在: %s" % self.root)
        self._files = _list_files(self.root)
        if not self._files:
            raise DataFileError("%s 下没有 NN.nc 文件" % self.root)
        self._order = sorted(self._files)          # 文件名序号升序
        self.n_leads = len(self._order)
        self.lead_hours = (np.arange(self.n_leads, dtype="f8") + 1.0) * 6.0
        first = self._files[self._order[0]]
        self.lat, self.lon, self.channels = _read_meta(first, engine)
        self.var2d = not _has_channel_axis(first, engine)
        self.levels = sorted(c.lower() for c in self.channels)
        self._ch_idx = {c.lower(): i for i, c in enumerate(self.channels)}
        self._name = {c.lower(): c for c in self.channels}
        self.members = ["det"]
        self.lon_fixed = False
        self.init_date = _init_from_dirname(self.root)

    @property
    def time(self):
        base = self.init_date or _dt.datetime(1970, 1, 1)
        return [base + _dt.timedelta(hours=float(h))
                for h in self.lead_hours]

    @property
    def time_hours(self):
        return np.array([(t - _EPOCH).total_seconds() / 3600.0
                         for t in self.time])

    def read(self, varname, time_idx=None):
        v = str(varname).lower()
        if v not in self._ch_idx:
            raise DataFileError("%s 中没有要素 %r，现有: %s"
                                % (self.root, varname, self.levels))
        idx = (list(range(self.n_leads)) if time_idx is None
               else [int(k) for k in np.asarray(time_idx)])
        if self.var2d:
            arrs = [_read_field2d(self._files[self._order[k]],
                                  self._name[v], self.engine)
                    for k in idx]
        else:
            i = self._ch_idx[v]
            arrs = [_read_var(self._files[self._order[k]], self.var_name, i,
                              self.engine)
                    for k in idx]
        a = np.stack(arrs, axis=0)                 # (T, lat, lon) f4
        return a if self.as_target else a[None]


def scan_aifs_dates(root):
    """root 下所有含 NN.nc 的起报点（YYYY-MM-DD 或 YYYYMMDD，升序）。"""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and (re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)
                                 or re.fullmatch(r"\d{8}", d)) \
                and _list_files(p):
            out.append(d)
    return out
