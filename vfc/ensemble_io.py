# -*- coding: utf-8 -*-
"""集合预报读取层：member_000…003/001.nc…061.nc → 堆叠数组。

本批 20250106 数据格式（与老 data(time,level,lat,lon) 不同）：
  * 一个文件 = 一个时效步，2D 场，变量各自独立成 dataset（Z500/MSL/TP/U200/V200）；
  * 无 time 轴，时效由文件名 NN.nc × lead_step 决定（本批 6h，001=+6h）；
  * 网格为标准 0.25°（lat 90→-90，lon 0→359.75），坐标无需修正。
"""
from __future__ import annotations

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

# 坐标/元数据 dataset 名（读取时跳过，不当作要素场）
_META_NAMES = {"lat", "lon", "time", "level", "channel", "data"}


def _pick_engine(engine):
    if engine == "netcdf4":
        if _nc is None:
            raise DataFileError("指定 netCDF4 引擎但未安装 netCDF4")
        return "netcdf4"
    if engine == "h5py":
        if _h5 is None:
            raise DataFileError("指定 h5py 引擎但未安装 h5py")
        return "h5py"
    if _nc is not None:
        return "netcdf4"
    if _h5 is not None:
        return "h5py"
    raise DataFileError("netCDF4 与 h5py 均不可用，无法读取 nc")


def read_step_nc(path, engine="auto", only=None):
    """读一个单时步 2D 文件。

    Parameters
    ----------
    path : str
    engine : "auto" | "netcdf4" | "h5py"
    only : str | None   只读指定要素（小写名）；None=读全部 2D 要素场。

    Returns
    -------
    (fields, lat, lon)：
        fields : {小写要素名: (lat, lon) float32}
        lat/lon : (nlat,)/(nlon,) float64
    """
    eng = _pick_engine(engine)
    only = [x.lower() for x in only] if isinstance(only, (list, tuple, set)) else (only.lower() if only else None)
    if eng == "netcdf4":
        import netCDF4
        ds = netCDF4.Dataset(path)
        try:
            lat = np.asarray(ds.variables["lat"][:], dtype="f8")
            lon = np.asarray(ds.variables["lon"][:], dtype="f8")
            fields = {}
            for name in ds.variables:
                key = name.lower()
                if key in _META_NAMES:
                    continue
                v = ds.variables[name]
                if v.ndim != 2:
                    continue
                if only is not None and key not in only:
                    continue
                fields[key] = np.ascontiguousarray(
                    np.ma.filled(v[:], np.nan), dtype="f4")
        finally:
            ds.close()
    else:
        import h5py
        with h5py.File(path, "r") as h:
            lat = np.asarray(h["lat"][:], dtype="f8")
            lon = np.asarray(h["lon"][:], dtype="f8")
            fields = {}
            for name in h:
                key = name.lower()
                if key in _META_NAMES:
                    continue
                d = h[name]
                if getattr(d, "ndim", 0) != 2:
                    continue
                if only is not None and key not in only:
                    continue
                fields[key] = np.ascontiguousarray(
                    np.where(np.abs(d[:]) > 1.0e30, np.nan, d[:]), dtype="f4")
    if not fields:
        raise DataFileError("%s 中没有 2D 要素场" % path)
    return fields, lat, lon


class EnsembleForecast(object):
    """一个集合预报：root_dir/member_xxx/NN.nc。

    read(varname) → (n_member, n_lead, n_lat, n_lon) float32；
    lead_hours = first_lead + arange(n_lead) * lead_step。
    只读所需要素（逐文件只切那个变量），内存可控。
    """

    def __init__(self, root_dir, members=None, first_lead=6.0, lead_step=6.0,
                 engine="auto"):
        self.root = str(root_dir)
        self.first_lead = float(first_lead)
        self.lead_step = float(lead_step)
        self.engine = engine
        if not os.path.isdir(self.root):
            raise DataFileError("集合预报根目录不存在: %s" % self.root)
        member_dirs = self._find_member_dirs(members)
        if not member_dirs:
            sub = sorted(d for d in os.listdir(self.root)
                         if os.path.isdir(os.path.join(self.root, d)))
            raise DataFileError(
                "在 %s 下没有找到含 NN.nc 步文件的成员目录。\n"
                "  直接子目录: %s\n"
                "  支持 member_000 平铺，或 20250106/20250106 式嵌套（自动向上找一层）。"
                "若数据已移动/删除，请检查 --pred-dir 路径。"
                % (self.root, sub[:30]))
        self.members = member_dirs
        self._files_cache = {}
        files = self._files(self.members[0])
        self.n_leads = len(files)
        if self.n_leads == 0:
            raise DataFileError("%s 中没有 NN.nc 步文件" % self.root)
        self.lead_hours = (np.arange(self.n_leads, dtype="f8")
                           * self.lead_step + self.first_lead)
        fields, lat, lon = read_step_nc(files[0], engine=engine)
        self.lat, self.lon = lat, lon
        self.levels = sorted(fields)
        self.lon_fixed = False
        self._engine = engine

    @staticmethod
    def _list_files(d):
        pat = re.compile(r"(\d+)\.nc$")
        items = []
        for fn in os.listdir(d):
            m = pat.search(fn)
            if m:
                items.append((int(m.group(1)), os.path.join(d, fn)))
        items.sort()
        return [p for _, p in items]

    def _find_member_dirs(self, members):
        """找成员目录，三种布局：
          1) root 直接放 NN.nc 步文件（单确定性，如 fuxi_single_output/<日期>/001.nc…）
             → 单成员 "det"；
          2) root/member_xxx/NN.nc 平铺；
          3) root/<日期>/member_xxx/NN.nc 嵌套（自动向上找一层）。
        只认确实含 NN.nc 步文件的目录（避免 out/缓存目录误入）。
        返回相对 self.root 的路径列表（"det" 表示 root 本身）。
        """
        if self._list_files(self.root):
            if members is None or any(m in ("det", "single") for m in members):
                return ["det"]

        def _cands(base):
            out = []
            for d in sorted(os.listdir(base)):
                p = os.path.join(base, d)
                if os.path.isdir(p) and self._list_files(p):
                    out.append(d)
            return out

        cands = _cands(self.root)
        if not cands:                          # 嵌套布局：根/20250106/member_xxx
            cands = []
            for d in sorted(os.listdir(self.root)):
                p = os.path.join(self.root, d)
                if not os.path.isdir(p):
                    continue
                for e in _cands(p):
                    cands.append(os.path.join(d, e))
        if members is None:
            return cands
        out = []
        for d in cands:
            base = os.path.basename(d)
            if any(base == m or base.endswith("_" + m) or base.endswith(m)
                   for m in members):
                out.append(d)
        return out

    def _files(self, member):
        if member not in self._files_cache:
            d = self.root if member == "det" else os.path.join(self.root,
                                                               member)
            self._files_cache[member] = self._list_files(d)
        return self._files_cache[member]


    def read_vars(self, names):
        """一次读多个要素（每个步文件只打开一次）。

        返回 {小写名: (n_member, n_lead, n_lat, n_lon) float32}；
        供派生要素（如 ws850 = sqrt(u850^2+v850^2)）同时取 u/v。"""
        names = [str(n).lower() for n in names]
        for n in names:
            if n not in self.levels:
                raise DataFileError("%s 中没有要素 %r，现有: %s"
                                    % (self.root, n, self.levels))
        acc = {m: {n: [] for n in names} for m in self.members}
        for m in self.members:
            for p2 in self._files(m):
                fields, _, _ = read_step_nc(p2, engine=self._engine,
                                            only=names)
                for n in names:
                    acc[m][n].append(fields[n])
        out = {}
        for n in names:
            out[n] = np.stack([np.stack(acc[m][n], axis=0)
                               for m in self.members], axis=0)
        return out


    def read_leads(self, varname, lead_indices):
        """(n_member, len(lead_indices), nlat, nlon) float32。

        只读指定 lead 下标对应的 NN.nc 里的该要素，
        避免一次把全部 member × 全部时效装进内存
        （多成员大集合时每进程峰值内存大幅降低）。
        """
        v = varname.lower()
        if v not in self.levels:
            raise DataFileError("%s 中没有要素 %r，现有: %s"
                                % (self.root, varname, self.levels))
        idx = [int(x) for x in lead_indices]
        if any(k < 0 for k in idx):
            raise DataFileError("read_leads 下标需 >= 0: %r" % (idx,))
        out = []
        for m in self.members:
            files = self._files(m)
            arrs = [read_step_nc(files[k], engine=self._engine, only=v)[0][v]
                    for k in idx]
            out.append(np.stack(arrs, axis=0))
        return np.stack(out, axis=0)

    def read_block(self, varnames, lead_indices):
        """一次开一个文件读多个要素。

        返回 {v: (n_member, len(lead_indices), nlat, nlon) float32}；
        每个 (member, lead) 文件只打开一次，同时取出本次需要的
        全部要素（减少小文件重复打开）。
        """
        want = [v.lower() for v in varnames]
        if not want:
            return {}
        idx = [int(x) for x in lead_indices]
        if any(k < 0 for k in idx):
            raise DataFileError("read_block 下标需 >= 0: %r" % (idx,))
        missing = [v for v in want if v not in self.levels]
        if missing:
            raise DataFileError("%s 中没有要素 %r，现有: %s"
                                % (self.root, missing, self.levels))
        per_var = {v: [] for v in want}
        for m in self.members:
            files = self._files(m)
            holder = {v: [] for v in want}
            for k in idx:
                fields, _, _ = read_step_nc(files[k], engine=self._engine,
                                           only=want)
                for v in want:
                    holder[v].append(fields[v])
            for v in want:
                per_var[v].append(np.stack(holder[v], axis=0))
        return {v: np.stack(arrs, axis=0) for v, arrs in per_var.items()}
    def read(self, varname):
        """(n_member, n_lead, n_lat, n_lon) float32。"""
        v = varname.lower()
        if v not in self.levels:
            raise DataFileError("%s 中没有要素 %r，现有: %s"
                                % (self.root, varname, self.levels))
        out = []
        for m in self.members:
            arrs = []
            for p in self._files(m):
                fields, _, _ = read_step_nc(p, engine=self._engine, only=v)
                arrs.append(fields[v])
            out.append(np.stack(arrs, axis=0))
        return np.stack(out, axis=0)
