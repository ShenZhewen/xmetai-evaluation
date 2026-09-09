# -*- coding: utf-8 -*-
"""分类管线的读取层：确定性 / 集合预报的输出读取。

只放分类管线自己的读取类：

* ``ClassDataset`` —— 自动识别确定性（``起报日期/001~060.nc``）与
  集合（``起报日期/member_*/001~060.nc``）两种布局，对外给出
  ``read_mean``（集合成员平均场，喂 TS）与 ``read_members``（未平均的
  成员场，喂 AROC/BS/BSS）。

分类指标（TS / AROC / BS / BSS）统一放在 ``vfc.metric_categorical``。
"""
from __future__ import annotations

import datetime as dt
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

class DataFileError(RuntimeError):
    """数据文件结构/一致性问题。"""

try:
    import netCDF4 as _nc
except Exception:                                    # pragma: no cover
    _nc = None
try:
    import h5py as _h5
except Exception:                                    # pragma: no cover
    _h5 = None


def _parse_init_date(name, init_hour):
    """目录名 YYYYMMDD → 起报 datetime（UTC），失败返回 None。"""
    try:
        base = dt.datetime.strptime(name, "%Y%m%d")
    except ValueError:
        return None
    return base + dt.timedelta(hours=init_hour)


def _read_var_f4(args):
    """读单个文件变量 → (lat, lon) float32（子进程用，与串行 read 逐位一致）。

    args: (path, var)。与 ``ClassDataset._read_var`` + float32 赋值等价：读
    float32 直接得到与 float64→float32 相同的值（float32 原值往返 float64 无损）。
    """
    path, var = args
    if _nc is not None:
        ds = _nc.Dataset(path)
        try:
            if var not in ds.variables:
                raise DataFileError("%s 无变量 %r（现有 %s）"
                                    % (path, var, sorted(ds.variables)))
            return np.ascontiguousarray(ds.variables[var][:], dtype="f4")
        finally:
            ds.close()
    if _h5 is not None:
        h = _h5.File(path, "r")
        try:
            if var not in h:
                raise DataFileError("%s 无变量 %r（现有 %s）"
                                    % (path, var, sorted(h.keys())))
            return np.ascontiguousarray(h[var][:], dtype="f4")
        finally:
            h.close()
    raise DataFileError("netCDF4 与 h5py 均不可用，无法读取")


class ClassDataset(object):
    """分类管线的读取层（支持多起报日期、确定性或集合成员）。

    Parameters
    ----------
    root : str
        根目录，其下为 YYYYMMDD 起报日期子目录。若子目录内是
        ``member_*/`` 子目录则视为集合预报，否则为确定性预报。
    var : str
        降水变量名（默认 "TP"）。
    step : float
        时效步长（小时），文件序号 n 对应 lead = n * step。
    init_hour : float
        起报时刻（UTC 小时，默认 0）。
    workers : int
        并行读文件的线程数（默认 1 串行；集合数据建议 8）。多线程只并发
        读不同文件，结果与串行逐位一致。
    """

    def __init__(self, root, var="TP", step=6.0, init_hour=0.0, workers=1):
        self.root = str(root)
        self.var = var
        self.step = float(step)
        self.init_hour = float(init_hour)
        self.workers = int(workers)
        self._pool = (ProcessPoolExecutor(max_workers=self.workers)
                      if self.workers > 1 else None)
        self._init_dates = []
        self._files = {}          # init_date -> 单值 list[str] / 集合 list[list[str]]
        self._n_members = 1       # 集合成员数（无 member 层为 1）
        self._lead_hours = None
        self._lat = None
        self._lon = None
        self._scan()

    # -- 扫描 --------------------------------------------------------------

    def _scan(self):
        if not os.path.isdir(self.root):
            raise DataFileError("根目录不存在: %s" % self.root)
        day_dirs = {}
        for name in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, name)
            if not os.path.isdir(path):
                continue
            idt = _parse_init_date(name, self.init_hour)
            if idt is None:
                continue
            day_dirs[name] = (idt, path)
        if not day_dirs:
            raise DataFileError("%s 下无 YYYYMMDD 起报日期子目录" % self.root)

        # 以第一个起报日期判断是否含集合成员（member_* 子目录）
        first_dir = day_dirs[sorted(day_dirs)[0]][1]
        self._n_members = len(self._list_members(first_dir)) or 1

        for name in sorted(day_dirs):
            idt, path = day_dirs[name]
            if self._n_members == 1:
                self._files[idt] = self._scan_nc_files(path)
            else:
                members = self._list_members(path)
                if len(members) != self._n_members:
                    raise DataFileError(
                        "起报日期 %s 成员数 %d 与 %s 的 %d 不一致"
                        % (name, len(members), sorted(day_dirs)[0],
                           self._n_members))
                self._files[idt] = [self._scan_nc_files(os.path.join(path, m))
                                    for m in members]
            self._init_dates.append(idt)

        # 网格与时效：以第一个起报日期为准
        n0 = self._n_lead_of(self._init_dates[0])
        self._lead_hours = np.arange(1, n0 + 1, dtype="f8") * self.step
        self._lat, self._lon = self._read_grid(
            self._first_file(self._init_dates[0]))
        for idt in self._init_dates[1:]:
            if self._n_lead_of(idt) != n0:
                raise DataFileError(
                    "起报日期 %s 时效文件数 %d 与 %s 的 %d 不一致"
                    % (idt.strftime("%Y%m%d"), self._n_lead_of(idt),
                       self._init_dates[0].strftime("%Y%m%d"), n0))
            lat, lon = self._read_grid(self._first_file(idt))
            if not np.array_equal(lat, self._lat) or \
                    not np.array_equal(lon, self._lon):
                raise DataFileError(
                    "起报日期 %s 的网格与 %s 不一致"
                    % (idt.strftime("%Y%m%d"),
                       self._init_dates[0].strftime("%Y%m%d")))

    @staticmethod
    def _member_index(name):
        m = re.match(r"member_(\d+)", name)
        return int(m.group(1)) if m else -1

    def _list_members(self, dirpath):
        """列出 dirpath 下 member_* 子目录（按成员编号升序）；无则返回 []。"""
        out = [d for d in os.listdir(dirpath)
               if os.path.isdir(os.path.join(dirpath, d))
               and d.startswith("member_")]
        out.sort(key=self._member_index)
        return out

    def _scan_nc_files(self, dirpath):
        """扫目录下 001..060.nc（序号连续），返回 [按序号的路径列表]。"""
        files = {}
        for fn in os.listdir(dirpath):
            if not fn.endswith(".nc"):
                continue
            base = os.path.splitext(fn)[0]
            try:
                idx = int(base)
            except ValueError:
                continue
            files[idx] = os.path.join(dirpath, fn)
        if not files:
            raise DataFileError("%s 下无 .nc 文件" % dirpath)
        ordered = sorted(files)
        if ordered[0] != 1 or ordered != list(range(1, len(ordered) + 1)):
            raise DataFileError("%s 文件序号不连续（期望 1..%d，实际 %s）"
                                % (dirpath, len(ordered), ordered))
        return [files[i] for i in ordered]

    def _first_file(self, idt):
        f = self._files[idt]
        return f[0][0] if self._n_members > 1 else f[0]

    def _n_lead_of(self, idt):
        f = self._files[idt]
        return len(f[0]) if self._n_members > 1 else len(f)

    # -- 文件读取 ----------------------------------------------------------

    def _read_var(self, path):
        """读单文件的一个变量场，返回 (lat, lon) float64。"""
        if _nc is not None:
            ds = _nc.Dataset(path)
            try:
                if self.var not in ds.variables:
                    raise DataFileError("%s 无变量 %r（现有 %s）"
                                        % (path, self.var,
                                           sorted(ds.variables)))
                return np.ascontiguousarray(
                    ds.variables[self.var][:], dtype="f8")
            finally:
                ds.close()
        if _h5 is not None:
            h = _h5.File(path, "r")
            try:
                if self.var not in h:
                    raise DataFileError("%s 无变量 %r（现有 %s）"
                                        % (path, self.var, sorted(h.keys())))
                return np.ascontiguousarray(h[self.var][:], dtype="f8")
            finally:
                h.close()
        raise DataFileError("netCDF4 与 h5py 均不可用，无法读取")

    def _read_grid(self, path):
        """读 lat/lon 坐标，返回 (lat, lon)。"""
        if _nc is not None:
            ds = _nc.Dataset(path)
            try:
                lat = np.asarray(ds.variables["lat"][:], dtype="f8")
                lon = np.asarray(ds.variables["lon"][:], dtype="f8")
                return lat, lon
            finally:
                ds.close()
        if _h5 is not None:
            h = _h5.File(path, "r")
            try:
                lat = np.asarray(h["lat"][:], dtype="f8")
                lon = np.asarray(h["lon"][:], dtype="f8")
                return lat, lon
            finally:
                h.close()
        raise DataFileError("netCDF4 与 h5py 均不可用，无法读取")

    # -- 对外接口 ----------------------------------------------------------

    @property
    def init_dates(self):
        return list(self._init_dates)

    @property
    def lead_hours(self):
        return self._lead_hours

    @property
    def lat(self):
        return self._lat

    @property
    def lon(self):
        return self._lon

    @property
    def n_members(self):
        """集合成员数（确定性 = 1）。"""
        return self._n_members

    def read_members(self, init_date):
        """读某起报日期的成员场，返回 (n_members, n_lead, lat, lon) float32。

        确定性时返回 (1, n_lead, lat, lon)。workers>1 时并行读文件，结果与
        串行逐位一致（读同一批文件、同样 float64→float32 赋值）。
        """
        if init_date not in self._files:
            raise DataFileError(
                "未知起报日期 %s（现有 %s）"
                % (init_date, [d.strftime("%Y%m%d") for d in self._init_dates]))
        files = self._files[init_date]
        n_lead = self._lead_hours.size
        if self._n_members == 1:
            out = np.empty((1, n_lead, self._lat.size, self._lon.size),
                           dtype="f4")
            tasks = [(0, i, p) for i, p in enumerate(files)]
        else:
            out = np.empty((self._n_members, n_lead,
                            self._lat.size, self._lon.size), dtype="f4")
            tasks = [(mi, i, p) for mi, mf in enumerate(files)
                     for i, p in enumerate(mf)]
        if self._pool is None:
            for mi, i, p in tasks:
                out[mi, i] = self._read_var(p)
        else:
            args = [(p, self.var) for _, _, p in tasks]
            for (mi, i, _p), val in zip(tasks, self._pool.map(
                    _read_var_f4, args, chunksize=16)):
                out[mi, i] = val
        return out

    def read_mean(self, init_date):
        """读某起报日期的平均场，返回 (n_lead, lat, lon) float32。

        确定性 = 单场；集合 = 各成员逐格点算术平均。
        """
        if self._n_members == 1:
            if init_date not in self._files:
                raise DataFileError(
                    "未知起报日期 %s（现有 %s）"
                    % (init_date,
                       [d.strftime("%Y%m%d") for d in self._init_dates]))
            files = self._files[init_date]
            n_lead = self._lead_hours.size
            out = np.empty((n_lead, self._lat.size, self._lon.size),
                           dtype="f4")
            if self._pool is None:
                for i, p in enumerate(files):
                    out[i] = self._read_var(p)
            else:
                for i, val in enumerate(self._pool.map(
                        _read_var_f4, [(p, self.var) for p in files],
                        chunksize=16)):
                    out[i] = val
            return out
        return self.read_members(init_date).mean(axis=0).astype("f4")

    def close(self):
        """关闭并行读进程池（如有）。"""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
