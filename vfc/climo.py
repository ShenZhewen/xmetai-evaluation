# -*- coding: utf-8 -*-
"""逐日（day-of-year）气候态：ACC / 预报活跃度（FA）的公共前置。

支持两种 nc 布局（自动识别）：
  A. 旧格式 data(time, level, lat, lon)，level 存字符串要素名（走 FieldFile）；
  B. 新格式 lat/lon/time + 每要素独立数据集（如 era5_clim_phys_5.nc：
     变量 msl/tp/u200/v200/z500），time 轴为 365 或 366 天（6h 子日则 ×4）。

口径决定：
  * 来源**两套都支持**（做成可配置）：QX 规范 §5.6 规定 CRA40 1991—2020
    逐日平均；对表 FDP 时更宜与 obs 同源的 ERA5 逐日气候态。来源只是
    ``source`` 标签 + 不同的文件，读取与计算完全一致；
  * 必须**逐日/子日**气候态：时效跨 60 天，单一平均会严重失真。取值按
    "起报日 + 时效"对应的验证日期滑动，配 ±window/2 的循环平滑窗
    （默认 15 天），6h 时效落在分数日上做线性内插。

网格约定：0.25° 全球；升降序 / lon ±180 差异由调用方（run_ensemble）自动对齐。
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import os

from .io_nc import FieldFile, DataFileError, attr_str, lead_hours
from .ensemble_io import _pick_engine


def _has_data_var(path):
    """旧格式（有 data 变量）与新格式（每变量独立）识别。"""
    eng = _pick_engine("auto")
    try:
        if eng == "netcdf4":
            import netCDF4
            with netCDF4.Dataset(path) as ds:
                return "data" in ds.variables
        else:
            import h5py
            with h5py.File(path, "r") as h:
                return "data" in h
    except Exception:
        return False


def _simple_lead_hours(values, units):
    """新格式 time 轴 → 小时数。兼容 "days since ..." / "hours since ..." /
    无 since 的纯 "days"/"hours" 等。"""
    u = str(units or "").strip().lower()
    if "since" in u:
        return lead_hours(values, u)
    for prefix, fac in (("hour", 1.0), ("day", 24.0),
                        ("second", 1.0 / 3600.0), ("minute", 1.0 / 60.0)):
        if u.startswith(prefix):
            return np.asarray(values, dtype="f8") * fac
    raise DataFileError("无法解析气候态 time:units = %r" % (units,))


class _SimpleClimoFile(object):
    """新格式气候态：lat/lon/time + 每要素独立数据集。"""

    def __init__(self, path, engine="auto"):
        self.path = str(path)
        self.engine = engine
        eng = _pick_engine(engine)
        if eng == "netcdf4":
            import netCDF4
            with netCDF4.Dataset(path) as ds:
                self.lat = np.asarray(ds.variables["lat"][:], dtype="f8")
                self.lon = np.asarray(ds.variables["lon"][:], dtype="f8")
                t = ds.variables["time"]
                tvals = np.ma.filled(t[:], np.nan)
                units = attr_str(t.units)
                self.var_names = sorted(n for n in ds.variables
                                        if n not in ("lat", "lon", "time"))
        else:
            import h5py
            with h5py.File(path, "r") as h:
                self.lat = np.asarray(h["lat"][:], dtype="f8")
                self.lon = np.asarray(h["lon"][:], dtype="f8")
                t = h["time"]
                tvals = np.asarray(t[:])
                units = attr_str(t.attrs["units"])
                self.var_names = sorted(n for n in h
                                        if n not in ("lat", "lon", "time"))
        if not self.var_names:
            raise DataFileError("%s 没有要素变量（只有 lat/lon/time）" % path)
        self.lead_hours = _simple_lead_hours(tvals, units)
        self.levels = list(self.var_names)

    def read(self, varname):
        """(T, lat, lon) float32；前导维度折叠为 time（兼容 (1,1,lat,lon)）。"""
        eng = self.engine if self.engine != "auto" else _pick_engine("auto")
        if eng == "netcdf4":
            import netCDF4
            with netCDF4.Dataset(self.path) as ds:
                a = np.ma.filled(ds.variables[varname][:], np.nan)
        else:
            import h5py
            with h5py.File(self.path, "r") as h:
                a = np.asarray(h[varname][:])
        a = np.asarray(a, dtype="f8").reshape(-1, self.lat.size, self.lon.size)
        return np.ascontiguousarray(a, dtype="f4")

    def read_slice(self, varname, i0, i1):
        """只读 time 行区间 [i0, i1]（含端点），(i1-i0+1, lat, lon) f4。
        用于滚动窗口加载（避免整年常驻内存）。"""
        eng = self.engine if self.engine != "auto" else _pick_engine("auto")
        if eng == "netcdf4":
            import netCDF4
            with netCDF4.Dataset(self.path) as ds:
                a = np.ma.filled(ds.variables[varname][i0:i1 + 1], np.nan)
        else:
            import h5py
            with h5py.File(self.path, "r") as h:
                a = np.asarray(h[varname][i0:i1 + 1])
        a = np.asarray(a, dtype="f8").reshape(-1, self.lat.size, self.lon.size)
        return np.ascontiguousarray(a, dtype="f4")
class DailyClimatology(object):
    """加载一次，之后按 (要素, 验证时刻) 取气候场。

    自动识别旧/新两种 nc 布局；暴露 lat/lon/steps_per_day/levels 与
    field_for(varname, lead_hours, init_date)。
    """

    def __init__(self, path, engine="auto", window=15, fix_lon=True,
                 source="未标注"):
        self.path = str(path)
        self.source = str(source)
        self.window = int(window)
        self._ff = None
        self._simple = None
        if _has_data_var(path):
            self._ff = FieldFile(path, engine=engine, fix_lon=fix_lon)
            self.lat = self._ff.lat
            self.lon = self._ff.lon
            self.levels = list(self._ff.levels)
            leads = self._ff.lead_hours
        else:
            self._simple = _SimpleClimoFile(path, engine=engine)
            self.lat = self._simple.lat
            self.lon = self._simple.lon
            self.levels = list(self._simple.levels)
            leads = self._simple.lead_hours

        if leads.size < 2:
            raise DataFileError("气候态文件 time 轴过短（%d 步）" % leads.size)
        d = np.diff(leads)
        if not np.allclose(d, d[0], rtol=1e-6, atol=1e-9):
            raise DataFileError("气候态文件 time 轴步长不均匀")
        step_h = float(d[0])
        spd = 24.0 / step_h
        if spd < 1 or not np.isclose(spd, round(spd), rtol=0, atol=1e-9):
            raise DataFileError("气候态文件步长 %.4g h 非常规（需整除 24h）"
                                % step_h)
        self.steps_per_day = int(round(spd))
        n_days_f = leads.size / self.steps_per_day
        if not np.isclose(n_days_f, round(n_days_f), rtol=0, atol=1e-9):
            raise DataFileError("气候态文件步数 %d 无法整除 步/日 %d"
                                % (leads.size, self.steps_per_day))
        n_days = int(round(n_days_f))
        if n_days not in (365, 366):
            raise DataFileError(
                "气候态文件有 %d 天（%d 步 × %d/日），只支持 365 或 366 天"
                "（365：闰日并入 2/28；366：含闰日 2/29）。若 time 轴是 "
                "12 个月等其它约定，请把结构贴给开发适配。"
                % (n_days, leads.size, self.steps_per_day))
        self.n_days = n_days
        self.days = leads / 24.0                # 0…364（366 天则 0…365）
        self._cache = {}   # varname -> 平滑后的 (n_days*spd, lat, lon)

        # 滚动窗口加载（VFC_CLIMO_ROLLING=1 时启用；仅新格式单变量文件支持）
        _r = os.environ.get("VFC_CLIMO_ROLLING", "").strip().lower()
        self._roll = _r in ("1", "true", "yes", "on") and \
            self._simple is not None
        self._w = {}   # varname -> {"lo": int, "arr": (L, lat, lon) f4}

    # ------------------------------------------------------------ 接口

    def _read_var(self, varname):
        if self._ff is not None:
            return self._ff.read(varname)
        return self._simple.read(varname)

    def check_grid(self, obs: FieldFile):
        """（旧 verify 用；run_ensemble 走自动对齐）"""
        if not np.array_equal(self.lat, obs.lat) or not np.array_equal(
                self.lon, obs.lon):
            raise DataFileError(
                "气候态(%s)与 obs 网格不一致（lat %s..%s/%d lon %s..%s/%d vs "
                "lat %s..%s/%d lon %s..%s/%d）——run_ensemble 会自动对齐升降序/"
                "lon 半圈，请确认所用网格一致"
                % ((self.path, self.lat[0], self.lat[-1], self.lat.size,
                    self.lon[0], self.lon[-1], self.lon.size, obs.lat[0],
                    obs.lat[-1], obs.lat.size, obs.lon[0], obs.lon[-1],
                    obs.lon.size)))

    def has(self, varname) -> bool:
        return varname in self.levels

    # ------------------------------------------------------------ 取气候场

    def _smoothed(self, varname) -> np.ndarray:
        """±window/2 天（换算到步）循环平滑的气候态场（缓存）。window<=1 原样。"""
        if varname not in self._cache:
            raw = self._read_var(varname)    # (n, lat, lon) f4
            n = self.n_days * self.steps_per_day
            if raw.shape[0] == 1 and n > 1:
                raw = np.broadcast_to(
                    raw, (n, raw.shape[1], raw.shape[2])).copy()
            if raw.shape[0] != n:
                raise DataFileError("气候态 %s 的 %s 时效步数 %d != 期望 %d"
                                    % (self.path, varname, raw.shape[0], n))
            if self.window > 1:
                # 圆形窗口均值（cumsum 一次完成，替代 61 次循环；结果等价）
                half = (self.window * self.steps_per_day) // 2
                ksize = 2 * half + 1
                a = raw.astype("f8")
                if half:
                    d = np.concatenate([a[-half:], a, a[:half]], axis=0)
                else:
                    d = a
                cs = np.cumsum(d, axis=0)
                cs = np.concatenate(
                    [np.zeros((1,) + a.shape[1:], dtype="f8"), cs], axis=0)
                sm = (cs[ksize:ksize + a.shape[0]] - cs[:a.shape[0]]) / ksize
                self._cache[varname] = sm.astype("f4")
            else:
                self._cache[varname] = raw
        return self._cache[varname]

    def field_for(self, varname, lead_hours, init_date) -> np.ndarray:
        """验证日期 = init_date + lead_hours 处的气候场，(T,lat,lon)。

        按步索引：x = (doy + lead/24) × steps_per_day；对齐时次精确命中，
        非对齐在相邻两步间线性内插，年末岁初循环。
        """
        n = self.n_days * self.steps_per_day
        if self.n_days == 366:
            # 含闰日：真实日序号；非闰年 3/1 起跳过 2/29 槽
            doy0 = (_day_count(init_date)
                    + np.asarray(lead_hours, "f8") / 24.0)
            if not _is_leap(init_date.year):
                doy0 = np.where(doy0 >= 59.0, doy0 + 1.0, doy0)
        else:
            doy0 = (_day_of_year(init_date)
                    + np.asarray(lead_hours, "f8") / 24.0)
        x = np.mod(doy0 * self.steps_per_day, n)             # 循环折算
        i0 = np.floor(x).astype(int) % n
        i1 = (i0 + 1) % n
        w = (x - np.floor(x))[:, None, None]
        if self._roll:
            return self._roll_field(varname, i0, i1, w)
        base = self._smoothed(varname).astype("f8")
        return (1 - w) * base[i0] + w * base[i1]

    def _raw_segments(self, i0, i1):
        """行区间 [i0,i1]（可越年界）拆成若干 [a,b]（0<=a<=b<n，按展开顺序）。"""
        n = self.n_days * self.steps_per_day
        segs = []
        if i0 < 0:
            segs.append((i0 + n, n - 1))
            i0 = 0
        if i1 >= n:
            if i0 < n:
                segs.append((i0, n - 1))
            segs.append((0, i1 - n))
        elif i0 < n:
            segs.append((i0, i1))
        return [(a, b) for a, b in segs if a <= b]

    def _rolling_raw(self, varname, lo, hi):
        """确保缓存含原始行 [lo, hi]（含端点），返回 (clo, arr)。"""
        w = self._w.get(varname)
        if w is None:
            parts = [self._simple.read_slice(varname, a, b)
                     for a, b in self._raw_segments(lo, hi)]
            arr = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            w = {"lo": int(lo), "arr": arr}
            self._w[varname] = w
        clo = w["lo"]
        arr = w["arr"]
        if lo < clo:
            parts = [self._simple.read_slice(varname, a, b)
                     for a, b in self._raw_segments(lo, clo - 1)]
            head = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            arr = np.concatenate([head, arr], axis=0)
            clo = int(lo)
        chi = clo + arr.shape[0] - 1
        if hi > chi:
            parts = [self._simple.read_slice(varname, a, b)
                     for a, b in self._raw_segments(chi + 1, hi)]
            tail = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            arr = np.concatenate([arr, tail], axis=0)
        drop = max(0, lo - clo)                # 截掉已不需的前端，控制内存
        if drop:
            arr = arr[drop:]
            clo = int(lo)
        w["lo"], w["arr"] = clo, arr
        return clo, arr

    def _rolling_smoothed(self, varname, s_idx):
        """局部循环窗口平滑：与整年 cumsum 平滑逐位一致。
        所需窗口接近整年时退化为整年加载（仅年界日期）。"""
        n = self.n_days * self.steps_per_day
        s_idx = np.asarray(s_idx, dtype="i8")
        half = (self.window * self.steps_per_day) // 2
        if self.window <= 1 or half == 0:
            lo, hi = int(s_idx.min()), int(s_idx.max())
            clo, raw = self._rolling_raw(varname, lo, hi)
            return raw[s_idx - clo].astype("f8")
        ksize = 2 * half + 1
        lo = int(s_idx.min()) - half
        hi = int(s_idx.max()) + half
        if (hi - lo + 1) >= int(0.75 * n):       # 年界日期：退化整年
            full = self._smoothed(varname).astype("f8")
            return full[s_idx]
        clo, raw = self._rolling_raw(varname, lo, hi)
        a = raw.astype("f8")
        cs = np.concatenate(
            [np.zeros((1,) + a.shape[1:], dtype="f8"), np.cumsum(a, axis=0)],
            axis=0)
        rel = s_idx - clo
        return (cs[rel + half + 1] - cs[rel - half]) / ksize

    def _roll_field(self, varname, i0, i1, w):
        """滚动窗口下的 i0/i1 线性内插（与整年路径同一公式）。"""
        s_all = np.unique(np.concatenate([i0, i1]).astype("i8"))
        sm = self._rolling_smoothed(varname, s_all)
        pos = {int(vv): k for k, vv in enumerate(s_all)}
        j0 = np.asarray([pos[int(k)] for k in i0], dtype="i8")
        j1 = np.asarray([pos[int(k)] for k in i1], dtype="i8")
        return (1.0 - w) * sm[j0] + w * sm[j1]

    def close(self):
        if self._ff is not None:
            self._ff.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _day_count(date) -> float:
    """真实日序号（0 基，含闰日；2024-02-29 → 59.0，2025-03-01 → 59.0）。"""
    if not isinstance(date, _dt.datetime):
        date = _dt.datetime(date.year, date.month, date.day)
    jan1 = _dt.datetime(date.year, 1, 1)
    return (date - jan1).total_seconds() / 86400.0


def _day_of_year(date) -> float:
    """datetime → day-of-year（0 基，含当日 00:00 = 整数；2/29 向 2/28 折算
    到 365 天日历）。"""
    if not isinstance(date, _dt.datetime):
        date = _dt.datetime(date.year, date.month, date.day)
    jan1 = _dt.datetime(date.year, 1, 1)
    d = (date - jan1).total_seconds() / 86400.0
    if _is_leap(date.year) and d >= 59:                      # 2/29(=59) 及以后
        d -= 1.0
    return d


def _is_leap(y):
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
