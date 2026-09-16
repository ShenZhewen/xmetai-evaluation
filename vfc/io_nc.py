# -*- coding: utf-8 -*-
"""读取层：netCDF4 优先、h5py 自动回退。

针对本批数据的约定与坑：
  * 变量名为 ``data(time, level, lat, lon)``，**level 维存的是字符串要素名**
    （如 z500/t2m/tp/msl），按名字定位切片，天然支持任意要素组合；
  * **lon 坐标是错的**：文件里存的是 linspace(0,360,n)（首点 0、末点 360），
    而数据本身实际在标准 0.25° 网格（0…359.75）上（上游已确认）。检测到该
    错误模式时替换为 ``arange(n)*0.25``；文件头修正后（末点≈359.75）自动直通；
  * time:units 属性可能是 bytes / 单元素数组 / str，统一解码为小时数时效。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    import netCDF4 as _nc
except Exception:                                    # pragma: no cover
    _nc = None
try:
    import h5py as _h5
except Exception:                                    # pragma: no cover
    _h5 = None

# ---------------------------------------------------------------- 兼容小工具

def attr_str(value) -> str:
    """属性值统一成 str：兼容 bytes / np.bytes_ / 单元素数组 / str。"""
    a = value
    if isinstance(a, np.ndarray):
        a = a.ravel()[0] if a.size else ""
    if isinstance(a, (bytes, np.bytes_)):
        return a.decode("utf-8", "ignore")
    return str(a)


def decode_levels(arr) -> list:
    """level 坐标（字符串要素名）统一成 list[str]。

    兼容：变长字符串（object/bytes 一维）、经典 char 数组 (n, len) 的 S1 二维。
    """
    arr = np.asarray(arr)
    if arr.dtype == "S1" and arr.ndim == 2:
        return [b"".join(r).decode("utf-8", "ignore").strip("\x00 \t")
                for r in arr]
    out = []
    for x in arr.ravel():
        if isinstance(x, (bytes, np.bytes_)):
            x = x.decode("utf-8", "ignore")
        elif isinstance(x, np.ndarray):
            x = b"".join(x.astype("S1")).decode("utf-8", "ignore")
        out.append(str(x).strip("\x00 \t"))
    return out


_UNIT_TO_HOURS = (("second", 1.0 / 3600), ("minute", 1.0 / 60),
                  ("hour", 1.0), ("day", 24.0))


def lead_hours(values, units: str) -> np.ndarray:
    """按 time:units 把时间数值换算成“距起报的小时数”数组。"""
    u = units.strip().lower()
    for prefix, fac in _UNIT_TO_HOURS:               # startswith 兼容复数
        if u.startswith(prefix):
            return np.asarray(values, dtype="f8") * fac
    raise DataFileError("无法解析 time:units = %r" % (units,))


def parse_since(units: str):
    """解析 time:units 的 "since <时刻>" 部分，返回 datetime；失败返回 None。"""
    import datetime as dt
    m = re.search(r"since\s+(.+)$", units or "", re.IGNORECASE)
    if not m:
        return None
    s = m.group(1).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return None


# ---------------------------------------------------------------- 坐标修正

def fix_lon_if_buggy(lon: np.ndarray):
    """检测已知的 lon 写入错误并修正。

    错误模式：首点 0.0、末点 ≈360.0、步长均匀且名义 0.25°（实际 360/(n-1)）。
    该网格实为 linspace(0,360,n)，而数据本身在标准 0.25° 网格上 ——
    替换为 arange(n)*0.25（不删任何列，0 与 360 是两条不同经线）。

    返回 (lon, fixed)。标准网格（末点≈359.75）或任何非匹配模式原样返回。
    """
    lon = np.asarray(lon, dtype="f8")
    if lon.size < 2 or not np.isclose(lon[0], 0.0, atol=1e-3):
        return lon, False
    if not np.isclose(lon[-1], 360.0, atol=1e-3):
        return lon, False
    d = np.diff(lon)
    if not np.allclose(d, d[0], rtol=1e-6, atol=1e-9):
        return lon, False
    if not (0.24 <= d[0] <= 0.26):                   # 名义 0.25° 数据才处理
        return lon, False
    return np.arange(lon.size) * 0.25, True


# ---------------------------------------------------------------- 文件封装

class DataFileError(RuntimeError):
    """数据文件结构/一致性问题。"""


class FieldFile(object):
    """打开一个 obs/pred 文件，暴露网格、时效与按要素读场。

    Parameters
    ----------
    path : str
    engine : "auto" | "netcdf4" | "h5py"
    fix_lon : bool   是否应用 lon 坐标修正（默认开）。
    forecast_type : "det" | "ens"   确定性/集合预报模式。
    """

    def __init__(self, path, engine="auto", fix_lon=True,
                 forecast_type="det"):
        if forecast_type not in ("det", "ens"):
            raise ValueError("forecast_type 须为 'det' 或 'ens'")
        self.path = str(path)
        self.forecast_type = forecast_type
        self.engine = self._pick_engine(engine)
        self._close = None
        if self.engine == "netcdf4":
            if Path(self.path).is_dir() and forecast_type == "ens":
                self._read_ens_netcdf4(self.path)
            elif Path(self.path).is_dir(): # check if the data is path
                ds = _nc.Dataset(self._comb_netcdf4_fuxi(self.path))
                self._close = ds.close
                self._read_netcdf4(ds)
            else:
                ds = _nc.Dataset(self.path)
                self._close = ds.close
                self._read_netcdf4(ds)

        else:
            h = _h5.File(self.path, "r")
            self._close = h.close
            self._read_h5py(h)
        self._check_lat()
        if fix_lon:
            lon, fixed = fix_lon_if_buggy(self.lon)
            if fixed:
                import sys
                print("[vfc] 警告: %s 的 lon 坐标为已知的 linspace(0,360,n) "
                      "错误模式，已按上游确认替换为标准 0.25° 网格"
                      "（未删任何列）" % self.path, file=sys.stderr)
                self.lon, self.lon_fixed = lon, True
            else:
                self.lon_fixed = False
        else:
            self.lon_fixed = False
        self._init_date = None
        self._check_time_step()

    def _check_lat(self):
        """纬度坐标体检：单位必须是度、且步长为常规网格间隔。

        传弧度时 |lat| <= pi 不会越界，但步长会比度制小 ~57 倍，
        这里用步长兜底拒绝，避免 cos(lat) 权重静默算错。
        """
        lat = np.asarray(self.lat, dtype="f8")
        if lat.size == 0:
            return
        vmax = float(np.nanmax(np.abs(lat)))
        if vmax > 90.0 + 1e-6:
            raise DataFileError(
                "lat 超出 [-90, 90]：max|lat|=%.4f（%s）" % (vmax, self.path))
        if lat.size > 1:
            dlat = float(np.nanmean(np.abs(np.diff(lat))))
            if dlat < 0.005:
                raise DataFileError(
                    "lat 步长均值 %.5f 过小，疑似单位为弧度（本库要求「度」）：%s"
                    % (dlat, self.path))

    def _check_time_step(self):
        """时效步长为非常规值（非 1/3/6/12/24h 的均匀步长）时告警。

        本批 0829 组的 time 轴即 linspace(0,1008,88)、步长 11.586h
        （本意 12h，上游已确认）——与 lon 同族的文件头写入错误。
        """
        d = np.diff(self.lead_hours)
        if d.size and np.allclose(d, d[0], rtol=1e-6, atol=1e-9):
            step = float(d[0])
            if not any(abs(step - c) <= 0.01 * c for c in (1.0, 3.0, 6.0,
                                                            12.0, 24.0)):
                import sys
                print("[vfc] 警告: %s 的 time 轴步长 %.4g h 非常规"
                      "（1/3/6/12/24h），疑似文件头 linspace 类错误；"
                      "若已知真实步长请用 --lead-step H 重标时效"
                      % (self.path, step), file=sys.stderr)

    # -- 引擎选择 ----------------------------------------------------------

    @staticmethod
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
        raise DataFileError("netCDF4 与 h5py 均不可用，无法读取")

    # -- 两个引擎各自的元数据读取 ------------------------------------------

    def _read_netcdf4(self, ds):
        try:
            var = ds.variables["data"]
            tvar = ds.variables["time"]
            self.levels = decode_levels(ds.variables["level"][:])
            units = attr_str(tvar.units)
            tvals = np.ma.filled(tvar[:], np.nan)
            self.lat = np.asarray(ds.variables["lat"][:], dtype="f8")
            self.lon = np.asarray(ds.variables["lon"][:], dtype="f8")
        except KeyError as e:
            raise DataFileError(
                "%s 缺少变量 %s，现有变量: %s"
                % (self.path, e, sorted(ds.variables))) from e
        self.time_units = units
        self.lead_hours = lead_hours(tvals, units)
        self._reader = lambda i: np.ma.filled(var[:, i, :, :], np.nan)

    def _read_h5py(self, h):
        if "data" not in h:
            raise DataFileError("%s 缺少变量 data，现有: %s"
                                % (self.path, sorted(h.keys())))
        var = h["data"]
        try:
            self.levels = decode_levels(h["level"][:])
            units = attr_str(h["time"].attrs["units"])
            tvals = h["time"][:]
            self.lat = np.asarray(h["lat"][:], dtype="f8")
            self.lon = np.asarray(h["lon"][:], dtype="f8")
        except KeyError as e:
            raise DataFileError("%s 缺少坐标 %s" % (self.path, e)) from e
        self.time_units = units
        self.lead_hours = lead_hours(tvals, units)
        self._reader = lambda i: var[:, i, :, :]

    #适应哲文的Fuxi运行Period：[06， 12， 18， 00]
    def _comb_netcdf4_fuxi(self, path, first_lead_hours=6, dt_hours=6,):
        from datetime import datetime, timedelta

        root = Path(path)
        out = root / "combined.nc"
        if out.exists():
            return str(out)

        files = sorted(Path(path).glob("*.nc"), key=lambda file: int(file.stem))
        if not files:
            raise DataFileError(f"目录 {path} 下未找到 .nc 文件")

        names = {"msl": "MSL", "u10": "U10M", "v10": "V10M"}
        self.levels = list(names)
        init = datetime.strptime(Path(path).name, "%Y%m%d")

        with _nc.Dataset(files[0]) as s:
            lat, lon = s.variables["lat"][:], s.variables["lon"][:]

        ds = str(Path(path) / "combined.nc")
        dst = _nc.Dataset(ds, "w")
        dst.createDimension("time", len(files))
        dst.createDimension("level", len(names))
        dst.createDimension("lat", lat.shape[0])
        dst.createDimension("lon", lon.shape[0])

        dst.createVariable("time", "f8", ("time",))
        init = datetime.strptime(Path(path).name, "%Y%m%d")
        dst.variables["time"].units = init.strftime("hours since %Y-%m-%d 00:00:00")
        # for datetime
        dst.createVariable("level", str, ("level",))[:] = np.array(self.levels)
        dst.createVariable("lat", "f8", ("lat",))[:] = lat
        dst.createVariable("lon", "f8", ("lon",))[:] = lon
        dst.createVariable("data", "f4", ("time", "level", "lat", "lon"))

        for i, f in enumerate(files):
            dst.variables["time"][i] = first_lead_hours + i * dt_hours
            with _nc.Dataset(f) as s:
                for k, src in enumerate(names.values()):
                    dst.variables["data"][i, k] = s.variables[src][:]

        dst.close()
        return ds

    def _read_ens_netcdf4(self, path, dt_hours=6):
        """直接读取一个集合成员目录，不生成 combined.nc 中间文件。"""
        root = Path(path)
        files = sorted(root.glob("*.nc"), key=lambda file: int(file.stem))
        if not files:
            raise DataFileError("集合成员目录 %s 下未找到 .nc 文件" % path)

        with _nc.Dataset(files[0]) as ds:
            try:
                self.lat = np.asarray(ds.variables["lat"][:], dtype="f8")
                self.lon = np.asarray(ds.variables["lon"][:], dtype="f8")
            except KeyError as e:
                raise DataFileError("%s 缺少坐标 %s" % (files[0], e)) from e
            fields = [name for name, var in ds.variables.items()
                      if var.dimensions == ("lat", "lon")]

        source_names = {name.lower(): name for name in fields}
        self.levels = list(source_names)
        date_part = next((part for part in reversed(root.parts)
                          if re.fullmatch(r"\d{8}", part)), None)
        if date_part is None:
            raise DataFileError("无法从集合成员路径 %s 解析 YYYYMMDD 起报日期"
                                % path)
        self.time_units = "hours since %s-%s-%s 00:00:00" % (
            date_part[:4], date_part[4:6], date_part[6:])
        self.lead_hours = np.asarray(
            [int(file.stem) * dt_hours for file in files], dtype="f8")

        def read_series(index):
            source_name = source_names[self.levels[index]]
            data = np.empty((len(files), self.lat.size, self.lon.size),
                            dtype="f4")
            for time_index, file in enumerate(files):
                with _nc.Dataset(file) as ds:
                    if source_name not in ds.variables:
                        raise DataFileError("%s 缺少要素 %s"
                                            % (file, source_name))
                    data[time_index] = np.ma.filled(
                        ds.variables[source_name][:], np.nan)
            return data

        self._reader = read_series



    # -- 对外接口 ----------------------------------------------------------

    @property
    def shape(self):
        """(n_lead, n_lat, n_lon)"""
        return (self.lead_hours.size, self.lat.size, self.lon.size)

    @property
    def init_date(self):
        """起报时刻：解析 time:units 的 since 部分；失败为 None。"""
        if self._init_date is None:
            self._init_date = parse_since(self.time_units)
        return self._init_date

    def read(self, varname) -> np.ndarray:
        """读一个要素的整条时效序列，返回 (n_lead, n_lat, n_lon) float32。

        只切 level 那一片，不触及其他要素的数据。
        """
        if varname not in self.levels:
            raise DataFileError("%s 中没有要素 %r，现有: %s"
                                % (self.path, varname, self.levels))
        a = self._reader(self.levels.index(varname))
        return np.ascontiguousarray(a, dtype="f4")

    def close(self):
        if self._close is not None:
            self._close()
            self._close = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def check_pair(obs: FieldFile, pred: FieldFile):
    """校验 obs/pred 网格、时效、要素逐点一致；不一致给出可读报错。"""
    if obs.lead_hours.shape != pred.lead_hours.shape or not np.allclose(
            obs.lead_hours, pred.lead_hours, rtol=0, atol=1e-6):
        raise DataFileError(
            "obs/pred 时效轴不一致: %s %s步[%g..%g]h vs %s %s步[%g..%g]h"
            % (obs.path, obs.lead_hours.size, obs.lead_hours[0],
               obs.lead_hours[-1], pred.path, pred.lead_hours.size,
               pred.lead_hours[0], pred.lead_hours[-1]))
    for attr in ("lat", "lon"):
        a, b = getattr(obs, attr), getattr(pred, attr)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise DataFileError("obs/pred 的 %s 坐标不一致（%s vs %s）"
                                % (attr, obs.path, pred.path))
    if set(obs.levels) != set(pred.levels):
        raise DataFileError(
            "obs/pred 要素集不一致: 仅obs有%s 仅pred有%s，公共要素=%s"
            % (sorted(set(obs.levels) - set(pred.levels)),
               sorted(set(pred.levels) - set(obs.levels)),
               sorted(set(obs.levels) & set(pred.levels))))
