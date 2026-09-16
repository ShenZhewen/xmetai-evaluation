# -*- coding: utf-8 -*-
"""ERA5 zarr（foundation store 格式）目标读取层。

实测结构（era5_sfc_2025.01-2026.07.c15.p25.h6.zarr，2026-08-31）：
  channel_names.json   要素名列表，与 data 的 channel 轴一一对应
  data                 形状 (time, channel, lat, lon)，float32，**已是物理量**（非归一化）
  time                 int64，units "hours since 2025-01-01 00:00:00"，6h 步
  lat                  90→-90（0.25°）；lon -180→179.75（0.25°，与预报 0→359.75 差半圈）
  mean.npy / std.npy   每通道训练统计量（本读取层默认**不**反算，见 ``denorm``）
  mask.npy / weight.npy  本层不使用

约定：
  * tp 为**逐 6h 时段量**（非自起报累积），单位 m（与预报 mm 差 1000，由调用方换算）；
  * 元素据中 GRIB_stepType='instant'、units='K' 等是首通道残留属性，勿据此判断。
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from .io_nc import DataFileError, parse_since

_EPOCH = _dt.datetime(1970, 1, 1)


def _require(name):
    try:
        __import__(name)
    except Exception:
        raise DataFileError(
            "读取 zarr 需要 %r 库；请在目标环境安装（pip install %s）" % (name, name))


def _np_load(path):
    return np.load(path) if os.path.exists(path) else None


def _dt64_to_datetime(arr):
    """datetime64 数组 → list[datetime]（ns/ms/us/s 均可）。"""
    arr = np.asarray(arr)
    if arr.dtype.kind != "M":
        raise DataFileError("不是 datetime64: %s" % arr.dtype)
    arr = arr.astype("datetime64[ns]")
    return [t.astype(_dt.datetime) for t in arr]


class ZarrTarget(object):
    """打开一个 ERA5 zarr 目标，暴露 levels/lat/lon/time 与 read(varname)。

    Parameters
    ----------
    path : str
    denorm : bool   若 True 且存在 mean.npy/std.npy，按 x*std+mean 反算（默认 False：
        实测 data 已是物理量）。
    tp_exp : bool   若 True，对 tp 通道做 expm1（还原 log1p 存储；默认 False）。
    time_units : str | None   覆盖 time:units（默认取数组 attrs）。
    channel_axis : int | None   显式指定 channel 轴（默认按通道数自动识别）。
    """

    def __init__(self, path, denorm=False, tp_exp=False, time_units=None,
                 channel_axis=None):
        _require("zarr")
        import zarr
        self.path = str(path)
        root = zarr.open(self.path, mode="r")
        if "data" not in root:
            raise DataFileError("%s 缺少 data 数组（现有: %s）"
                                % (self.path, sorted(root.keys())))
        self._data = root["data"]
        self.shape = tuple(self._data.shape)

        self.channel_names = self._read_channel_names(root)
        n_ch = len(self.channel_names)
        self.axis = self._detect_channel_axis(self.shape, n_ch, channel_axis)

        self.lat = np.asarray(root["lat"][:], dtype="f8")
        self.lon = np.asarray(root["lon"][:], dtype="f8")
        self.time, self.time_hours = self._read_time(root, time_units)

        # 反算统计量（存在才启用）
        self.mean = _np_load(os.path.join(self.path, "mean.npy"))
        self.std = _np_load(os.path.join(self.path, "std.npy"))
        if self.mean is not None:
            self.mean = np.asarray(self.mean, dtype="f8").ravel()
        if self.std is not None:
            self.std = np.asarray(self.std, dtype="f8").ravel()
        if self.mean is not None and self.mean.size != n_ch:
            raise DataFileError("mean.npy 长度 %d 与通道数 %d 不符"
                                % (self.mean.size, n_ch))
        self.denorm = bool(denorm) and self.mean is not None and self.std is not None
        self.tp_exp = bool(tp_exp)

        # 识别 time/lat/lon 轴
        self.time_axis = self._find_axis(self.shape, len(self.time), "time")
        self.lat_axis = self._find_axis(self.shape, self.lat.size, "lat")
        self.lon_axis = self._find_axis(self.shape, self.lon.size, "lon")
        self._red = [i for i in range(len(self.shape)) if i != self.axis]
        self._pos = {orig: k for k, orig in enumerate(self._red)}
        self.lon_fixed = False
        self.engine = "zarr"

    # ------------------------------------------------------------ 元数据

    @staticmethod
    def _read_channel_names(root):
        try:
            with open(os.path.join(root.path, "channel_names.json"), "r",
                      encoding="utf-8") as f:
                names = json.load(f)
        except Exception:
            try:
                names = root["channel"][:]
            except Exception as e:
                raise DataFileError("无法读取 channel_names.json 或 channel 坐标: %s"
                                    % e)
        return [str(n).strip().lower() for n in names]

    @staticmethod
    def _detect_channel_axis(shape, n_ch, channel_axis):
        if channel_axis is not None:
            if not (0 <= int(channel_axis) < len(shape)):
                raise DataFileError("channel_axis=%s 越界（shape=%s）"
                                    % (channel_axis, shape))
            return int(channel_axis)
        cands = [i for i, s in enumerate(shape) if s == n_ch]
        if len(cands) == 1:
            return cands[0]
        if not cands:
            raise DataFileError("data.shape=%s 中找不到长度为 %d（通道数）的轴"
                                % (shape, n_ch))
        if 1 in cands:                       # 常见 (time, channel, lat, lon)
            return 1
        return cands[-1]

    @staticmethod
    def _find_axis(shape, size, label):
        hits = [i for i, s in enumerate(shape) if s == size]
        if len(hits) != 1:
            raise DataFileError("data.shape=%s 无法唯一识别 %s 轴（size=%d，候选 %s）"
                                % (shape, label, size, hits))
        return hits[0]

    def _read_time(self, root, time_units):
        t = root["time"]
        arr = np.asarray(t[:])
        attrs = dict(t.attrs)
        if np.issubdtype(arr.dtype, np.datetime64):
            dt_list = _dt64_to_datetime(arr)
            return dt_list, np.array(
                [(d - _EPOCH).total_seconds() / 3600.0 for d in dt_list])
        units = time_units or attrs.get("units")
        if isinstance(units, (bytes, np.bytes_)):
            units = units.decode("utf-8", "ignore")
        if isinstance(units, str) and "since" in units:
            base = parse_since(units)
            if base is None:
                raise DataFileError("无法解析 time:units=%r" % units)
            low = units.lower()
            if low.startswith("hour"):
                fac = 1.0
            elif low.startswith("day"):
                fac = 24.0
            elif low.startswith("second"):
                fac = 1.0 / 3600.0
            elif low.startswith("minute"):
                fac = 1.0 / 60.0
            else:
                raise DataFileError("无法解析 time:units=%r" % units)
            dt_list = [base + _dt.timedelta(hours=float(v) * fac) for v in arr]
            return dt_list, np.array(
                [(d - _EPOCH).total_seconds() / 3600.0 for d in dt_list])
        # 启发式：试常见量纲，取结果落在 1850-2150 年者
        v = np.asarray(arr, dtype="f8")
        if v.size == 0:
            return [], np.array([], dtype="f8")
        cands = [
            (_dt.datetime(1900, 1, 1), 1.0),      # hours since 1900
            (_dt.datetime(1970, 1, 1), 1.0),      # hours since 1970
            (_dt.datetime(1900, 1, 1), 24.0),     # days since 1900
            (_dt.datetime(1970, 1, 1), 24.0),     # days since 1970
            (_dt.datetime(1970, 1, 1), 1.0 / 3600.0),   # seconds since 1970
            (_dt.datetime(1900, 1, 1), 1.0 / 3600.0),   # seconds since 1900
        ]
        for base, fac in cands:
            try:
                dt_list = [base + _dt.timedelta(hours=float(x) * fac) for x in v]
            except OverflowError:
                continue
            ys = [d.year for d in dt_list]
            if ys and 1850 <= min(ys) and max(ys) <= 2150:
                return dt_list, np.array(
                    [(d - _EPOCH).total_seconds() / 3600.0 for d in dt_list])
        raise DataFileError(
            "无法识别 time 轴（dtype=%s，前几值=%s，attrs=%s）。"
            "请用 --target-time-units 指定，如 'hours since 1900-01-01'"
            % (arr.dtype, arr[:5].tolist(), attrs))

    # ------------------------------------------------------------ 读场

    @property
    def levels(self):
        return list(self.channel_names)

    def read(self, varname, time_idx=None):
        """读一个要素，返回 (T, lat, lon) float32。

        time_idx : array_like | None   目标 time 轴上的位置（需单调）。
            None=全时次；否则只读 [min,max] 一段再取所需位置，避免整轴读入。
        """
        v = str(varname).lower()
        if v not in self.channel_names:
            raise DataFileError("%s 中没有要素 %r，现有: %s"
                                % (self.path, varname, self.channel_names))
        i = self.channel_names.index(v)
        sl = [slice(None)] * len(self.shape)
        sl[self.axis] = i
        if time_idx is not None:
            idx = np.asarray(time_idx, dtype="i8")
            i0, i1 = int(idx.min()), int(idx.max())
            if i0 < 0 or i1 >= self.shape[0]:
                raise DataFileError("time_idx 越界（%d..%d，目标 %d 步）"
                                    % (i0, i1, self.shape[0]))
            sl[self.time_axis] = slice(i0, i1 + 1)
            a = np.asarray(self._data[tuple(sl)])
            a = np.take(a, idx - i0, axis=self.time_axis)
        else:
            a = np.asarray(self._data[tuple(sl)])
        a = np.transpose(a, (self._pos[self.time_axis],
                             self._pos[self.lat_axis],
                             self._pos[self.lon_axis]))
        a = np.where(np.abs(a) > 1.0e30, np.nan, a)   # GRIB 缺失值掩成 NaN
        if self.denorm:
            a = a * self.std[i] + self.mean[i]
        if self.tp_exp and v == "tp":
            a = np.exp(np.clip(a, 0.0, 7.0)) - 1.0
        return np.ascontiguousarray(a, dtype="f4")

    def read_multi(self, names, time_idx=None):
        """一次取多个通道（底层 data 只切片一次），返回 {小写名: (T,lat,lon)}。"""
        names = [str(n).lower() for n in names]
        for n in names:
            if n not in self.channel_names:
                raise DataFileError("%s 中没有要素 %r，现有: %s"
                                    % (self.path, n, self.channel_names))
        idxs = [self.channel_names.index(n) for n in names]
        sl = [slice(None)] * len(self.shape)
        sl[self.axis] = list(idxs)
        if time_idx is not None:
            idx = np.asarray(time_idx, dtype="i8")
            i0, i1 = int(idx.min()), int(idx.max())
            if i0 < 0 or i1 >= self.shape[0]:
                raise DataFileError("time_idx 越界（%d..%d，目标 %d 步）"
                                    % (i0, i1, self.shape[0]))
            sl[self.time_axis] = slice(i0, i1 + 1)
            a = np.asarray(self._data[tuple(sl)])
            a = np.take(a, idx - i0, axis=self.time_axis)
        else:
            a = np.asarray(self._data[tuple(sl)])
        out = {}
        for j, n in enumerate(names):
            aj = np.take(a, j, axis=self.axis)
            aj = np.transpose(aj, (self._pos[self.time_axis],
                                   self._pos[self.lat_axis],
                                   self._pos[self.lon_axis]))
            aj = np.where(np.abs(aj) > 1.0e30, np.nan, aj)
            if self.denorm:
                ci = self.channel_names.index(n)
                aj = aj * self.std[ci] + self.mean[ci]
            if self.tp_exp and n == "tp":
                aj = np.exp(np.clip(aj, 0.0, 7.0)) - 1.0
            out[n] = np.ascontiguousarray(aj, dtype="f4")
        return out
