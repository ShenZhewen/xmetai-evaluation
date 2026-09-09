# -*- coding: utf-8 -*-
"""站点层：diamond3 逐小时观测 + BSS 气候概率参考（ref/MMDDHH.000）。

观测（diamond3 样例 r1_samples/*.000，GBK 编码）：
  第 1 行  diamond 3 <描述文字>
  第 2 行  <yy mm dd hh level ... ... 站数>     # 末位数字 = 数据行数
  数据行   站号  经度  纬度  站高(m)  1h降水量(mm)   # 降水 0 也有记录

约定与坑：
  * 文件时刻为**北京时**，小时量记在"该小时结束"的整点（如 21 时的文件
    = 20—21 时累计）；pred（ERA5 系）为 UTC，对时需 +8h 换算（--tz-shift）；
  * 累积窗（结束于 V、长 W 小时）= 站点文件 V−W+1 … V 共 W 个；
  * **完整性剔除**：站号必须在窗内**每个**小时文件里都有记录行，
    缺任一小时即整站剔除（上游规则：降水 0 也应报文，缺行=该时次缺测）。

参考（ref 目录）：
  * ``MMDDHH.000`` 逐 6h 气候概率文件，每行一个站：
    ``站号 p≥0.1 p≥4 p≥13 p≥25``（4 列概率，值 [0,1]，供 BSS 参考）。
  * 文件名为**北京时**，HH 为 6h 窗口的**结束时刻**。
"""
from __future__ import annotations

import datetime as _dt
import glob
import os

import numpy as np


class StationDataError(RuntimeError):
    """站点文件结构/一致性问题。"""


class HourlyStations(object):
    """一个时次的站点观测表。"""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            line1 = f.readline().decode("gbk", "ignore")
            line2 = f.readline().decode("gbk", "ignore")
        if "diamond" not in line1.lower():
            raise StationDataError("%s 第 1 行不是 diamond 头: %r"
                                   % (self.path, line1[:40]))
        head = line2.split()
        try:
            yy, mm, dd, hh = (int(float(x)) for x in head[:4])
            self.n_declared = int(float(head[-1]))
        except (ValueError, IndexError):
            raise StationDataError("%s 第 2 行无法解析: %r"
                                   % (self.path, line2[:60]))
        self.time = _dt.datetime(2000 + yy, mm, dd, hh)      # 两位年 +2000
        rows = np.loadtxt(self.path, skiprows=2, encoding="gbk",
                          ndmin=2)
        if rows.shape[1] < 5:
            raise StationDataError("%s 数据列不足 5 列（站号/经度/纬度/站高/降水）"
                                   % self.path)
        if rows.shape[0] != self.n_declared:
            raise StationDataError("%s 声明 %d 站，实际 %d 行"
                                   % (self.path, self.n_declared,
                                      rows.shape[0]))
        self.id = rows[:, 0].astype("i8")
        order = np.argsort(self.id)
        self.id = self.id[order]
        self.lon, self.lat, self.alt, self.rain = (
            rows[order, 1], rows[order, 2], rows[order, 3], rows[order, 4])
        bad = ~np.isfinite(self.rain)
        if bad.any():
            raise StationDataError("%s 有 %d 个非数值降水量"
                                   % (self.path, int(bad.sum())))

    def __len__(self):
        return self.id.size


def load_hour(path) -> HourlyStations:
    return HourlyStations(path)


def accumulate_window(hour_paths, start, end):
    """把 [start, end]（闭区间，逐时）站点文件累加成一个观测表。

    返回 (HourlyStations 视图 dict)：id/lon/lat/alt/rain(累计 mm)、
    n_excluded(因缺任一小时被剔除的站数)。任一小时文件缺失则抛错。
    """
    hours = [start + _dt.timedelta(hours=k)
             for k in range(int((end - start).total_seconds() // 3600) + 1)]
    tables = {}
    for t in hours:
        if t not in hour_paths:
            raise StationDataError("缺少 %s 时次的站点文件" % t)
        hs = hour_paths[t]
        if hs.time != t:
            raise StationDataError("站点文件 %s 头时间 %s 与索引时刻 %s 不符"
                                   % (hs.path, hs.time, t))
        tables[t] = hs
    common = set(tables[hours[0]].id.tolist())
    universe = set(tables[hours[0]].id.tolist())
    for t in hours[1:]:
        ids = set(tables[t].id.tolist())
        common &= ids
        universe |= ids
    first = tables[hours[0]]
    keep = np.isin(first.id, np.array(sorted(common), dtype="i8"))
    n_excluded = len(universe) - len(common)          # 相对全体小时并集
    rain = first.rain[keep].copy()
    for t in hours[1:]:
        hs = tables[t]
        sel = np.isin(hs.id, first.id[keep])
        rain += hs.rain[sel]                    # 站号已排序，两表同序
    return {
        "time": end,
        "window_hours": len(hours),
        "id": first.id[keep], "lon": first.lon[keep], "lat": first.lat[keep],
        "alt": first.alt[keep], "rain": rain,
        "n_excluded": n_excluded,
    }


def scan_station_dir(directory, prefix_pattern="%Y%m%d%H"):
    """扫描目录建立 {datetime: 路径}；文件名（去扩展名）按模式解析为时刻。"""
    import re
    idx = {}
    for fn in os.listdir(directory):
        base, ext = os.path.splitext(fn)
        try:
            t = _dt.datetime.strptime(base, prefix_pattern)
        except ValueError:
            continue
        idx[t] = os.path.join(directory, fn)
    return idx


def load_station_whitelist(path):
    """读站号清单（zd_sta_*.dat），返回站号 set[int]。

    文件为 UTF-8 文本，每行 5 列：``站号 经度 纬度 站号 地区``。
    取第 1 列作为站号（主站号）。
    """
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            try:
                ids.add(int(float(p[0])))
            except (ValueError, IndexError):
                continue
    return ids


def subset_stations(acc, keep_ids):
    """按站号白名单过滤 accumulate_window 的返回表，只保留 keep_ids 内的站。

    被白名单剔除的站数并入 n_excluded（与完整性剔除合并计数）。
    """
    keep_arr = np.asarray(sorted({int(x) for x in keep_ids}), dtype="i8")
    keep = np.isin(acc["id"], keep_arr)
    return {
        "time": acc["time"],
        "window_hours": acc["window_hours"],
        "id": acc["id"][keep],
        "lon": acc["lon"][keep],
        "lat": acc["lat"][keep],
        "alt": acc["alt"][keep],
        "rain": acc["rain"][keep],
        "n_excluded": acc["n_excluded"] + int((~keep).sum()),
    }


class RefProb(object):
    """BSS 的逐 6h 气候概率参考（按验证时刻北京时查表）。"""

    def __init__(self, ref_dir):
        self.ref_dir = str(ref_dir)
        self._index = {}          # MMDDHH -> 文件路径
        self._cache = {}          # MMDDHH -> {站号: np.array(4)}
        for p in glob.glob(os.path.join(self.ref_dir, "*.000")):
            self._index[os.path.basename(p)[:6]] = p
        if not self._index:
            raise ValueError("ref 目录 %s 下无 .000 文件" % self.ref_dir)

    def _load(self, key):
        if key not in self._cache:
            a = np.loadtxt(self._index[key], ndmin=2)
            self._cache[key] = {int(sid): a[i, 1:]
                                for i, sid in enumerate(a[:, 0])}
        return self._cache[key]

    def probabilities(self, when, station_ids):
        """when: 北京时 datetime；station_ids: 站号数组 → (n, 4) 概率矩阵。

        缺 ref 的站对应行置 NaN（4 列全 NaN），由调用方决定跳过。
        """
        key = when.strftime("%m%d%H")
        if key not in self._index:
            raise ValueError("ref 缺少时次 %s（%s）" % (key, when))
        table = self._load(key)
        out = np.full((len(station_ids), 4), np.nan, dtype="f8")
        for i, sid in enumerate(station_ids):
            row = table.get(int(sid))
            if row is not None:
                out[i] = row
        return out

