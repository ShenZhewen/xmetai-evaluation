"""
Diamond 格式站点数据 Reader

文件格式：/station_dir/YYYYMMDD/HH.000
每个文件包含一个小时的所有站点观测。

文件头：
diamond  3 2025年1月1日0时1小时降水(逐时)
25 1 1 0 1000 0 0 0 0 1 77017

数据行：
站号 经度 纬度 高度(m) 降水(mm)
45004 114.1728 22.3119 66.4 0.0
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

from xmetai_evaluation.io.base import Reader, DataCatalog
from xmetai_evaluation.core.contracts import (
    DataRequest,
    DataIndex,
    DataBundle,
    DataKind,
    SemanticMetadata,
    Provenance,
)
from xmetai_evaluation.core.errors import DiscoveryError, DecodeError
from xmetai_evaluation.core.variables import TemporalKind


class DiamondStationCatalog(DataCatalog):
    """
    Diamond 站点文件目录

    扫描 station_dir/YYYYMMDD/HH.000 文件。
    """

    def __init__(self, station_dir: Path, station_whitelist: Optional[List[int]] = None):
        """
        Args:
            station_dir: 站点数据根目录
            station_whitelist: 站点白名单（站号列表），None 表示全部站点
        """
        self.station_dir = Path(station_dir)
        self.station_whitelist = set(station_whitelist) if station_whitelist else None

    def discover(self, request: DataRequest) -> DataIndex:
        """
        发现指定时间范围的站点文件

        支持两种布局：
        1. 扁平布局：所有 .000 文件直接在 station_dir 下（如 YYYYMMDDHH.000）
        2. 分层布局：station_dir/YYYYMMDD/HH.000

        Args:
            request: 数据请求

        Returns:
            DataIndex，available 包含文件路径列表
        """
        if not self.station_dir.exists():
            raise DiscoveryError(
                f"Station directory does not exist: {self.station_dir}",
                source_id=request.source_id,
            )

        # 首先尝试扁平布局：直接查找 .000 文件
        flat_files = sorted(self.station_dir.glob("*.000"))

        if flat_files:
            # 扁平布局
            return DataIndex(
                source_id=request.source_id,
                available=flat_files,
                ambiguous=[],
            )

        # 尝试分层布局：扫描日期子目录
        date_dirs = sorted([d for d in self.station_dir.iterdir()
                           if d.is_dir() and d.name.isdigit()])

        if not date_dirs:
            raise DiscoveryError(
                f"No .000 files or date directories found in {self.station_dir}",
                source_id=request.source_id,
            )

        # 收集所有 .000 文件
        all_files = []
        for date_dir in date_dirs:
            files = sorted(date_dir.glob("*.000"))
            all_files.extend(files)

        if not all_files:
            raise DiscoveryError(
                f"No .000 files found in {self.station_dir}",
                source_id=request.source_id,
            )

        return DataIndex(
            source_id=request.source_id,
            available=all_files,
            ambiguous=[],
        )


class DiamondStationReader(Reader):
    """
    Diamond 格式站点观测 Reader

    读取多个小时文件，拼接为时间序列。

    输出：xr.Dataset with dims (time, station)
    coords: time, station_id, lat, lon, altitude
    data_vars: precipitation (mm)
    """

    def __init__(self, source_id: str = "diamond_station", version: str = "1.0.0"):
        super().__init__(source_id=source_id, version=version)

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        """
        读取站点文件并拼接为 xarray Dataset

        Args:
            request: 数据请求
            index: 文件索引

        Returns:
            DataBundle(kind=STATION_OBSERVATION)
        """
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)

        # 使用参考实现同样的 NumPy 路径组装二维站点数组。
        # 不把所有时次拼成超大 DataFrame，也不使用 pandas Python engine。
        parsed = []
        pbar = tqdm(
            index.available,
            desc="解析站点文件",
            unit="文件",
            ncols=100,
        )
        for fpath in pbar:
            try:
                item = self._parse_diamond_file(fpath)
                if item is not None:
                    parsed.append(item)
            except Exception as exc:
                raise DecodeError(
                    f"Failed to parse station file {fpath}: {exc}",
                    source_id=request.source_id,
                    path=str(fpath),
                    cause=exc,
                ) from exc
        pbar.close()

        if not parsed:
            raise DecodeError("No valid station data parsed", source_id=request.source_id)

        time_index = sorted(item["time"] for item in parsed)
        station_ids = set()
        for item in parsed:
            station_ids.update(int(sid) for sid in item["station_id"])
        station_index = np.asarray(sorted(station_ids), dtype="i8")
        if station_index.size == 0:
            raise DecodeError("No stations in requested files", source_id=request.source_id)

        n_times = len(parsed)
        n_stations = len(station_index)
        precip_data = np.full((n_times, n_stations), np.nan, dtype="f4")
        lat_data = np.full(n_stations, np.nan, dtype="f4")
        lon_data = np.full(n_stations, np.nan, dtype="f4")
        alt_data = np.full(n_stations, np.nan, dtype="f4")
        station_map = {int(sid): i for i, sid in enumerate(station_index)}

        for time_idx, item in enumerate(parsed):
            source_positions = np.asarray(
                [i for i, sid in enumerate(item["station_id"]) if int(sid) in station_map],
                dtype="i8",
            )
            positions = np.asarray(
                [station_map[int(item["station_id"][i])] for i in source_positions],
                dtype="i8",
            )
            precip_data[time_idx, positions] = item["precipitation"][source_positions]
            if time_idx == 0:
                lat_data[positions] = item["lat"][source_positions]
                lon_data[positions] = item["lon"][source_positions]
                alt_data[positions] = item["altitude"][source_positions]

        # 统一按时间排序；Catalog 通常已经排序，但不能依赖文件系统顺序。
        order = np.argsort(np.asarray(time_index))
        precip_data = precip_data[order]
        time_index = [time_index[i] for i in order]

        ds = xr.Dataset(
            {"precipitation": (["time", "station"], precip_data)},
            coords={
                "time": time_index,
                "station": station_index,
                "station_id": ("station", station_index),
                "lat": ("station", lat_data),
                "lon": ("station", lon_data),
                "altitude": ("station", alt_data),
            },
        )
        ds["precipitation"].attrs.update(units="mm", long_name="Hourly precipitation")

        semantic = SemanticMetadata(
            units={"precipitation": "mm"},
            temporal_kind=TemporalKind.INTERVAL_ACCUMULATION,
            grid_type="stations",
        )
        provenance = Provenance(
            input_files=[str(item["path"]) for item in parsed],
            reader_id=self.source_id,
            reader_version=self.version,
        )
        return DataBundle(
            payload=ds,
            kind=DataKind.STATION_OBSERVATION,
            source_id=request.source_id,
            standard_vars={"precipitation": "tp"},
            semantic=semantic,
            provenance=provenance,
        )

    def _parse_diamond_file(self, fpath: Path) -> Optional[Dict[str, Any]]:
        """
        解析单个 Diamond 文件

        支持两种文件名格式：
        1. YYYYMMDDHH.000 - 扁平布局
        2. HH.000 - 分层布局（需要从父目录名获取日期）

        Args:
            fpath: 文件路径

        Returns:
            {'time': datetime, 'df': DataFrame} 或 None（文件无效）
        """
        try:
            fname = fpath.stem  # 去掉 .000

            # 尝试解析扁平布局：YYYYMMDDHH.000
            if len(fname) == 10 and fname.isdigit():
                year = int(fname[0:4])
                month = int(fname[4:6])
                day = int(fname[6:8])
                hour = int(fname[8:10])
                obs_time = datetime(year, month, day, hour)
            # 尝试解析分层布局：日期目录/HH.000
            elif len(fname) == 2 and fname.isdigit():
                parent_dir = fpath.parent.name
                if len(parent_dir) == 8 and parent_dir.isdigit():
                    year = int(parent_dir[0:4])
                    month = int(parent_dir[4:6])
                    day = int(parent_dir[6:8])
                    hour = int(fname)
                    obs_time = datetime(year, month, day, hour)
                else:
                    return None
            else:
                return None

            # 参考实现使用 NumPy 直接读取数值列，避免 pandas Python engine 的开销。
            rows = np.loadtxt(fpath, skiprows=2, encoding="gbk", ndmin=2)
            if rows.shape[1] < 5:
                raise ValueError("数据列不足 5 列")
            if rows.shape[0] == 0:
                return None
            order = np.argsort(rows[:, 0].astype("i8"))
            rows = rows[order]
            return {
                "path": fpath,
                "time": obs_time,
                "station_id": rows[:, 0].astype("i8"),
                "lon": rows[:, 1].astype("f4"),
                "lat": rows[:, 2].astype("f4"),
                "altitude": rows[:, 3].astype("f4"),
                "precipitation": rows[:, 4].astype("f4"),
            }

        except Exception as e:
            # 调试：可以打开这行看哪些文件解析失败
            # print(f"Failed to parse {fpath}: {e}")
            return None

    @classmethod
    def with_catalog(
        cls,
        station_dir: Path,
        source_id: str = "diamond_station",
        station_whitelist: Optional[List[int]] = None,
    ):
        """
        便捷构造器：自动创建配套的 Catalog

        Returns:
            (reader, catalog) tuple
        """
        reader = cls(source_id=source_id)
        catalog = DiamondStationCatalog(
            station_dir=station_dir,
            station_whitelist=station_whitelist,
        )
        return reader, catalog
