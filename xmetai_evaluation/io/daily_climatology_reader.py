# -*- coding: utf-8 -*-
"""日序气候态 Reader（单文件，按"年内第几天"索引）。

与 ``io/climatology_reader.py`` 的区别是索引方式：

    ClimatologyReader   目录里一堆文件，按"月日+时次"找文件
    DailyClimatology    一个文件，时间轴就是年内日序（0…364 或 0…365）

**不能混用**：把单文件日序气候态接到按 MMDDHH 找文件的 Catalog 上，会在每个
有效时刻都把整份文件当成同一时刻，静默算错且不报错。

时间轴约定（与参考实现一致）：

    365 天文件    闰年的 2/29 并到 2/28（日序 0-based，闰年 d>=59 减 1）
    366 天文件    真日序（含 2/29）；平年 d>=59 加 1，跳过 2/29 那一格

``time:units`` 只取**单位**（days / hours / …），起算时刻被忽略——气候态索引的是
"年内位置"，不是绝对时刻。15 天环形平滑在读取时算（``--climo-window`` 口径）。

内存：整年 × 全球 × 14 要素很大，所以按纬度分块惰性读，峰值只与分块有关，
不把整年整块载入。
"""

from __future__ import annotations

import calendar
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import (
    DataBundle,
    DataIndex,
    DataRequest,
    Provenance,
    SemanticMetadata,
)
from xmetai_evaluation.core.errors import DecodeError, DiscoveryError
from xmetai_evaluation.core.variables import DataKind, TemporalKind
from xmetai_evaluation.io.base import DataCatalog, Reader

log = logging.getLogger(__name__)

#: 默认平滑窗口（天），与参考实现 ``--climo-window`` 一致
DEFAULT_WINDOW = 15

#: time:units 的单位 -> 小时换算系数
_HOURS_PER_UNIT = {
    "hour": 1.0,
    "hours": 1.0,
    "hr": 1.0,
    "h": 1.0,
    "day": 24.0,
    "days": 24.0,
    "d": 24.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "min": 1.0 / 60.0,
    "second": 1.0 / 3600.0,
    "seconds": 1.0 / 3600.0,
    "s": 1.0 / 3600.0,
}

#: 除 lat/lon/time 之外不作为变量名的元数据键
_NON_VARIABLE_KEYS = {"lat", "lon", "latitude", "longitude", "time", "level", "channel", "data"}


def _hours_per_unit(units: Any, path: Path) -> float:
    """从 ``time:units`` 里取出单位对应的小时系数（忽略起算时刻）。"""
    text = str(units or "").strip().lower()
    unit = text.partition(" since ")[0].strip()
    factor = _HOURS_PER_UNIT.get(unit)
    if factor is None:
        raise DecodeError(
            f"{path} 的 time 轴单位无法识别（需要 days / hours / minutes / seconds，"
            f"可带 ' since <时刻>'，实际是 {units!r}）"
        )
    return factor


def _day_of_year(moment: datetime) -> int:
    """365 天口径的 0-based 日序：闰年 d>=59 减 1，把 2/29 并到 2/28。"""
    day = moment.timetuple().tm_yday - 1
    if calendar.isleap(moment.year) and day >= 59:
        day -= 1
    return day


def _day_count(moment: datetime) -> int:
    """366 天口径的 0-based 日序：平年 d>=59 加 1，跳过 2/29 那一格。"""
    day = moment.timetuple().tm_yday - 1
    if not calendar.isleap(moment.year) and day >= 59:
        day += 1
    return day


def _day_fraction(moment: datetime) -> float:
    """当天已过的比例（时/分/秒），起报时次因此参与日序插值。"""
    return (
        moment.hour * 3600 + moment.minute * 60 + moment.second
    ) / 86400.0


def _decode_levels(values: Any) -> List[str]:
    """把 ``level`` 轴上的变量名解码成字符串（兼容 vlen 字符串和字符数组）。"""
    items = np.asarray(values)
    if items.ndim == 2:  # (n, 字符数) 的字符数组，逐行拼起来
        if items.dtype.kind == "U":
            return ["".join(str(cell) for cell in row).strip() for row in items]
        return [b"".join(row).decode("utf-8", "replace").strip() for row in items]
    return [
        item.decode("utf-8", "replace").strip() if isinstance(item, bytes) else str(item).strip()
        for item in items
    ]


class DailyClimatologyCatalog(DataCatalog):
    """日序气候态的"发现"：这份数据就是一个文件，所以只做路径解析与存在性校验。

    ``root_dir`` 可以是那个 .nc 文件本身，也可以是不含其它 .nc 的目录。
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def _resolve(self, request: DataRequest) -> Path:
        target = self.root_dir
        if target.is_file():
            return target
        if not target.exists():
            raise DiscoveryError(
                f"气候态路径不存在: {target}", source_id=request.source_id
            )
        matches = sorted(target.glob("*.nc"))
        if not matches:
            raise DiscoveryError(
                f"{target} 里没有 .nc 文件", source_id=request.source_id
            )
        if len(matches) > 1:
            raise DiscoveryError(
                f"{target} 里有 {len(matches)} 个 .nc 文件，无法确定用哪个: "
                + ", ".join(item.name for item in matches[:5]),
                source_id=request.source_id,
            )
        return matches[0]

    def discover(self, request: DataRequest) -> DataIndex:
        return DataIndex(
            source_id=request.source_id, available=[{"path": str(self._resolve(request))}]
        )


class DailyClimatologyReader(Reader):
    """把单文件日序气候态读成 ``(valid_time, lat, lon)`` 的参考场。

    按 ``request.init_times``（协议传的是有效时刻）取场，日序索引 + 线性插值，
    取场前先做 ``window`` 天的环形滑动平均。
    """

    def __init__(
        self,
        source_id: str = "daily_climatology",
        version: str = "1.0.0",
        window: int = DEFAULT_WINDOW,
        scales: Optional[Dict[str, float]] = None,
        units: Optional[Dict[str, str]] = None,
        chunk_points: int = 200_000_000,
    ):
        super().__init__(source_id=source_id, version=version)
        self.window = int(window)
        self.scales = {str(key): float(value) for key, value in (scales or {}).items()}
        self.units = dict(units or {})
        #: 单次读取的点数预算。取到 200M（f4 约 800MB）是为了让常见的几十步窗口
        #: **一次读完整个纬度范围**——纬度分块每多一块，就多一遍对同一批时间块的
        #: 重复解压。这里的默认值只在时间窗口大到装不下时才会真的切块。
        self.chunk_points = int(chunk_points)

    # -- 时间轴 -------------------------------------------------------------

    def _time_hours(self, dataset: xr.Dataset, path: Path) -> np.ndarray:
        """时间轴 -> 各时刻对应的小时数（0 = 年内第 0 天 00:00）。"""
        if "time" not in dataset.coords:
            raise DecodeError(f"{path} 没有 time 坐标，无法做日序索引")
        coord = dataset["time"]
        values = np.asarray(coord.values)
        if np.issubdtype(values.dtype, np.datetime64):
            return self._hours_from_datetime(values, path)
        factor = _hours_per_unit(coord.attrs.get("units"), path)
        return np.asarray(values, dtype="f8") * factor

    @staticmethod
    def _hours_from_datetime(values: np.ndarray, path: Path) -> np.ndarray:
        """datetime64 时间轴 -> 距首年 1/1 00:00 的小时数。

        合成年气候态把年内日序写成某一年（如 1990）的连续时刻，日序信息完整编码
        在时刻里，锚到该年 1/1 就与数值轴同口径。

        允许跨年：366 天口径从平年 1/1 起算时，第 366 天必然落到次年 1/1
        （如 ``era5_clim_phys_14.nc`` 的 1990-01-01 … 1991-01-01 18:00），
        那是正常的，不是错误。
        """
        moments = values.astype("datetime64[h]").astype("int64")
        first_year = int(values[0].astype("datetime64[Y]").astype("int64")) + 1970
        start = np.datetime64(f"{first_year:04d}-01-01", "h").astype("int64")
        hours = (moments - start).astype("f8")
        if hours[0] != 0.0:
            raise DecodeError(
                f"{path} 的 time 从 {values[0]} 起算，不是所在年的 1/1；"
                "日序气候态必须从年初开始，否则日序会对错格"
            )
        return hours

    def _steps_per_day(self, hours: np.ndarray, path: Path) -> int:
        if hours.size < 2:
            raise DecodeError(f"{path} 的时间轴至少要有 2 个时次才能判断步长")
        steps = np.diff(hours)
        if not np.allclose(steps, steps[0], atol=1e-6):
            raise DecodeError(f"{path} 的时间轴步长不均匀，无法做日序索引")
        step = float(steps[0])
        if step <= 0:
            raise DecodeError(f"{path} 的时间轴不是递增的")
        per_day = 24.0 / step
        if abs(per_day - round(per_day)) > 1e-6 or round(per_day) < 1:
            raise DecodeError(
                f"{path} 的时间轴步长 {step}h 不能整除 24 小时（每天 {per_day} 步）"
            )
        return int(round(per_day))

    def _year_length(self, hours: np.ndarray, steps_per_day: int, path: Path) -> int:
        total = hours.size / steps_per_day
        days = int(round(total))
        if abs(total - days) > 1e-6 or days not in (365, 366):
            raise DecodeError(
                f"{path} 的时间轴折合 {total} 天，只支持 365 天（闰年 2/29 并到 2/28）"
                f"或 366 天（含 2/29）两种约定"
            )
        return days

    # -- 变量场 -------------------------------------------------------------

    def _variable_names(self, dataset: xr.Dataset) -> List[str]:
        """变量名清单：legacy ``data`` 布局取自 level 轴，其余取顶层变量。"""
        if "data" in dataset.data_vars:
            if "level" not in dataset.coords:
                raise DecodeError(
                    "legacy 布局（data(time, level, lat, lon)）缺少 level 坐标，"
                    "无法确定变量名"
                )
            return _decode_levels(dataset["level"].values)
        return sorted(
            str(name)
            for name in dataset.data_vars
            if str(name).lower() not in _NON_VARIABLE_KEYS
        )

    def _field(self, dataset: xr.Dataset, name: str) -> Optional[xr.DataArray]:
        """取一个变量的 ``(time, lat, lon)`` 惰性视图（不立即载入）。"""
        if "data" in dataset.data_vars:
            levels = _decode_levels(dataset["level"].values)
            if name not in levels:
                return None
            field = dataset["data"].isel(level=levels.index(name))
        elif name in dataset.data_vars:
            field = dataset[name]
        else:
            return None

        drop = [
            dim
            for dim in field.dims
            if dim not in ("time", "lat", "lon") and field.sizes[dim] == 1
        ]
        if drop:
            field = field.squeeze(dim=drop, drop=True)
        if not {"time", "lat", "lon"}.issubset(field.dims):
            raise DecodeError(
                f"变量 {name} 的维度是 {field.dims}，无法规范成 (time, lat, lon)"
            )
        return field.transpose("time", "lat", "lon")

    # -- 平滑 ---------------------------------------------------------------

    def _smooth(self, slab: np.ndarray, half: int) -> np.ndarray:
        """沿时间轴做环形滑动平均（年内首尾相接）。"""
        if half <= 0:
            return slab
        ksize = 2 * half + 1
        padded = np.concatenate([slab[-half:], slab, slab[:half]], axis=0)
        cumulative = np.cumsum(padded, axis=0, dtype="f8")
        cumulative = np.concatenate(
            [np.zeros((1,) + cumulative.shape[1:], dtype="f8"), cumulative], axis=0
        )
        return ((cumulative[ksize:] - cumulative[:-ksize]) / ksize).astype("f4")

    # -- Reader 接口 --------------------------------------------------------

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        if not request.init_times:
            raise DecodeError(
                "日序气候态需要显式给出要取的有效时刻（request.init_times）"
            )
        path = Path(index.available[0]["path"])
        times = sorted(request.init_times)

        with xr.open_dataset(path) as dataset:
            hours = self._time_hours(dataset, path)
            steps_per_day = self._steps_per_day(hours, path)
            year_days = self._year_length(hours, steps_per_day, path)
            total_steps = hours.size
            half = int((self.window * steps_per_day) // 2)
            half = max(0, min(half, (total_steps - 1) // 2))

            available = self._variable_names(dataset)
            wanted_names = [str(name) for name in request.variables]
            resolved = [name for name in wanted_names if name in available]
            missing = [name for name in wanted_names if name not in available]
            if missing:
                log.warning("气候态里没有这些变量，已跳过: %s", ", ".join(missing))
            if not resolved:
                raise DecodeError(
                    f"气候态 {path} 里没有请求的变量 {wanted_names}；"
                    f"文件里有: {available}"
                )

            # 每个有效时刻 -> (i0, i1, 权重)，i0/i1 是年内的整数步号
            picks = [
                self._pick(moment, steps_per_day, total_steps, year_days)
                for moment in times
            ]
            wanted_steps = sorted({step for pick in picks for step in (pick[0], pick[1])})
            log.info(
                "气候态 %s: %d 个有效时刻，涉及 %d 个年内步号（全年 %d 步），"
                "平滑窗 ±%d 步，逐要素读取 %d 个",
                path.name, len(times), len(wanted_steps), total_steps, half,
                len(resolved),
            )

            lat = np.asarray(dataset["lat"].values, dtype="f8")
            lon = np.asarray(dataset["lon"].values, dtype="f8")
            fields: Dict[str, xr.DataArray] = {}
            units: Dict[str, str] = {}

            for order, name in enumerate(resolved, start=1):
                field = self._field(dataset, name)
                if field is None:
                    continue
                started = time.perf_counter()
                values = self._gather(
                    field, wanted_steps, picks, half, len(lat), len(lon)
                )
                log.info(
                    "气候态要素 %d/%d: %s (%.1fs)",
                    order, len(resolved), name, time.perf_counter() - started,
                )
                scale = self.scales.get(name, 1.0)
                if scale != 1.0:
                    values = values * scale
                unit = self.units.get(name, "unknown")
                fields[name] = xr.DataArray(
                    values,
                    dims=("valid_time", "lat", "lon"),
                    coords={"valid_time": times, "lat": lat, "lon": lon},
                    name=name,
                    attrs={"units": unit},
                )
                units[name] = unit

        if not fields:
            raise DecodeError(f"气候态 {path} 没有取到任何请求的变量")
        payload = xr.Dataset(fields)
        return DataBundle(
            payload=payload,
            kind=DataKind.REFERENCE,
            source_id=request.source_id,
            standard_vars={name: name for name in payload.data_vars},
            semantic=SemanticMetadata(
                units=units,
                temporal_kind=TemporalKind.INSTANTANEOUS,
                grid_type="regular_latlon",
            ),
            provenance=Provenance(
                input_files=[str(path)],
                reader_id="daily_climatology",
                reader_version=self.version,
            ),
        )

    def _pick(
        self, moment: datetime, steps_per_day: int, total_steps: int, year_days: int
    ) -> Tuple[int, int, float]:
        """有效时刻 -> (前一步号, 后一步号, 后一步权重)，首尾环形相接。"""
        day = _day_of_year(moment) if year_days == 365 else _day_count(moment)
        position = (day + _day_fraction(moment)) * steps_per_day
        floor = int(np.floor(position))
        weight = position - floor
        lower = floor % total_steps
        upper = (lower + 1) % total_steps
        return lower, upper, float(weight)

    def _gather(
        self,
        field: xr.DataArray,
        wanted_steps: Sequence[int],
        picks: Sequence[Tuple[int, int, float]],
        half: int,
        lat_size: int,
        lon_size: int,
    ) -> np.ndarray:
        """算出所有有效时刻的场。

        **只读 ``wanted_steps`` 前后各 ``half`` 步，不是整条时间轴。** 一年 1464
        步里真正用到的通常只有几十步；而 NetCDF/HDF5 基本都沿时间切块，读整条
        时间轴 = 把整份文件解压一遍。乘以纬度分块数（早先 20M 点的分块预算是
        9 行/块，721 行就是 81 块）就是几十次全量解压——几十 TB 的重复 IO，
        在日志上就是"卡死不动"。

        参考实现（``vfc/climo.py`` 的 ``read_slice`` / ``_rolling_raw``）同样是
        只读窗口内的行区间。

        ``half`` 步的余量是给平滑窗留的：``rows`` 按环形展开，取场的位置都在
        块内部，离块边至少 ``half`` 步，所以 ``_smooth`` 在块边界上的环形假设
        不会影响任何一个取到的点。
        """
        total = int(field.sizes["time"])
        wanted = np.asarray(wanted_steps, dtype=int)

        # 需要读的原始行：选中行各向两侧扩 half 步，按环形区间 [low, high] 展开。
        # 拆成至多两段**连续递增**行号，而不是一个取模后的整数数组：非单调索引
        # 会让后端退化成"读 min..max 再挑"，越年界那次照样把整年读一遍。
        low = int(wanted.min()) - half
        high = int(wanted.max()) + half
        if high - low + 1 >= total:
            segments = [(0, total - 1)]
        else:
            begin = low % total
            span = high - low + 1
            if begin + span <= total:
                segments = [(begin, begin + span - 1)]
            else:
                segments = [(begin, total - 1), (0, (begin + span - 1) % total)]
        rows = np.concatenate([np.arange(a, b + 1, dtype=int) for a, b in segments])
        spot = {int(step): order for order, step in enumerate(rows)}

        lower = np.asarray([spot[int(pick[0])] for pick in picks], dtype=int)
        upper = np.asarray([spot[int(pick[1])] for pick in picks], dtype=int)
        weights = np.asarray([pick[2] for pick in picks], dtype="f4")[:, None, None]

        out = np.empty((len(picks), lat_size, lon_size), dtype="f4")
        rows_per_chunk = max(1, self.chunk_points // max(1, rows.size * lon_size))
        for start in range(0, lat_size, rows_per_chunk):
            stop = min(start + rows_per_chunk, lat_size)
            pieces = [
                np.asarray(
                    field.isel(time=slice(a, b + 1), lat=slice(start, stop)).values,
                    dtype="f4",
                )
                for a, b in segments
            ]
            slab = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
            smoothed = self._smooth(slab, half)
            out[:, start:stop, :] = (
                (1.0 - weights) * smoothed[lower] + weights * smoothed[upper]
            )
        return out
