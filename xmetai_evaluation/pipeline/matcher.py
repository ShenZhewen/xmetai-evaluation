"""
Matcher：预报与观测配对

按照 README 第 5.3 节，Matcher 负责：
1. 时间对齐（valid_time = init_time + lead_time）
2. 空间对齐（插值到共同网格）
3. 生成共同有效掩码
4. 计算权重
5. 构建 EvaluationBatch
"""

from dataclasses import replace
from typing import Dict, Any, List, Optional
import logging
import xarray as xr
import numpy as np

from xmetai_evaluation.core.contracts import (
    DataBundle,
    EvaluationBatch,
)
from xmetai_evaluation.core.errors import AlignmentError
from xmetai_evaluation.transforms.regrid import compute_latitude_weights

log = logging.getLogger(__name__)

#: 由分量合成的派生变量：名字 -> (u 分量, v 分量)。风速 = sqrt(u²+v²)，m/s。
DERIVED_VARIABLES: Dict[str, Any] = {
    "ws10m": ("u10m", "v10m"),
    "ws850": ("u850", "v850"),
    "ws200": ("u200", "v200"),
}


def align_lon_to(ds: xr.Dataset, target_lon: Any) -> xr.Dataset:
    """把 ``ds`` 的经度折算到 ``target_lon`` 所在的那一圈。

    不同数据源的经度原点会差 180°（如 ERA5 zarr 用 -180..180，预报文件用 0..360）。
    xarray 的 ``interp`` 在目标坐标超出源坐标范围时**直接返回 NaN**，不报错，
    所以不折算就会静默配对出一整片空场。经度是周期坐标，折算等价且不丢信息。

    折算后必须 ``sortby``：``[-180..180)`` 取模到 ``[0..360)`` 会变成降序，
    而 ``spectrum`` / ``zonal_spectrum`` 的 rFFT 依赖经度升序。
    """
    if "lon" not in ds.coords:
        return ds
    lon = np.asarray(ds["lon"].values)
    target = np.asarray(target_lon)
    if lon.size == 0 or target.size == 0:
        return ds
    lon_min = float(lon.min())
    target_min = float(target.min())
    if lon_min < 0.0 <= target_min:
        return ds.assign_coords(lon=lon % 360.0).sortby("lon")
    if lon_min >= 0.0 > target_min:
        return ds.assign_coords(lon=((lon + 180.0) % 360.0) - 180.0).sortby("lon")
    return ds


def align_to_grid(
    ds: xr.Dataset, target_lat: Any, target_lon: Any, atol: float = 1e-6
) -> Optional[xr.Dataset]:
    """把 ``ds`` 按**索引**搬到 ``target_lat`` / ``target_lon`` 这套网格上。

    只认「同一套网格，可能上下颠倒（lat 升序 vs 降序）或差半圈（经度原点不同）」
    这一种情况：lat 要么逐点相等、要么逐点反转；lon 要么相等、要么存在一个环形
    位移 ``k`` 能把源经度对齐到目标经度。对不上返回 ``None``，由调用方决定是否
    退回插值。

    这里一个插值都不做，是参考实现（``vfc/regr_ens.py:84-118``）的做法：两边本
    就是同一套全球网格时，``a[:, ::-1, :]`` 和 ``np.roll`` 是精确搬运——既没有
    双线性权重误差，也不会把 (时效 × 要素) 展平成 GB 级的中间数组。
    """
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return None
    lat = np.asarray(ds["lat"].values, dtype="f8")
    lon = np.asarray(ds["lon"].values, dtype="f8")
    want_lat = np.asarray(target_lat, dtype="f8")
    want_lon = np.asarray(target_lon, dtype="f8")

    if lat.shape != want_lat.shape or lon.shape != want_lon.shape or lon.size == 0:
        return None

    if np.allclose(lat, want_lat, rtol=0.0, atol=atol):
        flip = False
    elif np.allclose(lat, want_lat[::-1], rtol=0.0, atol=atol):
        flip = True
    else:
        return None

    # 位移量先用首元素定位候选，再整体比对确认：均匀网格上是 O(n)，
    # 逐个 k 去试是 O(n²)（1440² 也还好，但没必要）。
    roll = next(
        (
            int(k)
            for k in np.flatnonzero(np.isclose(lon, want_lon[0], rtol=0.0, atol=atol))
            if np.allclose(np.roll(lon, int(k)), want_lon, rtol=0.0, atol=atol)
        ),
        None,
    )
    if roll is None:
        return None

    if flip:
        ds = ds.isel(lat=slice(None, None, -1))
    if roll:
        ds = ds.roll(lon=roll, roll_coords=True)
    # 坐标按目标逐点写回：下游掩码/权重是按标签对齐的，浮点尾数级别的差异
    # 留在坐标里会让本可命中的配对悄悄落空。
    return ds.assign_coords(lat=want_lat, lon=want_lon)


def add_derived_variables(ds: xr.Dataset, wanted: List[str]) -> xr.Dataset:
    """按需合成派生变量（风速 = sqrt(u²+v²)）。

    只合成**被显式请求**且分量齐备的名字——不主动发明没人要的变量。
    """
    derived: Dict[str, Any] = {}
    for name, (u_name, v_name) in DERIVED_VARIABLES.items():
        if name in ds or name not in wanted:
            continue
        if u_name not in ds or v_name not in ds:
            continue
        field = np.sqrt(ds[u_name] ** 2 + ds[v_name] ** 2)
        field.attrs["units"] = "m/s"
        derived[name] = field
    return ds.assign(**derived) if derived else ds


def narrow_batch(batch: EvaluationBatch, variable: str) -> Optional[EvaluationBatch]:
    """把批次收窄到单个变量（按变量路由指标时用）。

    收窄后**必须重算 valid_mask**：批量掩码是所有变量的交集，十几个变量里
    只要有一个缺测，就会把其它变量的有效点一起砍掉。

    预报或观测里没有该变量时返回 None，调用方据此跳过这个 run。
    """

    def pick(source):
        if source is None:
            return None
        if isinstance(source, xr.Dataset):
            return source[variable] if variable in source.data_vars else None
        return source

    forecast = pick(batch.forecast)
    observation = pick(batch.observation)
    if forecast is None or observation is None:
        return None

    reference = pick(batch.reference)
    members = pick(batch.members)

    return replace(
        batch,
        forecast=forecast,
        observation=observation,
        reference=reference,
        members=members,
        valid_mask=forecast.notnull() & observation.notnull(),
    )


class Matcher:
    """
    预报与观测配对器

    负责时间、空间对齐，生成 EvaluationBatch。
    """

    def __init__(self, ensemble_reduction: str = "mean"):
        """
        Args:
            ensemble_reduction: 集合处理方式（'mean', 'median', 'none'）
        """
        self.ensemble_reduction = ensemble_reduction
        #: 经度口径是否做过折算，写进 batch.alignment 供排查"结果全 NaN"用
        self._lon_convention = "native"
        #: 空间对齐方式，同样写进 batch.alignment
        self._spatial = "forecast_regridded_to_observation"

    def match(
        self,
        forecast: DataBundle,
        observation: DataBundle,
        reference: Optional[DataBundle] = None,
        variables: Optional[List[str]] = None,
    ) -> List[EvaluationBatch]:
        """
        配对预报与观测

        Args:
            forecast: 预报数据（可能包含 init_time, lead_time, member 维度）
            observation: 观测数据（包含 valid_time 维度）
            reference: 可选的参考数据（如气候态）
            variables: 要评测的变量列表（None 表示全部）

        Returns:
            EvaluationBatch 列表（每个时次一个）
        """
        forecast_ds = forecast.payload if isinstance(forecast.payload, xr.Dataset) else forecast.payload.to_dataset()
        observation_ds = observation.payload if isinstance(observation.payload, xr.Dataset) else observation.payload.to_dataset()

        # 经度折算：观测在 -180..180 而预报在 0..360 时，不折算 interp 会全 NaN
        lon_before = (
            np.asarray(observation_ds["lon"].values)
            if "lon" in observation_ds.coords
            else None
        )
        observation_ds = align_lon_to(observation_ds, forecast_ds["lon"].values)
        if lon_before is not None and "lon" in observation_ds.coords:
            self._lon_convention = (
                "native"
                if np.array_equal(lon_before, np.asarray(observation_ds["lon"].values))
                else "observation_wrapped_to_forecast_circle"
            )

        # 派生变量必须在变量求交**之前**合成：ws* 任何文件里都没有，
        # 放到求交之后会被当成"源里没有这个变量"直接丢掉。
        wanted = (
            list(variables)
            if variables is not None
            else list(set(forecast_ds.data_vars) | set(observation_ds.data_vars))
        )
        forecast_ds = add_derived_variables(forecast_ds, wanted)
        observation_ds = add_derived_variables(observation_ds, wanted)

        # 确定变量
        if variables is None:
            variables = list(set(forecast_ds.data_vars) & set(observation_ds.data_vars))

        # 缺变量按参考实现语义跳过并告警：一个源少几个要素不该让整轮评测挂掉
        missing = [
            name
            for name in variables
            if name not in forecast_ds.data_vars or name not in observation_ds.data_vars
        ]
        if missing:
            log.warning("以下变量在预报或观测中不存在，已跳过: %s", ", ".join(missing))
            variables = [name for name in variables if name not in missing]

        if not variables:
            raise AlignmentError("No common variables between forecast and observation")

        # 只保留共同变量
        forecast_ds = forecast_ds[variables]
        observation_ds = observation_ds[variables]

        # 计算 valid_time（如果预报有 init_time 和 lead_time）
        sample_records = {}
        if "init_time" in forecast_ds.dims and "lead_time" in forecast_ds.dims:
            for init_time in forecast_ds.coords["init_time"].values:
                for lead_time in forecast_ds.coords["lead_time"].values:
                    valid_time = init_time + np.timedelta64(int(lead_time), "h")
                    sample_records[valid_time] = {
                        "init_time": str(init_time),
                        "lead_h": float(lead_time),
                    }
            forecast_ds = self._add_valid_time(forecast_ds)

        # 集合处理
        if "member" in forecast_ds.dims and self.ensemble_reduction == "mean":
            forecast_ds = forecast_ds.mean(dim="member")

        # 空间对齐：两个源本就是同一套全球网格时（lat 可能反向、lon 可能差半圈），
        # 按索引搬运即可，一格都不用插。
        #
        # 绝不能轻易落到 xarray.interp：它会把 (时效 × 要素) 一次性展平，
        # 60 时效 × 16 要素 = 6200 万个点逐个算双线性权重，结果数组 8GB 起步。
        # 参考实现（vfc/regr_ens.py 的 _align_info / _apply_target_align）从头到尾
        # 只做 a[:, ::-1, :] 和 np.roll，从不插值。
        aligned = align_to_grid(
            observation_ds, forecast_ds["lat"].values, forecast_ds["lon"].values
        )
        if aligned is not None:
            observation_ds = aligned
            self._spatial = "grids_identical"
            log.info(
                "预报与观测同网格（%d×%d），按索引对齐，跳过插值",
                forecast_ds.sizes.get("lat", 0),
                forecast_ds.sizes.get("lon", 0),
            )
            forecast_regridded = forecast_ds
        else:
            self._spatial = "forecast_regridded_to_observation"
            log.warning(
                "预报与观测不是同一套网格，退回双线性插值（很慢）。"
                "预报 lat %s..%s lon %s..%s；实况 lat %s..%s lon %s..%s",
                forecast_ds["lat"].values[0], forecast_ds["lat"].values[-1],
                forecast_ds["lon"].values[0], forecast_ds["lon"].values[-1],
                observation_ds["lat"].values[0], observation_ds["lat"].values[-1],
                observation_ds["lon"].values[0], observation_ds["lon"].values[-1],
            )
            forecast_regridded = forecast_ds.interp(
                lat=observation_ds.coords["lat"],
                lon=observation_ds.coords["lon"],
                method="linear",
            )

        # 时间对齐：找到共同的 valid_time
        if "valid_time" in forecast_regridded.dims and "valid_time" in observation_ds.dims:
            common_times = list(
                set(forecast_regridded.coords["valid_time"].values) &
                set(observation_ds.coords["valid_time"].values)
            )
            if not common_times:
                raise AlignmentError("No common valid_time between forecast and observation")

            forecast_regridded = forecast_regridded.sel(valid_time=common_times)
            observation_ds = observation_ds.sel(valid_time=common_times)

        # 计算权重（纬度余弦权重）
        weights = compute_latitude_weights(observation_ds)

        # 为每个时次创建 EvaluationBatch
        batches = []

        if "valid_time" in forecast_regridded.dims:
            moments = list(forecast_regridded.coords["valid_time"].values)
            for valid_time in moments:
                batch = self._create_batch(
                    forecast_regridded.sel(valid_time=valid_time),
                    observation_ds.sel(valid_time=valid_time),
                    weights,
                    valid_time,
                    reference,
                    sample_records.get(valid_time, {}),
                )
                batches.append(batch)
                if len(batches) % 20 == 0:
                    log.info("配对进度 %d/%d", len(batches), len(moments))
        else:
            # 单时次
            batch = self._create_batch(
                forecast_regridded,
                observation_ds,
                weights,
                None,
                reference,
            )
            batches.append(batch)

        return batches

    def _add_valid_time(self, forecast_ds: xr.Dataset) -> xr.Dataset:
        """添加 valid_time 坐标"""
        # valid_time = init_time + lead_time
        init_times = forecast_ds.coords["init_time"].values
        lead_times = forecast_ds.coords["lead_time"].values

        # 创建 valid_time 坐标
        valid_times = []
        for init_time in init_times:
            for lead_time in lead_times:
                valid_time = init_time + np.timedelta64(int(lead_time), 'h')
                valid_times.append(valid_time)

        # 重构数据集
        # 将 (init_time, lead_time) 展平为 valid_time
        forecast_flat = []
        for init_time in init_times:
            for lead_time in lead_times:
                ds_slice = forecast_ds.sel(init_time=init_time, lead_time=lead_time)
                valid_time = init_time + np.timedelta64(int(lead_time), 'h')
                ds_slice = ds_slice.expand_dims(valid_time=[valid_time])
                forecast_flat.append(ds_slice)

        forecast_ds = xr.concat(forecast_flat, dim="valid_time")

        return forecast_ds

    def _create_batch(
        self,
        forecast: xr.Dataset,
        observation: xr.Dataset,
        weights: xr.DataArray,
        valid_time: Optional[np.datetime64],
        reference: Optional[DataBundle],
        sample_record: Optional[Dict[str, Any]] = None,
    ) -> EvaluationBatch:
        """创建单个 EvaluationBatch"""
        # 计算共同有效掩码
        forecast_valid = ~forecast.isnull()
        observation_valid = ~observation.isnull()

        # 对所有变量取交集
        valid_mask = xr.DataArray(True, coords=observation.coords)
        for var in forecast.data_vars:
            valid_mask = valid_mask & forecast_valid[var] & observation_valid[var]

        # Sample key
        sample_keys = [
            {
                **(sample_record or {}),
                "valid_time": str(valid_time) if valid_time is not None else "unknown",
            }
        ]

        # 对齐记录
        alignment = {
            "spatial": self._spatial,
            "temporal": "valid_time_matched",
            "units": "assumed_consistent",
            "lon_convention": self._lon_convention,
        }

        return EvaluationBatch(
            forecast=forecast,
            observation=observation,
            sample_keys=sample_keys,
            valid_mask=valid_mask,
            reference=reference,
            weights=weights,
            sample_dim="grid",
            alignment=alignment,
        )
