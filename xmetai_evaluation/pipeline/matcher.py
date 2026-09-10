"""
Matcher：预报与观测配对

按照 README 第 5.3 节，Matcher 负责：
1. 时间对齐（valid_time = init_time + lead_time）
2. 空间对齐（插值到共同网格）
3. 生成共同有效掩码
4. 计算权重
5. 构建 EvaluationBatch
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

from xmetai_evaluation.core.contracts import (
    DataBundle,
    EvaluationBatch,
)
from xmetai_evaluation.core.errors import AlignmentError
from xmetai_evaluation.transforms.regrid import (
    regrid_to_target,
    compute_ensemble_mean,
    compute_latitude_weights,
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

        # 确定变量
        if variables is None:
            variables = list(set(forecast_ds.data_vars) & set(observation_ds.data_vars))

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

        # 空间插值：将预报插值到观测网格
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
            for valid_time in forecast_regridded.coords["valid_time"].values:
                batch = self._create_batch(
                    forecast_regridded.sel(valid_time=valid_time),
                    observation_ds.sel(valid_time=valid_time),
                    weights,
                    valid_time,
                    reference,
                    sample_records.get(valid_time, {}),
                )
                batches.append(batch)
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
            "spatial": "forecast_regridded_to_observation",
            "temporal": "valid_time_matched",
            "units": "assumed_consistent",
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
