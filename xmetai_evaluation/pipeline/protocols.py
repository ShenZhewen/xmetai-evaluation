# -*- coding: utf-8 -*-
"""验证协议：决定样本空间怎么遍历、配对怎么做。

一个协议只回答三个问题：

    prepare()      这次评测要先把哪些数据准备好（读观测、建索引）；
    samples()      要遍历哪些样本，每个样本对应长表里的哪个结果键；
    build_batch()  这个样本怎么变成可计算的配对数据。

样本循环、状态合并和落盘都在 ``pipeline/runner.py``，协议里不允许出现它们。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import DataIndex, DataRequest, EvaluationBatch
from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.pipeline.matcher import (
    Matcher,
    add_derived_variables,
    align_lon_to,
    align_to_grid,
)
from xmetai_evaluation.pipeline.spec import PipelineSpec, daily_times
from xmetai_evaluation.transforms.temporal import window_sum_at

log = logging.getLogger(__name__)

#: 作用于 DataBundle 的变换（在协议里按声明顺序执行）
BUNDLE_TRANSFORMS = ("ensemble_mean",)


def _never_requires_members() -> bool:
    """没有 needs_members() 的指标（如自定义实现）默认不需要成员。"""
    return False


@dataclass
class Sample:
    """一个待评测样本：结果键 + 长表坐标 + 协议内部负载。"""

    key: Tuple
    coordinates: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """执行上下文：数据源、变换链、指标由 Runner 组装后交给协议。"""

    spec: PipelineSpec
    forecast: Any
    observation: Any
    reference: Any = None
    transforms: Dict[str, Any] = field(default_factory=dict)
    metrics: List[Any] = field(default_factory=list)

    def transform(self, name: str) -> Any:
        return self.transforms.get(name)


class Protocol(ABC):
    """验证协议基类。"""

    name = "protocol"

    def __init__(self, spec: PipelineSpec):
        self.spec = spec

    def prepare(self, context: PipelineContext) -> None:
        """准备一次性的数据（观测、索引等）。"""

    @abstractmethod
    def samples(self, context: PipelineContext) -> Iterator[Sample]:
        """产出样本；生成器内部可以流式读取预报，控制内存峰值。"""

    @abstractmethod
    def build_batch(
        self, context: PipelineContext, sample: Sample
    ) -> Optional[EvaluationBatch]:
        """把样本配对成 EvaluationBatch；返回 None 表示这个样本跳过。"""

    def defaults(self, context: PipelineContext) -> Dict[str, Any]:
        """长表的兜底坐标。"""
        return {}

    def expected_samples(self) -> Optional[int]:
        """预计样本数（用于进度显示）；未知时返回 None。"""
        return None

    def summary(self) -> Dict[str, Any]:
        """追加进 manifest 的协议统计。"""
        return {}


class StationValidTimeProtocol(Protocol):
    """站点观测与格点预报按有效时刻配对。

    - 样本空间：每个起报的每个完整窗口时效（window, 2*window, ...）；
    - 结果键：``("lead_h", lead)``；
    - 观测：一次读入后按 [valid - window + 1h, valid] 闭区间累积；
    - 空间：预报双线性插值到站点。
    """

    name = "station_valid_time"

    def __init__(self, spec: PipelineSpec):
        super().__init__(spec)
        self.forecast_var = spec.variables.get("forecast") or spec.forecast.params.get("variable")
        self.observation_var = spec.variables.get("observation") or spec.observation.params.get(
            "variable"
        )
        if not self.forecast_var or not self.observation_var:
            raise ConfigError(
                "station_valid_time 协议需要在配置里声明 forecast/observation 的 variable"
            )
        self.window_hours = spec.window_hours
        self.sample_unit = str(spec.options.get("sample_unit", "station"))
        self.init_times: List[datetime] = []
        self.forecast_reader = None
        self.forecast_request: Optional[DataRequest] = None
        self.forecast_index: Optional[DataIndex] = None
        self.observation_bundle = None
        self.observation_ds: Optional[xr.Dataset] = None
        self.station_lats: Optional[np.ndarray] = None
        self.station_lons: Optional[np.ndarray] = None
        self.window_leads: Optional[List[float]] = None
        self.skipped_inits = 0
        self.ref_reader = None
        self._diagnostics_left = 3

    def prepare(self, context: PipelineContext) -> None:
        forecast = context.forecast
        observation = context.observation
        start, end = self.spec.period()
        self.forecast_reader = forecast.reader

        # 预报：先按日期范围探测可用起报，再按真实可用起报建立索引
        discovered = forecast.catalog.discover(
            DataRequest(
                source_id=forecast.source_id,
                variables=[self.forecast_var],
                init_times=daily_times(start, end),
            )
        )
        init_times = forecast.reader.available_init_times(discovered)
        if not init_times:
            raise ValueError(f"时间范围 {start} 到 {end} 内没有可用预报起报时间")
        if self.spec.limit:
            init_times = init_times[: self.spec.limit]
        self.init_times = init_times
        log.info(
            "起报时间: %s 到 %s，共 %d 个",
            init_times[0],
            init_times[-1],
            len(init_times),
        )

        self.forecast_request = DataRequest(
            source_id=forecast.source_id,
            variables=[self.forecast_var],
            init_times=init_times,
            lead_times=self.spec.forecast.params.get("lead_times"),
        )
        self.forecast_index = forecast.catalog.discover(self.forecast_request)
        lead_max = int(forecast.reader.max_lead_hours(self.forecast_index))
        log.info("预报文件发现完成：%d 个起报，最大时效 %sh", len(init_times), lead_max)

        # 观测：一次读取覆盖所有起报和时效，后续窗口只做内存索引
        offset = timedelta(hours=self.spec.local_utc_offset_hours)
        obs_start = min(init_times) + offset + timedelta(hours=1 - self.window_hours)
        obs_end = max(init_times) + offset + timedelta(hours=lead_max)
        obs_request = DataRequest(
            source_id=observation.source_id, variables=[self.observation_var]
        )
        catalog_files = observation.catalog.discover(obs_request).available
        selected = [
            path
            for path in catalog_files
            if obs_start <= observation.catalog.file_time(path) <= obs_end
        ]
        selected.sort(key=observation.catalog.file_time)
        log.info("观测窗口覆盖 %s 到 %s，共 %d 个文件", obs_start, obs_end, len(selected))
        if not selected:
            raise ValueError("没有找到评估所需的观测文件")

        self.observation_bundle = observation.reader.read(
            obs_request, DataIndex(source_id=observation.source_id, available=selected)
        )
        self.observation_ds = self.observation_bundle.payload
        self.region = self.spec.options.get("region")
        if self.region:
            lat_range = self.region.get("lat", (-90.0, 90.0))
            lon_range = self.region.get("lon", (0.0, 360.0))
            lats = self.observation_ds["lat"].values
            lons = self.observation_ds["lon"].values
            keep = (
                (lats >= lat_range[0])
                & (lats <= lat_range[1])
                & (lons >= lon_range[0])
                & (lons <= lon_range[1])
            )
            self.observation_ds = self.observation_ds.isel(station=keep)
            log.info("区域筛选 %s 后站点数: %d", self.region, int(keep.sum()))
        self.station_lats = self.observation_ds["lat"].values
        self.station_lons = self.observation_ds["lon"].values
        self.station_weights = None
        if str(self.spec.options.get("weights", "none")) == "cos_lat":
            self.station_weights = np.cos(np.deg2rad(np.abs(self.station_lats)))
        log.info("观测读取完成: %s", self.observation_ds.sizes)

        # 外部 BSS 气候概率参考（可选）：站点协议只认 ref_probability 这类逐站参考
        if context.reference is not None:
            self.ref_reader = getattr(context.reference, "reader", None)
            if self.ref_reader is None or not hasattr(self.ref_reader, "probabilities"):
                log.warning(
                    "站点协议忽略参考源 %s：不是逐站气候概率参考（需 ref_probability）",
                    context.reference.source_id,
                )
                self.ref_reader = None

    def samples(self, context: PipelineContext) -> Iterator[Sample]:
        accumulator = context.transform("time_window_accumulator")
        reader = context.forecast.reader
        # 只有声明了需要成员的概率指标时，才额外保留成员级窗口场（多约 1/5 内存）
        keep_members = any(
            getattr(metric, "needs_members", _never_requires_members)()
            for metric in (context.metrics or [])
        )

        for init_time in self.init_times:
            try:
                bundle = reader.read_one(
                    self.forecast_request, self.forecast_index, init_time
                )
                raw_field = bundle.payload[self.forecast_var].isel(init_time=0)
                bundle = self._apply_bundle_transforms(context, bundle)
                field = bundle.payload[self.forecast_var].isel(init_time=0)
                # 预报已是窗口累积量（如 6h 降水文件）时无需再累积
                windows = accumulator.transform(field) if accumulator is not None else field
                member_windows = None
                if keep_members:
                    if "member" in windows.dims:
                        member_windows = windows
                    elif "member" in raw_field.dims:
                        member_windows = (
                            accumulator.transform(raw_field) if accumulator is not None else raw_field
                        )
            except Exception as exc:
                self.skipped_inits += 1
                log.exception("起报 %s 预报读取失败: %s", init_time, exc)
                continue

            if self.window_leads is None:
                self.window_leads = [
                    float(lead)
                    for lead in windows.lead_time.values
                    if float(lead) >= self.window_hours
                    and abs(float(lead) % self.window_hours) < 1e-6
                ]
                log.info("评估时效: %s", self.window_leads)
                log.info("站点数量: %d", len(self.station_lats))

            for lead in self.window_leads:
                if lead not in windows.lead_time.values:
                    continue
                lead_key = int(lead) if float(lead).is_integer() else float(lead)
                yield Sample(
                    key=("lead_h", lead_key),
                    coordinates={
                        "lead_h": lead_key,
                        "window_h": self.window_hours,
                        "variable": self.observation_var,
                        "sample_unit": self.sample_unit,
                    },
                    payload={
                        "init_time": init_time,
                        "lead": lead,
                        "windows": windows,
                        "member_windows": member_windows,
                    },
                )

    def _apply_bundle_transforms(self, context: PipelineContext, bundle: Any) -> Any:
        """按声明顺序应用作用于 DataBundle 的变换（如集合降维）。"""
        for item in self.spec.transforms:
            if item.name not in BUNDLE_TRANSFORMS:
                continue
            transform = context.transform(item.name)
            if transform is not None and hasattr(transform, "transform"):
                bundle = transform.transform(bundle)
        return bundle

    def build_batch(
        self, context: PipelineContext, sample: Sample
    ) -> Optional[EvaluationBatch]:
        interpolator = context.transform("grid_to_station")
        if interpolator is None:
            raise ValueError("station_valid_time 协议需要 grid_to_station 变换")

        init_time = sample.payload["init_time"]
        lead = sample.payload["lead"]
        windows = sample.payload["windows"]
        valid_time = init_time + timedelta(
            hours=float(lead) + self.spec.local_utc_offset_hours
        )

        observation = window_sum_at(
            self.observation_ds,
            self.observation_var,
            valid_time,
            self.window_hours,
            time_dim="time",
            require_complete=str(
                self.spec.options.get("observation_window", "complete")
            ) != "reference",
        )
        forecast_at_lead = windows.sel(lead_time=lead)
        forecast_at_station = interpolator.transform(
            forecast_at_lead, self.station_lats, self.station_lons
        )
        members = None
        member_windows = sample.payload.get("member_windows")
        if member_windows is not None:
            members = interpolator.transform(
                member_windows.sel(lead_time=lead), self.station_lats, self.station_lons
            )
        valid_mask = xr.DataArray(
            np.isfinite(forecast_at_station.values) & np.isfinite(observation.values),
            dims=forecast_at_station.dims,
            coords=forecast_at_station.coords,
        )
        weights = None
        if self.station_weights is not None:
            weights = xr.DataArray(
                self.station_weights,
                dims=["station"],
                coords={"station": forecast_at_station["station"].values},
            )
        reference = None
        if self.ref_reader is not None:
            station_ids = self.observation_ds["station"].values
            probs = self.ref_reader.probabilities(valid_time, station_ids)  # (n, 4)
            reference = xr.DataArray(
                probs,
                dims=("station", "threshold"),
                coords={
                    "station": station_ids,
                    "threshold": list(self.ref_reader.thresholds),
                },
            )
        if int(valid_mask.sum()) == 0:
            return None
        if self._diagnostics_left > 0:
            self._diagnostics_left -= 1
            log.info(
                "样本诊断 init=%s lead=%s forecast_shape=%s obs_shape=%s n_valid=%d",
                init_time,
                lead,
                forecast_at_station.shape,
                observation.shape,
                int(valid_mask.sum()),
            )

        return EvaluationBatch(
            forecast=forecast_at_station,
            observation=observation,
            sample_keys=[
                {"init": init_time.isoformat(), "lead": float(lead)}
            ]
            * len(self.station_lats),
            valid_mask=valid_mask,
            members=members,
            reference=reference,
            weights=weights,
            sample_dim=self.sample_unit,
            alignment={
                "method": "bilinear",
                "window_hours": self.window_hours,
                "valid_time": valid_time.isoformat(),
                "timezone_offset_hours": self.spec.local_utc_offset_hours,
            },
            protocol_id=self.name,
        )

    def defaults(self, context: PipelineContext) -> Dict[str, Any]:
        return {
            "variable": self.observation_var,
            "sample_unit": self.sample_unit,
            "window_h": self.window_hours,
            "unit": self._unit(),
            "region": self.region.get("name", "") if self.region else "",
        }

    def expected_samples(self) -> Optional[int]:
        """起报数 × 评估时效数（时效要在读到第一个起报后才知道）。"""
        if not self.init_times or not self.window_leads:
            return None
        return len(self.init_times) * len(self.window_leads)

    def _unit(self) -> str:
        """评分单位取自观测语义（预报与观测在配对时已要求同单位）。"""
        bundle = self.observation_bundle
        if bundle is None:
            return ""
        return str((bundle.semantic.units or {}).get(self.observation_var, ""))

    def summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "n_init_times": len(self.init_times),
            "n_skipped_inits": self.skipped_inits,
            "window_hours": self.window_hours,
        }
        if self.forecast_index is not None:
            summary["forecast_input_files"] = self._forecast_input_files()
        if self.observation_bundle is not None:
            summary["observation_input_files"] = list(
                self.observation_bundle.provenance.input_files
            )
        return summary

    def _forecast_input_files(self) -> List[str]:
        """预报输入文件清单（由 Reader 从索引还原，不依赖索引内部形状）。"""
        reader = getattr(self.forecast_reader, "input_files", None)
        if reader is None or self.forecast_index is None:
            return []
        return reader(self.forecast_index)


class GridValidTimeProtocol(Protocol):
    """格点预报与格点实况按 valid_time 配对（连续场评测）。

    - 样本空间：预报 (init_time, lead_time) 展平出的有效时刻；
    - 结果键：``("valid_time", 有效时刻)``；
    - 起报：配置显式声明 ``init_times`` 时用它，否则按 ``start_date``/``end_date``
      探测预报源里实际可用的起报（与 ``station_valid_time`` 同一口径）；
    - 集合降维：由声明的 ``ensemble_mean`` 变换完成；
    - 空间：Matcher 把预报插值到实况网格。
    """

    name = "grid_valid_time"

    def __init__(self, spec: PipelineSpec):
        super().__init__(spec)
        self.forecast_vars = _variable_list(spec.forecast.params, "forecast")
        self.observation_vars = _variable_list(spec.observation.params, "observation")
        self.forecast_bundle = None
        self.member_bundle = None
        self.observation_bundle = None
        self.reference_bundle = None
        #: 需要成员场的指标所路由到的变量；空表示这次评测不取成员
        self.member_vars: List[str] = []
        self.batches: Dict[Any, EvaluationBatch] = {}

    def prepare(self, context: PipelineContext) -> None:
        forecast = context.forecast
        observation = context.observation
        # 起报：配置显式声明优先；没声明就按评测时段探测可用起报
        # （与 station_valid_time 同一口径，配置只需给 start_date/end_date）
        init_times = [
            datetime.fromisoformat(str(value))
            for value in self.spec.forecast.params.get("init_times", [])
        ]
        if not init_times:
            start, end = self.spec.period()
            discovered = forecast.catalog.discover(
                DataRequest(
                    source_id=forecast.source_id,
                    variables=self.forecast_vars,
                    init_times=daily_times(start, end),
                )
            )
            init_times = forecast.reader.available_init_times(discovered)
            if not init_times:
                raise ValueError(f"时间范围 {start} 到 {end} 内没有可用预报起报时间")
        if self.spec.limit:
            init_times = init_times[: self.spec.limit]
        log.info(
            "起报时间: %s 到 %s，共 %d 个", init_times[0], init_times[-1], len(init_times)
        )
        lead_times = [float(value) for value in self.spec.forecast.params.get("lead_times", [])]

        forecast_request = DataRequest(
            source_id=forecast.source_id,
            variables=self.forecast_vars,
            init_times=init_times,
            lead_times=lead_times or None,
        )
        bundle = forecast.reader.read(
            forecast_request, forecast.catalog.discover(forecast_request)
        )
        raw_bundle = bundle
        for item in self.spec.transforms:
            transform = context.transform(item.name)
            if transform is not None and hasattr(transform, "transform"):
                bundle = transform.transform(bundle)
        self.forecast_bundle = bundle
        log.info("预报读取完成: %s", bundle.payload.dims)

        # 概率/集合类指标需要原始成员：集合平均是把成员降到均值，不能替代成员
        keep_members = any(
            getattr(metric, "needs_members", _never_requires_members)()
            for metric in (context.metrics or [])
        )
        self.member_vars = self._member_variables(context) if keep_members else []
        if self.member_vars and "member" in raw_bundle.payload.dims:
            self.member_bundle = raw_bundle

        valid_times = sorted(
            {
                init_time + timedelta(hours=float(lead))
                for init_time in init_times
                for lead in (lead_times or self._available_leads(bundle.payload))
            }
        )
        observation_request = DataRequest(
            source_id=observation.source_id,
            variables=self.observation_vars,
            init_times=valid_times,
        )
        self.observation_bundle = observation.reader.read(
            observation_request, observation.catalog.discover(observation_request)
        )
        log.info("实况读取完成: %s", self.observation_bundle.payload.dims)

        if context.reference is not None:
            reference_request = DataRequest(
                source_id=context.reference.source_id,
                variables=self.observation_vars,
                init_times=valid_times,
            )
            self.reference_bundle = context.reference.reader.read(
                reference_request,
                context.reference.catalog.discover(reference_request),
            )
            log.info("气候态参考读取完成: %s", self.reference_bundle.payload.dims)

        matcher = Matcher(
            ensemble_reduction=str(self.spec.options.get("ensemble_reduction", "mean"))
        )
        for batch in matcher.match(
            forecast=self.forecast_bundle,
            observation=self.observation_bundle,
            variables=self.forecast_vars,
        ):
            key = batch.sample_keys[0].get("valid_time") if batch.sample_keys else None
            batch.members = self._members_for(batch)
            batch.reference = self._reference_for(batch)
            self.batches[key] = batch
            if len(self.batches) % 20 == 0:
                log.info("参考场装配进度 %d", len(self.batches))
        log.info("配对完成：%d 个批次", len(self.batches))

    @staticmethod
    def _available_leads(payload: xr.Dataset) -> List[float]:
        """没声明 ``lead_times`` 时，按预报文件自带的时效展开（单位：小时）。

        与 ``station_valid_time`` 同一口径（那边也是读 ``windows.lead_time.values``）。
        这里不能退化成 ``[0.0]``：那样实况和气候态只会被请求一个有效时刻，
        Matcher 随之只配出一个批次，评测静默缩水成 1/N 且不报错。
        """
        if "lead_time" not in payload.coords:
            return [0.0]
        return [float(value) for value in payload["lead_time"].values]

    def _member_variables(self, context: PipelineContext) -> List[str]:
        """需要成员场的指标路由到了哪些变量。

        只给这些变量取成员，而不是全部预报变量：成员场按 (init, lead) 展开后
        很大，且会一直留在 ``self.batches`` 里。
        """
        routed: List[str] = []
        for item, metric in zip(self.spec.metrics, context.metrics or []):
            if not getattr(metric, "needs_members", _never_requires_members)():
                continue
            for name in item.params.get("variables") or self.forecast_vars:
                if name not in routed:
                    routed.append(name)
        return routed

    def _members_for(self, batch: EvaluationBatch) -> Optional[xr.Dataset]:
        """把成员场按样本的 init/lead 取出来，插值到实况网格。

        始终返回 Dataset（哪怕只有一个变量），好让 ``narrow_batch`` 能按变量取用——
        返回裸 DataArray 会在路由到别的变量时被误用。
        """
        if self.member_bundle is None or not batch.sample_keys:
            return None
        available = [
            name for name in self.member_vars if name in self.member_bundle.payload.data_vars
        ]
        if not available:
            return None
        record = batch.sample_keys[0]
        try:
            field = self.member_bundle.payload[available].sel(
                init_time=np.datetime64(record["init_time"]),
                lead_time=float(record["lead_h"]),
            )
        except Exception as exc:
            log.debug("成员场取用失败（%s）：%s", record, exc)
            return None
        if "member" not in field.dims:
            return None
        # 对齐目标取**批次里的实况**，不是原始 bundle（理由同 ``_reference_for``）
        target = batch.observation
        aligned = align_to_grid(field, target["lat"].values, target["lon"].values)
        if aligned is not None:
            return aligned
        return field.interp(lat=target["lat"], lon=target["lon"])

    def _reference_for(self, batch: EvaluationBatch) -> Optional[xr.Dataset]:
        """取该有效时刻的气候态参考场，插值到实况网格。

        同样返回 Dataset（全部观测变量），由 ``narrow_batch`` 按变量取用。
        """
        if self.reference_bundle is None or not batch.sample_keys:
            return None
        available = [
            name
            for name in self.observation_vars
            if name in self.reference_bundle.payload.data_vars
        ]
        if not available:
            return None
        record = batch.sample_keys[0]
        try:
            field = self.reference_bundle.payload[available].sel(
                valid_time=np.datetime64(record["valid_time"])
            )
        except Exception as exc:
            log.debug("气候态取用失败（%s）：%s", record, exc)
            return None
        # 对齐目标取**批次里的实况**，不是原始 bundle：Matcher 会把实况经度折算到
        # 预报那一圈（ERA5 的 -180..180 -> 0..360），原始 bundle 仍是折算前的。
        # 拿原始 bundle 当目标，参考场和实况就会落在两套经度上，ACC/活跃度
        # 按标签对齐后全是 NaN。
        target = batch.observation
        # 气候态可能和预报不在同一圈经度上；缺 wsX 时用分量现合成（参考实现如此兜底）
        field = align_lon_to(field, target["lon"].values)
        field = add_derived_variables(field, self.observation_vars)
        aligned = align_to_grid(field, target["lat"].values, target["lon"].values)
        if aligned is not None:
            return aligned
        return field.interp(lat=target["lat"], lon=target["lon"])

    def samples(self, context: PipelineContext) -> Iterator[Sample]:
        variable = self.forecast_vars[0] if len(self.forecast_vars) == 1 else ""
        for key in sorted(self.batches):
            batch = self.batches[key]
            record = batch.sample_keys[0] if batch.sample_keys else {}
            yield Sample(
                key=("valid_time", key),
                coordinates={
                    "variable": variable,
                    "valid_time": key,
                    "init_time": record.get("init_time", ""),
                    "lead_h": record.get("lead_h", ""),
                    "sample_unit": batch.sample_dim,
                },
                payload={"valid_time": key},
            )

    def build_batch(
        self, context: PipelineContext, sample: Sample
    ) -> Optional[EvaluationBatch]:
        return self.batches.get(sample.payload["valid_time"])

    def defaults(self, context: PipelineContext) -> Dict[str, Any]:
        return {
            "variable": self.forecast_vars[0] if len(self.forecast_vars) == 1 else "",
            "sample_unit": "grid",
            "unit": self._unit(),
        }

    def expected_samples(self) -> Optional[int]:
        """配对批次在准备阶段就确定。"""
        return len(self.batches) or None

    def _unit(self) -> str:
        """评分单位取自实况语义。"""
        bundle = self.observation_bundle
        if bundle is None or len(self.forecast_vars) != 1:
            return ""
        return str((bundle.semantic.units or {}).get(self.forecast_vars[0], ""))

    def summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"n_batches": len(self.batches)}
        for name, bundle in (
            ("forecast", self.forecast_bundle),
            ("observation", self.observation_bundle),
        ):
            if bundle is None:
                continue
            provenance = bundle.provenance
            summary[f"{name}_input_files"] = list(provenance.input_files)
            summary[f"{name}_reader"] = f"{provenance.reader_id}@{provenance.reader_version}"
        return summary


def _variable_list(params: Dict[str, Any], side: str) -> List[str]:
    names = params.get("variables")
    if names:
        return [str(name) for name in names]
    if params.get("variable"):
        return [str(params["variable"])]
    raise ConfigError(f"{side} 数据源必须声明 variable 或 variables")
