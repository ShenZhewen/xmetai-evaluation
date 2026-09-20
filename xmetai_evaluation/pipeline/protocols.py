# -*- coding: utf-8 -*-
"""验证协议：决定样本空间怎么遍历、配对怎么做。

一个协议只回答三个问题：

    prepare()      这次评测要先把哪些数据准备好（读观测、建索引）；
    samples()      要遍历哪些样本，每个样本对应长表里的哪个结果键；
    build_batch()  这个样本怎么变成可计算的配对数据。

样本循环在 ``execution/executor.py``、状态合并与落盘在 ``pipeline/runner.py``，
协议里不允许出现它们。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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
from xmetai_evaluation.pipeline.spec import PipelineSpec, daily_times, hourly_times
from xmetai_evaluation.transforms.temporal import window_sum_at

log = logging.getLogger(__name__)

#: 作用于 DataBundle 的变换（在协议里按声明顺序执行）
BUNDLE_TRANSFORMS = ("ensemble_mean",)


def station_observation_span(
    init_times: List[datetime],
    offset_hours: float,
    window_hours: int,
    lead_max_hours: int,
) -> Tuple[datetime, datetime]:
    """站点观测需要覆盖的时间窗（协议与执行计划层共用的唯一口径）。

    最早起报的窗口起点在 valid - window + 1h，最晚起报的最长时效是收尾。
    """
    offset = timedelta(hours=offset_hours)
    return (
        min(init_times) + offset + timedelta(hours=1 - window_hours),
        max(init_times) + offset + timedelta(hours=lead_max_hours),
    )


def sample_leads(available_leads: Sequence[float], window_hours: float) -> List[float]:
    """可评的采样时效集：只取**完整累积窗**的末端时效。

    规则只此一份——计划层（切时效窗）与协议（遍历样本）都调它，避免两处各写
    一遍导致口径漂移。约束对没有累积变换的流程同样成立：``window_hours`` 以内
    的时效不是可评的窗口末端（预报已是 6h 累积量时 ``window_hours=6`` 只保留
    6 的倍数，天然排掉 0 时效的瞬时场，否则会与 6h 累积实况配出垃圾分）。
    """
    span = float(window_hours)
    return [
        float(lead)
        for lead in available_leads
        if float(lead) >= span and abs(float(lead) % span) < 1e-6
    ]


def select_observation_files(
    observation: Any, observation_var: str, start: datetime, end: datetime
) -> Tuple[DataRequest, List[Any]]:
    """按时间窗从 catalog 挑观测文件（协议直读与加载策略层共用）。"""
    request = DataRequest(
        source_id=observation.source_id, variables=[observation_var]
    )
    catalog_files = observation.catalog.discover(request).available
    selected = [
        path
        for path in catalog_files
        if start <= observation.catalog.file_time(path) <= end
    ]
    selected.sort(key=observation.catalog.file_time)
    return request, selected


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
    """执行上下文：数据源、变换链、指标由执行器组装后交给协议。"""

    spec: PipelineSpec
    forecast: Any
    observation: Any
    reference: Any = None
    transforms: Dict[str, Any] = field(default_factory=dict)
    metrics: List[Any] = field(default_factory=list)
    #: 数据加载策略层（分块执行时由执行器注入；None 表示协议自己直读）
    loader: Any = None

    def transform(self, name: str) -> Any:
        return self.transforms.get(name)


class Protocol(ABC):
    """验证协议基类。"""

    name = "protocol"

    #: 跨工作块出现重复结果键时的合并语义（执行层读，协议自己不用）：
    #:   append  同键的状态追加（如 station 协议：同 lead 下多个起报累积）；
    #:   replace 后块覆盖前块（如 grid 协议：同一 valid_time 只认最新起报）。
    duplicate_key_policy = "append"

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
        #: 预报读取失败的起报（ISO 串）。报列表不报计数：切时效后同一个起报
        #: 会出现在多个块里，跨块相加会把一个坏起报数成好几个。
        self.skipped_init_times: List[str] = []
        self.ref_reader = None
        #: 每块前 N 个样本打一条"样本诊断"（形状 + 有效站数），用来定位"样本
        #: 出不来"。**它是块口径、不是 run 口径**：这个协议实例每个工作块重建
        #: 一次（``executor.collect_chunk``），所以实际条数是 N × 块数——全年那趟
        #: 五千多块就是一万六千多行。默认关掉，排查样本层问题时再改回 3。
        self._diagnostics_left = 0

    def prepare(self, context: PipelineContext) -> None:
        forecast = context.forecast
        observation = context.observation
        self.forecast_reader = forecast.reader

        # 起报：显式声明优先（分块执行时计划层按块注入），否则按评测时段探测
        init_times = [
            datetime.fromisoformat(str(value))
            for value in self.spec.forecast.params.get("init_times", [])
        ]
        if not init_times:
            start, end = self.spec.period()
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
        # 逐块流水（每块一次）：INFO 留给块级进度，这些细节归 DEBUG，
        # 需要时用 --log-file（文件 handler 是 DEBUG）全量取回。
        log.debug(
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
        log.debug("预报文件发现完成：%d 个起报，最大时效 %sh", len(init_times), lead_max)

        # 观测：一次读取覆盖所有起报和时效，后续窗口只做内存索引。
        # 走加载策略层时（resident）整个 run 只读一次，块间共享同一份缓存。
        obs_start, obs_end = station_observation_span(
            init_times,
            self.spec.local_utc_offset_hours,
            self.window_hours,
            lead_max,
        )
        if context.loader is not None:
            self.observation_bundle = context.loader.materialize(
                "observation",
                DataRequest(
                    source_id=observation.source_id,
                    variables=[self.observation_var],
                    init_times=hourly_times(obs_start, obs_end),
                ),
            )
        else:
            obs_request, selected = select_observation_files(
                observation, self.observation_var, obs_start, obs_end
            )
            log.info(
                "观测窗口覆盖 %s 到 %s，共 %d 个文件",
                obs_start,
                obs_end,
                len(selected),
            )
            if not selected:
                raise ValueError("没有找到评估所需的观测文件")
            self.observation_bundle = observation.reader.read(
                obs_request,
                DataIndex(source_id=observation.source_id, available=selected),
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
        log.debug("观测读取完成: %s", self.observation_ds.sizes)

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
        # 这一块到底产出了什么：一个样本都没产出时，块层面只会记成「没有成功
        # 处理任何评测批次」，原因得由这里说清楚。
        emitted = 0
        seen_leads: set = set()
        read_failures = 0

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
                if "lead_time" in windows.coords:
                    seen_leads.update(
                        float(value) for value in windows.lead_time.values
                    )
                member_windows = None
                if keep_members:
                    if "member" in windows.dims:
                        member_windows = windows
                    elif "member" in raw_field.dims:
                        member_windows = (
                            accumulator.transform(raw_field) if accumulator is not None else raw_field
                        )
            except Exception as exc:
                self.skipped_init_times.append(init_time.isoformat())
                read_failures += 1
                log.exception("起报 %s 预报读取失败: %s", init_time, exc)
                continue

            if self.window_leads is None:
                # 分块执行时计划层按块声明采样时效（它已按采样格点切窗，是
                # (起报, 时效) 的纯划分）；没声明就按完整窗口末端自己推。
                # 这里刻意**不**与本次 init 的可用轴求交：这个列表只算一次，
                # 拿首个 init 的轴去交会把后面 init 本可评的时效静默截短。
                declared = self.spec.forecast.params.get("sample_leads")
                self.window_leads = (
                    [float(lead) for lead in declared]
                    if declared
                    else sample_leads(windows.lead_time.values, self.window_hours)
                )
                log.debug("评估时效: %s", self.window_leads)
                log.debug("站点数量: %d", len(self.station_lats))

            for lead in self.window_leads:
                if lead not in windows.lead_time.values:
                    continue
                emitted += 1
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

        if emitted == 0:
            # 一个样本都没产出：块会记成「没有成功处理任何评测批次」。原因无非
            # 两种——声明要评的时效根本没被累积器产出（多因该起报的预报文件不
            # 够窗口所需步数，且预报侧静默丢了帧），或者起报在读取阶段就全失败
            # 了（那种情况上面已有 ERROR）。这里把两者一起打出来，省得再猜。
            log.warning(
                "起报窗口无样本可评：声明采样时效 %s，累积器只产出时效 %s，"
                "预报读取失败 %d 个起报",
                self.window_leads,
                sorted(seen_leads)[:20],
                read_failures,
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

    def _unit(self) -> str:
        """评分单位取自观测语义（预报与观测在配对时已要求同单位）。"""
        bundle = self.observation_bundle
        if bundle is None:
            return ""
        return str((bundle.semantic.units or {}).get(self.observation_var, ""))

    def summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            # 报时刻列表而不是计数：切时效后同一个起报会出现在多个块里，
            # 跨块相加会放大（归并时列表取并集，计数由列表长度得出）
            "init_times": [value.isoformat() for value in self.init_times],
            "skipped_init_times": list(self.skipped_init_times),
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
    """格点预报与格点实况按时间配对（连续场评测）。

    采样键口径由流程参数 ``sample_by`` 决定：

    - ``valid_time``（缺省）：样本空间是预报 (init_time, lead_time) 展平出的
      有效时刻，**一个有效时刻一个样本**——多个起报够到同一时刻时只留最新
      起报；结果键 ``("valid_time", 有效时刻)``。
    - ``init_lead``：**每个 (起报, 时效) 各出一个样本**，同一有效时刻有几个
      起报就有几个样本（逐起报报满整段时效用它）；结果键
      ``("init_lead", 起报, 时效)``。

    其余相同：起报显式声明 ``init_times`` 时用它，否则按 ``start_date``/
    ``end_date`` 探测预报源里实际可用的起报（与 ``station_valid_time`` 同一
    口径）；集合降维由声明的 ``ensemble_mean`` 变换完成；空间上由 Matcher
    把预报插值到实况网格。
    """

    name = "grid_valid_time"

    #: 同一 valid_time 可能被多个起报够到（如逐日起报 + 长时效重叠）。
    #: 单段执行时 Matcher 内部"最新起报获胜"；分块执行下由执行层按块序
    #: 覆盖重现同一语义：同键只保留时间上最后一块（= 最新起报）的状态。
    #: ``init_lead`` 口径下键是 (起报, 时效)、不再撞车，``__init__`` 会把它
    #: 改成 append——那时再按同键覆盖就是把别的起报的样本丢掉。
    duplicate_key_policy = "replace"

    def __init__(self, spec: PipelineSpec):
        super().__init__(spec)
        #: 采样键口径，见类 docstring；缺省 valid_time，行为与加这个开关之前一致。
        self.sample_by = str(spec.options.get("sample_by", "valid_time") or "valid_time")
        if self.sample_by not in ("valid_time", "init_lead"):
            raise ConfigError(
                f"sample_by 只能是 valid_time 或 init_lead，收到 {self.sample_by!r}"
            )
        self._init_lead = self.sample_by == "init_lead"
        if self._init_lead:
            self.duplicate_key_policy = "append"
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
        # 逐块流水（每块一次）：INFO 留给块级进度，这些细节归 DEBUG，
        # 需要时用 --log-file（文件 handler 是 DEBUG）全量取回。
        log.debug(
            "起报时间: %s 到 %s，共 %d 个", init_times[0], init_times[-1], len(init_times)
        )
        # ``or []``：键写了但值是 None（"不限制时效"）时也要当"没声明"。
        lead_times = [
            float(value) for value in self.spec.forecast.params.get("lead_times") or []
        ]
        # 采样时效（窗口内要出分的样本）与读取时效（可能带累积预热）是两回事：
        # 分块执行时计划层只声明前者，后者是 lead_times。
        sample_lead_times = [
            float(value) for value in self.spec.forecast.params.get("sample_leads", [])
        ]

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
        log.debug("预报读取完成: %s", bundle.payload.dims)

        # 概率/集合类指标需要原始成员：集合平均是把成员降到均值，不能替代成员
        keep_members = any(
            getattr(metric, "needs_members", _never_requires_members)()
            for metric in (context.metrics or [])
        )
        self.member_vars = self._member_variables(context) if keep_members else []
        if self.member_vars and "member" in raw_bundle.payload.dims:
            self.member_bundle = raw_bundle

        # 用**采样**时效而不是读取时效：读取集可能带累积预热时效，把预热时效也
        # 算进 valid_time 会让 matcher 多配出批次，那些批次既不是本窗的样本、
        # 又会跨窗重复，破坏采样集的纯划分。
        valid_times = sorted(
            {
                init_time + timedelta(hours=float(lead))
                for init_time in init_times
                for lead in (
                    sample_lead_times
                    or lead_times
                    or self._available_leads(bundle.payload)
                )
            }
        )
        observation_request = DataRequest(
            source_id=observation.source_id,
            variables=self.observation_vars,
            init_times=valid_times,
        )
        if context.loader is not None:
            # resident/window 时整段共享缓存，逐块只取自己跨度内的切片
            self.observation_bundle = context.loader.materialize(
                "observation", observation_request
            )
        else:
            self.observation_bundle = observation.reader.read(
                observation_request, observation.catalog.discover(observation_request)
            )
        log.debug("实况读取完成: %s", self.observation_bundle.payload.dims)

        if context.reference is not None:
            reference_request = DataRequest(
                source_id=context.reference.source_id,
                variables=self.observation_vars,
                init_times=valid_times,
            )
            if context.loader is not None:
                self.reference_bundle = context.loader.materialize(
                    "reference", reference_request
                )
            else:
                self.reference_bundle = context.reference.reader.read(
                    reference_request,
                    context.reference.catalog.discover(reference_request),
                )
            log.debug("气候态参考读取完成: %s", self.reference_bundle.payload.dims)

        matcher = Matcher(
            ensemble_reduction=str(self.spec.options.get("ensemble_reduction", "mean"))
        )
        for batch in matcher.match(
            forecast=self.forecast_bundle,
            observation=self.observation_bundle,
            variables=self.forecast_vars,
            sample_by=self.sample_by,
        ):
            batch.members = self._members_for(batch)
            batch.reference = self._reference_for(batch)
            self.batches[self._batch_key(batch)] = batch
            if len(self.batches) % 20 == 0:
                log.info("参考场装配进度 %d", len(self.batches))
        log.debug("配对完成：%d 个批次", len(self.batches))

    def _batch_key(self, batch: EvaluationBatch) -> Any:
        """批次在 ``self.batches`` 里的键，口径由 ``sample_by`` 决定。"""
        record = batch.sample_keys[0] if batch.sample_keys else {}
        if self._init_lead:
            return (str(record.get("init_time", "")), float(record.get("lead_h") or 0.0))
        return record.get("valid_time")

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
                key=(
                    ("init_lead", key[0], key[1])
                    if self._init_lead
                    else ("valid_time", key)
                ),
                coordinates={
                    "variable": variable,
                    # 键是 (起报, 时效) 时有效时刻只能从记录里取
                    "valid_time": (
                        record.get("valid_time", "") if self._init_lead else key
                    ),
                    "init_time": record.get("init_time", ""),
                    "lead_h": record.get("lead_h", ""),
                    "sample_unit": batch.sample_dim,
                },
                payload={"batch_key": key},
            )

    def build_batch(
        self, context: PipelineContext, sample: Sample
    ) -> Optional[EvaluationBatch]:
        return self.batches.get(sample.payload["batch_key"])

    def defaults(self, context: PipelineContext) -> Dict[str, Any]:
        return {
            "variable": self.forecast_vars[0] if len(self.forecast_vars) == 1 else "",
            "sample_unit": "grid",
            "unit": self._unit(),
        }

    def _unit(self) -> str:
        """评分单位取自实况语义。"""
        bundle = self.observation_bundle
        if bundle is None or len(self.forecast_vars) != 1:
            return ""
        return str((bundle.semantic.units or {}).get(self.forecast_vars[0], ""))

    def summary(self) -> Dict[str, Any]:
        # 同 station 协议：报有效时刻列表而非批次数，同一个有效时刻可以由
        # (早起报, 长时效) 与 (晚起报, 短时效) 两条路径够到。
        #
        # init_lead 口径下 batches 的键是 (起报, 时效) 对，一个 run 两万个，
        # 拿去跨块做列表并集是平方级的；这里报去重后的有效时刻，语义不变。
        if self._init_lead:
            valid_times = sorted(
                {
                    str(record.get("valid_time", ""))
                    for batch in self.batches.values()
                    for record in batch.sample_keys
                }
                - {""}
            )
        else:
            valid_times = sorted(self.batches)
        summary: Dict[str, Any] = {"valid_times": valid_times}
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
