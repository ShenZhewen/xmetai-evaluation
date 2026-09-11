# -*- coding: utf-8 -*-
"""专项指标：Z500 活跃度比与纬向功率谱。

口径与参考实现 ``ref/fdp/verify/verify/activity_spectrum_verifier.py`` 一致：

* **活跃度比**：距平（场 − 气候态）的纬度加权标准差之比
  ``AR = std_w(F−C) / std_w(O−C)``，权重 ``w = cos(lat)``，标准差为
  "减加权均值"的加权标准差；
* **功率谱**：对**原始场**做二维 FFT 得到纬向波数谱
  ``P(k) = Σ_l [A_t(l,k)² + B_t(l,k)²]``（按参考实现的等效 FFT 推导），
  单边谱取 k=0..N/2，缺测用全场均值填充。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import (
    EvaluationBatch,
    MetricResult,
    ResultStatus,
)
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
)

DEFAULT_MAX_WAVENUMBER = 30


def _values(data: Any) -> np.ndarray:
    if isinstance(data, xr.DataArray):
        return np.asarray(data.values, dtype="f8")
    if isinstance(data, xr.Dataset):
        return np.asarray(data[list(data.data_vars)[0]].values, dtype="f8")
    if hasattr(data, "payload"):
        payload = data.payload
        if isinstance(payload, xr.Dataset):
            payload = payload[list(payload.data_vars)[0]]
        return np.asarray(payload.values, dtype="f8")
    return np.asarray(data, dtype="f8")


def _weights(batch: EvaluationBatch, shape) -> np.ndarray:
    if batch.weights is not None:
        weights = _values(batch.weights)
        if weights.shape == shape:
            return weights
    return np.ones(shape, dtype="f8")


def power_spectrum(field: np.ndarray, max_wavenumber: Optional[int] = None):
    """纬向波数功率谱 P(k)。

    Args:
        field: (M, N) 二维场（M=纬度方向格点数，N=经度方向格点数）。
        max_wavenumber: 返回的最大波数；None 表示 N//2。

    Returns:
        (wavenumbers, power)
    """
    values = np.asarray(field, dtype="f8")
    if values.ndim != 2:
        raise MetricError(f"功率谱需要二维场，实际 ndim={values.ndim}")
    if np.any(np.isnan(values)):
        values = np.where(np.isnan(values), np.nanmean(values), values)

    nlat, nlon = values.shape
    n_freq = nlon // 2 + 1
    max_k = n_freq - 1 if max_wavenumber is None else min(int(max_wavenumber), n_freq - 1)

    vertical = np.fft.ifft(values, axis=0)
    lat_index = np.arange(nlat)[:, None]
    lon_index = np.arange(nlon)[None, :]
    modulated = vertical * np.exp(-2j * np.pi * lat_index * lon_index / nlat)
    transformed = np.fft.ifft(modulated, axis=1)
    power_per_wavenumber = (nlat * nlon) ** 2 * np.abs(transformed) ** 2
    full = power_per_wavenumber.sum(axis=0)

    power = np.zeros(max_k + 1)
    for k in range(max_k + 1):
        if k == 0 or (nlon % 2 == 0 and k == nlon // 2):
            power[k] = full[k]
        else:
            power[k] = full[k] + full[nlon - k]
    return np.arange(max_k + 1), power


class ActivityRatio(Metric):
    """活跃度比（距平纬度加权标准差之比），需要气候态参考。"""

    PRODUCT_KIND = "specialized"

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="activity", version="1.0.0", params=params or {})

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    def _anomalies(self, batch: EvaluationBatch):
        if batch.reference is None:
            raise MetricError(
                "活跃度比是距平统计，需要气候态参考（请为流程配置 reference 数据源）",
                variable=self.name,
            )
        forecast = _values(batch.forecast)
        observation = _values(batch.observation)
        climatology = _values(batch.reference)
        if not (forecast.shape == observation.shape == climatology.shape):
            raise MetricError(
                f"形状不一致：fc={forecast.shape}, obs={observation.shape}, "
                f"clim={climatology.shape}",
                variable=self.name,
            )
        keep = (
            np.isfinite(forecast) & np.isfinite(observation) & np.isfinite(climatology)
        )
        weights = _weights(batch, forecast.shape)
        if batch.valid_mask is not None:
            mask = _values(batch.valid_mask)
            if mask.shape == keep.shape:
                keep = keep & mask.astype(bool)
        return forecast - climatology, observation - climatology, weights, keep

    @staticmethod
    def _moments(values, weights, keep):
        w = np.where(keep, weights, 0.0)
        v = np.where(keep, values, 0.0)
        return {
            "w": float(w.sum()),
            "w1": float((w * v).sum()),
            "w2": float((w * v * v).sum()),
        }

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        forecast_anomaly, observation_anomaly, weights, keep = self._anomalies(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast": self._moments(forecast_anomaly, weights, keep),
                "observation": self._moments(observation_anomaly, weights, keep),
                "n_valid": int(keep.sum()),
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        first = states[0]
        merged = {
            "forecast": {"w": 0.0, "w1": 0.0, "w2": 0.0},
            "observation": {"w": 0.0, "w1": 0.0, "w2": 0.0},
            "n_valid": 0,
        }
        for state in states:
            if state.metric_name != first.metric_name:
                raise MetricError(
                    f"Cannot merge states from different metrics: "
                    f"{first.metric_name} vs {state.metric_name}",
                    variable=self.name,
                )
            for side in ("forecast", "observation"):
                for key in ("w", "w1", "w2"):
                    merged[side][key] += state.data[side][key]
            merged["n_valid"] += state.data["n_valid"]
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data=merged,
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    @staticmethod
    def _std(moments):
        total_weight = moments["w"]
        if total_weight <= 0:
            return float("nan")
        mean = moments["w1"] / total_weight
        variance = moments["w2"] / total_weight - mean**2
        return float(np.sqrt(max(variance, 0.0)))

    def finalize(self, state: MetricState) -> MetricResult:
        forecast_std = self._std(state.data["forecast"])
        observation_std = self._std(state.data["observation"])
        if not np.isfinite(forecast_std) or not np.isfinite(observation_std):
            status = ResultStatus.NO_VALID_DATA
            ratio = float("nan")
        elif observation_std == 0:
            status = ResultStatus.UNDEFINED
            ratio = float("nan")
        else:
            status = ResultStatus.SUCCESS
            ratio = float(forecast_std / observation_std)

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value={
                "ACTIVITY_RATIO": ratio,
                "FC_ACTIVITY": forecast_std,
                "OBS_ACTIVITY": observation_std,
                "FC_OBS_BIAS": float(forecast_std - observation_std),
            },
            status=status,
            n_requested=int(state.data["n_valid"]),
            n_valid=int(state.data["n_valid"]),
            aggregation="area_weighted",
            unit="1",
            product_kind=self.PRODUCT_KIND,
        )


class PowerSpectrum(Metric):
    """纬向功率谱（对原始场）：曲线进明细表，总功率比进长表。"""

    PRODUCT_KIND = "specialized"

    def __init__(self, max_wavenumber: int = DEFAULT_MAX_WAVENUMBER, params=None):
        super().__init__(name="spectrum", version="1.0.0", params=params or {})
        self.max_wavenumber = int(max_wavenumber)

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    def _spectra(self, batch: EvaluationBatch):
        forecast = _values(batch.forecast)
        observation = _values(batch.observation)
        forecast_grid = (
            forecast.reshape(-1, forecast.shape[-1]) if forecast.ndim > 2 else forecast
        )
        observation_grid = (
            observation.reshape(-1, observation.shape[-1])
            if observation.ndim > 2
            else observation
        )
        _, forecast_power = power_spectrum(forecast_grid, self.max_wavenumber)
        _, observation_power = power_spectrum(observation_grid, self.max_wavenumber)
        return forecast_power, observation_power

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        forecast_power, observation_power = self._spectra(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_power": forecast_power,
                "observation_power": observation_power,
                "n_samples": 1,
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        forecast_total = np.zeros_like(states[0].data["forecast_power"])
        observation_total = np.zeros_like(states[0].data["observation_power"])
        for state in states:
            forecast_total = forecast_total + state.data["forecast_power"]
            observation_total = observation_total + state.data["observation_power"]
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_power": forecast_total,
                "observation_power": observation_total,
                "n_samples": sum(state.data["n_samples"] for state in states),
            },
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        count = max(int(state.data["n_samples"]), 1)
        forecast_power = state.data["forecast_power"] / count
        observation_power = state.data["observation_power"] / count

        # 曲线进 diagnostics（details 表），总功率比作为标量进长表
        value: Dict[str, Any] = {}
        for k in range(forecast_power.size):
            value[f"k={k}"] = {
                "wavenumber": int(k),
                "power_forecast": float(forecast_power[k]),
                "power_observation": float(observation_power[k]),
            }
        forecast_total = float(forecast_power.sum())
        observation_total = float(observation_power.sum())
        value["summary"] = {
            "total_power_forecast": forecast_total,
            "total_power_observation": observation_total,
            "power_ratio": (
                float(forecast_total / observation_total)
                if observation_total > 0
                else float("nan")
            ),
        }

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=value,
            status=ResultStatus.SUCCESS,
            n_requested=count,
            n_valid=count,
            aggregation="wavenumber_spectrum",
            unit="",
            product_kind=self.PRODUCT_KIND,
        )


def zonal_spectrum(
    field: np.ndarray,
    lat: np.ndarray,
    weighted: bool = True,
    remove_zonal_mean: bool = True,
) -> np.ndarray:
    """纬向功率谱（去纬向均值 → rfft → cos 纬度加权平均）。

    口径与参考实现 ``vfc/metrics/spectrum.py::zonal_spectrum`` 一致。

    Args:
        field: (..., lat, lon) 场，最后两维为 (lat, lon)。
        lat: 纬度数组（度），长度 = field 的 lat 维。
        weighted: True 时按 cos(lat) 加权平均。
        remove_zonal_mean: True 时先去除纬向均值、k=0 置零。

    Returns:
        (..., nlon // 2 + 1) 纬向功率谱。
    """
    x = np.asarray(field, dtype="f8")
    nlon = x.shape[-1]
    if remove_zonal_mean:
        x = x - x.mean(axis=-1, keepdims=True)
    F = np.fft.rfft(x, axis=-1)
    psd = (F.real**2 + F.imag**2) / (nlon**2)
    if remove_zonal_mean:
        psd[..., 0] = 0.0
    w = (
        np.cos(np.deg2rad(np.asarray(lat, dtype="f8")))
        if weighted
        else np.ones(lat.size, dtype="f8")
    )
    wshape = (1,) * (psd.ndim - 2) + (lat.size, 1)
    return (psd * w.reshape(wshape)).sum(axis=-2) / w.sum()


class ZonalSpectrum(Metric):
    """纬向功率谱（去纬向均值后 rfft，cos 纬度加权），逐样本平均曲线。"""

    PRODUCT_KIND = "specialized"

    def __init__(self, max_wavenumber: int = DEFAULT_MAX_WAVENUMBER, params=None):
        super().__init__(name="zonal_spectrum", version="1.0.0", params=params or {})
        self.max_wavenumber = int(max_wavenumber)

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    @staticmethod
    def _lat(batch: EvaluationBatch) -> np.ndarray:
        src = batch.forecast
        if hasattr(src, "payload"):
            src = src.payload
        if isinstance(src, (xr.DataArray, xr.Dataset)):
            lat = src.coords.get("lat")
            if lat is not None:
                return np.asarray(
                    lat.values if hasattr(lat, "values") else lat, dtype="f8"
                )
        raise MetricError(
            "纬向谱需要 forecast 携带 lat 坐标（用于 cos 纬度加权）",
            variable="zonal_spectrum",
        )

    def _spectra(self, batch: EvaluationBatch):
        forecast = _values(batch.forecast)
        observation = _values(batch.observation)
        if forecast.ndim < 2 or observation.ndim < 2:
            raise MetricError(
                f"纬向谱需要 (..., lat, lon) 场，实际 ndim={forecast.ndim}",
                variable=self.name,
            )
        lat = self._lat(batch)
        f = forecast.reshape(-1, forecast.shape[-2], forecast.shape[-1])
        o = observation.reshape(-1, observation.shape[-2], observation.shape[-1])
        forecast_psd = zonal_spectrum(f, lat).sum(axis=0)
        observation_psd = zonal_spectrum(o, lat).sum(axis=0)
        max_k = min(self.max_wavenumber, forecast_psd.size - 1)
        return (
            forecast_psd[: max_k + 1],
            observation_psd[: max_k + 1],
            int(f.shape[0]),
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        forecast_power, observation_power, n_samples = self._spectra(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_power": forecast_power,
                "observation_power": observation_power,
                "n_samples": n_samples,
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        forecast_total = np.zeros_like(states[0].data["forecast_power"])
        observation_total = np.zeros_like(states[0].data["observation_power"])
        for state in states:
            forecast_total = forecast_total + state.data["forecast_power"]
            observation_total = observation_total + state.data["observation_power"]
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_power": forecast_total,
                "observation_power": observation_total,
                "n_samples": sum(state.data["n_samples"] for state in states),
            },
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        count = max(int(state.data["n_samples"]), 1)
        forecast_power = state.data["forecast_power"] / count
        observation_power = state.data["observation_power"] / count

        value: Dict[str, Any] = {}
        for k in range(forecast_power.size):
            value[f"k={k}"] = {
                "wavenumber": int(k),
                "power_forecast": float(forecast_power[k]),
                "power_observation": float(observation_power[k]),
            }
        forecast_total = float(forecast_power.sum())
        observation_total = float(observation_power.sum())
        value["summary"] = {
            "total_power_forecast": forecast_total,
            "total_power_observation": observation_total,
            "power_ratio": (
                float(forecast_total / observation_total)
                if observation_total > 0
                else float("nan")
            ),
        }

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=value,
            status=ResultStatus.SUCCESS,
            n_requested=count,
            n_valid=count,
            aggregation="wavenumber_spectrum",
            unit="",
            product_kind=self.PRODUCT_KIND,
        )
