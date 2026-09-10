"""ACC（距平相关系数）的加权去均值口径与可合并性。"""

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.metrics.acc import ACC


def _batch(forecast, observation, weights=None, reference=None):
    forecast = xr.DataArray(np.asarray(forecast, dtype="f8"), dims=["lat", "lon"])
    observation = xr.DataArray(np.asarray(observation, dtype="f8"), dims=["lat", "lon"])
    batch = EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"sample": 0}],
        valid_mask=xr.DataArray(np.ones(forecast.shape, dtype=bool), dims=["lat", "lon"]),
        alignment={"method": "direct"},
    )
    if weights is not None:
        batch.weights = xr.DataArray(np.asarray(weights, dtype="f8"), dims=["lat", "lon"])
    if reference is not None:
        batch.reference = xr.DataArray(
            np.asarray(reference, dtype="f8"), dims=["lat", "lon"]
        )
    return batch


def test_acc_is_centered_so_a_constant_shift_scores_one():
    """只差一个常数偏移的两场，去均值后应完全相关（参考实现口径）。"""
    forecast = [[1.0, 2.0], [3.0, 4.0]]
    observation = [[11.0, 12.0], [13.0, 14.0]]

    result = ACC().compute(_batch(forecast, observation))

    assert result.value == pytest.approx(1.0)
    assert result.aggregation == "spatial_correlation"
    # 没有气候态时会给出警告（距平退化为原始场）
    assert any("climatology" in warning.lower() for warning in result.warnings)


def test_acc_uses_the_reference_as_anomaly_base():
    """给了气候态后，距平 = 场 - 气候态。"""
    climate = [[0.0, 0.0], [0.0, 0.0]]
    forecast = [[1.0, 2.0], [3.0, 4.0]]
    observation = [[2.0, 4.0], [6.0, 8.0]]

    # 距平 o = 2f → 完全相关
    result = ACC().compute(_batch(forecast, observation, reference=climate))
    assert result.value == pytest.approx(1.0)
    assert result.warnings == []  # 有气候态 → 无警告
    assert result.status == ResultStatus.SUCCESS


def test_acc_merge_matches_single_batch():
    metric = ACC()
    forecast = [[1.0, 2.0], [3.0, 4.0]]
    observation = [[2.0, 5.0], [4.0, 9.0]]
    whole = metric.compute(_batch(forecast, observation)).value

    left = metric.accumulate(_batch([[1.0, 2.0]], [[2.0, 5.0]]))
    right = metric.accumulate(_batch([[3.0, 4.0]], [[4.0, 9.0]]))
    merged = metric.finalize(metric.merge([left, right])).value

    assert merged == pytest.approx(whole)


def test_zero_variance_is_undefined():
    forecast = [[1.0, 1.0], [1.0, 1.0]]
    observation = [[1.0, 2.0], [3.0, 4.0]]

    result = ACC().compute(_batch(forecast, observation))

    assert result.status == ResultStatus.UNDEFINED
    assert np.isnan(result.value)
