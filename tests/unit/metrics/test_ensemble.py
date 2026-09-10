"""集合指标 CRPS / Spread-Error 的口径。"""

import math

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.ensemble import CRPS, SpreadError


def _batch(members, observation, weights=None):
    members = xr.DataArray(np.asarray(members, dtype="f8"), dims=["member", "lat", "lon"])
    observation = xr.DataArray(np.asarray(observation, dtype="f8"), dims=["lat", "lon"])
    shape = (members.sizes["member"], members.sizes["lat"], members.sizes["lon"])
    batch = EvaluationBatch(
        forecast=members,
        observation=observation,
        sample_keys=[{"sample": 0}],
        valid_mask=xr.DataArray(np.ones(shape, dtype=bool), dims=["member", "lat", "lon"]),
        members=members,
        alignment={"method": "direct"},
    )
    if weights is not None:
        batch.weights = xr.DataArray(np.asarray(weights, dtype="f8"), dims=["lat", "lon"])
    return batch


# 2 个成员、1x2 网格：格点0 成员 [0,2]，格点1 成员 [2,4]；实况 1.5 / 3.5；权重 1 / 0.5
MEMBERS = [[[0.0, 2.0]], [[2.0, 4.0]]]
OBSERVATION = [[1.5, 3.5]]
WEIGHTS = [[1.0, 0.5]]


def test_crps_matches_hand_computation():
    metric = CRPS()
    result = metric.compute(_batch(MEMBERS, OBSERVATION, WEIGHTS))

    # term1 = (|0-1.5|+|2-1.5|)/2 = 1.0；term2 = (-x0 + x1)/4 = 0.5 → 每点 0.5
    assert result.status == ResultStatus.SUCCESS
    assert result.value == pytest.approx(0.5)
    assert result.n_valid == 2
    assert result.aggregation == "area_weighted"


def test_spread_error_matches_hand_computation():
    metric = SpreadError()
    result = metric.compute(_batch(MEMBERS, OBSERVATION, WEIGHTS))

    # Spread = sqrt((1*2 + 0.5*2) / (1.5*(2-1))) = sqrt(2)；RMSE = sqrt(0.375/1.5) = 0.5
    assert result.value["SPREAD"] == pytest.approx(math.sqrt(2))
    assert result.value["RMSE"] == pytest.approx(0.5)
    assert result.value["RATIO"] == pytest.approx(2 * math.sqrt(2))


def test_merge_matches_single_batch():
    metric = SpreadError()
    whole = metric.compute(_batch(MEMBERS, OBSERVATION, WEIGHTS)).value

    left = metric.accumulate(_batch([[[0.0, 2.0]], [[2.0, 4.0]]], [[1.5, 3.5]], [[1.0, 1.0]]))
    right = metric.accumulate(_batch([[[0.0, 2.0]], [[2.0, 4.0]]], [[1.5, 3.5]], [[0.0, 0.5]]))
    merged = metric.finalize(metric.merge([left, right])).value

    for field in ("SPREAD", "RMSE", "RATIO"):
        assert merged[field] == pytest.approx(whole[field])


def test_missing_members_are_rejected():
    crps = CRPS()
    batch = _batch(MEMBERS, OBSERVATION, WEIGHTS)
    batch.members = None
    with pytest.raises(MetricError, match="需要集合成员"):
        crps.validate(batch)

    spread = SpreadError()
    with pytest.raises(MetricError, match="成员数不一致"):
        spread.merge(
            [
                spread.accumulate(_batch(MEMBERS, OBSERVATION, WEIGHTS)),
                spread.accumulate(_batch([[[0.0, 2.0]]], OBSERVATION, WEIGHTS)),
            ]
        )
