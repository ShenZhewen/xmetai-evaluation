"""集合概率评分 AROC/BS/BSS 的口径与可合并性。"""

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.probabilistic import EnsembleProbabilityScore


def _batch(members, observation):
    members = xr.DataArray(np.asarray(members, dtype="f8"), dims=["member", "station"])
    observation = xr.DataArray(np.asarray(observation, dtype="f8"), dims=["station"])
    valid_mask = xr.DataArray(
        np.ones((members.sizes["member"], members.sizes["station"]), dtype=bool),
        dims=["member", "station"],
    )
    return EvaluationBatch(
        forecast=members,
        observation=observation,
        sample_keys=[{"station": index} for index in range(members.sizes["station"])],
        valid_mask=valid_mask,
        members=members,
        alignment={"method": "direct"},
    )


# 4 个站点、2 个成员：p(>=0.1) 分别为 0 / 0.5 / 1 / 1，事件为 0 / 1 / 1 / 0
MEMBERS = [[0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 10.0, 10.0]]
OBSERVATION = [0.0, 10.0, 10.0, 0.0]


def test_probability_scores_match_hand_computation():
    metric = EnsembleProbabilityScore(thresholds=[("≥0.1", 0.1)])

    result = metric.compute(_batch(MEMBERS, OBSERVATION))
    entry = result.value["≥0.1"]

    assert result.status == ResultStatus.SUCCESS
    assert result.product_kind == "probabilistic"
    assert entry["n_points"] == 4
    assert entry["base_rate"] == pytest.approx(0.5)
    assert entry["BS"] == pytest.approx(0.3125)
    assert entry["BS_ref"] == pytest.approx(0.25)
    assert entry["BSS"] == pytest.approx(-0.25)
    # ROC 顶点 (0,0) (0.5,0.5) (0.5,1) (1,1) 的梯形面积
    assert entry["AROC"] == pytest.approx(0.625)


def test_perfect_forecast_scores_one():
    """成员全部命中/全部未命中：AROC=1，BSS=1。"""
    members = [[0.0, 10.0], [0.0, 10.0]]
    observation = [0.0, 10.0]

    result = EnsembleProbabilityScore(thresholds=[("≥0.1", 0.1)]).compute(
        _batch(members, observation)
    )
    entry = result.value["≥0.1"]

    assert entry["AROC"] == pytest.approx(1.0)
    assert entry["BS"] == pytest.approx(0.0)
    assert entry["BSS"] == pytest.approx(1.0)


def test_merge_matches_single_batch_and_checks_member_count():
    metric = EnsembleProbabilityScore(thresholds=[("≥0.1", 0.1)])
    whole = metric.compute(_batch(MEMBERS, OBSERVATION)).value["≥0.1"]

    first = metric.accumulate(_batch([[0.0, 0.0], [0.0, 10.0]], [0.0, 10.0]))
    second = metric.accumulate(_batch([[10.0, 10.0], [10.0, 10.0]], [10.0, 0.0]))
    merged = metric.finalize(metric.merge([first, second])).value["≥0.1"]

    for field in ("AROC", "BS", "BSS", "base_rate", "n_points"):
        assert merged[field] == pytest.approx(whole[field])

    with pytest.raises(MetricError, match="成员数不一致"):
        metric.merge(
            [first, metric.accumulate(_batch([[0.0], [0.0], [0.0]], [0.0]))]
        )


def test_requires_members_and_rejects_missing_event_class():
    metric = EnsembleProbabilityScore(thresholds=[("≥0.1", 0.1)])

    batch = _batch(MEMBERS, OBSERVATION)
    batch.members = None
    with pytest.raises(MetricError, match="需要集合成员"):
        metric.validate(batch)

    # 全是事件、没有非事件时 AROC 不可定义，但不能伪造数值
    result = metric.compute(_batch([[10.0, 10.0], [10.0, 10.0]], [10.0, 10.0]))
    assert np.isnan(result.value["≥0.1"]["AROC"])
    assert result.status == ResultStatus.PARTIAL
