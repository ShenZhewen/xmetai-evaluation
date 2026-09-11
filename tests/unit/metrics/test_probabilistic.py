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


def test_external_reference_bs_ref():
    """给了外部气候概率参考时，BSS 用 BS_ref=mean((p_clim-o)²) 而非样本频率。"""
    metric = EnsembleProbabilityScore(thresholds=[("≥0.1", 0.1)])
    batch = _batch(MEMBERS, OBSERVATION)
    batch.reference = xr.DataArray(
        np.full((4, 1), 0.2),
        dims=["station", "threshold"],
        coords={"station": [0, 1, 2, 3], "threshold": [0.1]},
    )

    entry = metric.compute(batch).value["≥0.1"]

    # o = [0,1,1,0]，p_clim 全 0.2 → BS_ref = (0.04+0.64+0.64+0.04)/4 = 0.34
    assert entry["BS_ref"] == pytest.approx(0.34)
    # 与样本频率回退（base_rate*(1-base_rate)=0.25）不同，证明外部参考确实生效
    assert entry["BSS"] == pytest.approx(1 - 0.3125 / 0.34)


def test_external_reference_missing_station_uses_zero():
    """缺 ref 站（NaN 气候概率）按 0 兜底，与参考实现 RefProb 口径一致。"""
    metric = EnsembleProbabilityScore(thresholds=[("≥0.1", 0.1)])
    batch = _batch(MEMBERS, OBSERVATION)
    ref = np.full((4, 1), 0.2)
    ref[0, 0] = np.nan
    batch.reference = xr.DataArray(
        ref, dims=["station", "threshold"], coords={"threshold": [0.1]}
    )

    entry = metric.compute(batch).value["≥0.1"]

    # 站 0 p_clim=0 → BS_ref = (0 + 0.64 + 0.64 + 0.04)/4 = 0.33
    assert entry["BS_ref"] == pytest.approx(0.33)


def test_external_reference_merges_across_batches():
    """外部参考的 bs_ref 累加（sum/count）与逐批计算一致。"""
    metric = EnsembleProbabilityScore(thresholds=[("≥0.1", 0.1)])
    ref = np.full((4, 1), 0.2)

    whole = _batch(MEMBERS, OBSERVATION)
    whole.reference = xr.DataArray(
        ref, dims=["station", "threshold"], coords={"threshold": [0.1]}
    )
    whole_bs_ref = metric.compute(whole).value["≥0.1"]["BS_ref"]

    # 让两个 batch 也带上各自的外部参考（各 2 站）
    first_batch = _batch([[0.0, 0.0], [0.0, 10.0]], [0.0, 10.0])
    first_batch.reference = xr.DataArray(
        ref[:2], dims=["station", "threshold"], coords={"threshold": [0.1]}
    )
    second_batch = _batch([[10.0, 10.0], [10.0, 10.0]], [10.0, 0.0])
    second_batch.reference = xr.DataArray(
        ref[2:], dims=["station", "threshold"], coords={"threshold": [0.1]}
    )
    first = metric.accumulate(first_batch)
    second = metric.accumulate(second_batch)

    merged_bs_ref = metric.finalize(metric.merge([first, second])).value["≥0.1"]["BS_ref"]

    assert merged_bs_ref == pytest.approx(whole_bs_ref)
