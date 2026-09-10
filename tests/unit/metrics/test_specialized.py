"""FSS / 活跃度比 / 功率谱的口径。"""

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.spatial import FractionsSkillScore, neighborhood_fraction
from xmetai_evaluation.metrics.specialized import ActivityRatio, PowerSpectrum, power_spectrum


def _batch(forecast, observation, reference=None, weights=None):
    forecast = xr.DataArray(np.asarray(forecast, dtype="f8"), dims=["lat", "lon"])
    observation = xr.DataArray(np.asarray(observation, dtype="f8"), dims=["lat", "lon"])
    batch = EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"sample": 0}],
        valid_mask=xr.DataArray(np.ones(forecast.shape, dtype=bool), dims=["lat", "lon"]),
        alignment={"method": "direct"},
    )
    if reference is not None:
        batch.reference = xr.DataArray(
            np.asarray(reference, dtype="f8"), dims=["lat", "lon"]
        )
    if weights is not None:
        batch.weights = xr.DataArray(np.asarray(weights, dtype="f8"), dims=["lat", "lon"])
    return batch


# ---------------------------------------------------------------- FSS

FORECAST = [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
OBSERVATION = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_fss_matches_hand_computation_for_a_single_cell_window():
    """窗口=1 时 PF/PO 即二值场：Σ(PF-PO)²=1，Σ(PF²+PO²)=3 → FSS=2/3。"""
    metric = FractionsSkillScore(thresholds=[("≥1", 1.0)], windows=[1])

    result = metric.compute(_batch(FORECAST, OBSERVATION))

    assert result.value["≥1|w1"]["FSS"] == pytest.approx(2.0 / 3.0)
    assert result.value["≥1|w1"]["n_points"] == 9
    assert result.status == ResultStatus.SUCCESS


def test_fss_is_one_for_identical_fields():
    metric = FractionsSkillScore(thresholds=[("≥1", 1.0)], windows=[1, 3])
    result = metric.compute(_batch(FORECAST, FORECAST))
    for entry in result.value.values():
        assert entry["FSS"] == pytest.approx(1.0)


def test_neighborhood_fraction_matches_hand_values_both_implementations():
    """3×3 邻域覆盖率：与手算一致，且 scipy 与积分图回退结果相同。"""
    import xmetai_evaluation.metrics.spatial as spatial

    binary = np.asarray(FORECAST, dtype="f8")
    hand = {  # (行, 列) -> 3×3 窗口内 1 的个数 / 9
        (0, 0): 2 / 9, (0, 1): 2 / 9, (0, 2): 1 / 9,
        (1, 0): 2 / 9, (1, 1): 2 / 9, (1, 2): 1 / 9,
        (2, 0): 0.0, (2, 2): 0.0,
    }

    with_scipy = neighborhood_fraction(binary, 3)
    saved = spatial._uniform_filter
    spatial._uniform_filter = None
    try:
        fallback = spatial.neighborhood_fraction(binary, 3)
    finally:
        spatial._uniform_filter = saved

    for (row, column), expected in hand.items():
        assert with_scipy[row, column] == pytest.approx(expected)
        assert fallback[row, column] == pytest.approx(expected)
    np.testing.assert_allclose(with_scipy, fallback)


def test_fss_merge_matches_single_batch():
    metric = FractionsSkillScore(thresholds=[("≥1", 1.0)], windows=[3])
    whole = metric.compute(_batch(FORECAST, OBSERVATION)).value["≥1|w3"]["FSS"]
    merged = metric.finalize(
        metric.merge(
            [
                metric.accumulate(_batch(FORECAST, OBSERVATION)),
                metric.accumulate(_batch(FORECAST, OBSERVATION)),
            ]
        )
    ).value["≥1|w3"]["FSS"]
    assert merged == pytest.approx(whole)


# ------------------------------------------------------- 活跃度比


def test_activity_ratio_matches_hand_computation():
    """obs = 2×fc 的距平 → 活跃度比 = 0.5。"""
    metric = ActivityRatio()
    result = metric.compute(
        _batch([[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]], reference=[[0.0, 0.0], [0.0, 0.0]])
    )

    assert result.value["ACTIVITY_RATIO"] == pytest.approx(0.5)
    assert result.value["FC_ACTIVITY"] == pytest.approx(np.sqrt(1.25))
    assert result.value["OBS_ACTIVITY"] == pytest.approx(np.sqrt(5.0))
    assert result.status == ResultStatus.SUCCESS


def test_activity_ratio_requires_climatology():
    with pytest.raises(MetricError, match="气候态"):
        ActivityRatio().compute(_batch([[1.0]], [[2.0]]))


# --------------------------------------------------------- 功率谱


def test_power_spectrum_peaks_at_the_sinusoid_wavenumber():
    """纯 k0=2 的纬向正弦场，功率应集中在波数 2。"""
    nlon, nlat, k0 = 16, 4, 2
    lon = np.arange(nlon)
    field = np.tile(np.cos(2 * np.pi * k0 * lon / nlon), (nlat, 1))

    wavenumbers, power = power_spectrum(field)

    assert list(wavenumbers) == list(range(nlon // 2 + 1))
    assert int(np.argmax(power[1:]) ) + 1 == k0


def test_power_spectrum_metric_reports_curve_and_ratio():
    metric = PowerSpectrum(max_wavenumber=4)
    # 场宽 12 → 单边谱 k=0..6，取 0..4
    wide_forecast = [list(range(12)), list(range(12)), list(range(12))]
    wide_observation = [[value * 2.0 for value in row] for row in wide_forecast]
    result = metric.compute(_batch(wide_forecast, wide_observation))

    assert "summary" in result.value
    assert result.value["summary"]["power_ratio"] > 0
    assert {f"k={k}" for k in range(5)}.issubset(set(result.value))


def test_power_spectrum_fills_nan_with_field_mean():
    field = np.array([[1.0, np.nan], [3.0, 4.0]])
    _, power = power_spectrum(field)
    assert np.all(np.isfinite(power))
