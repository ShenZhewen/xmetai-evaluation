"""Matcher 的三件事：经度折算、派生变量、缺变量容错。

这三件都是"接 ERA5 zarr 实况"暴露出来的：ERA5 的经度是 -180..180，
预报是 0..360；风速只在 u/v 分量里；一份配置 16 个要素、某个源少几个。
"""

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import (
    DataBundle,
    EvaluationBatch,
    Provenance,
    SemanticMetadata,
)
from xmetai_evaluation.core.variables import DataKind, TemporalKind
from xmetai_evaluation.pipeline.matcher import (
    DERIVED_VARIABLES,
    Matcher,
    add_derived_variables,
    align_lon_to,
    narrow_batch,
)

LATS = [10.0, 20.0]


def _bundle(dataset, source_id, kind, units=None):
    return DataBundle(
        payload=dataset,
        kind=kind,
        source_id=source_id,
        standard_vars={name: name for name in dataset.data_vars},
        semantic=SemanticMetadata(
            units=units or {},
            temporal_kind=TemporalKind.INSTANTANEOUS,
            grid_type="regular_latlon",
        ),
        provenance=Provenance(input_files=[], reader_id=source_id, reader_version="1.0.0"),
    )


def _forecast(zero_to_360=True):
    lon = np.arange(0.0, 360.0, 90.0) if zero_to_360 else np.arange(-180.0, 180.0, 90.0)
    return xr.Dataset(
        {"z500": (["lat", "lon"], np.full((len(LATS), len(lon)), 7.0))},
        coords={"lat": LATS, "lon": lon},
    )


def _observation(lon):
    return xr.Dataset(
        {"z500": (["lat", "lon"], np.full((len(LATS), len(lon)), 5.0))},
        coords={"lat": LATS, "lon": lon},
    )


def test_observation_on_the_other_circle_is_wrapped(tmp_path):
    """-180..180 的实况配 0..360 的预报：不折算 interp 会全 NaN。"""
    matcher = Matcher()
    batches = matcher.match(
        forecast=_bundle(_forecast(), "fuxi", DataKind.GRIDDED_FORECAST),
        observation=_bundle(
            _observation(np.arange(-180.0, 180.0, 90.0)), "era5", DataKind.GRIDDED_OBSERVATION
        ),
        variables=["z500"],
    )

    batch = batches[0]
    assert batch.alignment["lon_convention"] == "observation_wrapped_to_forecast_circle"
    # 折算后两个网格重合，插值不该产生 NaN
    assert np.isfinite(batch.forecast["z500"].values).all()
    assert np.isfinite(batch.observation["z500"].values).all()
    # 经度必须仍升序：rFFT 的谱指标依赖这一点
    lon = np.asarray(batch.observation["lon"].values)
    assert (np.diff(lon) > 0).all()
    assert lon.min() >= 0.0


def test_matching_circles_are_left_alone():
    matcher = Matcher()
    batches = matcher.match(
        forecast=_bundle(_forecast(zero_to_360=False), "fuxi", DataKind.GRIDDED_FORECAST),
        observation=_bundle(
            _observation(np.arange(-180.0, 180.0, 90.0)), "era5", DataKind.GRIDDED_OBSERVATION
        ),
        variables=["z500"],
    )

    assert batches[0].alignment["lon_convention"] == "native"


def test_align_lon_to_is_a_noop_when_it_cannot_tell():
    """没有 lon 坐标时原样返回，不猜。"""
    dataset = xr.Dataset({"z500": (["lat"], [1.0, 2.0])}, coords={"lat": LATS})
    assert align_lon_to(dataset, [0.0, 90.0]) is dataset


def test_derived_variables_are_synthesized_from_components():
    dataset = xr.Dataset(
        {
            "u10m": (["lat"], [3.0, 0.0]),
            "v10m": (["lat"], [4.0, 5.0]),
        },
        coords={"lat": LATS},
    )

    result = add_derived_variables(dataset, ["ws10m"])

    np.testing.assert_allclose(result["ws10m"].values, [5.0, 5.0])
    assert result["ws10m"].attrs["units"] == "m/s"


def test_derived_variables_are_not_synthesized_unless_requested():
    """没点名的派生变量不该凭空长出来——多出来的变量会污染变量求交。"""
    dataset = xr.Dataset(
        {"u10m": (["lat"], [3.0, 0.0]), "v10m": (["lat"], [4.0, 5.0])},
        coords={"lat": LATS},
    )

    assert "ws10m" not in add_derived_variables(dataset, ["u10m"])


def test_derived_variables_survive_the_intersection():
    """ws* 任何文件里都没有，必须在变量求交**之前**合成，否则会被丢掉。"""
    forecast = _bundle(_forecast(), "fuxi", DataKind.GRIDDED_FORECAST)
    observation = _bundle(
        _observation(np.arange(-180.0, 180.0, 90.0)), "era5", DataKind.GRIDDED_OBSERVATION
    )
    for bundle in (forecast, observation):
        bundle.payload["u10m"] = bundle.payload["z500"] * 0.0 + 3.0
        bundle.payload["v10m"] = bundle.payload["z500"] * 0.0 + 4.0
        bundle.standard_vars.update({"u10m": "u10m", "v10m": "v10m"})

    batches = Matcher().match(
        forecast=forecast, observation=observation, variables=["z500", "ws10m"]
    )

    assert "ws10m" in batches[0].forecast
    assert "ws10m" in batches[0].observation
    np.testing.assert_allclose(batches[0].forecast["ws10m"].values, 5.0)
    np.testing.assert_allclose(batches[0].observation["ws10m"].values, 5.0)


def test_missing_variable_is_skipped_not_fatal(caplog):
    """一个源少几个要素不该让整轮评测挂掉，但要告警。"""
    forecast = _bundle(_forecast(), "fuxi", DataKind.GRIDDED_FORECAST)
    observation = _bundle(
        _observation(np.arange(-180.0, 180.0, 90.0)), "era5", DataKind.GRIDDED_OBSERVATION
    )

    with caplog.at_level("WARNING"):
        batches = Matcher().match(
            forecast=forecast, observation=observation, variables=["z500", "q700"]
        )

    assert "q700" not in batches[0].forecast
    assert "q700" in caplog.text


def _batch(forecast, observation, mask):
    """直接造一个批次——narrow_batch 的行为不该被 interp 的细节干扰。"""
    return EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"valid_time": "2025-01-01T00:00:00"}],
        valid_mask=mask,
        alignment={"lon_convention": "native"},
    )


def test_narrow_batch_recomputes_the_valid_mask():
    """整批的掩码是所有变量的交集：一个变量缺测会砍掉其它变量的有效点。"""
    forecast = xr.Dataset(
        {"z500": (["lat"], [1.0, 2.0]), "q700": (["lat"], [np.nan, 4.0])},
        coords={"lat": LATS},
    )
    observation = xr.Dataset(
        {"z500": (["lat"], [1.0, 2.0]), "q700": (["lat"], [3.0, 4.0])},
        coords={"lat": LATS},
    )
    mask = (
        forecast["z500"].notnull()
        & observation["z500"].notnull()
        & forecast["q700"].notnull()
    )
    batch = _batch(forecast, observation, mask)
    # 整批掩码被 q700 的缺测拖累，z500 的第 0 个点也失效了
    assert not bool(batch.valid_mask.isel(lat=0))

    narrowed = narrow_batch(batch, "z500")

    assert narrowed.forecast.dims == ("lat",)
    assert bool(narrowed.valid_mask.isel(lat=0))
    assert bool(narrowed.valid_mask.isel(lat=1))


def test_narrow_batch_keeps_members_and_reference():
    forecast = xr.Dataset({"z500": (["lat"], [1.0, 2.0])}, coords={"lat": LATS})
    observation = xr.Dataset({"z500": (["lat"], [1.0, 2.0])}, coords={"lat": LATS})
    members = xr.Dataset(
        {"z500": (["member", "lat"], [[1.0, 2.0], [1.0, 2.0]])},
        coords={"member": [0, 1], "lat": LATS},
    )
    reference = xr.Dataset({"z500": (["lat"], [0.5, 0.5])}, coords={"lat": LATS})
    batch = EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"valid_time": "2025-01-01T00:00:00"}],
        valid_mask=forecast["z500"].notnull() & observation["z500"].notnull(),
        members=members,
        reference=reference,
        alignment={"lon_convention": "native"},
    )

    narrowed = narrow_batch(batch, "z500")

    assert narrowed.members.dims == ("member", "lat")
    assert narrowed.reference.dims == ("lat",)


def test_narrow_batch_returns_none_for_absent_variable():
    forecast = xr.Dataset({"z500": (["lat"], [1.0, 2.0])}, coords={"lat": LATS})
    observation = xr.Dataset({"z500": (["lat"], [1.0, 2.0])}, coords={"lat": LATS})
    batch = _batch(
        forecast,
        observation,
        forecast["z500"].notnull() & observation["z500"].notnull(),
    )

    assert narrow_batch(batch, "q700") is None


def test_derived_table_covers_the_xu_wind_variables():
    """xu 的 --vars 里有 ws10m/ws850/ws200，三个都得能合成。"""
    assert {"ws10m", "ws850", "ws200"}.issubset(DERIVED_VARIABLES)
