"""协议与参考场（气候态）的接线。"""

from datetime import datetime

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import (
    DataBundle,
    EvaluationBatch,
    Provenance,
    SemanticMetadata,
)
from xmetai_evaluation.core.variables import DataKind, TemporalKind
from xmetai_evaluation.pipeline.protocols import GridValidTimeProtocol
from xmetai_evaluation.pipeline.spec import PipelineSpec, SourceSpec

LATS = [10.0, 20.0]
LONS = [100.0, 110.0, 120.0]


def _bundle(dataset, source_id, kind):
    return DataBundle(
        payload=dataset,
        kind=kind,
        source_id=source_id,
        standard_vars={"z500": "z500"},
        semantic=SemanticMetadata(
            units={"z500": "m"},
            temporal_kind=TemporalKind.INSTANTANEOUS,
            grid_type="regular_latlon",
        ),
        provenance=Provenance(input_files=[], reader_id=source_id, reader_version="1.0.0"),
    )


def _spec():
    return PipelineSpec(
        name="protocol_test",
        forecast=SourceSpec("fengqing", {"variables": ["z500"]}),
        observation=SourceSpec("cra", {"variables": ["z500"]}),
    )


def test_reference_is_selected_by_valid_time_and_regridded():
    protocol = GridValidTimeProtocol(_spec())
    climatology = xr.Dataset(
        {"z500": (["valid_time", "lat", "lon"], np.full((1, 2, 3), 7.0))},
        coords={"valid_time": [np.datetime64("2026-08-20T06:00:00")], "lat": LATS, "lon": LONS},
    )
    protocol.reference_bundle = _bundle(climatology, "climatology", DataKind.REFERENCE)
    protocol.observation_bundle = _bundle(
        xr.Dataset(
            {"z500": (["lat", "lon"], np.zeros((2, 3)))},
            coords={"lat": LATS, "lon": LONS},
        ),
        "cra",
        DataKind.GRIDDED_OBSERVATION,
    )

    batch = _batch()
    field = protocol._reference_for(batch)

    assert field is not None
    assert field.dims == ("lat", "lon")
    np.testing.assert_allclose(field.values, 7.0)


def test_no_reference_source_means_no_reference_field():
    protocol = GridValidTimeProtocol(_spec())
    protocol.observation_bundle = _bundle(
        xr.Dataset(
            {"z500": (["lat", "lon"], np.zeros((2, 3)))},
            coords={"lat": LATS, "lon": LONS},
        ),
        "cra",
        DataKind.GRIDDED_OBSERVATION,
    )
    assert protocol._reference_for(_batch()) is None


def test_unknown_valid_time_degrades_to_no_reference():
    protocol = GridValidTimeProtocol(_spec())
    climatology = xr.Dataset(
        {"z500": (["valid_time", "lat", "lon"], np.full((1, 2, 3), 7.0))},
        coords={"valid_time": [np.datetime64("2026-08-20T06:00:00")], "lat": LATS, "lon": LONS},
    )
    protocol.reference_bundle = _bundle(climatology, "climatology", DataKind.REFERENCE)
    protocol.observation_bundle = _bundle(
        xr.Dataset(
            {"z500": (["lat", "lon"], np.zeros((2, 3)))},
            coords={"lat": LATS, "lon": LONS},
        ),
        "cra",
        DataKind.GRIDDED_OBSERVATION,
    )

    batch = _batch(valid_time="2026-08-21T00:00:00.000000000")
    assert protocol._reference_for(batch) is None


def test_reference_synthesizes_wind_speed_from_components():
    """气候态里只有 u/v 分量时，ws850 要现合成——xu 也是这么兜底的。"""
    protocol = GridValidTimeProtocol(_spec())
    protocol.observation_vars = ["u850", "v850", "ws850"]
    climatology = xr.Dataset(
        {
            "u850": (["valid_time", "lat", "lon"], np.full((1, 2, 3), 3.0)),
            "v850": (["valid_time", "lat", "lon"], np.full((1, 2, 3), 4.0)),
        },
        coords={
            "valid_time": [np.datetime64("2026-08-20T06:00:00")],
            "lat": LATS,
            "lon": LONS,
        },
    )
    protocol.reference_bundle = _bundle(climatology, "daily_climatology", DataKind.REFERENCE)
    protocol.observation_bundle = _bundle(
        xr.Dataset(
            {"u850": (["lat", "lon"], np.zeros((2, 3)))},
            coords={"lat": LATS, "lon": LONS},
        ),
        "era5_zarr",
        DataKind.GRIDDED_OBSERVATION,
    )

    field = protocol._reference_for(_batch())

    assert field is not None
    assert "ws850" in field
    np.testing.assert_allclose(field["ws850"].values, 5.0)


def _batch(valid_time="2026-08-20T06:00:00.000000000"):
    forecast = xr.DataArray(np.zeros((2, 3)), dims=["lat", "lon"], coords={"lat": LATS, "lon": LONS})
    return EvaluationBatch(
        forecast=forecast,
        observation=forecast.copy(),
        sample_keys=[{"valid_time": valid_time, "lead_h": 6.0, "init_time": str(datetime(2026, 8, 20))}],
        valid_mask=xr.DataArray(np.ones((2, 3), dtype=bool), dims=["lat", "lon"]),
        alignment={"method": "direct"},
    )
