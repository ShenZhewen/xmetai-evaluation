"""气候态参考场 Reader（按 月日+时次 索引）。"""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.core.errors import DiscoveryError
from xmetai_evaluation.core.variables import DataKind
from xmetai_evaluation.io.climatology_reader import (
    ClimatologyCatalog,
    ClimatologyReader,
)


def _write_climate(root, moment, value, variable="z500", suffix=".nc"):
    name = (
        f"ART_ATM_GLB_0P25_CLI_ANAL_{moment.month:02d}{moment.day:02d}"
        f"{moment.hour:02d}{suffix}"
    )
    dataset = xr.Dataset(
        {variable: (["lat", "lon"], np.full((2, 3), value), {"units": "m"})},
        coords={"lat": [10.0, 20.0], "lon": [100.0, 110.0, 120.0]},
    )
    dataset.to_netcdf(root / name)


def test_reads_by_month_day_and_hour(tmp_path):
    moments = [datetime(2026, 8, 20, 6), datetime(2026, 8, 20, 12)]
    _write_climate(tmp_path, moments[0], 5000.0)
    _write_climate(tmp_path, moments[1], 5100.0)

    catalog = ClimatologyCatalog(root_dir=tmp_path)
    reader = ClimatologyReader(engine="netcdf")
    request = DataRequest(source_id="climatology", variables=["z500"], init_times=moments)

    bundle = reader.read(request, catalog.discover(request))

    assert bundle.kind == DataKind.REFERENCE
    assert bundle.payload["z500"].dims == ("valid_time", "lat", "lon")
    assert bundle.payload.sizes["valid_time"] == 2
    np.testing.assert_allclose(bundle.payload["z500"].isel(valid_time=0).values, 5000.0)
    np.testing.assert_allclose(bundle.payload["z500"].isel(valid_time=1).values, 5100.0)
    assert len(bundle.provenance.input_files) == 2


def test_year_is_ignored(tmp_path):
    """气候态与年份无关：2026 的请求能命中 2001 年的气候态文件。"""
    _write_climate(tmp_path, datetime(2001, 8, 20, 6), 4900.0)

    catalog = ClimatologyCatalog(root_dir=tmp_path)
    request = DataRequest(
        source_id="climatology", variables=["z500"], init_times=[datetime(2026, 8, 20, 6)]
    )
    index = catalog.discover(request)

    assert list(index.available[0].values())[0].name.endswith("082006.nc")


def test_missing_files_fail_loudly(tmp_path):
    catalog = ClimatologyCatalog(root_dir=tmp_path)
    request = DataRequest(
        source_id="climatology", variables=["z500"], init_times=[datetime(2026, 8, 20, 6)]
    )
    with pytest.raises(DiscoveryError):
        catalog.discover(request)
