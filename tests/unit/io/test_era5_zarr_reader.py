"""ERA5 zarr 实况 Reader。

与通用格点 Reader 的差别是数据形态：一个 store 装全部时次，通道名旁挂在
channel_names.json 里，变量按 pl/sfc 分 store。这里造真 store 来验。
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.core.errors import DecodeError, DiscoveryError
from xmetai_evaluation.core.variables import DataKind
from xmetai_evaluation.io.era5_zarr_reader import (
    CHANNEL_NAMES_FILE,
    Era5ZarrCatalog,
    Era5ZarrReader,
)
from xmetai_evaluation.io.layouts import ERA5_ZARR_LAYOUT, GriddedLayout

pytest.importorskip("zarr")

LATS = [10.0, 20.0]
LONS = [-180.0, -90.0, 0.0]
START = datetime(2025, 1, 1)
STEP_HOURS = 6


def _make_store(root, name, channels, moments, fill):
    """造一个 (time, channel, lat, lon) 的 store + 旁挂通道名清单。

    ``fill(channel, step)`` 给出该通道该时次的值。
    """
    path = root / name
    data = np.stack(
        [
            np.stack(
                [
                    np.full((len(LATS), len(LONS)), fill(channel, step), dtype="f4")
                    for channel in channels
                ]
            )
            for step in range(len(moments))
        ]
    )

    dataset = xr.Dataset(
        {"data": (["time", "channel", "lat", "lon"], data)},
        coords={
            "time": np.arange(len(moments), dtype="f8") * STEP_HOURS,
            "lat": LATS,
            "lon": LONS,
        },
    )
    dataset["time"].attrs["units"] = "hours since 2025-01-01 00:00:00"
    dataset.to_zarr(path, mode="w")
    (path / CHANNEL_NAMES_FILE).write_text(
        json.dumps(list(channels)), encoding="utf-8"
    )
    return path


def _read(stores, variables, moments, layout=ERA5_ZARR_LAYOUT):
    reader = Era5ZarrReader(source_id="era5_zarr", stores=stores, layout=layout)
    request = DataRequest(source_id="era5_zarr", variables=variables, init_times=moments)
    return reader.read(request, Era5ZarrCatalog(stores).discover(request))


def _moments(count=3):
    return [START + timedelta(hours=STEP_HOURS * step) for step in range(count)]


def test_reads_across_two_stores_in_one_call(tmp_path):
    moments = _moments()
    pl = _make_store(tmp_path, "pl.zarr", ["z_500"], moments, lambda c, s: 500.0 + s)
    sfc = _make_store(tmp_path, "sfc.zarr", ["t2m"], moments, lambda c, s: 280.0 + s)

    bundle = _read({"pl": str(pl), "sfc": str(sfc)}, ["z500", "t2m"], moments)

    assert bundle.kind == DataKind.GRIDDED_OBSERVATION
    assert set(bundle.payload.data_vars) == {"z500", "t2m"}
    np.testing.assert_allclose(bundle.payload["z500"].isel(valid_time=1).values, 501.0)
    np.testing.assert_allclose(bundle.payload["t2m"].isel(valid_time=2).values, 282.0)
    assert set(bundle.provenance.input_files) == {
        str(pl), str(pl / CHANNEL_NAMES_FILE), str(sfc), str(sfc / CHANNEL_NAMES_FILE)
    }


def test_channel_name_is_resolved_by_underscore_alias(tmp_path):
    """store 里叫 z_500、配置写 z500——去掉下划线后要对得上。"""
    moments = _moments()
    pl = _make_store(tmp_path, "pl.zarr", ["z_500"], moments, lambda c, s: 500.0)

    # 布局里没声明变量，只能靠别名解析
    plain = GriddedLayout(name="plain", patterns=())
    bundle = _read({"pl": str(pl)}, ["z500"], moments, layout=plain)

    assert list(bundle.payload.data_vars) == ["z500"]
    assert bundle.standard_vars == {"z500": "z_500"}


def test_time_is_matched_exactly_and_misses_fail_loudly(tmp_path):
    moments = _moments()
    pl = _make_store(tmp_path, "pl.zarr", ["z_500"], moments, lambda c, s: 500.0)

    # 命中：顺序打乱也要对上
    shuffled = [moments[2], moments[0]]
    bundle = _read({"pl": str(pl)}, ["z500"], shuffled)
    assert list(bundle.payload["valid_time"].values) == [
        np.datetime64(moment) for moment in sorted(shuffled)
    ]

    # 未命中：必须报错并列出缺失时刻，静默错配比报错危险得多
    with pytest.raises(DecodeError, match="没有这些有效时刻"):
        _read({"pl": str(pl)}, ["z500"], [moments[0] + timedelta(hours=1)])


def test_layout_unit_conversions(tmp_path):
    """q kg/kg -> g/kg、tp m -> mm、z500 保持 m²/s²（不除 g）。"""
    moments = _moments(1)
    pl = _make_store(
        tmp_path,
        "pl.zarr",
        ["z_500", "q_700"],
        moments,
        lambda channel, step: {"z_500": 50000.0, "q_700": 0.008}[channel],
    )
    sfc = _make_store(tmp_path, "sfc.zarr", ["tp"], moments, lambda c, s: 0.002)

    bundle = _read({"pl": str(pl), "sfc": str(sfc)}, ["z500", "q700", "tp"], moments)

    np.testing.assert_allclose(bundle.payload["z500"].isel(valid_time=0).values, 50000.0)
    np.testing.assert_allclose(bundle.payload["q700"].isel(valid_time=0).values, 8.0)
    np.testing.assert_allclose(bundle.payload["tp"].isel(valid_time=0).values, 2.0)
    assert bundle.semantic.units["q700"] == "g/kg"
    assert bundle.semantic.units["tp"] == "mm"
    assert bundle.semantic.units["z500"] == "m^2/s^2"


def test_missing_sentinel_becomes_nan(tmp_path):
    moments = _moments(1)
    pl = _make_store(
        tmp_path,
        "pl.zarr",
        ["z_500"],
        moments,
        lambda c, s: 1e31,  # GRIB 缺测哨兵
    )

    bundle = _read({"pl": str(pl)}, ["z500"], moments)

    assert np.isnan(bundle.payload["z500"].isel(valid_time=0).values).all()


def test_missing_variable_is_skipped_not_fatal(tmp_path):
    moments = _moments(1)
    pl = _make_store(tmp_path, "pl.zarr", ["z_500"], moments, lambda c, s: 500.0)

    bundle = _read({"pl": str(pl)}, ["z500", "q700"], moments)

    assert list(bundle.payload.data_vars) == ["z500"]


def test_catalog_fails_loudly_on_missing_store(tmp_path):
    with pytest.raises(DiscoveryError, match="不存在"):
        _read({"pl": str(tmp_path / "nope.zarr")}, ["z500"], _moments(1))


def test_read_without_valid_times_fails_loudly(tmp_path):
    moments = _moments(1)
    pl = _make_store(tmp_path, "pl.zarr", ["z_500"], moments, lambda c, s: 500.0)

    with pytest.raises(DecodeError, match="有效时刻"):
        _read({"pl": str(pl)}, ["z500"], [])
