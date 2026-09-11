"""日序气候态 Reader（单文件，按年内日序索引）。

参考实现（xu）的两种布局都要认：
    legacy  data(time, level, lat, lon)，level 是变量名字符串
    常规    每个变量一个 dataset

时间轴也有两种写法，都要认：
    数值轴  time = 0,1,2,… + ``units: days since …``（since 起算时刻忽略）
    合成年  time = 某一年 1/1 起的连续时刻（日序编码在时刻里）；
            366 天口径会溢出到次年 1/1，那是正常的
"""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.core.errors import DecodeError, DiscoveryError
from xmetai_evaluation.core.variables import DataKind
from xmetai_evaluation.io.daily_climatology_reader import (
    DailyClimatologyCatalog,
    DailyClimatologyReader,
)

LATS = [10.0, 20.0]
LONS = [100.0, 110.0, 120.0]


def _write_per_variable(root, days=365, steps_per_day=1, variable="z500", name="clim.nc"):
    """常规布局：一个变量一个 dataset，时间轴是年内日序。

    第 n 步填 n，取到的值直接就是步号——日序命中、插值、环绕一眼可验。
    """
    count = days * steps_per_day
    hours = np.arange(count, dtype="f8") * (24.0 / steps_per_day)
    values = np.repeat(np.arange(count, dtype="f4")[:, None, None], len(LATS), axis=1)
    values = np.repeat(values, len(LONS), axis=2)
    dataset = xr.Dataset(
        {variable: (["time", "lat", "lon"], values, {"units": "m"})},
        coords={"time": hours, "lat": LATS, "lon": LONS},
    )
    dataset["time"].attrs["units"] = "hours since 2020-01-01 00:00:00"
    dataset.to_netcdf(root / name)


def _write_legacy(root, days=365, variables=("z500", "t2m"), name="legacy.nc"):
    """legacy 布局：data(time, level, lat, lon)，level 是变量名。"""
    count = days
    values = np.stack(
        [
            np.full((count, len(LATS), len(LONS)), float(order))
            for order in range(len(variables))
        ],
        axis=1,
    )
    dataset = xr.Dataset(
        {"data": (["time", "level", "lat", "lon"], values)},
        coords={
            "time": np.arange(count, dtype="f8"),
            "level": list(variables),
            "lat": LATS,
            "lon": LONS,
        },
    )
    dataset["time"].attrs["units"] = "days since 2020-01-01 00:00:00"
    dataset.to_netcdf(root / name)


def _read(root, variables, moments, window=1):
    catalog = DailyClimatologyCatalog(root_dir=root)
    reader = DailyClimatologyReader(window=window)
    request = DataRequest(source_id="daily_climatology", variables=variables, init_times=moments)
    return reader.read(request, catalog.discover(request))


def _field(bundle, variable, index=0):
    return bundle.payload[variable].isel(valid_time=index).values


def test_per_variable_layout_hits_day_of_year(tmp_path):
    _write_per_variable(tmp_path)
    bundle = _read(tmp_path, ["z500"], [datetime(2025, 3, 10)])

    assert bundle.kind == DataKind.REFERENCE
    assert bundle.payload["z500"].dims == ("valid_time", "lat", "lon")
    # 2025 非闰年，3/10 的 0-based 日序是 31+28+9
    np.testing.assert_allclose(_field(bundle, "z500"), 68.0)


def test_leap_day_collapses_onto_feb_28(tmp_path):
    """365 天文件里没有 2/29 那一格，闰年的 2/29 并到 2/28。"""
    _write_per_variable(tmp_path, days=365)
    bundle = _read(
        tmp_path, ["z500"], [datetime(2024, 2, 28), datetime(2024, 2, 29), datetime(2024, 3, 1)]
    )

    np.testing.assert_allclose(_field(bundle, "z500", 0), 58.0)  # 2/28
    np.testing.assert_allclose(_field(bundle, "z500", 1), 58.0)  # 2/29 -> 并到 2/28
    np.testing.assert_allclose(_field(bundle, "z500", 2), 59.0)  # 与平年的 3/1 对齐


def test_366_day_file_keeps_feb_29_and_shifts_common_years(tmp_path):
    """366 天文件含 2/29；平年请求要跳过那一格。"""
    _write_per_variable(tmp_path, days=366)
    bundle = _read(
        tmp_path, ["z500"], [datetime(2024, 2, 29), datetime(2025, 3, 1), datetime(2025, 2, 28)]
    )

    np.testing.assert_allclose(_field(bundle, "z500", 0), 59.0)  # 闰年 2/29 就是第 59 格
    np.testing.assert_allclose(_field(bundle, "z500", 1), 60.0)  # 平年 3/1 顺延到 60
    np.testing.assert_allclose(_field(bundle, "z500", 2), 58.0)  # 2/28 不受影响


def test_sub_daily_step_interpolates_linearly(tmp_path):
    """6 小时一次的气候态：请求时刻落在两步之间时要插值。"""
    _write_per_variable(tmp_path, days=365, steps_per_day=4)
    bundle = _read(tmp_path, ["z500"], [datetime(2025, 1, 2)])

    # 1/2 00:00 正好是第 4 步（不插值）
    np.testing.assert_allclose(_field(bundle, "z500"), 4.0)

    between = _read(tmp_path, ["z500"], [datetime(2025, 1, 2, 3)])
    # 1/2 03:00 -> 日序 1.125 天 = 第 4.5 步 -> 取第 4/5 步的中点
    np.testing.assert_allclose(_field(between, "z500"), 4.5)


def test_year_boundary_wraps_around(tmp_path):
    """12/31 12:00 要在"年末"和"年初"之间插值，不能越界。"""
    _write_per_variable(tmp_path, days=365)
    bundle = _read(tmp_path, ["z500"], [datetime(2025, 12, 31, 12)])

    np.testing.assert_allclose(_field(bundle, "z500"), 0.5 * 364.0)


def test_legacy_layout_reads_named_levels(tmp_path):
    _write_legacy(tmp_path, variables=("z500", "t2m"))
    bundle = _read(tmp_path, ["t2m", "z500"], [datetime(2025, 1, 11)])

    assert set(bundle.payload.data_vars) == {"z500", "t2m"}
    assert list(bundle.payload.data_vars) == ["t2m", "z500"]  # 保持请求顺序
    np.testing.assert_allclose(_field(bundle, "z500"), 0.0)  # level 顺序里的第 0 个
    np.testing.assert_allclose(_field(bundle, "t2m"), 1.0)


def test_missing_variable_is_skipped_not_fatal(tmp_path):
    """一个源少几个要素不该让整轮评测挂掉——与参考实现同一语义。"""
    _write_per_variable(tmp_path, variable="z500")
    bundle = _read(tmp_path, ["z500", "q700"], [datetime(2025, 1, 1)])

    assert list(bundle.payload.data_vars) == ["z500"]


def test_no_requested_variable_at_all_fails_loudly(tmp_path):
    _write_per_variable(tmp_path, variable="z500")
    with pytest.raises(DecodeError, match="没有请求的变量"):
        _read(tmp_path, ["q700"], [datetime(2025, 1, 1)])


def test_smoothing_averages_a_spike_over_the_window(tmp_path):
    """15 天环形平滑：年初那一个尖峰被摊到 15 天里。"""
    count = 365
    values = np.zeros((count, len(LATS), len(LONS)), dtype="f4")
    values[0] = 365.0
    dataset = xr.Dataset(
        {"z500": (["time", "lat", "lon"], values)},
        coords={"time": np.arange(count, dtype="f8"), "lat": LATS, "lon": LONS},
    )
    dataset["time"].attrs["units"] = "days since 2020-01-01 00:00:00"
    dataset.to_netcdf(tmp_path / "spike.nc")

    raw = _read(tmp_path, ["z500"], [datetime(2025, 1, 1)], window=1)
    smoothed = _read(tmp_path, ["z500"], [datetime(2025, 1, 1)], window=15)

    np.testing.assert_allclose(_field(raw, "z500"), 365.0)
    np.testing.assert_allclose(_field(smoothed, "z500"), 365.0 / 15.0)


def test_smoothing_wraps_across_year_end(tmp_path):
    """平滑是环形的：年末的值会摊到年初，反之亦然。"""
    count = 365
    values = np.zeros((count, len(LATS), len(LONS)), dtype="f4")
    values[-1] = 365.0  # 12/31
    dataset = xr.Dataset(
        {"z500": (["time", "lat", "lon"], values)},
        coords={"time": np.arange(count, dtype="f8"), "lat": LATS, "lon": LONS},
    )
    dataset["time"].attrs["units"] = "days since 2020-01-01 00:00:00"
    dataset.to_netcdf(tmp_path / "wrap.nc")

    bundle = _read(tmp_path, ["z500"], [datetime(2025, 1, 1)], window=15)

    np.testing.assert_allclose(_field(bundle, "z500"), 365.0 / 15.0)


def test_wrong_year_length_fails_loudly(tmp_path):
    _write_per_variable(tmp_path, days=300, name="short.nc")
    with pytest.raises(DecodeError, match="365|366"):
        _read(tmp_path, ["z500"], [datetime(2025, 1, 1)])


def test_uneven_step_fails_loudly(tmp_path):
    dataset = xr.Dataset(
        {"z500": (["time", "lat", "lon"], np.zeros((3, 2, 3), dtype="f4"))},
        coords={"time": [0.0, 6.0, 18.0], "lat": LATS, "lon": LONS},
    )
    dataset["time"].attrs["units"] = "hours since 2020-01-01 00:00:00"
    dataset.to_netcdf(tmp_path / "uneven.nc")

    with pytest.raises(DecodeError, match="步长不均匀"):
        _read(tmp_path, ["z500"], [datetime(2025, 1, 1)])


def _write_datetime_axis(root, times, name="synthetic_year.nc"):
    """把给定的时刻轴写成一份气候态，第 n 步填 n——日序/插值一眼可验。"""
    values = np.repeat(np.arange(times.size, dtype="f4")[:, None, None], len(LATS), axis=1)
    values = np.repeat(values, len(LONS), axis=2)
    xr.Dataset(
        {"z500": (["time", "lat", "lon"], values)},
        coords={"time": times, "lat": LATS, "lon": LONS},
    ).to_netcdf(root / name)


def test_synthetic_year_datetime_axis_is_indexed_by_day_of_year(tmp_path):
    """2021 年 365 个逐日时刻：日序就是「距 2021-01-01 的天数」。"""
    _write_datetime_axis(tmp_path, np.arange("2021-01-01", "2022-01-01", dtype="datetime64[D]"))
    bundle = _read(tmp_path, ["z500"], [datetime(2025, 3, 10)])

    np.testing.assert_allclose(_field(bundle, "z500"), 68.0)


def test_leap_synthetic_year_keeps_feb_29(tmp_path):
    """2020 是闰年，366 个时刻 -> 按 366 天口径索引。"""
    _write_datetime_axis(
        tmp_path, np.arange("2020-01-01", "2021-01-01", dtype="datetime64[D]"), name="leap.nc"
    )
    bundle = _read(tmp_path, ["z500"], [datetime(2024, 2, 29), datetime(2025, 3, 1)])

    np.testing.assert_allclose(_field(bundle, "z500", 0), 59.0)  # 闰年 2/29
    np.testing.assert_allclose(_field(bundle, "z500", 1), 60.0)  # 平年 3/1 顺延


def test_366_day_datetime_file_may_spill_into_the_next_calendar_year(tmp_path):
    """真实 ``era5_clim_phys_14.nc`` 的形状：1990-01-01 00:00 起、6 小时一次、1464 步。

    366 天口径从平年 1/1 起算，第 366 天必然落到次年 1/1——跨年是正常的。
    """
    steps = 366 * 4
    times = np.datetime64("1990-01-01T00") + np.arange(steps) * np.timedelta64(6, "h")
    _write_datetime_axis(tmp_path, times, name="phys14.nc")

    # 闰年 2/29 落在第 59 天（含 2/29 的口径里它就在那）-> 第 59*4 步
    leap = _read(tmp_path, ["z500"], [datetime(2024, 2, 29)])
    np.testing.assert_allclose(_field(leap, "z500"), 59.0 * 4)

    # 平年 3/1 要顺延过那个空出来的 2/29 -> 第 60 天
    shifted = _read(tmp_path, ["z500"], [datetime(2025, 3, 1)])
    np.testing.assert_allclose(_field(shifted, "z500"), 60.0 * 4)


def test_datetime_axis_starting_mid_year_fails_loudly(tmp_path):
    """日序锚在 1/1：从年中起算的文件会把 6 月当成 1 月，必须拒绝而不是对错格。"""
    times = np.arange("2021-06-01", "2022-06-01", dtype="datetime64[D]")
    _write_datetime_axis(tmp_path, times, name="midyear.nc")

    with pytest.raises(DecodeError, match="1/1"):
        _read(tmp_path, ["z500"], [datetime(2025, 1, 1)])


def test_catalog_accepts_file_or_directory(tmp_path):
    _write_per_variable(tmp_path)
    request = DataRequest(
        source_id="daily_climatology", variables=["z500"], init_times=[datetime(2025, 1, 1)]
    )

    from_dir = DailyClimatologyCatalog(root_dir=tmp_path).discover(request)
    from_file = DailyClimatologyCatalog(root_dir=tmp_path / "clim.nc").discover(request)

    assert from_dir.available[0]["path"] == from_file.available[0]["path"]


def test_catalog_rejects_missing_path(tmp_path):
    with pytest.raises(DiscoveryError):
        DailyClimatologyCatalog(root_dir=tmp_path / "nope.nc").discover(
            DataRequest(source_id="daily_climatology", variables=["z500"],
                        init_times=[datetime(2025, 1, 1)])
        )
