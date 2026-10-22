# -*- coding: utf-8 -*-
"""CRA c71 zarr 实况读取器（``cra_zarr``）的单元测试。

覆盖四件只在 CRA 上成立的事：

  1. store 存归一化值 -> 读出来是 ``x*std+mean``（与推理侧的 denormalize 同口径）；
  2. ``tp`` 在 store 里是 ``log1p(mm)`` -> 读出来做 ``expm1``；
  3. 时间轴是 zarr v3 原生 datetime64（没有 CF units）-> 照样对得上时刻；
  4. 有效时刻精确匹配：不在时间轴上就报错，不静默错配。

用临时目录现造一个同形状的小 store（与 ``tests/runner_weather_field.py`` 的
合成 store 同一手法），不依赖服务器上的真实数据。
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.core.errors import DecodeError
from xmetai_evaluation.io.era5_zarr_reader import CraZarrReader, Era5ZarrCatalog
from xmetai_evaluation.io.layouts import CRA_ZARR_LAYOUT, LAYOUTS

LAT = np.array([-10.0, 0.0, 10.0], dtype="f8")  # 南→北，与 CRA 一致
LON = np.array([0.0, 90.0, 180.0, 270.0], dtype="f8")
CHANNELS = ["z_500", "t2m", "tp"]
#: 归一化前的物理量（tp 那一层是 log1p(mm)，所以 max 很小）
PHYSICAL = np.stack(
    [
        np.full((2, 3, 4), 55000.0),
        np.full((2, 3, 4), 285.0),
        np.full((2, 3, 4), np.log1p(3.0)),
    ],
    axis=1,
)  # (time=2, channel=3, lat, lon)
MEAN = np.array([55000.0, 280.0, 0.0])
STD = np.array([500.0, 5.0, 1.0])
NORMALIZED = (PHYSICAL - MEAN[None, :, None, None]) / STD[None, :, None, None]


def _write_store(
    path: Path, *, time_as_datetime: bool = False, data_dtype: str = "f4"
) -> Path:
    import zarr

    group = zarr.open_group(str(path), mode="w")

    def create(name, *, data, **kwargs):
        method = getattr(group, "create_array", None)
        if method is None:
            method = group.create_dataset
        return method(name, data=data, **kwargs)

    create(
        "data",
        data=NORMALIZED.astype(data_dtype),
        chunks=(1, 1, len(LAT), len(LON)),
        overwrite=True,
    )
    if time_as_datetime:
        # 真实 CRA store 的写法：zarr v3 原生 datetime64，**没有** units 属性
        create(
            "time",
            data=np.array(
                ["2026-01-01T00:00", "2026-01-01T06:00"], dtype="datetime64[ns]"
            ),
            overwrite=True,
        )
    else:
        time = create("time", data=np.array([0.0, 6.0], dtype="f8"), overwrite=True)
        time.attrs["units"] = "hours since 2026-01-01 00:00:00"
    create("mean", data=MEAN.astype("f4"), overwrite=True)
    create("std", data=STD.astype("f4"), overwrite=True)
    create("lat", data=LAT, overwrite=True)
    create("lon", data=LON, overwrite=True)
    (path / "channel_names.json").write_text(json.dumps(CHANNELS), encoding="utf-8")
    return path


@pytest.fixture()
def store(tmp_path):
    return _write_store(tmp_path / "cra.zarr")


def _read(store, variables, times):
    reader = CraZarrReader(stores={"cra": str(store)})
    request = DataRequest(
        source_id="cra_zarr", variables=list(variables), init_times=list(times)
    )
    catalog = Era5ZarrCatalog({"cra": str(store)})
    return reader.read(request, catalog.discover(request))


def test_denormalize_and_expm1(store):
    """归一化值还原成物理量；tp 从 log1p 空间 expm1 回 mm。"""
    times = [datetime(2026, 1, 1, 0), datetime(2026, 1, 1, 6)]
    bundle = _read(store, ["z500", "t2m", "tp"], times)

    z500 = bundle.payload["z500"].values
    assert np.allclose(z500, 55000.0, atol=1e-2)
    assert np.allclose(bundle.payload["t2m"].values, 285.0, atol=1e-3)
    # log1p(3) -> 3 mm
    assert np.allclose(bundle.payload["tp"].values, 3.0, rtol=1e-5)
    assert bundle.semantic.units["tp"] == "mm"
    assert bundle.provenance.reader_id == "cra_zarr"


def test_datetime64_time_axis(tmp_path):
    """CRA store 的 time 是 zarr v3 原生 datetime64、没有 units，也要能对上时刻。

    真实 store（cra.19790101_20260714.c71.p25.h6.v1）就是这样：``time`` 的
    dtype 是 ``numpy.datetime64[ns]``，属性里只有 standard_name/axis，没有
    ``units``；按 CF 口径去解析会直接报错。
    """
    store = _write_store(tmp_path / "cra_datetime.zarr", time_as_datetime=True)
    times = [datetime(2026, 1, 1, 0), datetime(2026, 1, 1, 6)]
    bundle = _read(store, ["z500"], times)

    assert bundle.payload["z500"].shape == (2, len(LAT), len(LON))
    assert np.allclose(bundle.payload["z500"].values, 55000.0, atol=1e-2)


def test_float16_data_reads_without_overflow_warning(tmp_path):
    """真实 CRA store 的 data 是 float16；跟 1e30 的哨兵比较不能炸出溢出告警。"""
    import warnings

    store = _write_store(tmp_path / "cra_f16.zarr", data_dtype="f2")
    times = [datetime(2026, 1, 1, 0), datetime(2026, 1, 1, 6)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = _read(store, ["z500", "tp"], times)

    overflow = [item for item in caught if "overflow" in str(item.message)]
    assert not overflow, [str(item.message) for item in overflow]
    # float16 只有约三位有效数字：log1p(3) 存成 f2 再 expm1 会差到千分之几
    assert np.allclose(bundle.payload["tp"].values, 3.0, rtol=2e-3)


def test_missing_time_raises(store):
    """时间轴上没有的有效时刻必须报错，不能静默错配。"""
    with pytest.raises(DecodeError, match="有效时刻"):
        _read(store, ["z500"], [datetime(2026, 1, 2, 0)])


def test_denormalize_without_stats_raises(tmp_path):
    """开了 denormalize 但 store 没有 mean/std：直接报错，不猜。"""
    import zarr

    path = _write_store(tmp_path / "cra_nostats.zarr")
    group = zarr.open_group(str(path), mode="a")
    del group["mean"]
    with pytest.raises(DecodeError, match="mean/std"):
        _read(path, ["z500"], [datetime(2026, 1, 1, 0)])


def test_layout_covers_all_71_channels():
    """布局要覆盖 c71 store 的全部通道，层次名与推理侧一致。"""
    variables = CRA_ZARR_LAYOUT.variables
    assert len(variables) == 71
    for name in ("z50", "z500", "z1000", "t2m", "q2m", "u10m", "v10m", "msl", "tp"):
        assert name in variables
    assert LAYOUTS["cra_zarr"] is CRA_ZARR_LAYOUT
