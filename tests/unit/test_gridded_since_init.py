# -*- coding: utf-8 -*-
"""从起报累计的变量（AIFS 的 tp）在 Reader 内差分，以及计划层的多一步预热。

ECMWF/AIFS 集合的 ``tp`` 是 ``C(k) = Σ_{j≤k} I(j)``（自起报累计），而累积窗
和阈值判定要的是间隔量 ``I(k) = C(k) - C(k-1)``。不差分时 24h 窗会把 1..k 步
再加一遍，偏差随时效线性增长（实测 ≥10mm 的 BIAS +24h 2.63 → +360h 10.98）。
"""

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.execution.plan import _warmup_hours
from xmetai_evaluation.io.gridded import GriddedReader
from xmetai_evaluation.io.layouts import FUXI_ENS_LAYOUT


def _cumulative(values) -> xr.Dataset:
    """``(init_time, member, lead_time)`` 三维的累计降水，lead 6/12/18/24h。"""
    data = np.asarray(values, dtype="f8")
    return xr.Dataset(
        {"tp": (("init_time", "member", "lead_time"), data, {"units": "mm"})},
        coords={
            "init_time": [np.datetime64("2025-07-02T00:00")],
            "member": [0, 1],
            "lead_time": [6.0, 12.0, 18.0, 24.0],
        },
    )


def test_differences_per_member_and_keeps_first_step():
    # 成员 0 的累计 [1,3,6,10] -> 增量 [1,2,3,4]；成员 1 的累计 [0,0,2,5] -> [0,0,2,3]
    values = [[[1, 3, 6, 10], [0, 0, 2, 5]]]
    reader = GriddedReader("aifs", FUXI_ENS_LAYOUT, since_init_variables=["tp"])

    out = reader._difference_since_init(_cumulative(values))

    np.testing.assert_allclose(out["tp"].values[0, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(out["tp"].values[0, 1], [0, 0, 2, 3])
    np.testing.assert_allclose(out["lead_time"].values, [6.0, 12.0, 18.0, 24.0])
    assert out["tp"].attrs["units"] == "mm"


def test_first_step_equals_archive_first_interval():
    # 归档第一个文件（001.nc, +6h）的累计量恰好是第一个 6h 间隔，不能被差分掉
    values = [[[0.58, 1.18, 1.9, 2.6], [0.0, 0.0, 0.0, 0.0]]]
    reader = GriddedReader("aifs", FUXI_ENS_LAYOUT, since_init_variables=["tp"])

    out = reader._difference_since_init(_cumulative(values))

    assert out["tp"].values[0, 0, 0] == pytest.approx(0.58)
    np.testing.assert_allclose(out["tp"].values[0, 0, 1:], [0.60, 0.72, 0.70], rtol=1e-12)


def test_without_flag_values_untouched():
    values = [[[1, 3, 6, 10], [0, 0, 2, 5]]]
    reader = GriddedReader("fuxi", FUXI_ENS_LAYOUT)

    out = reader._difference_since_init(_cumulative(values))

    np.testing.assert_allclose(out["tp"].values, np.asarray(values, dtype="f8"))


def test_only_listed_variables_are_differenced():
    values = [[[1, 3, 6, 10], [0, 0, 2, 5]]]
    dataset = _cumulative(values)
    dataset["t2m"] = xr.zeros_like(dataset["tp"]) + 280.0
    reader = GriddedReader("aifs", FUXI_ENS_LAYOUT, since_init_variables=["tp"])

    out = reader._difference_since_init(dataset)

    np.testing.assert_allclose(out["tp"].values[0, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(out["t2m"].values, 280.0)


def test_single_lead_is_left_alone():
    # 读取集只有一个时效时无法差分（也没有会用到它的窗口），保留原值
    dataset = _cumulative([[[1, 3, 6, 10], [0, 0, 2, 5]]]).isel(lead_time=slice(0, 1))
    reader = GriddedReader("aifs", FUXI_ENS_LAYOUT, since_init_variables=["tp"])

    out = reader._difference_since_init(dataset)

    np.testing.assert_allclose(out["tp"].values, np.asarray([[[1.0], [0.0]]]))


def test_unknown_since_init_variable_raises():
    # 变量名写错时不能静默保留累计口径（偏差只会体现在结果里）
    with pytest.raises(ValueError, match="未声明"):
        GriddedReader("aifs", FUXI_ENS_LAYOUT, since_init_variables=["tp0"])


class _WindowSpec:
    window_hours = 24.0

    def transform(self, name):
        return object() if name == "time_window_accumulator" else None


class _Forecast:
    def __init__(self, needs_previous_step: bool):
        self.reader = type("_R", (), {"needs_previous_step": needs_previous_step})()


def test_warmup_adds_one_step_for_since_init():
    # 24h 窗 / 6h 步长：原有预热 18h；since-init 差分再要一个前驱 -> 24h
    assert _warmup_hours(_WindowSpec(), 6.0) == 18.0
    assert _warmup_hours(_WindowSpec(), 6.0, _Forecast(False)) == 18.0
    assert _warmup_hours(_WindowSpec(), 6.0, _Forecast(True)) == 24.0
