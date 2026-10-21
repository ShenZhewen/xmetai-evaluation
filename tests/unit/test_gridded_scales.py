# -*- coding: utf-8 -*-
"""格点 Reader 的配置级单位系数 ``scales``。

用于复用同一布局、但落盘单位不同的模型：AIFS / FengQing 的 q 是 kg/kg，
FuXi 物理口径声明的是 g/kg，config 里用 ``scales={"q700": 1000.0}`` 补齐。
"""

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.io.gridded import GriddedReader
from xmetai_evaluation.io.layouts import FUXI_PHYS_LAYOUT


def _q700_dataset(value: float = 0.005) -> xr.Dataset:
    return xr.Dataset(
        {"q700": (("lat", "lon"), np.full((2, 2), value), {"units": "g/kg"})},
        coords={"lat": [-10.0, 0.0], "lon": [100.0, 110.0]},
    )


def test_scales_multiplies_after_layout_conversion():
    reader = GriddedReader("aifs", FUXI_PHYS_LAYOUT, scales={"q700": 1000.0})

    out = reader._normalize(_q700_dataset(), ["q700"])

    np.testing.assert_allclose(out["q700"].values, 5.0)
    assert out["q700"].attrs["units"] == "g/kg"


def test_without_scales_values_untouched():
    reader = GriddedReader("fuxi", FUXI_PHYS_LAYOUT)

    out = reader._normalize(_q700_dataset(), ["q700"])

    np.testing.assert_allclose(out["q700"].values, 0.005)


def test_unknown_scale_key_raises():
    # 变量名写错时不能静默不生效（单位错会污染整批结果）
    with pytest.raises(ValueError, match="未声明"):
        GriddedReader("aifs", FUXI_PHYS_LAYOUT, scales={"q70": 1000.0})
