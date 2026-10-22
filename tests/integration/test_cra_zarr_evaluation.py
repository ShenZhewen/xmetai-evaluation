# -*- coding: utf-8 -*-
"""CRA 评测链路（Tongyun CRA 集合 × ``cra_zarr``）的合成数据端到端测试。

不碰服务器数据，用 9×16 的小网格现造两样东西：

* 集合预报 ``root/YYYYMMDD/member_XXX/00N.nc``（``fuxi_ens_phys`` 布局），
  两个成员在 z500 上各偏 +2 / +4，tp 分别 1 / 2 mm；
* CRA store（**归一化存储**、lat 南→北、lon 0–360、tp 在 log1p 空间），
  z500 真值无偏、tp 真值 3 mm。

于是集合均值 z500 恰好偏 +3、tp 均值 1.5，RMSE 应分别是 3.0 与 1.5——
任何一处（反归一化、expm1、lat 翻转、成员顺序）错了，数都对不上。

另外钉住配置口径：这条流程**不算 ACC / 活跃度**，scores.csv 里不该出现这两族。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.runner_weather_field import SYNTHETIC_FIELD_LAT, SYNTHETIC_FIELD_LON, _synthetic_fields

INIT = datetime(2025, 1, 2)
LEADS = (6, 12)
MEMBERS = {"member_000": 2.0, "member_001": 4.0}

#: 预报侧网格：lat 北→南、lon 0–360 升序（与生产一致）
FCST_LAT = SYNTHETIC_FIELD_LAT[::-1]
FCST_LON = np.sort(np.mod(SYNTHETIC_FIELD_LON, 360.0))
Z500, _ = _synthetic_fields(FCST_LAT, FCST_LON)

#: 实况侧网格：CRA 的 lat 是南→北
OBS_LAT = SYNTHETIC_FIELD_LAT
OBS_LON = FCST_LON

Z500_MEAN, Z500_STD = 50_000.0, 100.0


def _write_forecast(root: Path) -> Path:
    date_dir = root / INIT.strftime("%Y%m%d")
    for member, offset in MEMBERS.items():
        member_dir = date_dir / member
        member_dir.mkdir(parents=True, exist_ok=True)
        # 成员间 tp 取 1 / 2 mm（集合均值 1.5），与时效无关
        tp = 1.0 if member == "member_000" else 2.0
        for index, lead in enumerate(LEADS, start=1):
            dataset = xr.Dataset(
                {
                    "Z500": (("lat", "lon"), Z500 + offset),
                    "TP": (("lat", "lon"), np.full(Z500.shape, tp)),
                },
                coords={"lat": FCST_LAT, "lon": FCST_LON},
            )
            dataset["Z500"].attrs["units"] = "m^2/s^2"
            dataset["TP"].attrs["units"] = "mm"
            dataset.to_netcdf(member_dir / f"{index:03d}.nc")
    return root


def _write_cra_store(path: Path) -> Path:
    """归一化存储：``data = (物理量 - mean) / std``，tp 存 log1p(mm)。"""
    import zarr

    z500_true = np.stack([Z500] * len(LEADS), axis=0)  # (time, lat, lon)
    tp_true = np.stack([np.full(Z500.shape, 3.0)] * len(LEADS), axis=0)
    physical = np.stack([z500_true, np.log1p(tp_true)], axis=1)  # (time, channel, lat, lon)
    # 实况 lat 与预报相反：把北→南的场翻成南→北
    physical = physical[:, :, ::-1, :]
    mean = np.array([Z500_MEAN, 0.0])
    std = np.array([Z500_STD, 1.0])
    normalized = (physical - mean[None, :, None, None]) / std[None, :, None, None]

    group = zarr.open_group(str(path), mode="w")

    def create(name, *, data, **kwargs):
        method = getattr(group, "create_array", None)
        if method is None:
            method = group.create_dataset
        return method(name, data=data, **kwargs)

    create(
        "data",
        data=normalized.astype("f4"),
        chunks=(1, 1, len(OBS_LAT), len(OBS_LON)),
        overwrite=True,
    )
    time = create("time", data=np.array(LEADS, dtype="f8"), overwrite=True)
    time.attrs["units"] = "hours since 2025-01-02 00:00:00"
    create("mean", data=mean.astype("f4"), overwrite=True)
    create("std", data=std.astype("f4"), overwrite=True)
    create("lat", data=OBS_LAT, overwrite=True)
    create("lon", data=OBS_LON, overwrite=True)
    (path / "channel_names.json").write_text('["z_500", "tp"]', encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def scores(tmp_path_factory):
    root = tmp_path_factory.mktemp("cra_eval")
    forecast_root = _write_forecast(root / "forecast")
    store = _write_cra_store(root / "cra.zarr")

    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config("weather_rmse_ens_tongyun_cra")
    variables = ["z500", "tp"]
    cfg.forecast_reader = {
        **cfg.forecast_reader,
        "root_dir": str(forecast_root),
        "variables": variables,
        "lead_times": list(LEADS),
    }
    cfg.observation_reader = {
        **cfg.observation_reader,
        "path": str(store),
        "variables": variables,
    }
    cfg.periods = [("20250102", "20250102")]
    cfg.limit = None
    cfg.output_dir = str(root / "out")
    cfg.metric_options = {
        "rmse": {"variables": variables},
        "crps": {"variables": variables},
        "spread_error": {"variables": variables, "ddof": 0},
        "zonal_spectrum": {"variables": variables, "max_wavenumber": 3},
        "spherical_bands": {
            "variables": variables,
            "bands": [(1, 1), (2, 3)],
            "block": 1,
        },
    }
    cfg.writers = ["csv_long", "spectrum"]
    cfg.execution = {"mode": "serial", "n_workers": 1, "resume": False}
    run_evaluation(cfg)

    import pandas as pd

    return pd.read_csv(Path(cfg.output_dir) / "scores.csv")


def _value(scores, metric: str, variable: str, lead: int) -> float:
    rows = scores[
        (scores["metric"] == metric)
        & (scores["variable"] == variable)
        & (scores["lead_h"] == float(lead))
        & scores["region"].isna()
    ]
    assert len(rows) >= 1, f"scores.csv 里没有 {metric} / {variable} @ {lead}h"
    values = rows["value"].to_numpy(dtype=float)
    assert np.ptp(values) <= 1e-6 * max(abs(values[0]), 1e-12), values
    return float(values[0])


@pytest.mark.parametrize("lead", LEADS)
def test_rmse_after_denormalize(scores, lead):
    """z500：集合均值 +3 -> RMSE 3；tp：expm1 后均值 1.5 vs 3 -> RMSE 1.5。"""
    assert _value(scores, "rmse", "z500", lead) == pytest.approx(3.0, abs=1e-3)
    assert _value(scores, "rmse", "tp", lead) == pytest.approx(1.5, abs=1e-4)


@pytest.mark.parametrize("lead", LEADS)
def test_two_member_crps(scores, lead):
    """成员 {1,2} vs 真值 3：CRPS = 1.5 - (1/8)*2 = 1.25（闭式解，与时效无关）。"""
    assert _value(scores, "crps", "tp", lead) == pytest.approx(1.25, abs=1e-4)


def test_no_reference_metrics(scores):
    """没给气候态：ACC / 活跃度一律不该出现在结果里。"""
    metrics = set(scores["metric"].unique())
    assert not {"acc", "activity_ratio", "activity_forecast", "activity_bias"} & metrics
    assert {"rmse", "crps", "spread", "spread_error_ratio"} <= metrics
