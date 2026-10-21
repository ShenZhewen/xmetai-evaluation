"""空间分布：列名契约、宽表读取、出图行数、§4.6 文字与两份报告的图号契约。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xmetai_evaluation.visualization import spatial_report as sr

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "skills" / "xmetai-evaluation" / "scripts"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _spatial_common import target_grid  # noqa: E402
from extract_spatial_distribution_ens import read_ensemble_lead  # noqa: E402


LAT = np.array([60.0, 30.0, 0.0, -30.0, -60.0])
LON = np.array([0.0, 90.0, 180.0, 270.0])
MONTHS = ("202507", "202511")
LEADS = (24, 120)


def _frame(*, spread: bool = True, months=MONTHS, leads=LEADS,
           variables=("z500", "ws850")) -> pd.DataFrame:
    lat2d = np.repeat(LAT, LON.size)
    lon2d = np.tile(LON, LAT.size)
    columns = {"lat": lat2d, "lon": lon2d}
    for tag in months:
        for lead in leads:
            for variable in variables:
                shape = (LAT.size, LON.size)
                base = 5500.0 if variable == "z500" else 8.0
                noise = np.linspace(-1.0, 1.0, LAT.size * LON.size) * lead / 1000.0
                columns[f"{variable}_{tag}_h{lead:03d}"] = base + noise
                columns[f"{variable}_era5_{tag}_h{lead:03d}"] = base
                if spread:
                    columns[f"{variable}_spread_{tag}_h{lead:03d}"] = np.full(
                        LAT.size * LON.size, 1.0 + lead / 100.0
                    )
    return pd.DataFrame(columns)


def _write(tmp_path: Path, frame: pd.DataFrame, name="spatial.csv") -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- 列名契约


@pytest.mark.parametrize(
    "column, expected",
    [
        ("z500_202507_h024", ("z500", "model", "202507", 24)),
        ("z500_spread_202507_h120", ("z500", "spread", "202507", 120)),
        ("ws850_era5_202511_h072", ("ws850", "era5", "202511", 72)),
        ("msl_202501_h006", ("msl", "model", "202501", 6)),
    ],
)
def test_column_regex_parses_kinds(column, expected):
    matched = sr.COLUMN_RE.match(column)
    assert matched is not None
    var, kind, tag, lead = expected
    assert matched.group("var") == var
    assert (matched.group("kind") or "model") == kind
    assert matched.group("tag") == tag
    assert int(matched.group("lead")) == lead


@pytest.mark.parametrize(
    "column",
    ["z500_202507", "z500_spread_202507", "202507_h024", "z500_202507_h24"],
)
def test_column_regex_rejects_malformed(column):
    assert sr.COLUMN_RE.match(column) is None


# ---------------------------------------------------------------- 读表


def test_load_det_csv_has_no_spread(tmp_path):
    data = sr.load_spatial_csv(_write(tmp_path, _frame(spread=False)))
    assert data.has_spread is False
    assert data.variables == ("z500", "ws850")
    assert data.months == (("202507", "2025-07"), ("202511", "2025-11"))
    assert data.leads == (24, 120)
    assert data.figure_count == 4
    field = data.fields[("z500", "202507", 24, "model")]
    assert field.shape == (LAT.size, LON.size)
    assert np.isclose(np.median(field), 5500.0)


def test_load_ens_csv_has_spread(tmp_path):
    data = sr.load_spatial_csv(_write(tmp_path, _frame()))
    assert data.has_spread is True
    assert ("z500", "202507", 24, "spread") in data.fields


def test_load_converts_z500_m2s2_to_gpm(tmp_path):
    frame = _frame(spread=True)
    for column in frame.columns:
        if str(column).startswith("z500"):
            frame[column] = frame[column] * sr.GRAVITY
    data = sr.load_spatial_csv(_write(tmp_path, frame))
    model = data.fields[("z500", "202507", 24, "model")]
    spread = data.fields[("z500", "202507", 24, "spread")]
    assert 5000.0 < float(np.median(model)) < 6000.0
    assert float(np.median(spread)) < 10.0


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sr.load_spatial_csv(tmp_path / "nope.csv")


def test_load_rejects_missing_era5(tmp_path):
    frame = _frame(spread=False).drop(columns=["ws850_era5_202511_h120"])
    with pytest.raises(ValueError, match="缺少列"):
        sr.load_spatial_csv(_write(tmp_path, frame))


def test_load_rejects_incomplete_spread(tmp_path):
    frame = _frame().drop(columns=["ws850_spread_202511_h120"])
    with pytest.raises(ValueError, match="spread 列不完整"):
        sr.load_spatial_csv(_write(tmp_path, frame))


def test_load_rejects_unknown_column(tmp_path):
    frame = _frame(spread=False)
    frame["z500_bias_202507_h024"] = 0.0
    with pytest.raises(ValueError, match="无法识别"):
        sr.load_spatial_csv(_write(tmp_path, frame))


def test_load_rejects_wrong_row_order(tmp_path):
    frame = _frame(spread=False)
    shuffled = frame.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="行序"):
        sr.load_spatial_csv(_write(tmp_path, shuffled))


# ---------------------------------------------------------------- 出图


def _payload(tmp_path, *, mode="det", spread=True):
    data = sr.load_spatial_csv(_write(tmp_path, _frame(spread=spread)))
    out = tmp_path / "figs"
    paths = sr.render_spatial(data, out, model="Tongyun", mode=mode)
    return data, sr.SpatialPayload(
        data=data, model="Tongyun", mode=mode,
        figures={key: p.name for key, p in paths.items()},
    )


def test_render_det_writes_three_row_figures(tmp_path):
    data, payload = _payload(tmp_path, mode="det")
    assert len(payload.figures) == data.figure_count == 4
    for name in payload.figures.values():
        assert (tmp_path / "figs" / name).stat().st_size > 0
    assert len(sr._rows(data, "Tongyun", "det")) == 3


def test_render_ens_writes_four_row_figures(tmp_path):
    data, payload = _payload(tmp_path, mode="ens")
    assert len(payload.figures) == 4
    assert len(sr._rows(data, "Tongyun", "ens")) == 4


def test_ens_mode_requires_spread(tmp_path):
    data = sr.load_spatial_csv(_write(tmp_path, _frame(spread=False)))
    with pytest.raises(ValueError, match="spread"):
        sr.render_spatial(data, tmp_path / "figs", model="Tongyun", mode="ens")


# ---------------------------------------------------------------- §4.6 正文


def test_section_absent_keeps_slot():
    lines = sr.build_spatial_section(None, heading="## 4.6 空间分布图", figure_start=8)
    text = "\n".join(lines)
    assert text.startswith("## 4.6 空间分布图")
    assert sr.SPATIAL_ABSENT in text
    assert "本批未出" in text
    assert "图 8：" not in text


def test_section_present_numbers_figures(tmp_path):
    _data, payload = _payload(tmp_path, mode="ens")
    text = "\n".join(
        sr.build_spatial_section(payload, heading="## 4.6 空间分布图", figure_start=7)
    )
    assert "图 7：" in text and "图 10：" in text
    assert "图 11：" not in text
    assert "离散度" in text
    assert text.count("](tongyun_") == 4


def test_section_present_requires_all_figures(tmp_path):
    _data, payload = _payload(tmp_path, mode="det")
    broken = sr.SpatialPayload(
        data=payload.data, model=payload.model, mode=payload.mode,
        figures={},
    )
    with pytest.raises(ValueError, match="缺图"):
        sr.build_spatial_section(broken, heading="## 4.6", figure_start=8)


# ---------------------------------------------------------------- 图号契约


def _det_text(numbers, paired=()):
    return "\n".join(
        f"图 {index}：x" for index in sorted(list(numbers) + list(paired))
    )


def test_det_contract_k0_matches_previous_numbers():
    from xmetai_evaluation.visualization.det_report import _check_figure_contract

    _check_figure_contract(
        _det_text([1, 2, 3, 4, 5, 8], paired=(6, 7))
    )
    with pytest.raises(ValueError, match="缺失"):
        _check_figure_contract(_det_text([1, 2, 3, 4, 5]))
    with pytest.raises(ValueError, match="成对"):
        _check_figure_contract(_det_text([1, 2, 3, 4, 5, 8], paired=(6,)))


def test_det_contract_with_spatial_shifts_lead_figure():
    from xmetai_evaluation.visualization.det_report import _check_figure_contract

    _check_figure_contract(
        _det_text([1, 2, 3, 4, 5, 8, 9, 10, 11, 12], paired=(6, 7)),
        count=12, required=(1, 2, 3, 4, 5, 8, 9, 10, 11, 12), paired=(6, 7),
    )
    with pytest.raises(ValueError, match="缺失"):
        _check_figure_contract(
            _det_text([1, 2, 3, 4, 5, 8, 9, 10, 12], paired=(6, 7)),
            count=12, required=(1, 2, 3, 4, 5, 8, 9, 10, 11, 12), paired=(6, 7),
        )


def _ens_text(numbers):
    return "\n".join(f"图 {index}：x" for index in numbers)


def test_ens_contract_k0_matches_previous_numbers():
    from xmetai_evaluation.visualization.ens_report import _check_figure_contract

    _check_figure_contract(_ens_text(range(1, 10)))
    # 球谐带整块缺：8/9 一起没有是合法的
    _check_figure_contract(_ens_text(range(1, 8)))
    with pytest.raises(ValueError, match="成对"):
        _check_figure_contract(_ens_text(list(range(1, 8)) + [8]))


def test_ens_contract_with_spatial_shifts_spectrum_and_bands():
    from xmetai_evaluation.visualization.ens_report import _check_figure_contract

    numbers = list(range(1, 14))          # 1..13，k=4
    _check_figure_contract(
        _ens_text(numbers), count=13, required=11, paired=(12, 13)
    )
    _check_figure_contract(
        _ens_text(range(1, 12)), count=13, required=11, paired=(12, 13)
    )
    with pytest.raises(ValueError, match="成对"):
        _check_figure_contract(
            _ens_text(list(range(1, 12)) + [12]),
            count=13, required=11, paired=(12, 13),
        )


# ---------------------------------------------------------------- 集合取数


def _write_nc(path: Path, z, u, v):
    lat, lon = target_grid()[0], target_grid()[1]
    dataset = xr.Dataset(
        {
            "z500": (("lat", "lon"), np.full((lat.size, lon.size), float(z))),
            "u850": (("lat", "lon"), np.full((lat.size, lon.size), float(u))),
            "v850": (("lat", "lon"), np.zeros((lat.size, lon.size))),
        },
        coords={"lat": lat, "lon": lon},
    )
    dataset.to_netcdf(path)


def test_read_ensemble_lead_mean_and_spread(tmp_path):
    """逐成员月平均 → 成员均值与标准差（ddof=0），不是逐时次 std。"""
    entries = []
    for day_index, day in enumerate(("20250701", "20250702")):
        day_dir = tmp_path / day
        directories = {}
        for member, (z, u) in enumerate(((5500.0, 10.0), (5450.0, 6.0))):
            member_dir = day_dir / f"member_{member:03d}"
            member_dir.mkdir(parents=True)
            _write_nc(member_dir / "004.nc", z, u, 0.0)
            directories[member] = member_dir
        entries.append((day_dir, directories))

    lat_target, lon_target = target_grid()
    result, day_count, missing = read_ensemble_lead(
        entries, [0, 1], 24, lat_target, lon_target
    )

    assert missing == 0
    assert day_count == {0: 2, 1: 2}
    mean_z, spread_z, count_z = result["z500"]
    assert np.allclose(mean_z, 5475.0)
    assert np.allclose(spread_z, 25.0)
    assert int(count_z.min()) == 2
    mean_w, spread_w, _count = result["ws850"]
    assert np.allclose(mean_w, 8.0)
    assert np.allclose(spread_w, 2.0)
