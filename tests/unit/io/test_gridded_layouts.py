"""通用格点 Reader 的布局语义。

覆盖三套内置布局各自的"独特之处"：FuXi 的时效来自文件序号、
Fengqing 的单位换算与集合维、CRA 的单变量改名与按 valid_time 拼接。
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.core.variables import DataKind
from functools import partial

from xmetai_evaluation.io.gridded import GriddedCatalog, GriddedReader
from xmetai_evaluation.io.layouts import (
    GRAVITY,
    CRA_LAYOUT,
    FENGQING_LAYOUT,
    FUXI_ENS_LAYOUT,
    FUXI_LAYOUT,
)

# 数据源的差别只在布局声明（io/layouts.py），产品代码不再有"每模型一个 Reader 类"；
# 这里用 partial 保留原有的调用形态，测试关注点仍是"布局语义"。
FuXiCatalog = partial(GriddedCatalog, layout=FUXI_LAYOUT.with_step_hours(6.0))
FuXiReader = partial(GriddedReader, source_id="fuxi", layout=FUXI_LAYOUT.with_step_hours(6.0))
CRAReader = partial(GriddedReader, source_id="cra", layout=CRA_LAYOUT)
CRACatalog = partial(GriddedCatalog, layout=CRA_LAYOUT)
FengqingCatalog = partial(GriddedCatalog, layout=FENGQING_LAYOUT)


def _with_catalog(layout, root_dir, source_id):
    return GriddedReader(source_id=source_id, layout=layout), GriddedCatalog(
        Path(root_dir), layout
    )
from xmetai_evaluation.transforms.regrid import EnsembleMeanTransform

GRID_LATS = [-10.0, 0.0, 10.0]
GRID_LONS = [100.0, 105.0, 110.0, 115.0]


def _write_fengqing(path: Path, init: datetime, lead: int, group: str) -> None:
    name = (
        f"Fengqing_1.0_GLB_{group}_OP25_6HOR_ENS_FCST_"
        f"{init.strftime('%Y%m%d%H')}_{lead:03d}.nc"
    )
    if group == "PLEVELS":
        dataset = xr.Dataset(
            {"Z500": (["member", "level", "time", "dtime", "lat", "lon"],
                      np.full((2, 1, 1, 1, len(GRID_LATS), len(GRID_LONS)), 100.0 * GRAVITY))},
            coords={"member": [0, 1], "level": [500], "lat": GRID_LATS, "lon": GRID_LONS},
        )
    else:
        dataset = xr.Dataset(
            {"TP": (["member", "lat", "lon"],
                    np.full((2, len(GRID_LATS), len(GRID_LONS)), 2.5))},
            coords={"member": [0, 1], "lat": GRID_LATS, "lon": GRID_LONS},
        )
    dataset.to_netcdf(path / name)


def _write_fuxi(path: Path) -> None:
    for index in range(1, 5):
        dataset = xr.Dataset(
            {"TP": (["lat", "lon"], np.full((len(GRID_LATS), len(GRID_LONS)), 2.5),
                    {"units": "mm"})},
            coords={"lat": GRID_LATS, "lon": GRID_LONS},
        )
        dataset.to_netcdf(path / f"{index:03d}.nc")


def _fengqing_request(init, variables):
    return DataRequest(
        source_id="fengqing", variables=variables, init_times=[init], lead_times=[24]
    )


class TestFengqingLayout:
    def test_unit_conversion_and_member_dimension(self, tmp_path):
        init = datetime(2026, 8, 19, 0)
        init_dir = tmp_path / init.strftime("%Y%m%d")
        init_dir.mkdir()
        _write_fengqing(init_dir, init, 24, "PLEVELS")
        reader, catalog = _with_catalog(FENGQING_LAYOUT, tmp_path, "fengqing")

        request = _fengqing_request(init, ["z500"])
        index = catalog.discover(request)
        bundle = reader.read(request, index)

        assert set(bundle.payload.data_vars) == {"z500"}
        assert bundle.payload["z500"].dims == ("init_time", "lead_time", "member", "lat", "lon")
        assert bundle.payload.sizes["member"] == 2
        # m^2/s^2 -> m，且记录换算前单位
        np.testing.assert_allclose(bundle.payload["z500"].values, 100.0)
        assert bundle.payload["z500"].attrs["units"] == "m"
        assert bundle.payload["z500"].attrs["original_units"] == "m^2/s^2"
        assert bundle.semantic.units["z500"] == "m"
        assert bundle.semantic.member_count == 2
        assert bundle.kind == DataKind.GRIDDED_FORECAST

    def test_group_filter_reads_only_needed_files(self, tmp_path):
        init = datetime(2026, 8, 19, 0)
        init_dir = tmp_path / init.strftime("%Y%m%d")
        init_dir.mkdir()
        _write_fengqing(init_dir, init, 24, "PLEVELS")
        _write_fengqing(init_dir, init, 24, "SURFACE")

        reader, catalog = _with_catalog(FENGQING_LAYOUT, tmp_path, "fengqing")
        request = _fengqing_request(init, ["z500"])
        bundle = reader.read(request, catalog.discover(request))

        assert "PLEVELS" in bundle.provenance.input_files[0]
        assert len(bundle.provenance.input_files) == 1

    def test_multiple_variables_from_different_groups_are_merged(self, tmp_path):
        init = datetime(2026, 8, 19, 0)
        init_dir = tmp_path / init.strftime("%Y%m%d")
        init_dir.mkdir()
        _write_fengqing(init_dir, init, 24, "PLEVELS")
        _write_fengqing(init_dir, init, 24, "SURFACE")

        reader, catalog = _with_catalog(FENGQING_LAYOUT, tmp_path, "fengqing")
        request = _fengqing_request(init, ["z500", "tp"])
        bundle = reader.read(request, catalog.discover(request))

        assert set(bundle.payload.data_vars) == {"z500", "tp"}
        assert bundle.payload.sizes["lead_time"] == 1
        assert bundle.payload["z500"].dims == ("init_time", "lead_time", "member", "lat", "lon")

    def test_catalog_reports_file_groups(self, tmp_path):
        catalog = FengqingCatalog(root_dir=tmp_path)
        assert catalog._get_file_types_for_variables(["z500"]) == {"PLEVELS"}
        assert catalog._get_file_types_for_variables(["t2m"]) == {"SURFACE"}

    def test_pattern_does_not_match_other_models(self, tmp_path):
        catalog = FengqingCatalog(root_dir=tmp_path)
        assert catalog._parse_name(Path("NJU-Earth_v1_GLB_SURFACE_0P25_6HOR_FCST_2026081900_024.nc"),
                                   datetime(2026, 8, 19)) is None


class TestFuxiLayout:
    def test_lead_time_comes_from_file_order(self, tmp_path):
        init = datetime(2025, 1, 1, 0)
        init_dir = tmp_path / init.strftime("%Y%m%d")
        init_dir.mkdir()
        _write_fuxi(init_dir)

        request = DataRequest(source_id="fuxi", variables=["tp"], init_times=[init])
        catalog = FuXiCatalog(root_dir=tmp_path)
        index = catalog.discover(request)
        assert sorted(index.available[0]) == [init]
        assert len(index.available[0][init]) == 4

        reader = FuXiReader()
        bundle = reader.read_one(request, index, init)

        # 文件里是 TP，请求是 tp：大小写不敏感映射
        assert set(bundle.payload.data_vars) == {"tp"}
        assert bundle.payload["tp"].dims == ("init_time", "lead_time", "lat", "lon")
        np.testing.assert_allclose(bundle.payload["tp"].lead_time.values, [6, 12, 18, 24])
        assert bundle.payload["tp"].attrs["units"] == "mm"
        assert reader.step_hours == 6.0

    def test_lead_times_request_filters_records(self, tmp_path):
        init = datetime(2025, 1, 1, 0)
        init_dir = tmp_path / init.strftime("%Y%m%d")
        init_dir.mkdir()
        _write_fuxi(init_dir)

        request = DataRequest(
            source_id="fuxi", variables=["tp"], init_times=[init], lead_times=[12, 24]
        )
        index = FuXiCatalog(root_dir=tmp_path).discover(request)
        assert len(index.available[0][init]) == 2


def _write_member(directory: Path, value: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, 5):
        dataset = xr.Dataset(
            {
                "TP": (
                    ["lat", "lon"],
                    np.full((len(GRID_LATS), len(GRID_LONS)), value),
                    {"units": "mm"},
                )
            },
            coords={"lat": GRID_LATS, "lon": GRID_LONS},
        )
        dataset.to_netcdf(directory / f"{index:03d}.nc")


class TestFuXiEnsembleLayout:
    def test_members_form_a_member_dimension(self, tmp_path):
        init = datetime(2025, 1, 1, 0)
        day_dir = tmp_path / init.strftime("%Y%m%d")
        day_dir.mkdir()
        _write_member(day_dir / "member_01", 2.0)
        _write_member(day_dir / "member_02", 4.0)

        reader, catalog = _with_catalog(FUXI_ENS_LAYOUT, tmp_path, "fuxi_ens")
        request = DataRequest(source_id="fuxi_ens", variables=["tp"], init_times=[init])
        bundle = reader.read_one(request, catalog.discover(request), init)

        assert bundle.payload["tp"].dims == (
            "init_time",
            "lead_time",
            "member",
            "lat",
            "lon",
        )
        assert list(bundle.payload["member"].values) == [1, 2]
        np.testing.assert_allclose(bundle.payload["tp"].lead_time.values, [6, 12, 18, 24])
        np.testing.assert_allclose(bundle.payload["tp"].sel(member=1).values, 2.0)
        np.testing.assert_allclose(bundle.payload["tp"].sel(member=2).values, 4.0)
        assert bundle.semantic.member_count == 2
        assert bundle.source_id == "fuxi_ens"

    def test_ensemble_mean_reduces_the_member_dimension(self, tmp_path):
        init = datetime(2025, 1, 1, 0)
        day_dir = tmp_path / init.strftime("%Y%m%d")
        day_dir.mkdir()
        _write_member(day_dir / "member_01", 2.0)
        _write_member(day_dir / "member_02", 4.0)

        reader, catalog = _with_catalog(FUXI_ENS_LAYOUT, tmp_path, "fuxi_ens")
        request = DataRequest(source_id="fuxi_ens", variables=["tp"], init_times=[init])
        bundle = reader.read_one(request, catalog.discover(request), init)

        reduced = EnsembleMeanTransform().transform(bundle)

        assert "member" not in reduced.payload["tp"].dims
        np.testing.assert_allclose(reduced.payload["tp"].values, 3.0)
        assert reduced.semantic.member_count is None

    def test_lead_times_filter_limits_member_files(self, tmp_path):
        init = datetime(2025, 1, 1, 0)
        day_dir = tmp_path / init.strftime("%Y%m%d")
        day_dir.mkdir()
        _write_member(day_dir / "member_01", 2.0)
        _write_member(day_dir / "member_02", 4.0)

        request = DataRequest(
            source_id="fuxi_ens",
            variables=["tp"],
            init_times=[init],
            lead_times=[6, 24],
        )
        reader, catalog = _with_catalog(FUXI_ENS_LAYOUT, tmp_path, "fuxi_ens")
        index = catalog.discover(request)
        assert len(index.available[0]) == 4  # 2 个成员 × 2 个时效

        bundle = reader.read_one(request, index, init)
        np.testing.assert_allclose(bundle.payload["tp"].lead_time.values, [6, 24])
        assert reader.max_lead_hours(index) == 24.0
        assert len(reader.input_files(index)) == 4


class TestCRA_Layout:
    def test_parses_both_file_shapes(self, tmp_path):
        catalog = CRACatalog(root_dir=tmp_path)
        moment = datetime(2026, 8, 20, 0)

        assert catalog._parse_name(
            Path("ART_ATM_GLB_0P25_6HOR_ANAL_2026082000.grib2"), moment
        ) == (moment, None, "ATM")
        assert catalog._parse_name(
            Path("CRA40LAND_SURFACE_2026082000_GLB_0P25_HOUR_V1_0_0.grib"), moment
        ) == (moment, None, "SURFACE")
        assert catalog._parse_name(Path("random_file.nc"), moment) is None

    def test_index_is_keyed_by_valid_time_and_group(self, tmp_path):
        moment = datetime(2026, 8, 20, 0)
        date_dir = tmp_path / moment.strftime("%Y%m%d")
        date_dir.mkdir()
        (date_dir / "ART_ATM_GLB_0P25_6HOR_ANAL_2026082000.grib2").touch()
        (date_dir / "CRA40LAND_SURFACE_2026082000_GLB_0P25_HOUR_V1_0_0.grib").touch()

        catalog = CRACatalog(root_dir=tmp_path)
        request = DataRequest(source_id="cra", variables=["z500"], init_times=[moment])
        files = catalog.discover(request).available[0]
        assert (moment, "ATM") in files
        # z500 只属于 ATM 组，SURFACE 文件不应被纳入
        assert (moment, "SURFACE") not in files

    def test_single_variable_file_is_renamed_to_requested_name(self):
        reader = CRAReader()
        dataset = xr.Dataset(
            {"gh": (["latitude", "longitude"], np.full((2, 3), 42.0))},
            coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0, 120.0]},
        )

        normalized = reader._normalize(dataset, ["z500"])

        assert set(normalized.data_vars) == {"z500"}
        assert "lat" in normalized.coords and "lon" in normalized.coords
        assert normalized["z500"].dims == ("lat", "lon")
        np.testing.assert_allclose(normalized["z500"].values, 42.0)


def test_layouts_registry_is_complete():
    assert FUXI_LAYOUT.key_kind == "init"
    assert FENGQING_LAYOUT.key_kind == "init_lead_group"
    assert CRA_LAYOUT.key_kind == "time_group"
    assert CRA_LAYOUT.per_variable_open is True
    assert FENGQING_LAYOUT.per_variable_open is False
    assert FENGQING_LAYOUT.groups_for(["Z500"]) == {"PLEVELS"}
