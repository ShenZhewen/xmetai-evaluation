"""
FDP z500 连续场评测集成测试

测试 Fengqing + CRA 的完整评测流程。
"""

import pytest
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr

from functools import partial

from xmetai_evaluation.io.gridded import GriddedCatalog
from xmetai_evaluation.io.layouts import CRA_LAYOUT, FENGQING_LAYOUT

# 数据源 = 布局声明 + 通用 Reader（见 io/layouts.py）
CRACatalog = partial(GriddedCatalog, layout=CRA_LAYOUT)
FengqingCatalog = partial(GriddedCatalog, layout=FENGQING_LAYOUT)
from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.pipeline.matcher import Matcher
from xmetai_evaluation.metrics.rmse import RMSE
from xmetai_evaluation.metrics.bias import Bias
from xmetai_evaluation.metrics.acc import ACC


class TestFengqingReader:
    """测试 Fengqing Reader"""

    def test_catalog_discovery(self, tmp_path):
        """测试文件发现"""
        # 创建测试目录结构
        init_dir = tmp_path / "20260819"
        init_dir.mkdir()

        # 创建测试文件（不需要真实数据，只测试发现逻辑）
        test_file = init_dir / "Fengqing_1.0_GLB_PLEVELS_OP25_6HOR_ENS_FCST_2026081900_024.nc"
        test_file.touch()

        # 测试 Catalog
        catalog = FengqingCatalog(root_dir=tmp_path)
        request = DataRequest(
            source_id="fengqing",
            variables=["z500"],
            init_times=[datetime(2026, 8, 19, 0)],
            lead_times=[24],
        )

        index = catalog.discover(request)
        assert len(index.available) > 0
        files_by_key = index.available[0]
        assert (datetime(2026, 8, 19, 0), 24, "PLEVELS") in files_by_key

    def test_variable_file_type_mapping(self):
        """测试变量到文件类型的映射"""
        catalog = FengqingCatalog(root_dir=Path("/tmp"))

        # z500 需要 PLEVELS
        file_types = catalog._get_file_types_for_variables(["z500"])
        assert "PLEVELS" in file_types
        assert "SURFACE" not in file_types

        # t2m 需要 SURFACE
        file_types = catalog._get_file_types_for_variables(["t2m"])
        assert "SURFACE" in file_types
        assert "PLEVELS" not in file_types


class TestCRAReader:
    """测试 CRA Reader"""

    def test_catalog_discovery(self, tmp_path):
        """测试文件发现"""
        # 创建测试目录结构
        date_dir = tmp_path / "20260820"
        date_dir.mkdir()

        # 创建测试文件
        atm_file = date_dir / "ART_ATM_GLB_0P25_6HOR_ANAL_2026082000.grib2"
        atm_file.touch()

        # 测试 Catalog
        catalog = CRACatalog(root_dir=tmp_path)
        request = DataRequest(
            source_id="cra",
            variables=["z500"],
            init_times=[datetime(2026, 8, 20, 0)],
        )

        index = catalog.discover(request)
        assert len(index.available) > 0
        files_by_key = index.available[0]
        assert (datetime(2026, 8, 20, 0), "ATM") in files_by_key


class TestMatcher:
    """测试 Matcher"""

    def test_valid_time_computation(self):
        """测试 valid_time 计算"""
        matcher = Matcher(ensemble_reduction="mean")

        # 创建模拟数据
        forecast_ds = xr.Dataset(
            {
                "z500": (
                    ["init_time", "lead_time", "lat", "lon"],
                    np.random.rand(1, 2, 10, 10),
                )
            },
            coords={
                "init_time": [datetime(2026, 8, 19, 0)],
                "lead_time": [24, 48],
                "lat": np.linspace(-90, 90, 10),
                "lon": np.linspace(0, 359, 10),
            },
        )

        # 添加 valid_time
        forecast_with_vt = matcher._add_valid_time(forecast_ds)

        assert "valid_time" in forecast_with_vt.dims
        assert len(forecast_with_vt.coords["valid_time"]) == 2


class TestMetrics:
    """测试指标"""

    def test_rmse_accumulate_merge_finalize(self):
        """测试 RMSE 的 accumulate/merge/finalize"""
        from xmetai_evaluation.core.contracts import EvaluationBatch

        rmse = RMSE()

        # 创建模拟 batch
        forecast = xr.DataArray(
            np.array([1.0, 2.0, 3.0, 4.0]),
            dims=["sample"],
        )
        observation = xr.DataArray(
            np.array([1.1, 2.1, 2.9, 4.2]),
            dims=["sample"],
        )
        valid_mask = xr.DataArray(
            np.array([True, True, True, True]),
            dims=["sample"],
        )

        batch = EvaluationBatch(
            forecast=forecast,
            observation=observation,
            sample_keys=[{"sample": i} for i in range(4)],
            valid_mask=valid_mask,
            alignment={"test": "test"},
        )

        # Accumulate
        state = rmse.accumulate(batch)
        assert state.data["n_valid"] == 4

        # Finalize
        result = rmse.finalize(state)
        assert result.status.value == "success"
        assert result.value > 0

    def test_bias_positive_negative(self):
        """测试 Bias 正负值"""
        from xmetai_evaluation.core.contracts import EvaluationBatch

        bias = Bias()

        # 预报偏高
        forecast = xr.DataArray(np.array([2.0, 3.0]), dims=["sample"])
        observation = xr.DataArray(np.array([1.0, 2.0]), dims=["sample"])
        valid_mask = xr.DataArray(np.array([True, True]), dims=["sample"])

        batch = EvaluationBatch(
            forecast=forecast,
            observation=observation,
            sample_keys=[{"sample": i} for i in range(2)],
            valid_mask=valid_mask,
            alignment={"test": "test"},
        )

        state = bias.accumulate(batch)
        result = bias.finalize(state)
        assert result.value > 0  # 预报偏高

        # 预报偏低
        forecast = xr.DataArray(np.array([1.0, 2.0]), dims=["sample"])
        observation = xr.DataArray(np.array([2.0, 3.0]), dims=["sample"])

        batch = EvaluationBatch(
            forecast=forecast,
            observation=observation,
            sample_keys=[{"sample": i} for i in range(2)],
            valid_mask=valid_mask,
            alignment={"test": "test"},
        )

        state = bias.accumulate(batch)
        result = bias.finalize(state)
        assert result.value < 0  # 预报偏低

    def test_acc_without_climatology(self):
        """测试 ACC 在没有气候态时的行为"""
        from xmetai_evaluation.core.contracts import EvaluationBatch

        acc = ACC()

        forecast = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["sample"])
        observation = xr.DataArray(np.array([1.1, 2.1, 2.9]), dims=["sample"])
        valid_mask = xr.DataArray(np.array([True, True, True]), dims=["sample"])

        batch = EvaluationBatch(
            forecast=forecast,
            observation=observation,
            sample_keys=[{"sample": i} for i in range(3)],
            valid_mask=valid_mask,
            reference=None,  # 没有气候态
            alignment={"test": "test"},
        )

        state = acc.accumulate(batch)
        result = acc.finalize(state)

        # 应该有警告
        assert len(result.warnings) > 0
        assert any("climatology" in w.lower() for w in result.warnings)


class TestEndToEnd:
    """端到端测试（需要真实数据或模拟完整数据）"""

    @pytest.mark.skip(reason="Requires real data files")
    def test_full_evaluation_pipeline(self):
        """完整评测流程：流程模板（怎么算）+ 数据源（算什么）。"""
        from xmetai_evaluation.pipeline.runner import Runner
        from xmetai_evaluation.pipeline.spec import PipelineSpec, SourceSpec

        spec = PipelineSpec(
            name="test_z500",
            forecast=SourceSpec(
                "fengqing",
                {
                    "root_dir": "test_data/fengqing",
                    "variables": ["z500"],
                    "init_times": ["2026-08-19T00:00:00"],
                    "lead_times": [24],
                },
            ),
            observation=SourceSpec(
                "cra", {"root_dir": "test_data/cra", "variables": ["z500"]}
            ),
        )
        spec.use_pipeline("fdp_field_scores")

        result_bundle = Runner(spec).run()

        assert len(result_bundle.results) > 0
        assert result_bundle.run_id == "test_z500"
