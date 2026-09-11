"""
测试核心数据契约对象

验证 DataRequest、DataIndex、DataBundle、EvaluationBatch、MetricResult
符合 README 第 13.1 节的最小字段要求和基础校验规则。
"""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime

from xmetai_evaluation.core.contracts import (
    DataRequest,
    DataIndex,
    DataBundle,
    EvaluationBatch,
    MetricResult,
    ResultBundle,
    ResultStatus,
    SemanticMetadata,
    Provenance,
)
from xmetai_evaluation.core.variables import DataKind, TemporalKind


def test_data_request_minimal():
    """测试最小 DataRequest"""
    req = DataRequest(
        source_id="test_source",
        variables=["tp", "t2m"],
    )
    assert req.source_id == "test_source"
    assert req.variables == ["tp", "t2m"]
    assert req.init_times is None
    assert req.lead_times is None


def test_data_request_validation():
    """测试 DataRequest 必填字段校验"""
    with pytest.raises(ValueError, match="source_id"):
        DataRequest(source_id="", variables=["tp"])

    with pytest.raises(ValueError, match="variables"):
        DataRequest(source_id="test", variables=[])


def test_data_request_full():
    """测试完整 DataRequest"""
    req = DataRequest(
        source_id="fuxi_ens",
        variables=["tp"],
        init_times=[datetime(2025, 1, 6, 0)],
        lead_times=[24, 48],
        members=[1, 2, 3],
        levels=None,
        region={"lat_min": 20, "lat_max": 50, "lon_min": 100, "lon_max": 140},
    )
    assert len(req.init_times) == 1
    assert req.lead_times == [24, 48]
    assert req.members == [1, 2, 3]


def test_data_index_available():
    """测试可用数据索引"""
    idx = DataIndex(
        source_id="fuxi_ens",
        available=["/data/forecasts/fuxi_ens/20250106/member_001/001.nc"],
        ambiguous=[],
    )
    assert len(idx.available) == 1
    assert idx.source_id == "fuxi_ens"


def test_data_index_ambiguous():
    """测试歧义状态索引"""
    idx = DataIndex(
        source_id="test_source",
        available=[],
        ambiguous=["/path/a.nc", "/path/b.nc"],  # 有歧义的文件
    )
    assert len(idx.ambiguous) == 2


def test_data_bundle_minimal():
    """测试最小 DataBundle"""
    # 创建合成 xarray 数据
    data = xr.DataArray(
        np.random.randn(10, 20),
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(90, -90, 10),
            "lon": np.linspace(0, 360, 20),
        },
    )

    semantic = SemanticMetadata(
        units={"tp": "mm"},
        temporal_kind=TemporalKind.INTERVAL_ACCUMULATION,
    )

    provenance = Provenance(
        input_files=["/data/test.nc"],
        reader_id="test_reader",
        reader_version="0.1.0",
    )

    bundle = DataBundle(
        payload=data,
        kind=DataKind.GRIDDED_FORECAST,
        source_id="test_source",
        standard_vars={"TP": "tp"},
        semantic=semantic,
        provenance=provenance,
    )

    assert isinstance(bundle.payload, xr.DataArray)
    assert bundle.kind == DataKind.GRIDDED_FORECAST
    assert "tp" in bundle.semantic.units


def test_data_bundle_requires_xarray():
    """测试 DataBundle 必须使用 xarray"""
    with pytest.raises(TypeError, match="xarray"):
        DataBundle(
            payload=np.array([1, 2, 3]),  # 裸数组不允许
            kind=DataKind.GRIDDED_FORECAST,
            source_id="test",
            standard_vars={},
            semantic=SemanticMetadata(units={}, temporal_kind=TemporalKind.INSTANTANEOUS),
            provenance=Provenance([], "test", "0.1.0"),
        )


def test_evaluation_batch_requires_alignment():
    """测试 EvaluationBatch 必须有 alignment 记录"""
    forecast = xr.DataArray(np.random.randn(10, 20), dims=["lat", "lon"])
    observation = xr.DataArray(np.random.randn(10, 20), dims=["lat", "lon"])
    valid_mask = xr.DataArray(np.ones((10, 20), dtype=bool), dims=["lat", "lon"])

    # 缺少 alignment 应该失败
    with pytest.raises(ValueError, match="alignment"):
        EvaluationBatch(
            forecast=forecast,
            observation=observation,
            sample_keys=[{"init": "2025-01-06", "lead": 24}],
            valid_mask=valid_mask,
            alignment=None,  # 明确缺失
        )


def test_evaluation_batch_valid():
    """测试有效的 EvaluationBatch"""
    forecast = xr.DataArray(np.random.randn(10, 20), dims=["lat", "lon"])
    observation = xr.DataArray(np.random.randn(10, 20), dims=["lat", "lon"])
    valid_mask = xr.DataArray(np.ones((10, 20), dtype=bool), dims=["lat", "lon"])

    batch = EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"init": "2025-01-06", "lead": 24}],
        valid_mask=valid_mask,
        alignment={
            "time_aligned": True,
            "spatial_method": "bilinear",
            "unit_converted": {"tp": "mm -> mm"},
        },
    )

    assert batch.alignment is not None
    assert batch.alignment["time_aligned"] is True


def test_metric_result_minimal():
    """测试最小 MetricResult"""
    result = MetricResult(
        metric_name="rmse",
        metric_version="1.0.0",
        value=2.5,
        status=ResultStatus.SUCCESS,
        n_requested=100,
        n_valid=95,
    )

    assert result.metric_name == "rmse"
    assert result.value == 2.5
    assert result.status == ResultStatus.SUCCESS
    assert result.n_valid == 95


def test_metric_result_validation():
    """测试 MetricResult 校验规则"""
    # n_valid 不能超过 n_requested
    with pytest.raises(ValueError, match="cannot exceed"):
        MetricResult(
            metric_name="test",
            metric_version="1.0",
            value=1.0,
            status=ResultStatus.SUCCESS,
            n_requested=50,
            n_valid=100,  # 错误：超过请求数
        )

    # 负数检查
    with pytest.raises(ValueError, match="non-negative"):
        MetricResult(
            metric_name="test",
            metric_version="1.0",
            value=1.0,
            status=ResultStatus.SUCCESS,
            n_requested=-1,
            n_valid=0,
        )


def test_metric_result_with_partial_status():
    """测试部分完成状态"""
    result = MetricResult(
        metric_name="crps",
        metric_version="1.0.0",
        value=1.5,
        status=ResultStatus.PARTIAL,
        n_requested=100,
        n_valid=80,
        warnings=["20 samples had missing members"],
    )

    assert result.status == ResultStatus.PARTIAL
    assert len(result.warnings) == 1


def test_metric_result_status_not_value():
    """测试状态和值分离：按 README 第 16.2 节，value=NaN 不是状态协议"""
    # 可以有 NaN 值，但必须有明确状态
    result = MetricResult(
        metric_name="acc",
        metric_version="1.0.0",
        value=float("nan"),
        status=ResultStatus.UNDEFINED,  # 明确状态：数学未定义（零方差）
        n_requested=100,
        n_valid=100,
        warnings=["Zero variance in observation"],
    )

    assert result.status == ResultStatus.UNDEFINED
    assert np.isnan(result.value)
    assert "Zero variance" in result.warnings[0]


def test_result_bundle():
    """测试结果集合"""
    results = [
        MetricResult(
            metric_name="rmse",
            metric_version="1.0",
            value=2.5,
            status=ResultStatus.SUCCESS,
            n_requested=100,
            n_valid=100,
        ),
        MetricResult(
            metric_name="mae",
            metric_version="1.0",
            value=1.8,
            status=ResultStatus.SUCCESS,
            n_requested=100,
            n_valid=100,
        ),
    ]

    bundle = ResultBundle(
        run_id="test_run_001",
        results=results,
        manifest={"status": "completed", "start_time": "2025-01-06T00:00:00Z"},
        resolved_config={"source": "test"},
    )

    assert bundle.run_id == "test_run_001"
    assert len(bundle.results) == 2
