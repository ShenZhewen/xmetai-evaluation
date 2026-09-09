"""
测试核心错误分类

验证错误类别能够正确携带上下文信息，符合 README 第 14.2 节要求。
"""

import pytest
from xmetai_evaluation.core.errors import (
    EvaluationError,
    ConfigError,
    ContractError,
    DiscoveryError,
    DecodeError,
    AlignmentError,
    MetricError,
    ResourceError,
    OutputError,
)


def test_evaluation_error_basic():
    """测试基础错误消息"""
    err = EvaluationError("Something went wrong")
    assert str(err) == "Something went wrong"


def test_evaluation_error_with_context():
    """测试错误携带完整上下文"""
    err = EvaluationError(
        "Failed to read forecast",
        source_id="fuxi_ens",
        variable="tp",
        sample_key={"init": "2025-01-06", "lead": 24},
        path="/data/forecasts/fuxi_ens/20250106/member_001/001.nc",
        step_id="read_forecast",
    )

    error_str = str(err)
    assert "Failed to read forecast" in error_str
    assert "source=fuxi_ens" in error_str
    assert "variable=tp" in error_str
    assert "sample=" in error_str
    assert "path=" in error_str
    assert "step=read_forecast" in error_str


def test_evaluation_error_with_cause():
    """测试错误链"""
    cause = ValueError("Invalid NetCDF format")
    err = DecodeError("Cannot decode file", path="/data/test.nc", cause=cause)

    assert err.__cause__ is cause
    assert isinstance(err, EvaluationError)
    assert isinstance(err, DecodeError)


def test_config_error():
    """测试配置错误"""
    err = ConfigError("Missing required field 'source_id'")
    assert isinstance(err, EvaluationError)
    assert "Missing required field" in str(err)


def test_contract_error():
    """测试数据契约错误"""
    err = ContractError(
        "Unit mismatch: expected 'mm', got 'kg m-2'",
        source_id="test_source",
        variable="tp",
    )
    assert isinstance(err, EvaluationError)
    assert "Unit mismatch" in str(err)
    assert "source=test_source" in str(err)


def test_discovery_error():
    """测试发现错误"""
    err = DiscoveryError(
        "Ambiguous match: found 2 files for same sample",
        source_id="test_source",
        sample_key={"init": "2025-01-06", "lead": 24},
    )
    assert isinstance(err, EvaluationError)
    assert "Ambiguous match" in str(err)


def test_all_error_types_inherit_base():
    """测试所有错误类型都正确继承"""
    error_types = [
        ConfigError,
        ContractError,
        DiscoveryError,
        DecodeError,
        AlignmentError,
        MetricError,
        ResourceError,
        OutputError,
    ]

    for error_type in error_types:
        err = error_type("Test message")
        assert isinstance(err, EvaluationError)
        assert isinstance(err, Exception)
