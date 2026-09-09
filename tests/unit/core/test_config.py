"""
测试配置加载与校验

验证 ConfigLoader 符合 README 第 15 节要求：
- schema_version 检查
- 环境变量解析
- 相对路径解析
- 必填字段校验
"""

import pytest
import os
import tempfile
from pathlib import Path

from xmetai_evaluation.core.config import ConfigLoader, load_and_resolve_config
from xmetai_evaluation.core.errors import ConfigError


def test_config_loader_requires_schema_version():
    """测试配置必须有 schema_version"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("name: test\n")
        f.write("value: 123\n")
        config_path = Path(f.name)

    try:
        loader = ConfigLoader()
        with pytest.raises(ConfigError, match="schema_version"):
            loader.load(config_path)
    finally:
        config_path.unlink()


def test_config_loader_unsupported_version():
    """测试不支持的 schema_version"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("schema_version: 999\n")
        f.write("name: test\n")
        config_path = Path(f.name)

    try:
        loader = ConfigLoader()
        with pytest.raises(ConfigError, match="Unsupported schema_version"):
            loader.load(config_path)
    finally:
        config_path.unlink()


def test_config_loader_valid():
    """测试正常加载"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("schema_version: 1\n")
        f.write("id: test_config\n")
        f.write("value: 456\n")
        config_path = Path(f.name)

    try:
        loader = ConfigLoader()
        config = loader.load(config_path)

        assert config["schema_version"] == 1
        assert config["id"] == "test_config"
        assert config["value"] == 456
        assert "_config_file" in config
    finally:
        config_path.unlink()


def test_config_loader_invalid_yaml():
    """测试无效 YAML"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: syntax:\n")
        config_path = Path(f.name)

    try:
        loader = ConfigLoader()
        with pytest.raises(ConfigError, match="Invalid YAML"):
            loader.load(config_path)
    finally:
        config_path.unlink()


def test_config_loader_file_not_found():
    """测试文件不存在"""
    loader = ConfigLoader()
    with pytest.raises(ConfigError, match="not found"):
        loader.load(Path("/nonexistent/config.yaml"))


def test_resolve_env_vars():
    """测试环境变量解析"""
    # 设置测试环境变量
    os.environ["TEST_DATA_ROOT"] = "/data/test"
    os.environ["TEST_MODEL"] = "fuxi_ens"

    config = {
        "schema_version": 1,
        "path": {"root": "${TEST_DATA_ROOT}/forecasts/${TEST_MODEL}"},
        "nested": {"value": "${TEST_MODEL}"},
    }

    try:
        loader = ConfigLoader()
        resolved = loader.resolve_env_vars(config)

        assert resolved["path"]["root"] == "/data/test/forecasts/fuxi_ens"
        assert resolved["nested"]["value"] == "fuxi_ens"
    finally:
        del os.environ["TEST_DATA_ROOT"]
        del os.environ["TEST_MODEL"]


def test_resolve_env_vars_missing():
    """测试引用不存在的环境变量"""
    config = {
        "schema_version": 1,
        "path": "${NONEXISTENT_VAR}/data",
    }

    loader = ConfigLoader()
    with pytest.raises(ConfigError, match="Environment variable.*not found"):
        loader.resolve_env_vars(config)


def test_resolve_paths():
    """测试相对路径解析"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 创建配置文件
        config_file = tmpdir_path / "config.yaml"
        config_file.write_text(
            "schema_version: 1\n"
            "path:\n"
            "  root: ./data\n"
            "  file: ../output/result.nc\n"
        )

        loader = ConfigLoader()
        config = loader.load(config_file)
        resolved = loader.resolve_paths(config)

        # 相对路径应该被解析为绝对路径
        assert Path(resolved["path"]["root"]).is_absolute()
        assert Path(resolved["path"]["file"]).is_absolute()

        # 验证解析结果
        expected_root = (tmpdir_path / "data").resolve()
        expected_file = (tmpdir_path.parent / "output" / "result.nc").resolve()
        assert Path(resolved["path"]["root"]) == expected_root
        assert Path(resolved["path"]["file"]) == expected_file


def test_validate_required_fields():
    """测试必填字段校验"""
    config = {"schema_version": 1, "id": "test"}

    loader = ConfigLoader()

    # 缺少必填字段应该失败
    with pytest.raises(ConfigError, match="Missing required fields"):
        loader.validate_required_fields(config, ["id", "source_id", "variables"])

    # 所有必填字段存在应该通过
    config["source_id"] = "test_source"
    config["variables"] = ["tp"]
    loader.validate_required_fields(config, ["id", "source_id", "variables"])


def test_merge_with_defaults():
    """测试合并默认配置"""
    defaults = {
        "execution": {"timeout": 3600, "retry": 3},
        "output": {"format": "csv"},
    }

    config = {
        "execution": {"timeout": 7200},  # 覆盖默认值
        "output": {"format": "json", "compress": True},  # 部分覆盖 + 新增
    }

    loader = ConfigLoader()
    merged = loader.merge_with_defaults(config, defaults)

    assert merged["execution"]["timeout"] == 7200  # 用户值
    assert merged["execution"]["retry"] == 3  # 默认值保留
    assert merged["output"]["format"] == "json"  # 用户值
    assert merged["output"]["compress"] is True  # 用户新增


def test_load_and_resolve_config():
    """测试完整加载流程"""
    os.environ["TEST_ROOT"] = "/data"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        config_file = tmpdir_path / "test_config.yaml"

        config_file.write_text(
            "schema_version: 1\n"
            "id: integration_test\n"
            "path:\n"
            "  root: ${TEST_ROOT}/forecasts\n"
            "  relative: ./models\n"
        )

        try:
            config = load_and_resolve_config(config_file)

            # 验证 schema_version
            assert config["schema_version"] == 1

            # 验证环境变量解析（跨平台）
            assert "forecasts" in config["path"]["root"]
            assert config["path"]["root"].endswith("forecasts")

            # 验证相对路径解析
            assert Path(config["path"]["relative"]).is_absolute()
        finally:
            del os.environ["TEST_ROOT"]
