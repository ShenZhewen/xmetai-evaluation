"""
测试组件注册机制

验证 Registry 符合 README 第 13.5 节要求：
- 注册时发现重复名称必须失败
- 未注册名称必须在 validate 阶段失败
- 配置只能引用注册名
"""

import pytest
from xmetai_evaluation.core.registry import (
    Registry,
    ComponentType,
    register_reader,
    register_metric,
    get_registry,
)
from xmetai_evaluation.core.errors import ConfigError


def dummy_factory():
    """测试用工厂函数"""
    return "dummy_component"


def test_registry_register_basic():
    """测试基础注册"""
    registry = Registry()
    registry.register(
        name="netcdf",
        component_type=ComponentType.READER,
        version="1.0.0",
        factory=dummy_factory,
        description="NetCDF reader",
    )

    descriptor = registry.get("reader.netcdf")
    assert descriptor.name == "reader.netcdf"
    assert descriptor.version == "1.0.0"
    assert descriptor.factory is dummy_factory


def test_registry_duplicate_registration_fails():
    """测试重复注册必须失败"""
    registry = Registry()
    registry.register(
        name="test_component",
        component_type=ComponentType.READER,
        version="1.0.0",
        factory=dummy_factory,
    )

    # 再次注册同名组件应该失败
    with pytest.raises(ConfigError, match="already registered"):
        registry.register(
            name="test_component",
            component_type=ComponentType.READER,
            version="2.0.0",
            factory=dummy_factory,
        )


def test_registry_get_unregistered_fails():
    """测试获取未注册组件必须失败"""
    registry = Registry()

    with pytest.raises(ConfigError, match="not registered"):
        registry.get("reader.nonexistent")


def test_registry_get_with_type():
    """测试通过类型获取组件"""
    registry = Registry()
    registry.register("rmse", ComponentType.METRIC, "1.0.0", dummy_factory)

    # 可以用完整名称或简短名称+类型
    descriptor1 = registry.get("metric.rmse")
    descriptor2 = registry.get("rmse", ComponentType.METRIC)

    assert descriptor1.name == descriptor2.name
    assert descriptor1.name == "metric.rmse"


def test_registry_list_components():
    """测试列出组件"""
    registry = Registry()
    registry.register("netcdf", ComponentType.READER, "1.0.0", dummy_factory)
    registry.register("grib", ComponentType.READER, "1.0.0", dummy_factory)
    registry.register("rmse", ComponentType.METRIC, "1.0.0", dummy_factory)

    # 列出所有组件
    all_components = registry.list_components()
    assert len(all_components) == 3

    # 只列出 reader
    readers = registry.list_components(ComponentType.READER)
    assert len(readers) == 2
    assert all(c.component_type == ComponentType.READER for c in readers)

    # 只列出 metric
    metrics = registry.list_components(ComponentType.METRIC)
    assert len(metrics) == 1
    assert metrics[0].name == "metric.rmse"


def test_registry_validate_reference():
    """测试引用验证"""
    registry = Registry()
    registry.register("netcdf", ComponentType.READER, "1.0.0", dummy_factory)

    # 正确引用不应抛错
    registry.validate_reference("netcdf", ComponentType.READER)

    # 引用不存在的组件应该失败
    with pytest.raises(ConfigError, match="not registered"):
        registry.validate_reference("nonexistent", ComponentType.READER)


def test_registry_with_capabilities():
    """测试带能力描述的注册"""
    registry = Registry()
    capabilities = {
        "supported_formats": ["nc", "nc4"],
        "supported_dims": ["lat", "lon", "time"],
    }

    registry.register(
        "netcdf",
        ComponentType.READER,
        "1.0.0",
        dummy_factory,
        capabilities=capabilities,
    )

    descriptor = registry.get("reader.netcdf")
    assert descriptor.capabilities == capabilities
    assert "nc4" in descriptor.capabilities["supported_formats"]


def test_registry_with_dependencies():
    """测试带依赖提示的注册"""
    registry = Registry()
    dependencies = ["cfgrib>=0.9.10", "eccodes>=1.4"]

    registry.register(
        "grib",
        ComponentType.READER,
        "1.0.0",
        dummy_factory,
        dependencies=dependencies,
    )

    descriptor = registry.get("reader.grib")
    assert descriptor.dependencies == dependencies


def test_convenience_register_functions():
    """测试便捷注册函数"""
    # 清空全局注册表（仅用于测试）
    registry = get_registry()

    # 注册 reader
    register_reader("test_reader", "1.0.0", dummy_factory, description="Test reader")

    # 注册 metric
    register_metric("test_metric", "1.0.0", dummy_factory, description="Test metric")

    # 验证已注册
    reader_desc = registry.get("reader.test_reader")
    assert reader_desc.description == "Test reader"

    metric_desc = registry.get("metric.test_metric")
    assert metric_desc.description == "Test metric"
