"""组件注册表：配置只按注册名查表构造组件，代码里不做类型分支。"""

import pytest

from xmetai_evaluation.components import available_components, register_builtin_components
from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.core.registry import ComponentType, get_registry


def _build(kind: ComponentType, name: str, **params):
    register_builtin_components()
    return get_registry().build(kind, name, **params)


def test_builtin_components_are_registered():
    register_builtin_components()
    names = available_components()

    assert {"fuxi", "station", "diamond_station", "fengqing", "cra"}.issubset(
        set(names["reader"])
    )
    assert {"grid_to_station", "time_window_accumulator", "ensemble_mean"}.issubset(
        set(names["transform"])
    )
    assert {"rmse", "bias", "acc", "ts_score"}.issubset(set(names["metric"]))
    assert {"csv_long", "json", "categorical_wide"}.issubset(set(names["writer"]))
    assert {"station_valid_time", "grid_valid_time"}.issubset(set(names["protocol"]))


def test_registration_is_idempotent():
    register_builtin_components()
    register_builtin_components()
    assert "rmse" in available_components()["metric"]


def test_build_reader_returns_source_handle(tmp_path):
    handle = _build(ComponentType.READER, "station", root_dir=str(tmp_path))

    assert handle.source_id == "diamond_station"
    assert handle.catalog is not None
    assert handle.option("root_dir") == str(tmp_path)
    assert callable(handle.catalog.file_time)


def test_build_transform_from_registry():
    accumulator = _build(
        ComponentType.TRANSFORM, "time_window_accumulator", window_hours=24
    )
    assert accumulator.window_hours == 24.0
    assert accumulator.time_dim == "lead_time"

    interpolator = _build(ComponentType.TRANSFORM, "grid_to_station", method="nearest")
    assert interpolator.method == "nearest"

    ensemble = _build(ComponentType.TRANSFORM, "ensemble_mean")
    assert hasattr(ensemble, "transform")


def test_build_metric_normalizes_thresholds():
    metric = _build(ComponentType.METRIC, "ts_score", thresholds=[0.1, 10.0])
    assert [name for name, _ in metric.thresholds] == ["≥0.1", "≥10"]
    assert metric.PRODUCT_KIND == "categorical"


def test_unknown_component_name_fails_loudly():
    with pytest.raises(ConfigError):
        _build(ComponentType.METRIC, "not_a_metric")
