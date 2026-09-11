"""声明层与唯一执行内核。"""

from datetime import datetime

import pytest

from xmetai_evaluation.configs.base import EvalConfig
from xmetai_evaluation.core.errors import DiscoveryError
from xmetai_evaluation.pipeline.pipelines import get_template, list_pipelines
from xmetai_evaluation.pipeline.runner import Runner
from xmetai_evaluation.pipeline.spec import (
    MetricSpec,
    PipelineSpec,
    PipelineTemplate,
    SourceSpec,
    TransformSpec,
)


def _station_config(tmp_path) -> EvalConfig:
    return EvalConfig(
        name="synthetic_station_ts",
        description="声明层测试",
        forecast_reader={"type": "fuxi", "root_dir": str(tmp_path / "forecast"), "variable": "tp"},
        observation_reader={
            "type": "station",
            "root_dir": str(tmp_path / "stations"),
            "variable": "precipitation",
        },
        pipeline="weather_ts_det",
        transform_options={"time_window_accumulator": {"window_hours": 12}},
        metric_options={"ts_score": {"thresholds": [0.1]}},
        start_date="20250101",
        end_date="20250131",
        output_dir=str(tmp_path / "out"),
        writers=["csv_long", "categorical_wide"],
    )


def test_spec_from_config_maps_every_part(tmp_path):
    spec = PipelineSpec.from_config(_station_config(tmp_path))

    assert spec.name == "synthetic_station_ts"
    assert spec.forecast.reader == "fuxi"
    assert spec.observation.reader == "station"
    assert spec.forecast.params["variable"] == "tp"
    assert spec.variables == {"forecast": "tp", "observation": "precipitation"}
    assert spec.protocol == "station_valid_time"
    assert [item.name for item in spec.transforms] == [
        "time_window_accumulator",
        "grid_to_station",
    ]
    assert spec.window_hours == 12
    assert spec.output_writers == ["csv_long", "categorical_wide"]
    assert spec.period() == (datetime(2025, 1, 1), datetime(2025, 1, 31))


def test_spec_window_hours_falls_back_to_option(tmp_path):
    spec = PipelineSpec.from_config(_station_config(tmp_path))
    spec.template.transforms = []
    assert spec.window_hours == 24
    spec.config_options["window_hours"] = 6
    assert spec.window_hours == 6


def test_builtin_pipelines_resolve_to_declarations():
    presets = list_pipelines()
    assert {"weather_ts_det", "weather_ts_ens", "weather_field_scores", "weather_ens_crps", "fdp_ens_crps", "fdp_field_scores"}.issubset(set(presets))

    station = get_template("weather_ts_det")
    assert station.protocol == "station_valid_time"
    assert [item.name for item in station.transforms] == [
        "time_window_accumulator",
        "grid_to_station",
    ]

    grid = get_template("fdp_ens_crps")
    assert grid.protocol == "grid_valid_time"
    assert [item.name for item in grid.transforms] == ["ensemble_mean"]
    assert [item.name for item in grid.metrics] == ["crps", "spread_error"]


def test_pipeline_template_composes_with_data_config(tmp_path):
    """--config 给数据、--pipeline 给算法：两者正交。"""
    spec = PipelineSpec.from_config(_station_config(tmp_path))
    spec.use_pipeline("weather_ts_ens")

    assert spec.pipeline == "weather_ts_ens"
    assert [item.name for item in spec.metrics] == ["ts_score", "ensemble_probability"]
    # 数据仍来自配置
    assert spec.forecast.reader == "fuxi"
    assert spec.forecast.params["root_dir"] == str(tmp_path / "forecast")
    assert spec.start_date == "20250101"


def test_unknown_pipeline_name_fails_loudly():
    from xmetai_evaluation.core.errors import ConfigError

    with pytest.raises(ConfigError):
        get_template("no_such_pipeline")


def test_config_without_pipeline_fails_loudly(tmp_path):
    """配置必须声明走哪套流程——没有第二种写法。"""
    from xmetai_evaluation.core.errors import ConfigError

    cfg = _station_config(tmp_path)
    cfg.pipeline = ""
    with pytest.raises(ConfigError, match="pipeline"):
        PipelineSpec.from_config(cfg)


def test_runner_propagates_discovery_failure(tmp_path):
    """没有预报文件时必须在准备阶段报错，而不是静默产出空结果。"""
    spec = PipelineSpec(
        name="empty",
        forecast=SourceSpec(
            "fuxi", {"root_dir": str(tmp_path / "forecast"), "variable": "tp"}
        ),
        observation=SourceSpec(
            "station",
            {"root_dir": str(tmp_path / "stations"), "variable": "precipitation"},
        ),
        template=PipelineTemplate(
            name="inline:test",
            protocol="station_valid_time",
            transforms=[TransformSpec("time_window_accumulator", {"window_hours": 24})],
            metrics=[MetricSpec("ts_score", {"thresholds": [0.1]})],
        ),
        start_date="20250101",
        output_dir=str(tmp_path / "out"),
    )
    with pytest.raises(DiscoveryError):
        Runner(spec).run()


def test_cli_lists_builtin_pipelines(capsys):
    from xmetai_evaluation.cli import main

    assert main(["--list-pipelines"]) == 0
    output = capsys.readouterr().out
    assert "weather_ts_det" in output
    assert "weather_ts_ens" in output
    assert "fdp_ens_crps" in output


def test_cli_requires_a_config_or_pipeline():
    from xmetai_evaluation.cli import main

    with pytest.raises(SystemExit):
        main([])
