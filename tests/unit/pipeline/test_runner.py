"""声明层与唯一执行内核。"""

from datetime import datetime

import pytest

from xmetai_evaluation.configs.base import EvalConfig
from xmetai_evaluation.core.errors import DiscoveryError
from xmetai_evaluation.pipeline.pipelines import get_template, list_pipelines
from xmetai_evaluation.pipeline.runner import Runner, _metric_runs
from xmetai_evaluation.pipeline.spec import (
    MetricSpec,
    PipelineSpec,
    PipelineTemplate,
    SourceSpec,
    TransformSpec,
    specs_from_config,
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
    assert {
        "weather_ts_det", "weather_ts_ens", "weather_ts_ens_prob",
        "weather_field_scores", "weather_ens_crps", "weather_ens_field_scores",
        "fdp_ens_crps", "fdp_field_scores",
    }.issubset(set(presets))

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
    assert [item.name for item in spec.metrics] == ["ts_score"]
    # 数据仍来自配置
    assert spec.forecast.reader == "fuxi"
    assert spec.forecast.params["root_dir"] == str(tmp_path / "forecast")
    assert spec.start_date == "20250101"


def test_ens_ts_and_probability_are_two_templates(tmp_path):
    """TS 走集合平均（24h），概率评分走原始成员（6h）——窗口不同，只能分两条。"""
    cfg = _station_config(tmp_path)
    # transform_options 是各段共用的，带上它会把两段的窗口覆盖成同一个，
    # 这里要看模板自己的窗口口径，所以清掉
    cfg.transform_options = {}
    spec = PipelineSpec.from_config(cfg)
    spec.use_pipeline("weather_ts_ens_prob")

    assert [item.name for item in spec.transforms] == [
        "time_window_accumulator",
        "grid_to_station",
    ]
    assert [item.name for item in spec.metrics] == ["ensemble_probability"]
    assert spec.window_hours == 6

    spec.use_pipeline("weather_ts_ens")
    assert [item.name for item in spec.transforms][0] == "ensemble_mean"
    assert spec.window_hours == 24


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


# --------------------------------------------------------------------------
# 一份配置跑多段（pipeline 写成列表）
# --------------------------------------------------------------------------


def _two_segment_config(tmp_path) -> EvalConfig:
    cfg = _station_config(tmp_path)
    cfg.pipeline = ["weather_ts_ens", "weather_ts_ens_prob"]
    # 各段共用的 transform_options 会把两段的窗口都改掉，这里要看模板自己的口径
    cfg.transform_options = {}
    return cfg


def test_single_pipeline_name_yields_one_spec(tmp_path):
    specs = specs_from_config(_station_config(tmp_path))

    assert [spec.pipeline for spec in specs] == ["weather_ts_det"]


def test_pipeline_list_yields_one_spec_per_segment(tmp_path):
    specs = specs_from_config(_two_segment_config(tmp_path))

    assert [spec.pipeline for spec in specs] == ["weather_ts_ens", "weather_ts_ens_prob"]
    # 两段各带自己的窗口，共用同一份数据与同一个输出目录
    assert [spec.window_hours for spec in specs] == [24, 6]
    assert {spec.output_dir for spec in specs} == {str(tmp_path / "out")}
    # spec 本身仍是单段声明，asdict(resolved_config) 才不会炸
    assert all(isinstance(spec.pipeline, str) for spec in specs)


def test_empty_pipeline_list_fails_loudly(tmp_path):
    from xmetai_evaluation.core.errors import ConfigError

    cfg = _station_config(tmp_path)
    cfg.pipeline = []
    with pytest.raises(ConfigError, match="pipeline"):
        specs_from_config(cfg)


def test_metrics_declare_whether_they_need_a_reference():
    """参考源只在有指标要用时才建——不然 24h 的 TS 段会白查一遍气候概率。"""
    from xmetai_evaluation.components import register_builtin_components
    from xmetai_evaluation.core.registry import ComponentType, get_registry
    from xmetai_evaluation.pipeline.runner import _needs_reference

    register_builtin_components()
    registry = get_registry()

    def build(name, **params):
        return registry.build(ComponentType.METRIC, name, **params)

    probability = build("ensemble_probability", thresholds=[(0.1, 0.1)])
    ts_score = build("ts_score", thresholds=[0.1])
    acc = build("acc")
    activity = build("activity")

    assert _needs_reference([probability])
    assert _needs_reference([acc])
    assert _needs_reference([activity])
    assert not _needs_reference([ts_score])
    assert _needs_reference([ts_score, acc])
    # 没有 needs_reference() 的自定义指标按"不需要"处理
    assert not _needs_reference([object()])


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


# --------------------------------------------------------------------------
# 按变量路由指标（对标 xu 的 --var-metrics z500:rmse,acc）
# --------------------------------------------------------------------------


class _FakeMetric:
    """只占位的指标对象：_metric_runs 只看声明，不看实现。"""

    def __init__(self, name):
        self.name = name


def test_metric_runs_expands_variable_routing():
    specs = [
        MetricSpec("rmse", {"variables": ["z500", "t2m"]}),
        MetricSpec("acc", {"variables": ["z500"]}),
    ]
    metrics = [_FakeMetric("rmse"), _FakeMetric("acc")]

    runs = _metric_runs(specs, metrics)

    assert [(run.label, run.variable) for run in runs] == [
        ("rmse[z500]", "z500"),
        ("rmse[t2m]", "t2m"),
        ("acc[z500]", "z500"),
    ]
    assert runs[0].metric is metrics[0]
    assert runs[2].metric is metrics[1]


def test_metric_runs_without_routing_is_identity():
    """老配置没有 variables：指标仍只跑一次、评整批——行为完全不变。"""
    runs = _metric_runs([MetricSpec("bias")], [_FakeMetric("bias")])

    assert len(runs) == 1
    assert runs[0].variable is None
    assert runs[0].label == "bias"


def test_metric_runs_with_empty_routing_is_identity():
    runs = _metric_runs(
        [MetricSpec("bias", {"variables": []})], [_FakeMetric("bias")]
    )

    assert len(runs) == 1
    assert runs[0].variable is None


def test_single_variable_metric_uses_the_only_variable():
    import xarray as xr

    from xmetai_evaluation.metrics.base import single_variable

    dataset = xr.Dataset({"z500": (["lat"], [1.0, 2.0])}, coords={"lat": [10.0, 20.0]})

    assert single_variable(dataset, "forecast").name == "z500"


def test_multivariable_batch_fails_loudly_instead_of_scoring_the_first_one():
    """回归：以前是静默只算 data_vars 里的第一个变量，现在是显式报错。

    这也正是"路由必须写全"的原因——漏写一个指标的 variables，它就会拿到
    整批多变量数据，在这里炸掉而不是悄悄出一个假数。
    """
    import xarray as xr

    from xmetai_evaluation.core.errors import MetricError
    from xmetai_evaluation.metrics.base import single_variable

    dataset = xr.Dataset(
        {"z500": (["lat"], [1.0, 2.0]), "t2m": (["lat"], [3.0, 4.0])},
        coords={"lat": [10.0, 20.0]},
    )

    with pytest.raises(MetricError, match="variables"):
        single_variable(dataset, "forecast")


# --------------------------------------------------------------------------
# xu 复刻配置：声明层自洽性
# --------------------------------------------------------------------------

XU_CONFIGS = [
    "weather_field_scores_era5_fuxi",
    "weather_field_scores_era5_fengqing",
    "weather_ens_field_scores_era5_fuxi",
]


@pytest.mark.parametrize("name", XU_CONFIGS)
def test_xu_configs_load_and_route_every_metric(name):
    """模板里的每个指标都必须被 VAR_METRICS 点到名。

    漏一个，那个指标就会拿到整批多变量数据 → single_variable 直接报错。
    """
    from xmetai_evaluation.configs.base import load_config

    spec = PipelineSpec.from_config(load_config(name))
    routed = {
        item.name for item in spec.metrics if item.params.get("variables")
    }

    assert routed == {item.name for item in spec.metrics}


@pytest.mark.parametrize("name", XU_CONFIGS)
def test_xu_configs_use_era5_and_a_daily_climatology(name):
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config(name)

    assert cfg.observation_reader["type"] == "era5_zarr"
    assert set(cfg.observation_reader["stores"]) == {"pl", "sfc"}
    assert cfg.reference_reader["type"] == "daily_climatology"
    assert cfg.forecast_reader["type"].startswith(("fuxi", "fengqing"))
    assert cfg.forecast_reader["type"].endswith("_phys")


@pytest.mark.parametrize("name", XU_CONFIGS)
def test_xu_configs_use_the_uncentered_acc(name):
    """xu 的 ACC 是 uncentered（FDP/WeatherBench2 口径），差一点就对不上数。"""
    from xmetai_evaluation.configs.base import load_config

    options = load_config(name).metric_options

    assert options["acc"]["centered"] is False
    assert options["zonal_spectrum"]["max_wavenumber"] == 720


def test_ensemble_config_needs_the_combined_template():
    """集合那条要确定性指标和 CRPS 同跑，两个现有模板都装不下。"""
    from xmetai_evaluation.configs.base import load_config

    spec = PipelineSpec.from_config(
        load_config("weather_ens_field_scores_era5_fuxi")
    )
    names = [item.name for item in spec.metrics]

    assert names == ["rmse", "crps", "acc", "activity", "zonal_spectrum"]
    # 每个指标都点名了变量，crps 只给 z500
    assert spec.metric_options["crps"]["variables"] == ["z500"]


def test_xu_layouts_keep_the_report_units():
    """z500 报 m²/s²（不除 g）、q 报 g/kg——这是能和 xu 逐格对拍的前提。"""
    from xmetai_evaluation.io.layouts import (
        ERA5_ZARR_LAYOUT,
        FENGQING_PHYS_LAYOUT,
        FUXI_PHYS_LAYOUT,
    )

    for layout in (FUXI_PHYS_LAYOUT, FENGQING_PHYS_LAYOUT, ERA5_ZARR_LAYOUT):
        z500 = layout.spec_for("z500")
        assert z500.unit == "m^2/s^2"
        assert z500.scale == 1.0, f"{layout.name} 的 z500 不该除 g"

    assert ERA5_ZARR_LAYOUT.spec_for("q700").scale == 1000.0  # kg/kg -> g/kg
    assert ERA5_ZARR_LAYOUT.spec_for("tp").scale == 1000.0  # m -> mm
    # 预报侧源文件已是 g/kg，不能再换算
    for layout in (FUXI_PHYS_LAYOUT, FENGQING_PHYS_LAYOUT):
        assert layout.spec_for("q700").scale == 1.0


def test_phys_layouts_do_not_shadow_each_other():
    """同一个 source 只能有一个标准名，否则 u10m 永远解析不到。"""
    from xmetai_evaluation.io.layouts import FENGQING_PHYS_LAYOUT

    sources = [
        spec.source for spec in FENGQING_PHYS_LAYOUT.variables.values()
    ]
    assert len(sources) == len(set(sources))
    assert FENGQING_PHYS_LAYOUT.standard_name_for("U10") == "u10m"


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
