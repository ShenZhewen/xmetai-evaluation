"""执行层：能力画像与策略推导。"""

import pytest

from xmetai_evaluation.components import register_builtin_components
from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.core.registry import ComponentType, get_registry
from xmetai_evaluation.execution.profiles import union_profile
from xmetai_evaluation.execution.strategy import (
    ExecutionStrategy,
    derive_strategy,
    parse_load_policy,
)


def _build_metric(name, **params):
    register_builtin_components()
    return get_registry().build(ComponentType.METRIC, name, **params)


def test_union_profile_sums_requirement_dimensions():
    rmse = _build_metric("rmse")
    acc = _build_metric("acc")
    crps = _build_metric("crps")
    spectrum = _build_metric("spectrum", max_wavenumber=8)

    light = union_profile([rmse])
    assert light.compute_class == "light"
    assert not light.needs_reference
    assert not light.needs_members

    # ACC 带参考、CRPS 带成员、谱带整场且重——联合画像全部点亮
    combined = union_profile([rmse, acc, crps, spectrum])
    assert combined.needs_reference
    assert combined.needs_members
    assert combined.needs_full_field
    assert combined.compute_class == "heavy"


def test_union_profile_handles_custom_metrics_without_hooks():
    """没有 needs_* 钩子的自定义指标按"都不需要 + 表外轻量"处理。"""

    class Custom:
        name = "my_index"

    profile = union_profile([Custom()])
    assert not profile.needs_reference
    assert not profile.needs_members
    assert profile.compute_class == "light"


def test_parse_load_policy_accepts_all_forms():
    assert parse_load_policy("slice") == ("slice", 0)
    assert parse_load_policy("resident") == ("resident", 0)
    # 裸 window = 自动定窗（计划层按时效+并发跨度算出天数）；带数字显式指定
    assert parse_load_policy("window") == ("window", None)
    assert parse_load_policy("window:30") == ("window", 30)
    with pytest.raises(ConfigError):
        parse_load_policy("window:0")
    with pytest.raises(ConfigError):
        parse_load_policy("mmap")


def test_derive_strategy_loads_by_reader_shape():
    from xmetai_evaluation.execution.profiles import ResourceProfile

    light = ResourceProfile()
    # 站点观测 / 日序气候态默认驻留，era5_zarr 默认逐块直读
    strategy = derive_strategy(
        light,
        observation_reader="station",
        reference_reader="daily_climatology",
    )
    assert strategy.loads == {
        "observation": "resident",
        "reference": "resident",
    }
    strategy = derive_strategy(
        light,
        observation_reader="era5_zarr",
        reference_reader=None,
    )
    assert strategy.loads == {"observation": "slice"}


def test_derive_strategy_skips_unmanaged_reference_readers():
    """逐站概率参考没有整包可缓存，不该出现在 loads 里，也不许手动配。"""
    from xmetai_evaluation.execution.profiles import ResourceProfile

    strategy = derive_strategy(
        ResourceProfile(),
        observation_reader="station",
        reference_reader="ref_probability",
    )
    assert "reference" not in strategy.loads
    with pytest.raises(ConfigError):
        derive_strategy(
            ResourceProfile(),
            observation_reader="station",
            reference_reader="ref_probability",
            execution={"loads": {"reference": "resident"}},
        )


def test_derive_strategy_config_overrides_and_validates():
    from xmetai_evaluation.execution.profiles import ResourceProfile

    strategy = derive_strategy(
        ResourceProfile(),
        observation_reader="station",
        reference_reader=None,
        execution={
            "mode": "processes",
            "n_workers": 8,
            "chunk_days": 5,
            "loads": {"observation": "window:14"},
            "resume": True,
        },
    )
    assert strategy.mode == "processes"
    assert strategy.n_workers == 8
    assert strategy.chunk_days == 5
    assert strategy.loads["observation"] == "window:14"
    assert strategy.resume is True

    with pytest.raises(ConfigError):
        derive_strategy(
            ResourceProfile(),
            "station",
            None,
            execution={"mode": "asyncio"},
        )
    with pytest.raises(ConfigError):
        derive_strategy(ResourceProfile(), "station", None, execution={"chunk_days": 0})
    with pytest.raises(ConfigError):
        derive_strategy(
            ResourceProfile(), "station", None, execution={"n_worker": 4}
        )


def test_legacy_num_workers_field_is_honoured():
    from xmetai_evaluation.execution.profiles import ResourceProfile

    # execution 不给 n_workers 且旧字段 > 1：当作 n_workers（老配置不失效）
    strategy = derive_strategy(
        ResourceProfile(), "station", None, num_workers=48
    )
    assert strategy.n_workers == 48
    # execution 里的 n_workers 优先于旧字段
    strategy = derive_strategy(
        ResourceProfile(), "station", None, execution={"n_workers": 2}, num_workers=48
    )
    assert strategy.n_workers == 2
    # 都不给：缺省由形态决定
    strategy = derive_strategy(ResourceProfile(), "station", None)
    assert strategy.n_workers is None


def test_auto_mode_resolves_by_chunk_count_and_weight():
    from xmetai_evaluation.execution.profiles import ResourceProfile

    light = ResourceProfile(compute_class="light")
    heavy = ResourceProfile(compute_class="heavy")

    strategy = ExecutionStrategy(mode="auto")
    assert strategy.resolve_mode(1, light) == "serial"
    assert strategy.resolve_mode(8, light) == "threads"
    assert strategy.resolve_mode(8, heavy) == "processes"
    # 显式声明不被 auto 规则改写
    assert ExecutionStrategy(mode="serial").resolve_mode(64, heavy) == "serial"


def test_resolve_worker_defaults():
    strategy = ExecutionStrategy()
    assert strategy.resolve_workers("threads") == 4
    assert strategy.resolve_workers("processes") >= 1
    assert ExecutionStrategy(n_workers=3).resolve_workers("threads") == 3
