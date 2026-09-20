"""执行层：切块、加载驻留策略与归并语义。"""

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial

import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.execution.executor import ChunkOutcome, merge_outcomes
from xmetai_evaluation.execution.loader import RoleSpec, RunLoader


class _FakeHandle:
    """角色声明里的句柄替身（pickle 测试只走序列化，不调 reader）。"""

    source_id = "fake"
    reader = None
    catalog = None


# ---------------------------------------------------------------------------
# 加载策略：slice / resident / window
# ---------------------------------------------------------------------------


def _bundle(times):
    """一个最小 DataBundle 替身：只带 payload 与 provenance（loader 只用这些）。"""
    payload = xr.Dataset(
        {"value": (("valid_time",), [float(i) for i in range(len(times))])},
        coords={"valid_time": list(times)},
    )

    @dataclass
    class _Provenance:
        input_files: list = field(default_factory=list)

    @dataclass
    class _Bundle:
        payload: object
        provenance: object

    return _Bundle(payload=payload, provenance=_Provenance(input_files=[str(t) for t in times]))


def _record_builder(times, calls, span):
    """模块级 builder 本体：生产侧的 builder 也是模块级函数 + ``partial``。

    写成闭包会让 ``RoleSpec`` 整个 pickle 不了（spawn 形态下子进程要反序列化
    角色声明），而这条恰好是 ``test_run_loader_survives_pickle_round_trip``
    要守的东西——所以替身必须和生产同形。
    """
    selected = [t for t in times if span[0] <= t <= span[1]] if span else list(times)
    calls.append((span[0] if span else None, len(selected)))
    return _bundle(selected)


def _role(times, policy, window_days=1, count=None):
    """builder 会把每次调用记进 count（验证驻留/滚动有没有生效）。"""
    calls = count if count is not None else []
    times = list(times)

    return RoleSpec(
        role="observation",
        policy=policy,
        window_days=window_days,
        builder=partial(_record_builder, times, calls),
        times=times,
    ), calls


def _request(start, end):
    times = []
    current = start
    while current <= end:
        times.append(current)
        current += timedelta(hours=6)
    return DataRequest(source_id="obs", variables=["tp"], init_times=times)


BASE = datetime(2025, 1, 1)


def test_slice_policy_reads_every_request():
    times = [BASE + timedelta(hours=6 * i) for i in range(8)]
    spec, calls = _role(times, "slice")
    loader = RunLoader({"observation": spec})

    loader.materialize("observation", _request(BASE, BASE + timedelta(hours=12)))
    loader.materialize("observation", _request(BASE, BASE + timedelta(hours=12)))
    assert len(calls) == 2  # 现读现弃：两次请求两次读


def test_resident_policy_reads_once_and_slices():
    times = [BASE + timedelta(hours=6 * i) for i in range(8)]
    spec, calls = _role(times, "resident")
    loader = RunLoader({"observation": spec})

    first = loader.materialize(
        "observation", _request(BASE, BASE + timedelta(hours=12))
    )
    second = loader.materialize(
        "observation", _request(BASE + timedelta(hours=30), BASE + timedelta(hours=42))
    )
    assert len(calls) == 1  # 整段只物化一次
    assert first.payload.sizes["valid_time"] == 3
    assert second.payload.sizes["valid_time"] == 3
    # provenance 保留全量清单（manifest 口径：这个 run 读过哪些）
    assert len(second.provenance.input_files) == len(times)


def test_window_policy_loads_and_evicts_by_block():
    times = [BASE + timedelta(days=d, hours=h) for d in range(4) for h in (0, 12)]
    spec, calls = _role(times, "window", window_days=2)
    # 关掉预取：本条测的是按块加载与滚动淘汰，预取会额外多读一块干扰计数
    loader = RunLoader({"observation": spec}, prefetch=False)

    loader.materialize("observation", _request(BASE, BASE))
    loader.materialize("observation", _request(BASE + timedelta(days=3), BASE + timedelta(days=3)))
    # 第一窗（0-1 日）一个块、第二窗（2-3 日）一个块，互不重叠
    assert [call[1] for call in calls] == [4, 4]
    # 滚动淘汰：留的是"当前块 ±1"。第二窗请求后第一窗的块还在（它正是前一块），
    # 再往前没有别的块可留
    assert list(loader._window_blocks["observation"]) == [0, 1]


def test_window_keeps_the_previous_block_for_the_sawtooth():
    """观测读是锯齿（向前扫 L 天、再回跳 L-2 天）：留 ±1 块才接得住回跳。

    块序是「起报日外层、时效窗内层」，扫到尾部后下一个起报日组又从头开始。
    只留当前块时那个"从头"必然落空重读；留 ±1 块则块 0 一直躺在缓存里。
    """
    times = [BASE + timedelta(days=d) for d in range(6)]
    spec, calls = _role(times, "window", window_days=1)
    loader = RunLoader({"observation": spec}, prefetch=False)

    for offset in (0, 1):
        loader.materialize(
            "observation",
            _request(BASE + timedelta(days=offset), BASE + timedelta(days=offset)),
        )
    assert len(calls) == 2  # 块 0、块 1 各读一次

    loader.materialize("observation", _request(BASE, BASE))  # 回跳到块 0
    assert len(calls) == 2  # 块 0 还在缓存里，不重读
    assert list(loader._window_blocks["observation"]) == [0, 1]


def test_materialize_requires_known_role_and_times():
    spec, _ = _role([BASE], "resident")
    loader = RunLoader({"observation": spec})
    with pytest.raises(Exception):
        loader.materialize("reference", _request(BASE, BASE))
    with pytest.raises(Exception):
        loader.materialize(
            "observation", DataRequest(source_id="obs", variables=["tp"])
        )


# ---------------------------------------------------------------------------
# 归并：append / replace 语义
# ---------------------------------------------------------------------------


def _outcome(
    chunk_id, key_states, ok=True, processed=1, skipped=0, summary=None, init_time=""
):
    return ChunkOutcome(
        chunk_id=chunk_id,
        ok=ok,
        states=key_states,
        coordinates={
            key: {"lead_h": key[1], "init_time": init_time} for key in key_states
        },
        processed=processed,
        skipped=skipped,
        summary=summary or {},
    )


def _spec(protocol, name, transforms=()):
    """一段最小声明（只要协议与窗口口径对，组件都不用真连数据）。"""
    from xmetai_evaluation.components import register_builtin_components

    register_builtin_components()
    from xmetai_evaluation.pipeline.spec import (
        MetricSpec,
        PipelineSpec,
        PipelineTemplate,
        SourceSpec,
    )

    return PipelineSpec(
        name=name,
        forecast=SourceSpec("fuxi", {"root_dir": "unused", "variable": "tp"}),
        observation=SourceSpec(
            "station" if protocol == "station_valid_time" else "era5_zarr",
            {"root_dir": "unused", "variable": "rain"},
        ),
        template=PipelineTemplate(
            name="t",
            protocol=protocol,
            metrics=[MetricSpec("ts_score" if "station" in protocol else "rmse")],
            transforms=list(transforms),
        ),
    )


@pytest.fixture
def station_spec():
    return _spec("station_valid_time", "merge_test_station")


@pytest.fixture
def grid_spec():
    return _spec("grid_valid_time", "merge_test_grid")


def test_merge_outcomes_appends_station_states(station_spec):
    # station 协议：同 lead 下多个起报的状态追加
    first = _outcome("d1", {(0, ("lead_h", 24)): ["s1"]})
    second = _outcome("d2", {(0, ("lead_h", 24)): ["s2"]})

    merged = merge_outcomes([first, second], station_spec)

    assert merged.states[(0, ("lead_h", 24))] == ["s1", "s2"]
    assert merged.processed == 2
    assert not merged.failures


def test_merge_outcomes_replaces_grid_states_per_protocol(station_spec, grid_spec):
    # grid 协议声明 replace：同键只保留最新起报
    key = (0, ("valid_time", "2025-01-02T00:00:00"))
    first = _outcome("d1", {key: ["old-init"]}, init_time="2025-01-01T00:00:00")
    second = _outcome("d2", {key: ["new-init"]}, init_time="2025-01-02T00:00:00")

    merged = merge_outcomes([first, second], grid_spec)
    assert merged.states[key] == ["new-init"]

    # station 协议是 append：同一批 outcome 两种语义并存
    merged = merge_outcomes([first, second], station_spec)
    assert merged.states[key] == ["old-init", "new-init"]


def test_merge_outcomes_keeps_latest_init_even_when_older_chunk_comes_last(grid_spec):
    """replace 比的是 init_time，不是块序。

    同一个 valid_time 可以由 (早起报, 长时效) 和 (晚起报, 短时效) 两条路径够到，
    而**最新起报配的是最小 lead**——按"起报组升序 + 时效窗升序"排块时，晚起报的
    块反而排在前面。若归并按块序覆盖，长时效窗（早起报）会在最后把结果改掉。
    """
    key = (0, ("valid_time", "2025-01-03T00:00:00"))
    # 短时效窗：起报 01-02 + lead 24 → 块序在前
    short_window = _outcome("d0102_L24", {key: ["lead24@0102"]}, init_time="2025-01-02T00:00:00")
    # 长时效窗：起报 01-01 + lead 48 → 块序在后，但起报更早
    long_window = _outcome("d0101_L48", {key: ["lead48@0101"]}, init_time="2025-01-01T00:00:00")

    merged = merge_outcomes([short_window, long_window], grid_spec)

    assert merged.states[key] == ["lead24@0102"]
    assert merged.coordinates[key]["init_time"] == "2025-01-02T00:00:00"


def test_merge_outcomes_records_failures_but_keeps_good_chunks(station_spec):
    good = _outcome("d1", {(0, ("lead_h", 24)): ["s1"]}, processed=5, skipped=1)
    bad = ChunkOutcome(chunk_id="d2", ok=False, error="boom", skipped=2)
    summary = _outcome(
        "d3", {(0, ("lead_h", 24)): ["s3"]}, summary={"n_init_times": 2}
    )

    merged = merge_outcomes([good, bad, summary], station_spec)

    assert merged.failures == [("d2", "boom")]
    assert merged.processed == 6
    assert merged.skipped == 3
    assert merged.summary == {"n_init_times": 2}


def test_merge_summary_sums_counts_and_unions_lists():
    from xmetai_evaluation.execution.executor import _merge_summary

    merged = _merge_summary(
        {"n_init_times": 3, "files": ["a", "b"], "mode": "x", "window_hours": 24},
        {
            "n_init_times": 4,
            "files": ["b", "c"],
            "mode": "x",
            "window_hours": 24,
        },
    )
    # 计数键（n_ 前缀）跨块相加
    assert merged["n_init_times"] == 7
    # 输入文件清单取并集
    assert merged["files"] == ["a", "b", "c"]
    # 口径数值/字符串保留首个：window_hours 相加会变成 48 的假数
    assert merged["window_hours"] == 24
    assert merged["mode"] == "x"


# ---------------------------------------------------------------------------
# 切块：起报日组 × 时效窗，以及时效窗的采样格点与累积预热
# ---------------------------------------------------------------------------


def test_chunk_work_crosses_init_day_groups_with_lead_windows(grid_spec):
    """工作块 = 起报日组 × 时效窗；格点协议读到的每个时效都是样本。"""
    from xmetai_evaluation.execution.plan import _chunk_work

    inits = [
        datetime(2025, 1, 1, 0),
        datetime(2025, 1, 1, 12),
        datetime(2025, 1, 3, 0),
        datetime(2025, 1, 4, 0),
    ]
    leads = [float(step) for step in range(0, 49, 6)]  # 0..48

    chunks = _chunk_work(
        grid_spec, inits, chunk_days=2, leads=leads, lead_chunk_days=1, warmup_hours=0.0
    )

    # 起报日组：{01-01, 01-03}、{01-04}；组内按 24h 时效窗再切
    assert [c.chunk_id for c in chunks] == [
        "20250101-20250103_L0-18",
        "20250101-20250103_L24-42",
        "20250101-20250103_L48",
        "20250104_L0-18",
        "20250104_L24-42",
        "20250104_L48",
    ]
    assert [len(c.init_times) for c in chunks[:3]] == [3, 3, 3]
    # 格点协议：采样集 = 读取集，且是 (起报, 时效) 的纯划分
    assert chunks[1].sample_leads == chunks[1].read_leads == [24.0, 30.0, 36.0, 42.0]

    # chunk_days=1：一天一块（默认：按起报日开并发的口径）
    chunks = _chunk_work(
        grid_spec, inits, chunk_days=1, leads=leads, lead_chunk_days=0, warmup_hours=0.0
    )
    assert [c.chunk_id for c in chunks] == ["20250101_L0-48", "20250103_L0-48", "20250104_L0-48"]
    assert len(chunks[0].init_times) == 2  # 1 日的两个起报时次在一块里


def test_chunk_work_station_samples_complete_windows_with_warmup(station_spec):
    """站点协议：时效窗落在采样格点上，读取集往前带累积预热。

    window_hours=24 时只有 lead 24 的倍数才是完整窗口的末端；24h 宽的窗
    [0, 24) 里一个采样点都没有——按原始跨度切会切出空块（processed==0 判失败）。
    按采样格点分组，空窗结构上不会出现。
    """
    from xmetai_evaluation.execution.plan import _chunk_work

    leads = [float(step) for step in range(6, 49, 6)]  # 6..48
    chunks = _chunk_work(
        station_spec,
        [datetime(2025, 1, 1, 0)],
        chunk_days=1,
        leads=leads,
        lead_chunk_days=1,
        warmup_hours=18.0,  # (24/6 - 1) 步
    )

    assert [(c.sample_leads, c.read_leads) for c in chunks] == [
        ([24.0], [6.0, 12.0, 18.0, 24.0]),
        ([48.0], [30.0, 36.0, 42.0, 48.0]),
    ]

    # lead_chunk_days=0：整段时效一块（逃生口，与不切时效等价）
    chunks = _chunk_work(
        station_spec,
        [datetime(2025, 1, 1, 0)],
        chunk_days=1,
        leads=leads,
        lead_chunk_days=0,
        warmup_hours=18.0,
    )
    assert len(chunks) == 1
    assert chunks[0].sample_leads == [24.0, 48.0]
    assert chunks[0].read_leads == [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]


def test_chunk_work_reports_unusable_span_instead_of_empty_chunks(station_spec):
    """整段时效跨度撑不出一个完整窗口时明确报错，而不是切出一堆空块。"""
    from xmetai_evaluation.core.errors import EvaluationError
    from xmetai_evaluation.execution.plan import _chunk_work

    with pytest.raises(EvaluationError, match="没有任何完整窗口可评"):
        _chunk_work(
            station_spec,
            [datetime(2025, 1, 1, 0)],
            chunk_days=1,
            leads=[6.0, 12.0, 18.0],  # 都不到 window_hours=24
            lead_chunk_days=1,
            warmup_hours=18.0,
        )


def test_warmup_hours_only_for_sliding_accumulator():
    from xmetai_evaluation.execution.plan import _warmup_hours
    from xmetai_evaluation.pipeline.spec import TransformSpec

    plain = _spec("station_valid_time", "plain")
    assert _warmup_hours(plain, 6.0) == 0.0  # 预报本身就是窗口累积量

    accumulating = _spec(
        "station_valid_time",
        "accum",
        transforms=[TransformSpec("time_window_accumulator", {"window_hours": 24})],
    )
    assert _warmup_hours(accumulating, 6.0) == 18.0  # 24/6 - 1 步


def test_auto_window_days_covers_leads_and_thread_spread():
    from xmetai_evaluation.execution.plan import _auto_window_days
    from xmetai_evaluation.execution.profiles import ResourceProfile
    from xmetai_evaluation.execution.strategy import ExecutionStrategy

    leads = [0.0, 6.0, 12.0, 360.0]  # 最大时效 15 天
    heavy = ResourceProfile(compute_class="heavy")

    # 不切时效（lead_chunk_days=0，逃生口）：窗宽跟着整个时效跨度走
    # processes：段内串行一次一块，跨度 0 → 15 + 0 + 1
    strategy = ExecutionStrategy(
        mode="processes", n_workers=12, chunk_days=1, lead_chunk_days=0
    )
    assert _auto_window_days(strategy, 90, heavy, leads) == 16

    # 不切时效 + threads：5 个并发块时间相邻，跨度 = (5-1)*1 → 15 + 4 + 1
    strategy = ExecutionStrategy(
        mode="threads", n_workers=5, chunk_days=1, lead_chunk_days=0
    )
    assert _auto_window_days(strategy, 90, ResourceProfile(), leads) == 20

    # 没有时效信息时按 1 天兜底
    assert _auto_window_days(strategy, 90, ResourceProfile(), []) == 6

    # 切了时效（缺省 1 天）：窗宽仍按**整个时效跨度**算，不是按单块时效窗宽。
    # 块序是「起报日外层、时效窗内层」，一个起报日组要把所有时效窗顺次扫一遍，
    # 碰到的观测日跨度就是整个跨度；组与组之间还要回跳，窗宽不够回跳就落空重读。
    # 15（整段时效跨度）+ 0（预热）+ 1（块内起报跨度）+ 0（processes 无并发跨度）
    strategy = ExecutionStrategy(mode="processes", n_workers=12, chunk_days=1)
    assert _auto_window_days(strategy, 90, heavy, leads) == 16
    # 预热单独占一天：24h 累积窗 / 6h 步长 → 18h 预热
    assert _auto_window_days(strategy, 90, heavy, leads, warmup_hours=18.0) == 17

    # threads 下切时效：15（整段时效跨度）+ 1（起报跨度）+ (5-1)*1（并发跨度）
    strategy = ExecutionStrategy(mode="threads", n_workers=5, chunk_days=1)
    assert _auto_window_days(strategy, 90, ResourceProfile(), leads) == 20


def test_configured_window_is_capped_by_the_auto_minimum():
    """配置的窗宽当**上限**用：比自动值大就收到自动值，比它小则照用小值。"""
    from xmetai_evaluation.execution.plan import _resolve_window
    from xmetai_evaluation.execution.profiles import ResourceProfile
    from xmetai_evaluation.execution.strategy import ExecutionStrategy

    leads = [0.0, 6.0, 12.0, 360.0]  # 自动值 = 15（整段时效跨度）+ 1 = 16
    heavy = ResourceProfile(compute_class="heavy")

    def _resolve(value):
        strategy = ExecutionStrategy(
            mode="processes", n_workers=12, chunk_days=1, loads={"observation": value}
        )
        return _resolve_window("observation", strategy, 90, heavy, leads)

    assert _resolve("window") == ("window", 16)  # 裸写 = 自动值
    assert _resolve("window:16") == ("window", 16)
    assert _resolve("window:90") == ("window", 16)  # 比自动值大 → 收到自动值
    assert _resolve("window:4") == ("window", 4)  # 比自动值小 → 照用小值
    assert _resolve("slice") == ("slice", 0)



    from xmetai_evaluation.execution.executor import _partition_ranges

    # 10 块分 3 段：连续、段长差不超过 1
    assert _partition_ranges(list(range(10)), 3) == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    # 段数多于块数：退化为一段一块；空输入没有段
    assert _partition_ranges([0, 1], 5) == [[0], [1]]
    assert _partition_ranges([], 4) == []
    # resume 跳过的块留下空洞，剩余索引仍按时间连续分段
    assert _partition_ranges([2, 3, 7, 8, 9], 2) == [[2, 3, 7], [8, 9]]


def test_sub_spec_injects_chunk_init_times_and_clears_limit(station_spec):
    from xmetai_evaluation.execution.plan import WorkChunk, _sub_spec

    station_spec.start_date = "20250101"
    station_spec.limit = 10
    chunk = WorkChunk(
        chunk_id="20250101_L24",
        init_times=[datetime(2025, 1, 1)],
        read_leads=[6.0, 12.0, 18.0, 24.0],
        sample_leads=[24.0],
    )

    sub = _sub_spec(station_spec, chunk)

    assert sub.forecast.params["init_times"] == ["2025-01-01T00:00:00"]
    # 时效是**覆盖**不是合并：留着全时效声明，块会把整段跨度读回来
    assert sub.forecast.params["lead_times"] == [6.0, 12.0, 18.0, 24.0]
    assert sub.forecast.params["sample_leads"] == [24.0]
    assert sub.limit is None
    # 原声明不动（父进程口径保留）
    assert "init_times" not in station_spec.forecast.params
    assert station_spec.limit == 10


# ---------------------------------------------------------------------------
# 序列化：spawn 的子进程要靠它自建缓存
# ---------------------------------------------------------------------------


def test_role_declarations_are_picklable():
    """角色声明必须能 pickle，否则 spawn 的子进程只能拿 loader=None。

    原来是闭包（捕获 handle/variables/times），pickle 不了；改成模块级函数
    + ``partial`` 绑参之后才可序列化。这条一断，无 fork 平台就退回"每个块
    把观测重读一遍"，而且不会有任何报错。
    """
    from xmetai_evaluation.execution.plan import _grid_role, _station_role

    handle = _FakeHandle()
    times = [BASE + timedelta(hours=6 * i) for i in range(4)]
    roles = [
        _grid_role("observation", handle, ["tp"], times, "window", 30),
        _station_role(handle, "rain", (BASE, BASE + timedelta(days=1)), "resident", 0),
    ]

    for role in roles:
        clone = pickle.loads(pickle.dumps(role))
        assert clone.role == role.role
        assert clone.policy == role.policy
        assert clone.times == role.times


def test_run_loader_survives_pickle_round_trip():
    """锁与队列天生 pickle 不了（内部是 _thread.lock），靠 __getstate__ 重建。

    没有这两个钩子的话 _picklable(loader) 恒为 False，spawn 下"子进程自建
    缓存"会静默退回逐块直读——不报错，只是白读。
    """
    times = [BASE + timedelta(hours=6 * i) for i in range(8)]
    spec, _ = _role(times, "window", window_days=1)

    clone = pickle.loads(pickle.dumps(RunLoader({"observation": spec})))

    assert clone._roles["observation"].policy == "window"
    assert clone._roles["observation"].window_days == 1
    # 运行期物件各自新建，不与原对象共享
    assert clone._prefetch_queue is not None
    assert clone._prefetch_worker is None
    with clone._lock:
        pass  # 重建出来的锁是可用的


# ---------------------------------------------------------------------------
# 预取：读满一个窗块后，后台把下一个块也读进来
# ---------------------------------------------------------------------------


def test_next_block_span_is_the_block_after_the_request():
    from xmetai_evaluation.execution.loader import _blocks_for_span, _next_block_span

    times = [BASE + timedelta(days=d) for d in range(6)]
    spec, _ = _role(times, "window", window_days=2)

    first = _blocks_for_span(times, 2, (times[0], times[1]))
    assert [index for index, _ in first] == [0]
    assert _next_block_span(spec, first) == (1, (times[2], times[3]))

    # 已经是最后一个块：没有下一块可预取
    last = _blocks_for_span(times, 2, (times[4], times[5]))
    assert _next_block_span(spec, last) is None


def test_window_policy_prefetches_the_next_block():
    """``queue.join()`` 等预取落地——预取是异步的，直接断言会飘。"""
    times = [BASE + timedelta(days=d) for d in range(6)]
    spec, calls = _role(times, "window", window_days=2)
    loader = RunLoader({"observation": spec})

    loader.materialize("observation", _request(BASE, BASE + timedelta(days=1)))
    loader._prefetch_queue.join()

    # 请求只覆盖第 0 块，后台把第 1 块也读了
    assert [call[0] for call in calls] == [BASE, BASE + timedelta(days=2)]


def test_prefetched_block_is_reused_without_reading_again():
    times = [BASE + timedelta(days=d) for d in range(6)]
    spec, calls = _role(times, "window", window_days=2)
    loader = RunLoader({"observation": spec})

    loader.materialize("observation", _request(BASE, BASE + timedelta(days=1)))
    loader._prefetch_queue.join()
    assert len(calls) == 2

    # 请求第 1 块：吃预取好的那份，不再物化
    second = loader.materialize(
        "observation", _request(BASE + timedelta(days=2), BASE + timedelta(days=3))
    )
    # 第 1 块就是 times[2] / times[3] 两天
    assert second.payload.sizes["valid_time"] == 2
    assert loader._stats["cache_hits"] >= 1


def test_prefetch_only_for_window_and_only_when_enabled():
    times = [BASE + timedelta(days=d) for d in range(6)]

    # 关掉（threads 形态）：一个块都不多读，也不起线程
    window_spec, calls = _role(times, "window", window_days=2)
    loader = RunLoader({"observation": window_spec}, prefetch=False)
    loader.materialize("observation", _request(BASE, BASE + timedelta(days=1)))
    assert len(calls) == 1
    assert loader._prefetch_worker is None

    # slice 角色没有"块"可言，本来就不预取
    slice_spec, slice_calls = _role(times, "slice")
    loader = RunLoader({"observation": slice_spec})
    loader.materialize("observation", _request(BASE, BASE + timedelta(days=1)))
    assert len(slice_calls) == 1
    assert loader._prefetch_worker is None


# ---------------------------------------------------------------------------
# resume：只提醒，不改结果
# ---------------------------------------------------------------------------


def test_orphan_resume_states_are_warned_about(tmp_path, caplog):
    """计划缩水时旧块状态会变成孤儿：不复用、不报错，必须留下痕迹。"""
    from xmetai_evaluation.execution.executor import _warn_orphan_states

    for chunk_id in ("20250101_L24", "20250102_L24"):
        (tmp_path / f"seg01_{chunk_id}.pkl").write_bytes(b"")
    # 别的段的同名文件不算孤儿（前缀带段号）
    (tmp_path / "seg02_20250101_L24.pkl").write_bytes(b"")

    with caplog.at_level("WARNING"):
        _warn_orphan_states(tmp_path, 1, ["20250102_L24"])
    assert "20250101_L24" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        _warn_orphan_states(tmp_path, 1, ["20250101_L24", "20250102_L24"])
    assert caplog.text == ""  # 都在计划里，不该出声

    caplog.clear()
    with caplog.at_level("WARNING"):
        _warn_orphan_states(tmp_path / "不存在", 1, [])
    assert caplog.text == ""  # 目录都没有（首次跑）也不该出声
