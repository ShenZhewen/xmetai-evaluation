# -*- coding: utf-8 -*-
"""块执行器：工作块的收集、并发调度与统一定义归并。

一个工作块 = 一份子声明（起报已注入）+ 完整的一段流水线（构建组件 ->
协议 prepare -> 样本循环 -> 指标状态累积）。子声明是纯数据，可 pickle，
worker 拿到后自己重建组件——注册表架构让这件事不需要任何序列化技巧。

并发形态由策略层定（见 ``execution/strategy.py``）：

    serial     父进程逐块跑；
    threads    ThreadPoolExecutor，共享同一个 RunLoader（有锁）；
    processes  ProcessPoolExecutor。块按**时间连续分段**，一段一个任务、
               同一子进程顺序跑完——观测 window 缓存的复用依赖段内相邻块。
               Linux fork 下计划层把 loader 预热进模块全局再 fork，子进程
               写时复制共享 resident 数据；spawn 平台没有共享，但 loader 是
               可序列化的（角色声明用 partial 绑参而不是闭包），由池的
               initializer 给每个子进程装一份副本——缓存照常生效，代价是
               每个子进程各存一份（结果一致，内存 ×段数）。

归并（``merge_outcomes``）把各块的状态按**块序**合并：append 语义的协议
（station）同键追加，replace 语义的协议（grid，同一 valid_time 只认最新
起报）后块覆盖前块——保证切块大小不改变结果口径。
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from tqdm import tqdm

from xmetai_evaluation.components import register_builtin_components
from xmetai_evaluation.core.errors import ConfigError, EvaluationError, MetricError
from xmetai_evaluation.core.registry import ComponentType, get_registry
from xmetai_evaluation.pipeline.matcher import narrow_batch
from xmetai_evaluation.pipeline.protocols import PipelineContext
from xmetai_evaluation.pipeline.spec import PipelineSpec

log = logging.getLogger(__name__)

#: 进程模式下子进程每跑完这么多块打一行进度。段内是顺序跑的、父进程要等整段
#: 返回才拿到结果（父进程那条 tqdm 因此全程不动），中途唯一的进度来源就是这行。
#: 想更密/更疏直接改这个数：5568 块 / 5 段 ≈ 1114 块一段，50 → 每段约 22 行。
PROGRESS_EVERY = 50


class MetricRun(NamedTuple):
    """一次指标执行：指标对象 + 这次执行针对的变量（None 表示整批，不做路由）。"""

    metric: Any
    variable: Optional[str]
    label: str


def metric_runs(specs: List[Any], metrics: List[Any]) -> List[MetricRun]:
    """把指标声明展开成执行单元。

    指标声明了 ``variables`` 就逐变量各跑一次（对应参考实现的
    ``--var-metrics z500:rmse,acc``，同一指标只作用于指定变量）；
    没声明的保持原行为——整批跑一次，老配置完全不变。
    """
    runs: List[MetricRun] = []
    for item, metric in zip(specs, metrics):
        names = [str(name) for name in (item.params.get("variables") or [])]
        if not names:
            runs.append(MetricRun(metric=metric, variable=None, label=item.name))
            continue
        runs.extend(
            MetricRun(metric=metric, variable=name, label=f"{item.name}[{name}]")
            for name in names
        )
    return runs


def _never_requires_reference() -> bool:
    """没有 needs_reference() 的指标（如自定义实现）默认不需要参考源。"""
    return False


def needs_reference(metrics: List[Any]) -> bool:
    """这一段里有没有指标要用参考源。

    配置里的参考源是所有段共用的，但某一段可能压根用不上（比如 24h 的 TS 段
    配着 6h 概率段一起跑）。用不上就不该建：白建事小，参考目录缺时次时它抛的是
    ConfigError（不是 MetricError），会被样本循环的兜底 except 静默吞掉，
    表现成"这一段的结果莫名其妙少了一批样本"。
    """
    return any(
        getattr(metric, "needs_reference", _never_requires_reference)()
        for metric in metrics
    )


@dataclass
class ChunkOutcome:
    """一个工作块的产出（纯数据，可 pickle，供 resume 复用）。"""

    chunk_id: str
    ok: bool
    error: Optional[str] = None
    states: Dict[Tuple[int, Tuple], List[Any]] = field(default_factory=dict)
    coordinates: Dict[Tuple[int, Tuple], Dict[str, Any]] = field(default_factory=dict)
    processed: int = 0
    skipped: int = 0
    defaults: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergedOutcome:
    """一段全部块归并后的结果（Runner 拿它去收尾和落盘）。"""

    states: Dict[Tuple[int, Tuple], List[Any]] = field(default_factory=dict)
    coordinates: Dict[Tuple[int, Tuple], Dict[str, Any]] = field(default_factory=dict)
    processed: int = 0
    skipped: int = 0
    defaults: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    failures: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 块收集：一个块 = 完整的一段流水线（子声明口径）
# ---------------------------------------------------------------------------


def collect_chunk_states(
    spec: PipelineSpec, loader: Any = None, chunk_label: str = ""
) -> ChunkOutcome:
    """跑一个工作块，返回指标状态（不收尾、不落盘）。

    MetricError / ConfigError 原样抛出：口径与配置错误是全局的，换一块
    重试也没用，必须让整轮失败；其他异常算这一块失败，不拖垮别的块。
    """
    register_builtin_components()
    registry = get_registry()

    forecast = registry.build(
        ComponentType.READER, spec.forecast.reader, **spec.forecast.params
    )
    observation = registry.build(
        ComponentType.READER, spec.observation.reader, **spec.observation.params
    )
    transforms = {
        item.name: registry.build(ComponentType.TRANSFORM, item.name, **item.params)
        for item in spec.transforms
    }
    metrics = [
        registry.build(ComponentType.METRIC, item.name, **item.params)
        for item in spec.metrics
    ]
    reference = None
    if spec.reference is not None and needs_reference(metrics):
        reference = registry.build(
            ComponentType.READER, spec.reference.reader, **spec.reference.params
        )
    runs = metric_runs(spec.metrics, metrics)
    context = PipelineContext(
        spec=spec,
        forecast=forecast,
        observation=observation,
        reference=reference,
        transforms=transforms,
        metrics=metrics,
        loader=loader,
    )

    protocol = registry.build(ComponentType.PROTOCOL, spec.protocol, spec=spec)
    try:
        protocol.prepare(context)
        states, coordinates, processed, skipped = _sample_loop(protocol, context, runs)
    except (ConfigError, MetricError):
        raise
    except Exception as exc:
        label = chunk_label or "块"
        log.exception("工作块 %s 失败: %s", label, exc)
        return ChunkOutcome(chunk_id=label, ok=False, error=repr(exc))

    if processed == 0:
        return ChunkOutcome(
            chunk_id=chunk_label,
            ok=False,
            error="没有成功处理任何评测批次",
            skipped=skipped,
        )
    return ChunkOutcome(
        chunk_id=chunk_label,
        ok=True,
        states=states,
        coordinates=coordinates,
        processed=processed,
        skipped=skipped,
        defaults=protocol.defaults(context),
        summary=protocol.summary(),
    )


def _sample_loop(protocol: Any, context: PipelineContext, runs: List[MetricRun]):
    """样本循环（原 Runner._run_segment 的循环体，块口径不变）。"""
    states: Dict[Tuple[int, Tuple], List[Any]] = {}
    coordinates: Dict[Tuple[int, Tuple], Dict[str, Any]] = {}
    processed = 0
    skipped = 0
    for sample in protocol.samples(context):
        success = False
        try:
            batch = protocol.build_batch(context, sample)
            if batch is not None:
                for index, run in enumerate(runs):
                    # 路由到某个变量的指标只看该变量；数据源缺它就跳过这个 run
                    target = (
                        narrow_batch(batch, run.variable) if run.variable else batch
                    )
                    if target is None:
                        continue
                    run.metric.validate(target)
                    states.setdefault((index, sample.key), []).append(
                        run.metric.accumulate(target)
                    )
                    coordinates.setdefault(
                        (index, sample.key),
                        {
                            **sample.coordinates,
                            **(
                                {"variable": run.variable}
                                if run.variable
                                else {}
                            ),
                        },
                    )
                    success = True
        except MetricError:
            # 指标与输入不匹配属于配置错误（如集合未降维），必须立刻暴露，
            # 不能当成"这条样本坏了"跳过。
            raise
        except Exception as exc:
            log.debug("样本 %s 处理失败: %s", sample.key, exc)

        if success:
            processed += 1
        else:
            skipped += 1
        if processed and processed % 500 == 0:
            log.info("进度：已处理 %d 个样本（跳过 %d）", processed, skipped)
    return states, coordinates, processed, skipped


# ---------------------------------------------------------------------------
# 并发调度：serial / threads / processes
# ---------------------------------------------------------------------------

#: 进程模式下的加载器，走模块全局而不是 submit 参数：fork 时父进程先置位，
#: 子进程经写时复制拿到同一份 resident 缓存（当 submit 参数传会每块序列化
#: 一次缓存）；spawn 时由池的 initializer 反序列化一份副本装进来。
_SHARED_LOADER: Any = None


def _init_shared_loader(loader: Any) -> None:
    """spawn 子进程的池初始化：装一份父进程声明好的加载器。

    fork 与 spawn 的区别是这份 loader 是**反序列化出来的副本**，父子不共享
    内存；换来的是段内的 window 缓存仍然生效（spawn 原本完全没有缓存）。
    """
    global _SHARED_LOADER
    _SHARED_LOADER = loader


def _picklable(value: Any) -> bool:
    """能不能安全 pickle 进子进程——试一次，不靠猜。

    计划期的 loader 只有角色声明（缓存还空着），试序列化很便宜。
    """
    try:
        pickle.dumps(value)
        return True
    except Exception:
        return False


def _remote_collect_range(
    specs: List[PipelineSpec],
    chunk_ids: List[str],
    indices: List[int],
    segment: str,
) -> Tuple[List[Tuple[int, ChunkOutcome]], Dict[str, Any]]:
    """进程池分段任务入口（必须是模块顶层函数才能被 pickle 引用）。

    一个任务 = 一段**时间连续**的块，在同一个子进程里顺序跑完。观测的
    window 缓存在子进程自己的 loader 里：只有同一进程先后处理相邻块，
    滚动窗的重叠数据才命中缓存；反过来把块逐个丢进池里抢（随机顺序），
    每个块都要为不相邻的跨度重新物化窗块，缓存形同虚设。

    ConfigError / MetricError 原样抛回父进程（口径错误换块重试也没用；
    此时本段更早的块结果会丢，口径错本来就该整轮重来）；其余异常由
    collect_chunk_states 内部转成失败块结果，段继续往下跑。

    返回值带一份子进程的加载计时：fork 之后计数器是各自的页，父进程看不见
    （写时复制只共享"没被写过"的页），只能跟着返回值回去。spawn 下
    ``_SHARED_LOADER`` 为 None——子进程压根不走加载层，没有计时可报。

    ``segment`` 是给日志用的段标签（形如 ``"2/5"``）。第一块、每
    ``PROGRESS_EVERY`` 块、以及最后一块各打一行 INFO：段号、段内进度、实测速率、
    按当前速率的外推剩余、最新块号。外推只看本段已跑部分的均值，前面慢后面快时
    它会偏保守；第一块那行的速率含进程启动开销，会偏高，只当"活着"的信号看。
    """
    before = _SHARED_LOADER.stats() if _SHARED_LOADER is not None else {}
    outcomes: List[Tuple[int, ChunkOutcome]] = []
    total = len(indices)
    started = perf_counter()
    for done, (index, spec, chunk_id) in enumerate(
        zip(indices, specs, chunk_ids), start=1
    ):
        outcomes.append(
            (
                index,
                collect_chunk_states(spec, loader=_SHARED_LOADER, chunk_label=chunk_id),
            )
        )
        # 第一块也报：否则一段的开头要闷到第 PROGRESS_EVERY 块才有字，
        # 每块十几秒时那就是十几分钟的静默，看着像卡死。
        if done == 1 or done % PROGRESS_EVERY == 0 or done == total:
            elapsed = perf_counter() - started
            per_chunk = elapsed / done
            log.info(
                "段 %s 进度 %d/%d 块（已用 %.0fs，%.2fs/块，按此速率还需 %.0fs，最新块 %s）",
                segment,
                done,
                total,
                elapsed,
                per_chunk,
                per_chunk * (total - done),
                chunk_id,
            )
    after = _SHARED_LOADER.stats() if _SHARED_LOADER is not None else {}
    return outcomes, _stats_delta(before, after)


def _stats_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """子进程本次分段任务的加载增量（只保留可加的数值键）。

    fork 的子进程**继承**了父进程预热好的计数器（复制的是整个 loader 对象），
    直接回传会把预热耗时按子进程数重复计入。进程内先取一次基线再相减，得到的
    才是真正属于本段的增量；父进程的预热耗时由它自己那份 stats 报，各算一次。
    """
    return {
        key: float(value) - float(before.get(key) or 0.0)
        for key, value in after.items()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    }


def _sum_stats(*stats_maps: Dict[str, Any]) -> Dict[str, Any]:
    """把父进程与各子进程的加载统计按数值键相加（计数与耗时都是可加的）。"""
    total: Dict[str, Any] = {}
    for stats in stats_maps:
        for key, value in stats.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                total.setdefault(key, value)
            else:
                total[key] = total.get(key, 0) + value
    return total


def _log_load_share(workers: int, wall_seconds: float, stats: Dict[str, Any]) -> None:
    """报一次「读盘吃掉了多少 worker 时间」——预取值不值得做的依据。

    分母用 ``墙钟 × worker`` 而不是单份墙钟：并行时各 worker 的读盘是重叠的，
    除以单份墙钟会算出大于 100% 的假数。processes 的父进程预热时间与子进程
    逐块物化时间都会被加进来（前者经 plan.loader，后者经子进程回传）。
    """
    loader_seconds = float(stats.get("builder_seconds") or 0.0)
    if loader_seconds <= 0 or wall_seconds <= 0:
        return
    capacity = wall_seconds * max(1, workers)
    roles = sorted(
        (key.split(":", 1)[1], float(value))
        for key, value in stats.items()
        if key.startswith("builder_seconds:") and value
    )
    log.info(
        "读盘占比：builder 累计 %.1fs / 容量 %.1fs（%.1fs 墙钟 × %d worker）= %.0f%%"
        "；物化 %d 次、缓存命中 %d 次%s",
        loader_seconds,
        capacity,
        wall_seconds,
        max(1, workers),
        100.0 * loader_seconds / capacity,
        int(stats.get("builder_calls") or 0),
        int(stats.get("cache_hits") or 0),
        "；分类 " + "、".join(f"{role} {sec:.1f}s" for role, sec in roles)
        if roles
        else "",
    )


def _partition_ranges(indices: List[int], n: int) -> List[List[int]]:
    """把升序索引切成 n 个连续段（时间分段），段长差不超过 1，空段丢弃。"""
    n = max(1, min(int(n), len(indices)))
    size, extra = divmod(len(indices), n)
    ranges: List[List[int]] = []
    start = 0
    for position in range(n):
        length = size + (1 if position < extra else 0)
        if length:
            ranges.append(list(indices[start : start + length]))
            start += length
    return ranges


def execute_chunks(plan: Any, order: int = 1) -> List[ChunkOutcome]:
    """按策略执行一段的全部工作块，按块序返回结果。"""
    spec = plan.spec
    strategy = plan.strategy
    n_chunks = len(plan.chunks)
    mode = strategy.resolve_mode(n_chunks, plan.profile)
    workers = strategy.resolve_workers(mode)
    resume_dir = Path(spec.output_dir) / ".states" if strategy.resume else None

    log.info(
        "执行形态: %s × %d worker，%d 个工作块，加载策略: %s%s",
        mode,
        workers if mode != "serial" else 1,
        n_chunks,
        strategy.loads or "（逐块直读）",
        "，resume 开启" if resume_dir is not None else "",
    )

    outcomes: List[Optional[ChunkOutcome]] = [None] * n_chunks
    todo = list(range(n_chunks))
    if resume_dir is not None:
        todo = []
        for index, chunk in enumerate(plan.chunks):
            cached = _load_outcome(resume_dir, order, chunk.chunk_id)
            if cached is not None and cached.ok:
                outcomes[index] = cached
                log.info("复用已完成块 %s（resume）", chunk.chunk_id)
            else:
                todo.append(index)
        _warn_orphan_states(
            resume_dir, order, [chunk.chunk_id for chunk in plan.chunks]
        )

    def _done(index: int, outcome: ChunkOutcome) -> ChunkOutcome:
        outcomes[index] = outcome
        if resume_dir is not None and outcome.ok:
            _save_outcome(resume_dir, order, outcome)
        return outcome

    def _collect(index: int) -> ChunkOutcome:
        return collect_chunk_states(
            plan.sub_specs[index], loader=plan.loader, chunk_label=plan.chunks[index].chunk_id
        )

    def _collect_indexed(index: int) -> Tuple[List[Tuple[int, ChunkOutcome]], Dict[str, Any]]:
        """线程任务入口：一块一个任务（共享 loader 有锁，顺序无所谓）。

        加载计时不回传——线程与父进程共用同一个 loader，父进程直接读它。
        """
        return [(index, _collect(index))], {}

    if not todo:
        log.info("全部 %d 个块已有 resume 状态，跳过执行", n_chunks)
    elif mode == "serial":
        started = perf_counter()
        progress = tqdm(
            todo,
            desc=f"{spec.name} 评测块",
            unit="块",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for index in progress:
            _done(index, _collect(index))
        progress.close()
        _log_load_share(1, perf_counter() - started, plan.loader.stats())
    else:
        if mode == "threads":
            pool_obj = ThreadPoolExecutor(max_workers=workers)

            def submit(pool: Any, index: int) -> Future:
                return pool.submit(_collect_indexed, index)

        else:
            global _SHARED_LOADER
            fork = "fork" in multiprocessing.get_all_start_methods()
            initializer: Any = None
            init_args: Tuple[Any, ...] = ()
            if fork:
                # 预热 resident 缓存再 fork：子进程写时复制共享，零序列化
                _SHARED_LOADER = plan.loader
                plan.loader.warm()
                mp_context = multiprocessing.get_context("fork")
            else:
                # spawn（Windows）：没有写时复制，但 loader 现在是可序列化的
                # （角色声明用 partial 而不是闭包），子进程照样能自建一份缓存，
                # 段内相邻块仍然复用窗块——原本这里完全没有缓存，每块重读。
                mp_context = multiprocessing.get_context()
                if _picklable(plan.loader):
                    initializer = _init_shared_loader
                    init_args = (plan.loader,)
                    log.info(
                        "进程模式（spawn）：每个子进程自建加载器，段内窗块复用；"
                        "resident 角色会按段数各存一份（内存 ×段数）"
                    )
                else:
                    log.info(
                        "进程模式在无 fork 平台上退化为逐块直读"
                        "（加载器不可序列化：观测/参考句柄里有 pickle 不了的对象）"
                    )
            # 时间连续分段：一段一个任务、同一子进程顺序跑（理由见
            # _remote_collect_range 的注释——window 缓存靠段内相邻块复用）
            ranges = _partition_ranges(todo, min(workers, len(todo)))
            pool_obj = ProcessPoolExecutor(
                max_workers=len(ranges),
                mp_context=mp_context,
                initializer=initializer,
                initargs=init_args,
            )
            log.info(
                "进程分段：%d 段 × 平均 %.1f 块（段内顺序处理，观测 window 段内复用）",
                len(ranges),
                len(todo) / len(ranges),
            )

            def submit(pool: Any, range_index: int) -> Future:
                segment = ranges[range_index]
                return pool.submit(
                    _remote_collect_range,
                    [plan.sub_specs[i] for i in segment],
                    [plan.chunks[i].chunk_id for i in segment],
                    segment,
                    f"{range_index + 1}/{len(ranges)}",
                )

        started = perf_counter()
        #: 子进程回传的加载计时（threads 下恒为空，共用父进程的 loader）
        worker_stats: Dict[str, Any] = {}
        try:
            with pool_obj as pool:
                futures: Dict[Future, List[int]] = {}
                if mode == "threads":
                    futures = {submit(pool, index): [index] for index in todo}
                else:
                    futures = {
                        submit(pool, i): segment
                        for i, segment in enumerate(ranges)
                    }
                progress = tqdm(
                    total=len(todo),
                    desc=f"{spec.name} 评测块",
                    unit="块",
                    ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )
                for future in as_completed(futures):
                    segment = futures[future]
                    try:
                        pairs, stats = future.result()
                        worker_stats = _sum_stats(worker_stats, stats)
                    except (ConfigError, MetricError):
                        raise
                    except Exception as exc:
                        # 整个任务（线程的单块 / 进程的一整段）在收集之外崩了：
                        # 段内块全部记为失败，不拖垮其他段
                        pairs = [
                            (
                                index,
                                ChunkOutcome(
                                    chunk_id=plan.chunks[index].chunk_id,
                                    ok=False,
                                    error=repr(exc),
                                ),
                            )
                            for index in segment
                        ]
                    for index, outcome in pairs:
                        _done(index, outcome)
                        progress.update(1)
                    progress.set_postfix_str(
                        "失败=%d"
                        % sum(1 for o in outcomes if o is not None and not o.ok)
                    )
                progress.close()
        finally:
            if mode == "processes":
                _SHARED_LOADER = None
        wall = perf_counter() - started
        log.info(
            "块执行完成：%d 个（resume 复用 %d），耗时 %.1fs",
            len(todo),
            n_chunks - len(todo),
            wall,
        )
        # 进程模式：父进程的 stats 管预热，worker_stats 是各子进程回传的增量。
        # 无 fork 平台两边都是空的（父进程没预热、子进程不走加载层），这一行
        # 就不打印——那时也没有可比的读盘占比。
        _log_load_share(
            len(ranges) if mode == "processes" else min(workers, len(todo)),
            wall,
            _sum_stats(plan.loader.stats(), worker_stats),
        )

    merged_list = [outcome for outcome in outcomes if outcome is not None]
    if not any(outcome.ok for outcome in merged_list):
        reasons = "; ".join(
            f"{outcome.chunk_id}: {outcome.error}" for outcome in merged_list[:3]
        )
        raise EvaluationError(
            f"流程 {spec.pipeline or spec.name} 的 {n_chunks} 个工作块全部失败"
            + (f"（前三个原因: {reasons}）" if reasons else "")
        )
    return merged_list


def _outcome_path(resume_dir: Path, order: int, chunk_id: str) -> Path:
    return resume_dir / f"seg{order:02d}_{chunk_id}.pkl"


def _warn_orphan_states(resume_dir: Path, order: int, chunk_ids: List[str]) -> None:
    """点出 ``.states/`` 里本次计划不再请求的块状态。

    起报时刻是现扫出来的（没在配置里显式声明 ``init_times``）时，少一天
    数据就会让计划**静默缩水**：旧状态既不复用也不报错，run 照样"成功"，
    只是少了一段，而 ``failed_chunks`` 是空的，没人抓得住。这里只提醒，
    不删也不改结果。
    """
    prefix = f"seg{order:02d}_"
    expected = set(chunk_ids)
    orphans = sorted(
        path.name[len(prefix) : -len(".pkl")]
        for path in resume_dir.glob(f"{prefix}*.pkl")
        if path.name[len(prefix) : -len(".pkl")] not in expected
    )
    if not orphans:
        return
    head = "、".join(orphans[:3]) + (f" 等 {len(orphans)} 个" if len(orphans) > 3 else "")
    log.warning(
        "resume 目录里有 %d 个块的旧状态不属于本次计划（%s），本次不复用；"
        "若这是计划缩水（而不是切块口径变了），这些块覆盖的时段会静默缺失",
        len(orphans),
        head,
    )


def _save_outcome(resume_dir: Path, order: int, outcome: ChunkOutcome) -> None:
    try:
        resume_dir.mkdir(parents=True, exist_ok=True)
        with _outcome_path(resume_dir, order, outcome.chunk_id).open("wb") as stream:
            pickle.dump(outcome, stream)
    except Exception as exc:
        log.warning("块 %s 的 resume 状态写入失败: %s", outcome.chunk_id, exc)


def _load_outcome(resume_dir: Path, order: int, chunk_id: str) -> Optional[ChunkOutcome]:
    path = _outcome_path(resume_dir, order, chunk_id)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except Exception as exc:
        log.warning("块 %s 的 resume 状态读取失败，重跑该块: %s", chunk_id, exc)
        return None


# ---------------------------------------------------------------------------
# 统一归并
# ---------------------------------------------------------------------------


def _init_sort_key(outcome: Any, key: Any) -> str:
    """同键比较用的起报键：coordinates 里的 ``init_time``。

    协议写进去的是 ISO-8601 串，等宽时字典序即时间序。缺这个坐标时给空串
    （最小），让带起报信息的一方获胜——总比静默留旧的强。
    """
    return str((outcome.coordinates.get(key) or {}).get("init_time") or "")


def merge_outcomes(outcomes: List[ChunkOutcome], spec: PipelineSpec) -> MergedOutcome:
    """把一段全部块的状态按块序归并。

    所有内置指标的状态都是和式（sum/count/moment），归并可交换；唯一的
    例外由协议的 ``duplicate_key_policy`` 声明：replace 语义的协议同键
    只保留**最新起报**那一块（grid 协议重现"最新起报获胜"的配对口径）。
    """
    registry = get_registry()
    protocol = registry.build(ComponentType.PROTOCOL, spec.protocol, spec=spec)
    policy = str(getattr(protocol, "duplicate_key_policy", "append"))

    merged = MergedOutcome()
    for outcome in outcomes:
        if not outcome.ok:
            merged.failures.append((outcome.chunk_id, outcome.error or "未知原因"))
            # 失败块仍计入跳过数，manifest 的口径才对得上"请求了多少"
            merged.skipped += int(outcome.skipped or 0)
            continue
        merged.processed += int(outcome.processed or 0)
        merged.skipped += int(outcome.skipped or 0)
        if not merged.defaults and outcome.defaults:
            merged.defaults = dict(outcome.defaults)
        for key, states in outcome.states.items():
            if policy == "replace" and key in merged.states:
                # 同键只认**最新起报**。不能靠块序：同一个 valid_time 可以由
                # (早起报, 长时效) 和 (晚起报, 短时效) 两条路径够到，而最新起报
                # 配的是最小 lead——按时效窗升序排块时它反而最先落盘。比
                # init_time 就把正确性与块序解耦了。同一 init 时按块序后者胜。
                if _init_sort_key(outcome, key) >= _init_sort_key(merged, key):
                    merged.states[key] = list(states)
                    merged.coordinates[key] = dict(outcome.coordinates.get(key, {}))
            else:
                merged.states.setdefault(key, []).extend(states)
                merged.coordinates.setdefault(
                    key, outcome.coordinates.get(key, {})
                )
        merged.summary = _merge_summary(merged.summary, outcome.summary)
    if merged.failures:
        log.warning(
            "段 %s 有 %d 个工作块失败（其余 %d 个成功块的结果继续）",
            spec.pipeline or spec.name,
            len(merged.failures),
            len(outcomes) - len(merged.failures),
        )
    return merged


def _merge_summary(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """协议 summary 的跨块合并。

    计数键（``n_`` 前缀）跨块相加；列表（输入文件清单、``*_times`` 时刻清单）
    取并集；其余数值与字符串都是口径描述（如 window_hours=24、forecast_reader），
    保留首个——相加反而会把 window_hours 变成"窗口数 × 块数"的假数。

    ``*_times`` 列表的计数**不能**靠相加：切时效后同一个起报/有效时刻会出现在
    多个块里，相加得到的是"块数 × 时刻数"。这里统一由并集后的列表长度得出，
    协议因此报列表而不报 ``n_`` 计数。
    """
    merged = dict(base)
    for key, value in extra.items():
        if key not in merged:
            merged[key] = value
            continue
        current = merged[key]
        if isinstance(current, bool) or isinstance(value, bool):
            continue  # 布尔口径块间一致，保留即可
        if (
            key.startswith("n_")
            and isinstance(current, (int, float))
            and isinstance(value, (int, float))
        ):
            merged[key] = current + value
        elif isinstance(current, list) and isinstance(value, list):
            for item in value:
                if item not in current:
                    current.append(item)
        # 其余类型（字符串、口径数值）块间理应一致，不一致就保留首个
    for key, value in list(merged.items()):
        if isinstance(value, list) and key.endswith("_times"):
            merged[f"n_{key}"] = len(value)
    return merged
