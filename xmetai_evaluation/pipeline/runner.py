# -*- coding: utf-8 -*-
"""唯一执行内核。

整个框架只有一个样本循环、一处状态合并、一处落盘调用：都在这里。
pipeline / 协议 / 指标都不允许再写循环。

    for sample in protocol.samples(context):
        batch = protocol.build_batch(context, sample)
        for metric in metrics:
            metric.validate(batch); states[...].append(metric.accumulate(batch))
    results = [metric.finalize(metric.merge(states)) for ...]
    store.write(results)
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from tqdm import tqdm

from xmetai_evaluation.components import register_builtin_components
from xmetai_evaluation.core.contracts import MetricResult, ResultBundle
from xmetai_evaluation.core.errors import EvaluationError, MetricError
from xmetai_evaluation.core.registry import ComponentType, get_registry
from xmetai_evaluation.pipeline.matcher import narrow_batch
from xmetai_evaluation.pipeline.protocols import PipelineContext
from xmetai_evaluation.pipeline.spec import PipelineSpec
from xmetai_evaluation.results import ResultStore, RunContext

log = logging.getLogger(__name__)


class MetricRun(NamedTuple):
    """一次指标执行：指标对象 + 这次执行针对的变量（None 表示整批，不做路由）。"""

    metric: Any
    variable: Optional[str]
    label: str


def _metric_runs(specs: List[Any], metrics: List[Any]) -> List[MetricRun]:
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


def _needs_reference(metrics: List[Any]) -> bool:
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


class Runner:
    """按声明执行一次评测。

    可以一次跑多段（``spec`` 传列表）：每段有自己的流程模板与协议，结果合并后
    统一落盘到同一个 ``output_dir``——长表靠 ``window_h`` 之类的列区分是哪一段，
    所以配置写 ``pipeline=["a", "b"]`` 就只有一份 scores.csv。
    """

    def __init__(self, spec: Union[PipelineSpec, Sequence[PipelineSpec]]):
        self.specs: List[PipelineSpec] = (
            list(spec) if isinstance(spec, (list, tuple)) else [spec]
        )
        if not self.specs:
            raise EvaluationError("没有要执行的流程段")
        #: 第一段。单段时就是它；多段时只用来取两段共有的东西
        #: （output_dir / name / description / 数据源），各段的算法口径一律看段自己。
        self.spec = self.specs[0]

    def run(self) -> ResultBundle:
        """执行评测并写出标准产物。

        Returns:
            ResultBundle（同时已落盘到 ``spec.output_dir``）。

        Raises:
            EvaluationError: 某一段一个有效评测批次都没跑出来。宁可整轮报错，
                也不静默少一段结果。
        """
        register_builtin_components()
        registry = get_registry()
        started = perf_counter()

        results: List[MetricResult] = []
        contexts: List[PipelineContext] = []
        protocols: List[Any] = []
        processed = 0
        skipped = 0
        for order, spec in enumerate(self.specs, start=1):
            segment = self._run_segment(spec, registry, order, len(self.specs))
            results.extend(segment[0])
            contexts.append(segment[1])
            protocols.append(segment[2])
            processed += segment[3]
            skipped += segment[4]

        return self._persist(contexts, protocols, results, processed, skipped, started)

    def _run_segment(
        self, spec: PipelineSpec, registry: Any, order: int, total: int
    ) -> Tuple[List[MetricResult], PipelineContext, Any, int, int]:
        """跑一段（一套流程模板 + 一份数据）：准备、配对、累积、收尾。

        返回 ``(结果, 上下文, 协议, 成功样本数, 跳过样本数)``。
        """
        log.info("=" * 80)
        if total > 1:
            log.info(
                "开始评测任务: %s（第 %d/%d 段: %s，窗口 %dh）",
                spec.name,
                order,
                total,
                spec.pipeline,
                spec.window_hours,
            )
        else:
            log.info("开始评测任务: %s", spec.name)
            log.info("描述: %s", spec.description)
        log.info("协议: %s", spec.protocol)

        forecast = registry.build(
            ComponentType.READER, spec.forecast.reader, **spec.forecast.params
        )
        observation = registry.build(
            ComponentType.READER,
            spec.observation.reader,
            **spec.observation.params,
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
        if spec.reference is not None and _needs_reference(metrics):
            reference = registry.build(
                ComponentType.READER,
                spec.reference.reader,
                **spec.reference.params,
            )
        elif spec.reference is not None:
            log.info("本段指标都不需要参考源，跳过构建 %s", spec.reference.reader)
        runs = _metric_runs(spec.metrics, metrics)
        log.info("指标: %s", [run.label for run in runs])
        context = PipelineContext(
            spec=spec,
            forecast=forecast,
            observation=observation,
            reference=reference,
            transforms=transforms,
            metrics=metrics,
        )

        protocol = registry.build(ComponentType.PROTOCOL, spec.protocol, spec=spec)
        protocol.prepare(context)

        states: Dict[Tuple[int, Tuple], List[Any]] = {}
        coordinates: Dict[Tuple[int, Tuple], Dict[str, Any]] = {}
        processed = 0
        skipped = 0

        progress = tqdm(
            desc="评测进度",
            unit="样本",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        try:
            for sample in protocol.samples(context):
                expected = protocol.expected_samples()
                if expected and progress.total != expected:
                    progress.total = expected
                    progress.refresh()

                success = False
                try:
                    batch = protocol.build_batch(context, sample)
                    if batch is not None:
                        for index, run in enumerate(runs):
                            # 路由到某个变量的指标只看该变量；数据源缺它就跳过这个 run
                            target = (
                                narrow_batch(batch, run.variable)
                                if run.variable
                                else batch
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
                progress.update(1)
                progress.set_postfix_str(f"成功={processed} 跳过={skipped}")
                if processed and processed % 500 == 0:
                    log.info(
                        "进度：已处理 %d 个样本（跳过 %d），最近样本 %s",
                        processed,
                        skipped,
                        sample.key,
                    )
        finally:
            progress.close()

        log.info("样本处理完成：成功=%d，跳过=%d", processed, skipped)
        if processed == 0:
            raise EvaluationError(
                f"流程 {spec.pipeline or spec.name} 没有成功处理任何评测批次"
            )

        results = self._finalize(spec, protocol, context, runs, states, coordinates)
        return results, context, protocol, processed, skipped

    def _finalize(
        self,
        spec: PipelineSpec,
        protocol: Any,
        context: PipelineContext,
        runs: List[MetricRun],
        states: Dict[Tuple[int, Tuple], List[Any]],
        coordinates: Dict[Tuple[int, Tuple], Dict[str, Any]],
    ) -> List[MetricResult]:
        defaults = protocol.defaults(context)
        results: List[MetricResult] = []
        for index, run in enumerate(runs):
            keys = sorted(key for (metric_index, key) in states if metric_index == index)
            if not keys:
                # 路由到某个变量却一个结果都没有，通常是数据源里没这个变量——
                # 不告警的话表现是"结果里静静少了一项"。
                log.warning(
                    "指标 %s 没有产生任何结果%s（数据源里没有该变量？）",
                    run.metric.name,
                    f"（变量 {run.variable}）" if run.variable else "",
                )
                continue
            for key in keys:
                merged = run.metric.merge(states[(index, key)])
                result = run.metric.finalize(merged)
                result.coordinates = {**defaults, **coordinates.get((index, key), {})}
                result.product_kind = result.product_kind or getattr(
                    run.metric, "PRODUCT_KIND", ""
                )
                result.unit = result.unit or str(defaults.get("unit") or "")
                result.protocol_id = result.protocol_id or spec.protocol
                results.append(result)
        if not results:
            raise EvaluationError("没有可输出的指标结果")
        return results

    def _persist(
        self,
        contexts: List[PipelineContext],
        protocols: List[Any],
        results: List[MetricResult],
        processed: int,
        skipped: int,
        started: float,
    ) -> ResultBundle:
        """把各段的结果合并成一次落盘。

        只能写一次：``ResultStore`` 是平铺写 ``output_dir`` 的（没有 run_id 子
        目录），两段各写一次 scores.csv / manifest.json 会互相覆盖。
        """
        context = contexts[0]
        protocol = protocols[0]
        forecast = context.forecast
        observation = context.observation
        # 视图取各段的并集：24h 段要宽表 A，6h 段要宽表 B，合并后两个都要出。
        # 每个宽表 writer 自己按 product_kind 过滤，所以只会取到属于它的那一段。
        writers: List[str] = []
        for spec in self.specs:
            for name in spec.output_writers:
                if name not in writers:
                    writers.append(name)
        store = ResultStore(
            Path(self.spec.output_dir),
            RunContext(
                run_id=self.spec.name,
                model_id=forecast.source_id,
                dataset_id=observation.source_id,
                protocol_id=self.spec.protocol,
            ),
            defaults=protocol.defaults(context),
        )
        manifest = {
            "description": self.spec.description,
            # 各段之和才与 scores_rows 对得上
            "n_processed_batches": processed,
            "n_skipped_batches": skipped,
            "forecast_source": forecast.source_id,
            "observation_source": observation.source_id,
            **protocol.summary(),
        }
        if len(self.specs) > 1:
            manifest["segments"] = [
                {
                    "pipeline": spec.pipeline,
                    "protocol": spec.protocol,
                    "window_h": spec.window_hours,
                }
                for spec in self.specs
            ]
        resolved_config = asdict(self.spec)
        artifacts = store.write(
            results,
            manifest=manifest,
            resolved_config=resolved_config,
            writers=writers,
        )
        for name, path in artifacts.items():
            log.info("产物 %s -> %s", name, path)
        log.info("评测完成：耗时 %.1fs", perf_counter() - started)
        return ResultBundle(
            run_id=self.spec.name,
            results=results,
            manifest=manifest,
            resolved_config=resolved_config,
        )
