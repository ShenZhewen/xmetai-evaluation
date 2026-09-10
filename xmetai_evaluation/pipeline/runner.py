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
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from xmetai_evaluation.components import register_builtin_components
from xmetai_evaluation.core.contracts import MetricResult, ResultBundle
from xmetai_evaluation.core.errors import EvaluationError, MetricError
from xmetai_evaluation.core.registry import ComponentType, get_registry
from xmetai_evaluation.pipeline.protocols import PipelineContext
from xmetai_evaluation.pipeline.spec import PipelineSpec
from xmetai_evaluation.results import ResultStore, RunContext

log = logging.getLogger(__name__)


class Runner:
    """按声明执行一次评测。"""

    def __init__(self, spec: PipelineSpec):
        self.spec = spec

    def run(self) -> ResultBundle:
        """执行评测并写出标准产物。

        Returns:
            ResultBundle（同时已落盘到 ``spec.output_dir``）。

        Raises:
            EvaluationError: 没有任何有效评测批次。
        """
        register_builtin_components()
        registry = get_registry()
        started = perf_counter()
        log.info("=" * 80)
        log.info("开始评测任务: %s", self.spec.name)
        log.info("描述: %s", self.spec.description)
        log.info(
            "协议: %s；指标: %s",
            self.spec.protocol,
            [item.name for item in self.spec.metrics],
        )

        forecast = registry.build(
            ComponentType.READER, self.spec.forecast.reader, **self.spec.forecast.params
        )
        observation = registry.build(
            ComponentType.READER,
            self.spec.observation.reader,
            **self.spec.observation.params,
        )
        reference = None
        if self.spec.reference is not None:
            reference = registry.build(
                ComponentType.READER,
                self.spec.reference.reader,
                **self.spec.reference.params,
            )
        transforms = {
            item.name: registry.build(ComponentType.TRANSFORM, item.name, **item.params)
            for item in self.spec.transforms
        }
        metrics = [
            registry.build(ComponentType.METRIC, item.name, **item.params)
            for item in self.spec.metrics
        ]
        context = PipelineContext(
            spec=self.spec,
            forecast=forecast,
            observation=observation,
            reference=reference,
            transforms=transforms,
            metrics=metrics,
        )

        protocol = registry.build(ComponentType.PROTOCOL, self.spec.protocol, spec=self.spec)
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
                        for index, metric in enumerate(metrics):
                            metric.validate(batch)
                            states.setdefault((index, sample.key), []).append(
                                metric.accumulate(batch)
                            )
                            coordinates.setdefault(
                                (index, sample.key), dict(sample.coordinates)
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
            raise EvaluationError("没有成功处理任何评测批次")

        results = self._finalize(protocol, context, states, coordinates)
        return self._persist(context, protocol, results, processed, skipped, started)

    def _finalize(
        self,
        protocol: Any,
        context: PipelineContext,
        states: Dict[Tuple[int, Tuple], List[Any]],
        coordinates: Dict[Tuple[int, Tuple], Dict[str, Any]],
    ) -> List[MetricResult]:
        defaults = protocol.defaults(context)
        results: List[MetricResult] = []
        for index, metric in enumerate(context.metrics):
            keys = sorted(key for (metric_index, key) in states if metric_index == index)
            for key in keys:
                merged = metric.merge(states[(index, key)])
                result = metric.finalize(merged)
                result.coordinates = {**defaults, **coordinates.get((index, key), {})}
                result.product_kind = result.product_kind or getattr(
                    metric, "PRODUCT_KIND", ""
                )
                result.unit = result.unit or str(defaults.get("unit") or "")
                result.protocol_id = result.protocol_id or self.spec.protocol
                results.append(result)
        if not results:
            raise EvaluationError("没有可输出的指标结果")
        return results

    def _persist(
        self,
        context: PipelineContext,
        protocol: Any,
        results: List[MetricResult],
        processed: int,
        skipped: int,
        started: float,
    ) -> ResultBundle:
        forecast = context.forecast
        observation = context.observation
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
            "n_processed_batches": processed,
            "n_skipped_batches": skipped,
            "forecast_source": forecast.source_id,
            "observation_source": observation.source_id,
            **protocol.summary(),
        }
        resolved_config = asdict(self.spec)
        artifacts = store.write(
            results,
            manifest=manifest,
            resolved_config=resolved_config,
            writers=self.spec.output_writers,
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
