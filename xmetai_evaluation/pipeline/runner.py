# -*- coding: utf-8 -*-
"""唯一执行内核。

整个框架只有一处编排：这里。它把一段声明交给执行层（``execution/``）切成
工作块、按策略并发执行、统一归并，然后做指标收尾与一次性落盘。样本循环
本体在 ``execution/executor.py``（块口径），协议里不允许出现循环。

    plan  = build_plan(spec)                 # 起报探测、切块、加载声明
    merged = execute_chunks(plan) 合并        # 各块流水线 + 统一归并
    results = [metric.finalize(metric.merge(states)) for ...]
    store.write(results)
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm import tqdm

from xmetai_evaluation.components import register_builtin_components
from xmetai_evaluation.core.contracts import MetricResult, ResultBundle
from xmetai_evaluation.core.errors import EvaluationError
from xmetai_evaluation.execution import (
    MergedOutcome,
    WorkPlan,
    build_plan,
    execute_chunks,
    merge_outcomes,
    metric_runs,
    needs_reference,
)
from xmetai_evaluation.execution.executor import MetricRun
from xmetai_evaluation.pipeline.spec import PipelineSpec
from xmetai_evaluation.output import ResultStore, RunContext

log = logging.getLogger(__name__)


class Runner:
    """按声明执行一次评测。

    可以一次跑多段（``spec`` 传列表）：每段有自己的流程模板与协议，结果合并后
    统一落盘到同一个 ``output_dir``——长表靠 ``window_h`` 之类的列区分是哪一段，
    所以配置写 ``pipeline=["a", "b"]`` 就只有一份 scores.csv。

    Args:
        spec: 一段或一段列表（来自 ``specs_from_config``）。
        execution: 配置里的 ``execution`` 字典（并发与数据加载覆盖项；
            不写就按指标族与数据源形态推导，见 ``execution/strategy.py``）。
        num_workers: 旧字段 ``EvalConfig.num_workers``：execution 没给
            n_workers 且它 > 1 时当作 n_workers 用。
    """

    def __init__(
        self,
        spec: Union[PipelineSpec, Sequence[PipelineSpec]],
        execution: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
    ):
        self.specs: List[PipelineSpec] = (
            list(spec) if isinstance(spec, (list, tuple)) else [spec]
        )
        if not self.specs:
            raise EvaluationError("没有要执行的流程段")
        #: 第一段。单段时就是它；多段时只用来取两段共有的东西
        #: （output_dir / name / description / 数据源），各段的算法口径一律看段自己。
        self.spec = self.specs[0]
        self.execution = dict(execution or {})
        self.num_workers = int(num_workers or 1)

    def run(self) -> ResultBundle:
        """执行评测并写出标准产物。

        Returns:
            ResultBundle（同时已落盘到 ``spec.output_dir``）。

        Raises:
            EvaluationError: 某一段一个有效评测批次都没跑出来。宁可整轮报错，
                也不静默少一段结果。
        """
        register_builtin_components()
        started = perf_counter()

        results: List[MetricResult] = []
        plans: List[WorkPlan] = []
        merged_segments: List[MergedOutcome] = []
        processed = 0
        skipped = 0
        for order, spec in enumerate(self.specs, start=1):
            plan = build_plan(
                spec,
                execution=self.execution,
                num_workers=self.num_workers,
                order=order,
                total=len(self.specs),
            )
            outcomes = execute_chunks(plan, order)
            # 这一刻 24 个子进程已经全退了，后面三步（归并 / 指标收尾 / 落盘）
            # 全在本进程单线程跑，峰值内存也全压在本进程头上——没有这行的话，
            # 收尾期间日志里一个字都没有，看着就像"卡住了"。
            log.info(
                "第 %d/%d 段 %s 进入收尾：归并 %d 个块的状态，随后收尾指标并落盘",
                order,
                len(self.specs),
                spec.pipeline or spec.name,
                len(outcomes),
            )
            merged = merge_outcomes(outcomes, spec)
            if merged.processed == 0:
                raise EvaluationError(
                    f"流程 {spec.pipeline or spec.name} 没有成功处理任何评测批次"
                )
            runs = metric_runs(spec.metrics, plan.metrics)
            results.extend(
                self._finalize(
                    spec, runs, merged.states, merged.coordinates, merged.defaults
                )
            )
            plans.append(plan)
            merged_segments.append(merged)
            processed += merged.processed
            skipped += merged.skipped

        return self._persist(plans, merged_segments, results, processed, skipped, started)

    def _finalize(
        self,
        spec: PipelineSpec,
        runs: List[MetricRun],
        states: Dict[Tuple[int, Tuple], List[Any]],
        coordinates: Dict[Tuple[int, Tuple], Dict[str, Any]],
        defaults: Dict[str, Any],
    ) -> List[MetricResult]:
        """指标收尾：每键合并状态 -> finalize -> 补长表坐标。"""
        results: List[MetricResult] = []
        # 进度按 (指标, 键) 的总数报：这条循环在一段几万个键时要跑几分钟，
        # 期间一个字都不出的话，日志和"卡死"没法区分。
        progress = tqdm(
            total=len(states),
            desc=f"{spec.pipeline or spec.name} 指标收尾",
            unit="项",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        try:
            for index, run in enumerate(runs):
                keys = sorted(
                    key for (metric_index, key) in states if metric_index == index
                )
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
                    result.coordinates = {
                        **defaults,
                        **coordinates.get((index, key), {}),
                    }
                    result.product_kind = result.product_kind or getattr(
                        run.metric, "PRODUCT_KIND", ""
                    )
                    result.unit = result.unit or str(defaults.get("unit") or "")
                    result.protocol_id = result.protocol_id or spec.protocol
                    results.append(result)
                    progress.update(1)
        finally:
            progress.close()
        if not results:
            raise EvaluationError("没有可输出的指标结果")
        return results

    def _persist(
        self,
        plans: List[WorkPlan],
        merged_segments: List[MergedOutcome],
        results: List[MetricResult],
        processed: int,
        skipped: int,
        started: float,
    ) -> ResultBundle:
        """把各段的结果合并成一次落盘。

        只能写一次：``ResultStore`` 是平铺写 ``output_dir`` 的（没有 run_id 子
        目录），两段各写一次 scores.csv / manifest.json 会互相覆盖。
        """
        spec = self.spec
        forecast = plans[0].handles["forecast"]
        observation = plans[0].handles["observation"]
        merged = merged_segments[0]
        # 视图取各段的并集：24h 段要宽表 A，6h 段要宽表 B，合并后两个都要出。
        # 每个宽表 writer 自己按 product_kind 过滤，所以只会取到属于它的那一段。
        writers: List[str] = []
        for item in self.specs:
            for name in item.output_writers:
                if name not in writers:
                    writers.append(name)
        store = ResultStore(
            Path(spec.output_dir),
            RunContext(
                run_id=spec.name,
                model_id=forecast.source_id,
                dataset_id=observation.source_id,
                protocol_id=spec.protocol,
            ),
            defaults=merged.defaults,
        )
        manifest = {
            "description": spec.description,
            # 各段之和才与 scores_rows 对得上
            "n_processed_batches": processed,
            "n_skipped_batches": skipped,
            "forecast_source": forecast.source_id,
            "observation_source": observation.source_id,
            # 执行口径进 manifest：结果可比的前提是知道它是在什么策略下算的
            "execution": self._execution_summary(plans),
            **merged.summary,
        }
        failures = [
            {"chunk": chunk_id, "error": reason}
            for segment in merged_segments
            for chunk_id, reason in segment.failures
        ]
        if failures:
            manifest["failed_chunks"] = failures
        if len(self.specs) > 1:
            manifest["segments"] = [
                {
                    "pipeline": item.pipeline,
                    "protocol": item.protocol,
                    "window_h": item.window_hours,
                }
                for item in self.specs
            ]
        resolved_config = asdict(spec)
        log.info(
            "写入产物：%d 个指标结果，视图 %s -> %s",
            len(results),
            "、".join(writers) or "（无）",
            spec.output_dir,
        )
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
            run_id=spec.name,
            results=results,
            manifest=manifest,
            resolved_config=resolved_config,
        )

    @staticmethod
    def _execution_summary(plans: List[WorkPlan]) -> Dict[str, Any]:
        """manifest 里的执行口径摘要（按段）。"""
        summary: Dict[str, Any] = {}
        for plan in plans:
            mode = plan.strategy.resolve_mode(len(plan.chunks), plan.profile)
            summary[plan.spec.pipeline or plan.spec.name] = {
                "mode": mode,
                "n_workers": plan.strategy.resolve_workers(mode),
                "chunk_days": plan.strategy.chunk_days,
                "lead_chunk_days": plan.strategy.lead_chunk_days,
                "n_chunks": len(plan.chunks),
                "loads": dict(plan.strategy.loads),
                "profile": plan.profile.as_dict(),
            }
        return summary
