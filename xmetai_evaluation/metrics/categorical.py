"""
分类检验指标

TS (Threat Score / Critical Success Index)
POD (Probability of Detection)
FAR (False Alarm Ratio)
BIAS (Frequency Bias)

基于 2×2 列联表计算
"""

import numpy as np
import xarray as xr
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from xmetai_evaluation.metrics.base import Metric, MetricRequirements, MetricState, ProductType
from xmetai_evaluation.core.contracts import EvaluationBatch, MetricResult, ResultStatus
from xmetai_evaluation.core.errors import MetricError


@dataclass
class ContingencyTable:
    """2×2 列联表"""
    hits: int  # 命中（预报Yes，观测Yes）
    misses: int  # 漏报（预报No，观测Yes）
    false_alarms: int  # 空报（预报Yes，观测No）
    correct_negatives: int  # 正确否定（预报No，观测No）

    @property
    def n_total(self) -> int:
        """有效配对总数（包含 correct_negatives）。"""
        return self.hits + self.misses + self.false_alarms + self.correct_negatives

    def to_dict(self) -> Dict[str, int]:
        """转为字典"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "correct_negatives": self.correct_negatives,
            "n_total": self.n_total,
        }


class TSScore(Metric):
    """
    威胁评分（Threat Score / Critical Success Index）

    TS = hits / (hits + misses + false_alarms)

    同时计算：
    - POD (Probability of Detection) = hits / (hits + misses)
    - FAR (False Alarm Ratio) = false_alarms / (hits + false_alarms)
    - BIAS = (hits + false_alarms) / (hits + misses)

    支持多个阈值同时计算。
    """

    def __init__(self, thresholds: List[Tuple[str, float]], params: Dict[str, Any] = None):
        """
        Args:
            thresholds: 阈值列表，格式 [(name, value), ...]
                       例如 [('≥0.1', 0.1), ('≥10', 10.0), ('≥25', 25.0)]
            params: 可选参数
        """
        super().__init__(name="ts", version="1.0.0", params=params or {})
        self.thresholds = thresholds

    def requirements(self) -> MetricRequirements:
        """TS 需要确定性场"""
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
            can_merge_along=["sample", "time", "init_time", "station"],
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        """
        累积列联表

        Args:
            batch: 评测批次，forecast 和 observation 应该已经对齐（同样的维度和坐标）

        Returns:
            MetricState，包含各阈值的列联表计数
        """
        # 提取数据
        if isinstance(batch.forecast, xr.DataArray):
            forecast = batch.forecast.values
        else:
            forecast = batch.forecast.payload.values

        if isinstance(batch.observation, xr.DataArray):
            observation = batch.observation.values
        else:
            observation = batch.observation.payload.values

        # 应用掩码
        mask = batch.valid_mask.values

        # 展平为一维（只处理有效点）
        fcst_flat = forecast[mask]
        obs_flat = observation[mask]

        # 对每个阈值计算列联表
        tables = {}
        for name, threshold in self.thresholds:
            # 预报和观测的二值化
            fcst_event = fcst_flat >= threshold
            obs_event = obs_flat >= threshold

            # 计算列联表
            hits = int(np.sum(fcst_event & obs_event))
            misses = int(np.sum(~fcst_event & obs_event))
            false_alarms = int(np.sum(fcst_event & ~obs_event))
            correct_negatives = int(np.sum(~fcst_event & ~obs_event))

            tables[name] = ContingencyTable(
                hits=hits,
                misses=misses,
                false_alarms=false_alarms,
                correct_negatives=correct_negatives,
            )

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={"tables": tables},
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        """
        合并多个状态的列联表

        Args:
            states: 状态列表

        Returns:
            合并后的状态
        """
        if not states:
            raise MetricError("Cannot merge empty states list")

        # 验证状态来自同一指标
        first_state = states[0]
        for state in states[1:]:
            if state.metric_name != first_state.metric_name:
                raise MetricError(
                    f"Cannot merge states from different metrics: "
                    f"{first_state.metric_name} vs {state.metric_name}"
                )

        # 合并列联表（逐阈值相加）
        merged_tables = {}
        for name, _ in self.thresholds:
            total_hits = 0
            total_misses = 0
            total_false_alarms = 0
            total_correct_negatives = 0

            for state in states:
                table = state.data["tables"][name]
                total_hits += table.hits
                total_misses += table.misses
                total_false_alarms += table.false_alarms
                total_correct_negatives += table.correct_negatives

            merged_tables[name] = ContingencyTable(
                hits=total_hits,
                misses=total_misses,
                false_alarms=total_false_alarms,
                correct_negatives=total_correct_negatives,
            )

        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={"tables": merged_tables},
            n_accumulated=sum(s.n_accumulated for s in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        """
        从列联表计算最终指标

        Args:
            state: 最终状态

        Returns:
            MetricResult，value 为字典，包含各阈值的指标
        """
        tables = state.data["tables"]

        # 计算各阈值的指标
        results = {}
        for name, threshold in self.thresholds:
            table = tables[name]

            # 计算指标
            hits = table.hits
            misses = table.misses
            false_alarms = table.false_alarms
            n_total = table.n_total

            # TS = hits / (hits + misses + false_alarms)
            denominator_ts = hits + misses + false_alarms
            ts = float(hits) / denominator_ts if denominator_ts > 0 else np.nan

            # POD = hits / (hits + misses)
            denominator_pod = hits + misses
            pod = float(hits) / denominator_pod if denominator_pod > 0 else np.nan

            # FAR = false_alarms / (hits + false_alarms)
            denominator_far = hits + false_alarms
            far = float(false_alarms) / denominator_far if denominator_far > 0 else np.nan

            # BIAS = (hits + false_alarms) / (hits + misses)
            bias = float(hits + false_alarms) / denominator_pod if denominator_pod > 0 else np.nan

            results[name] = {
                "threshold": threshold,
                "threshold_name": name,
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": table.correct_negatives,
                "n_pairs": n_total,
                "TS": ts,
                "POD": pod,
                "FAR": far,
                "miss_rate": 1.0 - pod if not np.isnan(pod) else np.nan,
                "BIAS": bias,
            }

        # 确定整体状态
        if all(r["n_pairs"] == 0 for r in results.values()):
            status = ResultStatus.NO_VALID_DATA
        elif any(np.isnan(r["TS"]) for r in results.values()):
            status = ResultStatus.PARTIAL
        else:
            status = ResultStatus.SUCCESS

        # 计算总有效样本数
        total_n_valid = sum(r["n_pairs"] for r in results.values())

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=results,  # 字典，每个阈值的结果
            status=status,
            n_requested=total_n_valid,
            n_valid=total_n_valid,
            aggregation="contingency_table",
        )
