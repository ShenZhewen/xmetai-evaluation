"""
测试 TS 分类指标

验证列联表计算和指标计算
"""

import pytest
import numpy as np
import xarray as xr

from xmetai_evaluation.metrics.categorical import TSScore, ContingencyTable
from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus


class TestContingencyTable:
    """测试列联表"""

    def test_contingency_table_basic(self):
        """测试基本列联表"""
        table = ContingencyTable(
            hits=10,
            misses=5,
            false_alarms=3,
            correct_negatives=82,
        )

        # n_total 不含 correct_negatives（符合气象业务规范）
        assert table.n_total == 18  # hits + misses + false_alarms
        assert table.to_dict()["hits"] == 10


class TestTSScore:
    """测试 TS 指标"""

    def test_ts_single_threshold(self):
        """测试单个阈值的 TS 计算"""
        # 创建简单数据
        # 预报：[0, 1, 2, 5, 10, 15]
        # 观测：[0, 0, 3, 4, 11, 14]
        # 阈值 ≥5：预报[F,F,F,T,T,T]，观测[F,F,F,F,T,F]
        # hits=1 (位置4: 10≥5 且 11≥5), misses=0, false_alarms=2 (位置3: 5≥5但4<5, 位置5: 15≥5但14<5)
        # 实际上位置5: 15≥5 且 14≥5，所以也是 hit
        # 正确的：hits=2 (位置4,5), misses=0, false_alarms=1 (位置3)
        fcst = xr.DataArray(
            np.array([0, 1, 2, 5, 10, 15], dtype=float),
            dims=["sample"],
        )
        obs = xr.DataArray(
            np.array([0, 0, 3, 4, 11, 14], dtype=float),
            dims=["sample"],
        )
        mask = xr.DataArray(
            np.ones(6, dtype=bool),
            dims=["sample"],
        )

        batch = EvaluationBatch(
            forecast=fcst,
            observation=obs,
            sample_keys=[{"id": i} for i in range(6)],
            valid_mask=mask,
            alignment={"method": "direct"},
        )

        # 计算 TS
        ts_metric = TSScore(thresholds=[("≥5", 5.0)])
        result = ts_metric.compute(batch)

        assert result.status == ResultStatus.SUCCESS
        assert "≥5" in result.value

        metrics = result.value["≥5"]
        assert metrics["hits"] == 2  # 位置 4,5
        assert metrics["misses"] == 0
        assert metrics["false_alarms"] == 1  # 位置 3

        # TS = 2 / (2+0+1) = 0.667
        assert metrics["TS"] == pytest.approx(2.0 / 3.0)
        # POD = 2 / (2+0) = 1.0
        assert metrics["POD"] == pytest.approx(1.0)
        # FAR = 1 / (2+1) = 0.333
        assert metrics["FAR"] == pytest.approx(1.0 / 3.0)

    def test_ts_multiple_thresholds(self):
        """测试多个阈值"""
        fcst = xr.DataArray(
            np.array([0, 5, 10, 15, 20, 25], dtype=float),
            dims=["sample"],
        )
        obs = xr.DataArray(
            np.array([0, 6, 9, 16, 19, 26], dtype=float),
            dims=["sample"],
        )
        mask = xr.DataArray(
            np.ones(6, dtype=bool),
            dims=["sample"],
        )

        batch = EvaluationBatch(
            forecast=fcst,
            observation=obs,
            sample_keys=[{"id": i} for i in range(6)],
            valid_mask=mask,
            alignment={"method": "direct"},
        )

        ts_metric = TSScore(thresholds=[("≥0.1", 0.1), ("≥10", 10.0), ("≥20", 20.0)])
        result = ts_metric.compute(batch)

        assert result.status == ResultStatus.SUCCESS
        assert len(result.value) == 3
        assert "≥0.1" in result.value
        assert "≥10" in result.value
        assert "≥20" in result.value

        # 阈值 ≥0.1：几乎所有点都超过
        assert result.value["≥0.1"]["hits"] >= 4

    def test_ts_perfect_forecast(self):
        """测试完美预报"""
        fcst = xr.DataArray(
            np.array([0, 10, 20, 30], dtype=float),
            dims=["sample"],
        )
        obs = fcst.copy()  # 完美预报
        mask = xr.DataArray(
            np.ones(4, dtype=bool),
            dims=["sample"],
        )

        batch = EvaluationBatch(
            forecast=fcst,
            observation=obs,
            sample_keys=[{"id": i} for i in range(4)],
            valid_mask=mask,
            alignment={"method": "direct"},
        )

        ts_metric = TSScore(thresholds=[("≥10", 10.0)])
        result = ts_metric.compute(batch)

        metrics = result.value["≥10"]
        # 完美预报：TS=1, POD=1, FAR=0, BIAS=1
        assert metrics["TS"] == pytest.approx(1.0)
        assert metrics["POD"] == pytest.approx(1.0)
        assert metrics["FAR"] == pytest.approx(0.0)
        assert metrics["BIAS"] == pytest.approx(1.0)

    def test_ts_with_mask(self):
        """测试带掩码的计算"""
        fcst = xr.DataArray(
            np.array([0, 10, 20, 30, 40], dtype=float),
            dims=["sample"],
        )
        obs = xr.DataArray(
            np.array([0, 11, 19, 31, 39], dtype=float),
            dims=["sample"],
        )
        # 掩码掉第 3 个点
        mask = xr.DataArray(
            np.array([True, True, False, True, True], dtype=bool),
            dims=["sample"],
        )

        batch = EvaluationBatch(
            forecast=fcst,
            observation=obs,
            sample_keys=[{"id": i} for i in range(5)],
            valid_mask=mask,
            alignment={"method": "direct"},
        )

        ts_metric = TSScore(thresholds=[("≥10", 10.0)])
        result = ts_metric.compute(batch)

        # 只有 4 个有效点（第 3 个被掩码），但 n_pairs 不含 correct_negatives
        # hits=3, misses=0, false_alarms=0, correct_negatives=1
        # n_pairs = 3（不含 CN）
        metrics = result.value["≥10"]
        assert metrics["n_pairs"] == 3

    def test_ts_merge_states(self):
        """测试状态合并"""
        # 创建两个批次
        fcst1 = xr.DataArray(np.array([0, 10, 20], dtype=float), dims=["sample"])
        obs1 = xr.DataArray(np.array([0, 11, 19], dtype=float), dims=["sample"])
        mask1 = xr.DataArray(np.ones(3, dtype=bool), dims=["sample"])

        batch1 = EvaluationBatch(
            forecast=fcst1,
            observation=obs1,
            sample_keys=[{"id": i} for i in range(3)],
            valid_mask=mask1,
            alignment={"method": "direct"},
        )

        fcst2 = xr.DataArray(np.array([30, 40], dtype=float), dims=["sample"])
        obs2 = xr.DataArray(np.array([31, 39], dtype=float), dims=["sample"])
        mask2 = xr.DataArray(np.ones(2, dtype=bool), dims=["sample"])

        batch2 = EvaluationBatch(
            forecast=fcst2,
            observation=obs2,
            sample_keys=[{"id": i} for i in range(2)],
            valid_mask=mask2,
            alignment={"method": "direct"},
        )

        # 分别计算
        ts_metric = TSScore(thresholds=[("≥10", 10.0)])
        state1 = ts_metric.accumulate(batch1)
        state2 = ts_metric.accumulate(batch2)

        # 合并
        merged_state = ts_metric.merge([state1, state2])
        merged_result = ts_metric.finalize(merged_state)

        # 验证合并结果（n_pairs 不含 correct_negatives）
        # batch1: hits=2, misses=0, false_alarms=0, CN=1 -> n_pairs=2
        # batch2: hits=2, misses=0, false_alarms=0, CN=0 -> n_pairs=2
        # merged: hits=4, misses=0, false_alarms=0, CN=1 -> n_pairs=4
        assert merged_result.value["≥10"]["n_pairs"] == 4

    def test_ts_no_valid_data(self):
        """测试无有效数据"""
        fcst = xr.DataArray(np.array([0, 10], dtype=float), dims=["sample"])
        obs = xr.DataArray(np.array([0, 11], dtype=float), dims=["sample"])
        mask = xr.DataArray(np.array([False, False], dtype=bool), dims=["sample"])

        batch = EvaluationBatch(
            forecast=fcst,
            observation=obs,
            sample_keys=[{"id": i} for i in range(2)],
            valid_mask=mask,
            alignment={"method": "direct"},
        )

        ts_metric = TSScore(thresholds=[("≥10", 10.0)])
        result = ts_metric.compute(batch)

        assert result.status == ResultStatus.NO_VALID_DATA
        assert result.value["≥10"]["n_pairs"] == 0
