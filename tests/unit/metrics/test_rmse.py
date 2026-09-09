"""
测试 Metric 基类和 RMSE 实现

验证整批与分块合并一致性（README Change 3 关键要求）
"""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime

from xmetai_evaluation.metrics.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
)
from xmetai_evaluation.metrics.rmse import RMSE
from xmetai_evaluation.core.contracts import (
    EvaluationBatch,
    MetricResult,
    ResultStatus,
)
from xmetai_evaluation.core.errors import MetricError


def create_synthetic_batch(
    n_samples: int = 10,
    n_lat: int = 5,
    n_lon: int = 5,
    forecast_bias: float = 0.0,
    forecast_noise: float = 1.0,
    seed: int = 42,
) -> EvaluationBatch:
    """
    创建合成评测批次

    Args:
        n_samples: 样本数
        n_lat: 纬度数
        n_lon: 经度数
        forecast_bias: 预报偏差
        forecast_noise: 预报噪声标准差
        seed: 随机种子

    Returns:
        EvaluationBatch
    """
    np.random.seed(seed)

    # 创建观测场（真值）
    obs_data = np.random.randn(n_samples, n_lat, n_lon) * 10 + 20  # 均值20，标准差10
    observation = xr.DataArray(
        obs_data,
        dims=["sample", "lat", "lon"],
        coords={
            "sample": range(n_samples),
            "lat": np.linspace(-10, 10, n_lat),
            "lon": np.linspace(100, 120, n_lon),
        },
    )

    # 创建预报场（观测 + 偏差 + 噪声）
    forecast_data = obs_data + forecast_bias + np.random.randn(n_samples, n_lat, n_lon) * forecast_noise
    forecast = xr.DataArray(
        forecast_data,
        dims=["sample", "lat", "lon"],
        coords=observation.coords,
    )

    # 创建掩码（全部有效）
    valid_mask = xr.DataArray(
        np.ones((n_samples, n_lat, n_lon), dtype=bool),
        dims=["sample", "lat", "lon"],
        coords=observation.coords,
    )

    # 创建样本键
    sample_keys = [{"sample_id": i} for i in range(n_samples)]

    # 创建对齐信息
    alignment = {
        "method": "direct",
        "n_matched": n_samples * n_lat * n_lon,
    }

    return EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=sample_keys,
        valid_mask=valid_mask,
        alignment=alignment,
    )


class TestMetricBase:
    """测试 Metric 基类"""

    def test_rmse_requirements(self):
        """测试 RMSE requirements 方法"""
        rmse = RMSE()
        reqs = rmse.requirements()

        assert reqs.product_type == ProductType.DETERMINISTIC_FIELD
        assert "*" in reqs.variables
        assert "sample" in reqs.can_merge_along

    def test_rmse_compute_simple(self):
        """测试 RMSE 简单计算"""
        rmse = RMSE()
        batch = create_synthetic_batch(n_samples=10, forecast_noise=2.0)

        result = rmse.compute(batch)

        assert result.status == ResultStatus.SUCCESS
        assert result.metric_name == "rmse"
        assert result.metric_version == "1.0.0"
        assert result.n_valid > 0
        assert result.value > 0  # RMSE 应该大于 0
        assert not np.isnan(result.value)

    def test_rmse_zero_error(self):
        """测试零误差情况"""
        rmse = RMSE()
        batch = create_synthetic_batch(n_samples=5, forecast_noise=0.0, forecast_bias=0.0)

        # 强制预报 = 观测
        batch.forecast.values[:] = batch.observation.values

        result = rmse.compute(batch)

        assert result.status == ResultStatus.SUCCESS
        assert result.value == pytest.approx(0.0, abs=1e-10)

    def test_rmse_known_error(self):
        """测试已知误差值"""
        rmse = RMSE()

        # 创建简单场景：观测全为 0，预报全为 3
        # RMSE 应该正好是 3
        obs = xr.DataArray(
            np.zeros((2, 3, 3)),
            dims=["sample", "lat", "lon"],
            coords={
                "sample": [0, 1],
                "lat": [0, 1, 2],
                "lon": [0, 1, 2],
            },
        )
        fcst = xr.DataArray(
            np.full((2, 3, 3), 3.0),
            dims=["sample", "lat", "lon"],
            coords=obs.coords,
        )
        mask = xr.DataArray(
            np.ones((2, 3, 3), dtype=bool),
            dims=["sample", "lat", "lon"],
            coords=obs.coords,
        )

        batch = EvaluationBatch(
            forecast=fcst,
            observation=obs,
            sample_keys=[{"id": 0}, {"id": 1}],
            valid_mask=mask,
            alignment={"method": "direct", "n_matched": 18},
        )

        result = rmse.compute(batch)

        assert result.status == ResultStatus.SUCCESS
        assert result.value == pytest.approx(3.0, abs=1e-10)


class TestMetricMerge:
    """测试整批与分块合并一致性（Change 3 核心要求）"""

    def test_whole_batch_vs_chunked_merge(self):
        """
        测试整批计算与分块合并产生相同结果

        这是 Change 3 的核心验证：验证可合并统计状态的正确性
        """
        rmse = RMSE()

        # 创建大批次
        whole_batch = create_synthetic_batch(n_samples=20, forecast_noise=2.0, seed=42)

        # 整批计算
        whole_result = rmse.compute(whole_batch)

        # 分块计算
        chunk_size = 5
        states = []
        for i in range(0, 20, chunk_size):
            # 创建子批次
            chunk_batch = EvaluationBatch(
                forecast=whole_batch.forecast.isel(sample=slice(i, i + chunk_size)),
                observation=whole_batch.observation.isel(sample=slice(i, i + chunk_size)),
                sample_keys=whole_batch.sample_keys[i : i + chunk_size],
                valid_mask=whole_batch.valid_mask.isel(sample=slice(i, i + chunk_size)),
                alignment={"method": "direct", "n_matched": chunk_size * 5 * 5},
            )

            # 累积状态
            state = rmse.accumulate(chunk_batch)
            states.append(state)

        # 合并状态
        merged_state = rmse.merge(states)
        merged_result = rmse.finalize(merged_state)

        # 验证结果一致
        assert merged_result.status == ResultStatus.SUCCESS
        assert whole_result.status == ResultStatus.SUCCESS
        assert merged_result.value == pytest.approx(whole_result.value, rel=1e-10)
        assert merged_result.n_valid == whole_result.n_valid

    def test_merge_different_chunk_sizes(self):
        """测试不同分块方式产生相同结果"""
        rmse = RMSE()
        whole_batch = create_synthetic_batch(n_samples=15, forecast_noise=1.5, seed=123)

        # 整批
        whole_result = rmse.compute(whole_batch)

        # 分块方式1: 5 + 5 + 5
        states_1 = []
        for i in [0, 5, 10]:
            chunk = EvaluationBatch(
                forecast=whole_batch.forecast.isel(sample=slice(i, i + 5)),
                observation=whole_batch.observation.isel(sample=slice(i, i + 5)),
                sample_keys=whole_batch.sample_keys[i : i + 5],
                valid_mask=whole_batch.valid_mask.isel(sample=slice(i, i + 5)),
                alignment={"method": "direct", "n_matched": 5 * 5 * 5},
            )
            states_1.append(rmse.accumulate(chunk))

        merged_1 = rmse.finalize(rmse.merge(states_1))

        # 分块方式2: 3 + 3 + 3 + 3 + 3
        states_2 = []
        for i in range(0, 15, 3):
            chunk = EvaluationBatch(
                forecast=whole_batch.forecast.isel(sample=slice(i, i + 3)),
                observation=whole_batch.observation.isel(sample=slice(i, i + 3)),
                sample_keys=whole_batch.sample_keys[i : i + 3],
                valid_mask=whole_batch.valid_mask.isel(sample=slice(i, i + 3)),
                alignment={"method": "direct", "n_matched": 3 * 5 * 5},
            )
            states_2.append(rmse.accumulate(chunk))

        merged_2 = rmse.finalize(rmse.merge(states_2))

        # 三种方式结果应该完全一致
        assert merged_1.value == pytest.approx(whole_result.value, rel=1e-10)
        assert merged_2.value == pytest.approx(whole_result.value, rel=1e-10)

    def test_merge_with_partial_mask(self):
        """测试部分掩码场景的合并一致性"""
        rmse = RMSE()

        # 创建批次，部分点无效
        batch = create_synthetic_batch(n_samples=10, forecast_noise=1.0, seed=999)

        # 随机掩码掉 30% 的点
        np.random.seed(999)
        mask_array = np.random.rand(10, 5, 5) > 0.3
        batch.valid_mask.values[:] = mask_array

        # 整批
        whole_result = rmse.compute(batch)

        # 分块
        states = []
        for i in range(0, 10, 2):
            chunk = EvaluationBatch(
                forecast=batch.forecast.isel(sample=slice(i, i + 2)),
                observation=batch.observation.isel(sample=slice(i, i + 2)),
                sample_keys=batch.sample_keys[i : i + 2],
                valid_mask=batch.valid_mask.isel(sample=slice(i, i + 2)),
                alignment={"method": "direct", "n_matched": 2 * 5 * 5},
            )
            states.append(rmse.accumulate(chunk))

        merged_result = rmse.finalize(rmse.merge(states))

        assert merged_result.value == pytest.approx(whole_result.value, rel=1e-10)
        assert merged_result.n_valid == whole_result.n_valid

    def test_merge_empty_states_fails(self):
        """测试合并空状态列表应该失败"""
        rmse = RMSE()

        with pytest.raises(MetricError, match="Cannot merge empty states"):
            rmse.merge([])

    def test_merge_incompatible_metrics_fails(self):
        """测试合并不兼容指标的状态应该失败"""
        state1 = MetricState(
            metric_name="rmse",
            metric_version="1.0.0",
            data={"sum_squared_error": 10.0, "weights_sum": 5.0, "n_valid": 5},
            n_accumulated=1,
        )

        state2 = MetricState(
            metric_name="mae",  # 不同的指标
            metric_version="1.0.0",
            data={"sum_absolute_error": 8.0, "n_valid": 5},
            n_accumulated=1,
        )

        rmse = RMSE()
        with pytest.raises(MetricError, match="Cannot merge states from different metrics"):
            rmse.merge([state1, state2])

    def test_no_valid_data(self):
        """测试无有效数据的情况"""
        rmse = RMSE()
        batch = create_synthetic_batch(n_samples=5)

        # 全部掩码掉
        batch.valid_mask.values[:] = False

        result = rmse.compute(batch)

        assert result.status == ResultStatus.NO_VALID_DATA
        assert np.isnan(result.value)
        assert result.n_valid == 0
        assert "No valid data points" in result.warnings


class TestMetricWithWeights:
    """测试加权指标计算"""

    def test_rmse_with_uniform_weights(self):
        """测试均匀权重应该与无权重结果相同"""
        rmse = RMSE()
        batch = create_synthetic_batch(n_samples=8, forecast_noise=1.5, seed=777)

        # 无权重
        result_no_weight = rmse.compute(batch)

        # 均匀权重
        weights = xr.DataArray(
            np.ones_like(batch.forecast.values),
            dims=batch.forecast.dims,
            coords=batch.forecast.coords,
        )
        batch.weights = weights
        result_with_weight = rmse.compute(batch)

        assert result_with_weight.value == pytest.approx(result_no_weight.value, rel=1e-10)

    def test_rmse_with_area_weights(self):
        """测试面积权重影响结果"""
        rmse = RMSE()
        batch = create_synthetic_batch(n_samples=5, forecast_noise=2.0, seed=888)

        # 无权重
        result_no_weight = rmse.compute(batch)

        # 面积权重（纬度加权）
        lat_weights = np.cos(np.deg2rad(batch.forecast.lat))
        weights = xr.DataArray(
            np.ones_like(batch.forecast.values) * lat_weights.values[None, :, None],
            dims=batch.forecast.dims,
            coords=batch.forecast.coords,
        )
        batch.weights = weights
        result_with_weight = rmse.compute(batch)

        # 结果应该不同（因为权重不均匀）
        assert result_with_weight.value != pytest.approx(result_no_weight.value, rel=1e-6)
        assert result_with_weight.status == ResultStatus.SUCCESS
