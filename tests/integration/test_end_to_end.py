"""
端到端集成测试：确定性降水 RMSE 评测

验证链路：合成数据 → DataBundle → RMSE（整批/分块一致）→ ResultStore 长表产物。
不依赖真实气象数据，也不依赖任何具体 Reader 实现（本地 helper 直接包装 DataBundle）。
"""

import pytest
import pandas as pd
import xarray as xr
import numpy as np

from xmetai_evaluation.metrics.rmse import RMSE
from xmetai_evaluation.core.contracts import (
    EvaluationBatch,
    DataBundle,
    DataKind,
    SemanticMetadata,
    Provenance,
)
from xmetai_evaluation.core.variables import TemporalKind
from xmetai_evaluation.results import SCORE_COLUMNS, ResultStore, RunContext


def _load_bundle(path, source_id):
    """把合成 NetCDF 文件直接包装成 DataBundle（替代已删除的 SimpleNetCDFReader）。"""
    ds = xr.open_dataset(path)
    return DataBundle(
        payload=ds,
        kind=DataKind.GRIDDED_FORECAST,
        source_id=source_id,
        standard_vars={var: var for var in ds.data_vars},
        semantic=SemanticMetadata(
            units={var: str(ds[var].attrs.get("units", "unknown")) for var in ds.data_vars},
            temporal_kind=TemporalKind.INSTANTANEOUS,
            grid_type="regular_latlon",
        ),
        provenance=Provenance(
            input_files=[str(path)],
            reader_id=source_id,
            reader_version="1.0.0",
        ),
    )


@pytest.fixture
def synthetic_forecast_obs_pair(tmp_path):
    """
    创建合成的预报-观测数据对

    模拟简单场景：
    - 预报场有系统性偏差 +2°C
    - 预报场有随机噪声 ~1°C
    - 观测场是"真值"
    """
    np.random.seed(42)

    # 时空网格
    n_samples = 10
    n_lat = 20
    n_lon = 30

    # 生成观测"真值"
    rng1 = np.random.RandomState(42)
    obs_data = rng1.randn(n_samples, n_lat, n_lon) * 5 + 15  # 均值15°C，标准差5°C
    obs_ds = xr.Dataset(
        {"t2m": (["time", "lat", "lon"], obs_data)},
        coords={
            "time": np.arange(n_samples),
            "lat": np.linspace(-10, 10, n_lat),
            "lon": np.linspace(100, 130, n_lon),
        },
    )
    obs_ds["t2m"].attrs["units"] = "degC"
    obs_ds["t2m"].attrs["long_name"] = "2m temperature observation"

    # 生成预报（观测 + 偏差 + 噪声）
    bias = 2.0
    noise_std = 1.0
    rng2 = np.random.RandomState(123)
    fcst_data = obs_data + bias + rng2.randn(n_samples, n_lat, n_lon) * noise_std
    fcst_ds = xr.Dataset(
        {"t2m": (["time", "lat", "lon"], fcst_data)},
        coords=obs_ds.coords,
    )
    fcst_ds["t2m"].attrs["units"] = "degC"
    fcst_ds["t2m"].attrs["long_name"] = "2m temperature forecast"

    # 保存文件到不同目录
    fcst_dir = tmp_path / "forecast"
    obs_dir = tmp_path / "observation"
    fcst_dir.mkdir()
    obs_dir.mkdir()

    fcst_path = fcst_dir / "forecast.nc"
    obs_path = obs_dir / "observation.nc"
    fcst_ds.to_netcdf(fcst_path)
    obs_ds.to_netcdf(obs_path)

    return {
        "forecast_path": fcst_path,
        "observation_path": obs_path,
        "true_bias": bias,
        "noise_std": noise_std,
        "n_samples": n_samples,
        "n_lat": n_lat,
        "n_lon": n_lon,
    }


class TestEndToEndDeterministicRMSE:
    """端到端测试：确定性预报 RMSE 评测"""

    def test_manual_workflow(self, synthetic_forecast_obs_pair):
        """
        测试手动组装的工作流

        步骤：
        1. 读取预报和观测为 DataBundle
        2. 构建 EvaluationBatch
        3. 用 RMSE 计算指标
        4. 验证结果合理
        """
        # Step 1: 读取数据
        fcst_bundle = _load_bundle(
            synthetic_forecast_obs_pair["forecast_path"], "synthetic_forecast"
        )
        obs_bundle = _load_bundle(
            synthetic_forecast_obs_pair["observation_path"], "synthetic_obs"
        )

        # 验证读取成功
        assert fcst_bundle.source_id == "synthetic_forecast"
        assert obs_bundle.source_id == "synthetic_obs"
        assert "t2m" in fcst_bundle.payload.data_vars
        assert "t2m" in obs_bundle.payload.data_vars

        # Step 2: 构建 EvaluationBatch
        fcst_array = fcst_bundle.payload["t2m"]
        obs_array = obs_bundle.payload["t2m"]

        # 创建有效掩码（全部有效）
        valid_mask = xr.DataArray(
            np.ones_like(fcst_array.values, dtype=bool),
            dims=fcst_array.dims,
            coords=fcst_array.coords,
        )

        # 样本键
        n_samples = synthetic_forecast_obs_pair["n_samples"]
        sample_keys = [{"sample_id": i} for i in range(n_samples)]

        batch = EvaluationBatch(
            forecast=fcst_array,
            observation=obs_array,
            sample_keys=sample_keys,
            valid_mask=valid_mask,
            alignment={
                "method": "direct",
                "time_matched": True,
                "grid_matched": True,
            },
        )

        # Step 3: 计算 RMSE
        rmse_metric = RMSE()
        result = rmse_metric.compute(batch)

        # Step 4: 验证结果
        assert result.status.value == "success"
        assert result.n_valid > 0

        # RMSE 应该接近 sqrt(bias^2 + noise_std^2)
        # bias = 2.0, noise_std = 1.0 => expected RMSE ≈ sqrt(4 + 1) = 2.24
        expected_rmse = np.sqrt(
            synthetic_forecast_obs_pair["true_bias"] ** 2
            + synthetic_forecast_obs_pair["noise_std"] ** 2
        )

        # 允许一些随机性导致的偏差（±20%）
        assert result.value == pytest.approx(expected_rmse, rel=0.2)

    def test_with_partial_masking(self, synthetic_forecast_obs_pair):
        """测试部分掩码场景"""
        # 读取数据
        fcst_bundle = _load_bundle(
            synthetic_forecast_obs_pair["forecast_path"], "synthetic_forecast"
        )
        obs_bundle = _load_bundle(
            synthetic_forecast_obs_pair["observation_path"], "synthetic_obs"
        )

        fcst_array = fcst_bundle.payload["t2m"]
        obs_array = obs_bundle.payload["t2m"]

        # 随机掩码掉 30% 的点
        np.random.seed(123)
        mask_array = np.random.rand(*fcst_array.shape) > 0.3
        valid_mask = xr.DataArray(
            mask_array,
            dims=fcst_array.dims,
            coords=fcst_array.coords,
        )

        n_samples = synthetic_forecast_obs_pair["n_samples"]
        sample_keys = [{"sample_id": i} for i in range(n_samples)]

        batch = EvaluationBatch(
            forecast=fcst_array,
            observation=obs_array,
            sample_keys=sample_keys,
            valid_mask=valid_mask,
            alignment={"method": "direct"},
        )

        # 计算 RMSE
        rmse_metric = RMSE()
        result = rmse_metric.compute(batch)

        # 验证结果
        assert result.status.value == "success"
        # 有效点数应该少于总点数
        total_points = (
            synthetic_forecast_obs_pair["n_samples"]
            * synthetic_forecast_obs_pair["n_lat"]
            * synthetic_forecast_obs_pair["n_lon"]
        )
        assert result.n_valid < total_points
        # RMSE 仍应合理
        assert 0 < result.value < 10  # 合理范围

    def test_chunked_processing(self, synthetic_forecast_obs_pair):
        """
        测试分块处理与整批处理一致性

        模拟大数据场景：数据太大无法一次加载，需要分块处理。
        """
        # 读取数据
        fcst_bundle = _load_bundle(
            synthetic_forecast_obs_pair["forecast_path"], "synthetic_forecast"
        )
        obs_bundle = _load_bundle(
            synthetic_forecast_obs_pair["observation_path"], "synthetic_obs"
        )

        fcst_array = fcst_bundle.payload["t2m"]
        obs_array = obs_bundle.payload["t2m"]

        # 整批处理
        valid_mask_full = xr.DataArray(
            np.ones_like(fcst_array.values, dtype=bool),
            dims=fcst_array.dims,
            coords=fcst_array.coords,
        )

        n_samples = synthetic_forecast_obs_pair["n_samples"]
        sample_keys_full = [{"sample_id": i} for i in range(n_samples)]

        batch_full = EvaluationBatch(
            forecast=fcst_array,
            observation=obs_array,
            sample_keys=sample_keys_full,
            valid_mask=valid_mask_full,
            alignment={"method": "direct"},
        )

        rmse_metric = RMSE()
        result_full = rmse_metric.compute(batch_full)

        # 分块处理（按时间维度分块）
        chunk_size = 3
        states = []
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            fcst_chunk = fcst_array.isel(time=slice(i, end))
            obs_chunk = obs_array.isel(time=slice(i, end))
            mask_chunk = valid_mask_full.isel(time=slice(i, end))
            keys_chunk = [{"sample_id": j} for j in range(i, end)]

            batch_chunk = EvaluationBatch(
                forecast=fcst_chunk,
                observation=obs_chunk,
                sample_keys=keys_chunk,
                valid_mask=mask_chunk,
                alignment={"method": "direct"},
            )

            state = rmse_metric.accumulate(batch_chunk)
            states.append(state)

        # 合并状态
        merged_state = rmse_metric.merge(states)
        result_chunked = rmse_metric.finalize(merged_state)

        # 验证一致性
        assert result_full.value == pytest.approx(result_chunked.value, rel=1e-10)
        assert result_full.n_valid == result_chunked.n_valid


class TestEndToEndResultsTable:
    """测试"指标结果 -> 统一长表 -> 标准产物"的完整输出链路"""

    def test_results_are_written_as_canonical_long_table(self, tmp_path, synthetic_forecast_obs_pair):
        """任意评测的结果都应落成同一种长表"""
        fcst_bundle = _load_bundle(
            synthetic_forecast_obs_pair["forecast_path"], "synthetic_forecast"
        )
        obs_bundle = _load_bundle(
            synthetic_forecast_obs_pair["observation_path"], "synthetic_obs"
        )

        fcst_array = fcst_bundle.payload["t2m"]
        obs_array = obs_bundle.payload["t2m"]
        valid_mask = xr.DataArray(
            np.ones_like(fcst_array.values, dtype=bool),
            dims=fcst_array.dims,
            coords=fcst_array.coords,
        )

        batch = EvaluationBatch(
            forecast=fcst_array,
            observation=obs_array,
            sample_keys=[{"sample_id": index} for index in range(fcst_array.shape[0])],
            valid_mask=valid_mask,
            sample_dim="grid",
            alignment={"method": "direct"},
            protocol_id="grid_valid_time",
        )

        result = RMSE().compute(batch)
        result.coordinates = {"variable": "t2m", "lead_h": 24, "sample_unit": "grid"}
        result.product_kind = result.product_kind or "deterministic"

        output_dir = tmp_path / "results"
        store = ResultStore(
            output_dir,
            RunContext(
                run_id="synthetic_rmse",
                model_id="synthetic_forecast",
                dataset_id="synthetic_obs",
                protocol_id="grid_valid_time",
            ),
        )
        artifacts = store.write(
            [result],
            resolved_config={"schema_version": 1},
            writers=("csv_long", "json", "coverage"),
        )

        scores = pd.read_csv(artifacts["scores"])
        assert list(scores.columns) == SCORE_COLUMNS
        assert scores.loc[0, "metric"] == "rmse"
        assert scores.loc[0, "variable"] == "t2m"
        assert scores.loc[0, "lead_h"] == 24
        assert scores.loc[0, "model_id"] == "synthetic_forecast"
        assert scores.loc[0, "dataset_id"] == "synthetic_obs"
        assert scores.loc[0, "protocol_id"] == "grid_valid_time"
        assert scores.loc[0, "status"] == "success"
        assert scores.loc[0, "value"] == pytest.approx(result.value)

        # 标准产物齐备
        assert artifacts["coverage"].exists()
        assert artifacts["manifest"].exists()
        assert artifacts["json"].exists()
        assert (output_dir / "coverage.csv").exists()
