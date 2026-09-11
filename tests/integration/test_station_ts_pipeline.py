"""端到端：站点降水 TS 评测走完整 CLI 链路。

用合成的 FuXi 时效文件和 Diamond 站点文件，验证
配置 -> 注册表 -> Reader -> Transform -> Metric -> 统一长表 的整条链路。
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xmetai_evaluation.cli import run_evaluation
from xmetai_evaluation.configs.base import EvalConfig
from xmetai_evaluation.results import SCORE_COLUMNS

GRID_LATS = np.array([-10.0, 0.0, 10.0])
GRID_LONS = np.array([100.0, 105.0, 110.0])
STATIONS = [(45004, 100.0, 0.0, 10.0), (45005, 110.0, 10.0, 20.0)]

INIT_TIME = datetime(2025, 1, 1, 0)
STEP_HOURS = 6
WINDOW_HOURS = 24
FORECAST_INTERVAL_MM = 2.5  # 4 个时效之和 = 10mm
OBSERVATION_HOURLY_MM = 10.0 / WINDOW_HOURS


def _write_forecast(root):
    init_dir = root / INIT_TIME.strftime("%Y%m%d")
    init_dir.mkdir(parents=True)
    for index in range(WINDOW_HOURS // STEP_HOURS):
        dataset = xr.Dataset(
            {
                "TP": (
                    ["lat", "lon"],
                    np.full((GRID_LATS.size, GRID_LONS.size), FORECAST_INTERVAL_MM),
                    {"units": "mm"},
                )
            },
            coords={"lat": GRID_LATS, "lon": GRID_LONS},
        )
        dataset.to_netcdf(init_dir / f"{index + 1:03d}.nc")


def _write_observations(root):
    # 预报后时次 = 起报 + 时效 + 8h（北京时），实况窗口在它之前取满 WINDOW_HOURS 小时
    end_time = INIT_TIME + timedelta(hours=8 + WINDOW_HOURS)
    start_time = end_time - timedelta(hours=WINDOW_HOURS - 1)
    for step in range(WINDOW_HOURS):
        moment = start_time + timedelta(hours=step)
        path = root / f"{moment.strftime('%Y%m%d%H')}.000"
        lines = [
            f"diamond  3 {moment.strftime('%Y年%m月%d日%H时')}1小时降水(逐时)",
            "25 1 1 0 1000 0 0 0 0 1 77017",
        ]
        for station_id, lon, lat, altitude in STATIONS:
            lines.append(f"{station_id} {lon} {lat} {altitude} {OBSERVATION_HOURLY_MM}")
        path.write_text("\n".join(lines) + "\n", encoding="gbk")


@pytest.fixture
def synthetic_station_run(tmp_path):
    forecast_root = tmp_path / "forecast"
    station_root = tmp_path / "stations"
    station_root.mkdir()
    _write_forecast(forecast_root)
    _write_observations(station_root)
    return {"forecast": forecast_root, "stations": station_root, "output": tmp_path / "out"}


def _config(paths, writers):
    return EvalConfig(
        name="synthetic_station_ts",
        description="合成站点降水 TS 端到端测试",
        pipeline="weather_ts_det",
        forecast_reader={
            "type": "fuxi",
            "root_dir": str(paths["forecast"]),
            "variable": "tp",
            "step_hours": STEP_HOURS,
        },
        observation_reader={
            "type": "station",
            "root_dir": str(paths["stations"]),
            "variable": "precipitation",
        },
        transform_options={"time_window_accumulator": {"window_hours": WINDOW_HOURS}},
        metric_options={"ts_score": {"thresholds": [0.1]}},
        start_date=INIT_TIME.strftime("%Y%m%d"),
        end_date=INIT_TIME.strftime("%Y%m%d"),
        output_dir=str(paths["output"]),
        writers=writers,
    )


def test_station_ts_pipeline_writes_canonical_scores(synthetic_station_run):
    exit_code = run_evaluation(
        _config(synthetic_station_run, ["csv_long", "categorical_wide"])
    )
    assert exit_code == 0

    output_dir = synthetic_station_run["output"]
    scores = pd.read_csv(output_dir / "scores.csv")

    assert list(scores.columns) == SCORE_COLUMNS
    # 分类检验的每个评分各一行：ts / pod / far / miss_rate / frequency_bias
    assert set(scores["metric"]) == {"ts", "pod", "far", "miss_rate", "frequency_bias"}
    assert (scores["lead_h"] == WINDOW_HOURS).all()
    assert (scores["window_h"] == WINDOW_HOURS).all()
    assert (scores["model_id"] == "fuxi").all()
    assert (scores["sample_unit"] == "station").all()
    assert (scores["protocol_id"] == "station_valid_time").all()
    assert (scores["status"] == "success").all()

    ts_row = scores[scores["metric"] == "ts"].iloc[0]
    assert ts_row["value"] == pytest.approx(1.0)
    assert ts_row["threshold"] == pytest.approx(0.1)
    assert ts_row["n_valid"] == len(STATIONS)
    assert ts_row["product_kind"] == "categorical"


def test_station_ts_pipeline_writes_diagnostics_and_manifest(synthetic_station_run):
    run_evaluation(
        _config(
            synthetic_station_run,
            ["csv_long", "categorical_wide", "coverage", "details"],
        )
    )
    output_dir = synthetic_station_run["output"]

    wide = pd.read_csv(output_dir / "diagnostics" / "categorical_wide.csv")
    assert len(wide) == 1
    assert wide.iloc[0]["TS"] == pytest.approx(1.0)
    assert wide.iloc[0]["hits"] == len(STATIONS)
    assert wide.iloc[0]["misses"] == 0
    assert wide.iloc[0]["false_alarms"] == 0

    details = pd.read_csv(output_dir / "diagnostics" / "scores_detail.csv")
    assert {"hits", "misses", "false_alarms", "n_pairs"}.issubset(set(details["field"]))

    coverage = pd.read_csv(output_dir / "coverage.csv")
    assert len(coverage) == 1
    assert coverage.iloc[0]["n_valid"] == len(STATIONS)

    manifest = (output_dir / "manifest.json").read_text(encoding="utf-8")
    assert "station_valid_time" in manifest
    assert "resolved_config" in manifest
    assert not (output_dir / "resolved_config.yaml").exists()


CONFIG_TEMPLATE = '''
from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="synthetic_station_ts_cli",
    description="CLI 端到端",
    pipeline="weather_ts_det",
    forecast_reader={{
        "type": "fuxi",
        "root_dir": r"{forecast}",
        "variable": "tp",
        "step_hours": 6,
    }},
    observation_reader={{
        "type": "station",
        "root_dir": r"{stations}",
        "variable": "precipitation",
    }},
    transform_options={{"time_window_accumulator": {{"window_hours": 24}}}},
    metric_options={{"ts_score": {{"thresholds": [0.1]}}}},
    start_date="20250101",
    end_date="20250101",
    output_dir=r"{output}",
    writers=["csv_long"],
)
'''


def test_cli_runs_a_config_file_end_to_end(synthetic_station_run, tmp_path):
    """统一入口：--config 指向配置文件即可跑完整流程。"""
    from xmetai_evaluation.cli import main

    config_path = tmp_path / "synthetic_cli_config.py"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            forecast=synthetic_station_run["forecast"],
            stations=synthetic_station_run["stations"],
            output=synthetic_station_run["output"],
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "--log-level", "ERROR"]) == 0

    scores = pd.read_csv(synthetic_station_run["output"] / "scores.csv")
    assert set(scores["metric"]) == {"ts", "pod", "far", "miss_rate", "frequency_bias"}
    assert (scores["status"] == "success").all()


def _write_ensemble_forecast(root):
    """两个成员的 FuXi 集合输出；成员均值等于单模型预报的逐 6h 量。"""
    init_dir = root / INIT_TIME.strftime("%Y%m%d")
    for member, value in ((1, 0.0), (2, 2 * FORECAST_INTERVAL_MM)):
        member_dir = init_dir / f"member_{member:02d}"
        member_dir.mkdir(parents=True)
        for index in range(WINDOW_HOURS // STEP_HOURS):
            dataset = xr.Dataset(
                {
                    "TP": (
                        ["lat", "lon"],
                        np.full((GRID_LATS.size, GRID_LONS.size), value),
                        {"units": "mm"},
                    )
                },
                coords={"lat": GRID_LATS, "lon": GRID_LONS},
            )
            dataset.to_netcdf(member_dir / f"{index + 1:03d}.nc")


@pytest.fixture
def synthetic_ensemble_run(tmp_path):
    forecast_root = tmp_path / "forecast_ens"
    station_root = tmp_path / "stations"
    station_root.mkdir()
    _write_ensemble_forecast(forecast_root)
    _write_observations(station_root)
    return {"forecast": forecast_root, "stations": station_root, "output": tmp_path / "out"}


def _ensemble_config(paths):
    return EvalConfig(
        name="synthetic_fuxi_ens_ts",
        description="集合平均 TS 端到端测试",
        pipeline="weather_ts_ens",
        forecast_reader={
            "type": "fuxi_ens",
            "root_dir": str(paths["forecast"]),
            "variable": "tp",
            "step_hours": STEP_HOURS,
        },
        observation_reader={
            "type": "station",
            "root_dir": str(paths["stations"]),
            "variable": "precipitation",
        },
        transform_options={"time_window_accumulator": {"window_hours": WINDOW_HOURS}},
        metric_options={"ts_score": {"thresholds": [0.1]}},
        start_date=INIT_TIME.strftime("%Y%m%d"),
        end_date=INIT_TIME.strftime("%Y%m%d"),
        output_dir=str(paths["output"]),
        writers=["csv_long", "categorical_wide"],
    )


def test_ensemble_ts_is_computed_on_the_ensemble_mean(synthetic_ensemble_run):
    assert run_evaluation(_ensemble_config(synthetic_ensemble_run)) == 0

    output_dir = synthetic_ensemble_run["output"]
    scores = pd.read_csv(output_dir / "scores.csv")
    ts_row = scores[scores["metric"] == "ts"].iloc[0]
    assert ts_row["value"] == pytest.approx(1.0)
    assert ts_row["n_valid"] == len(STATIONS)
    assert ts_row["sample_unit"] == "station"
    assert ts_row["status"] == "success"

    wide = pd.read_csv(output_dir / "diagnostics" / "categorical_wide.csv")
    assert wide.iloc[0]["hits"] == len(STATIONS)
    assert wide.iloc[0]["misses"] == 0


def _write_mixed_ensemble(root):
    """成员 1 全 0；成员 2 只在最北一行有雨 —— 构造"有事件/无事件"混合样本。"""
    init_dir = root / INIT_TIME.strftime("%Y%m%d")
    for member in (1, 2):
        member_dir = init_dir / f"member_{member:02d}"
        member_dir.mkdir(parents=True)
        values = np.zeros((GRID_LATS.size, GRID_LONS.size))
        if member == 2:
            values[-1, :] = 20.0
        for index in range(WINDOW_HOURS // STEP_HOURS):
            dataset = xr.Dataset(
                {"TP": (["lat", "lon"], values, {"units": "mm"})},
                coords={"lat": GRID_LATS, "lon": GRID_LONS},
            )
            dataset.to_netcdf(member_dir / f"{index + 1:03d}.nc")


def _write_split_observations(root):
    """站点 A 无雨、站点 B 有雨。"""
    end_time = INIT_TIME + timedelta(hours=8 + WINDOW_HOURS)
    start_time = end_time - timedelta(hours=WINDOW_HOURS - 1)
    for step in range(WINDOW_HOURS):
        moment = start_time + timedelta(hours=step)
        lines = [
            f"diamond  3 {moment.strftime('%Y年%m月%d日%H时')}1小时降水(逐时)",
            "25 1 1 0 1000 0 0 0 0 1 77017",
        ]
        for position, (station_id, lon, lat, altitude) in enumerate(STATIONS):
            value = 0.0 if position == 0 else OBSERVATION_HOURLY_MM
            lines.append(f"{station_id} {lon} {lat} {altitude} {value}")
        (root / f"{moment.strftime('%Y%m%d%H')}.000").write_text(
            "\n".join(lines) + "\n", encoding="gbk"
        )


def test_ensemble_probability_scores_are_written(tmp_path):
    """同一批成员：TS 走集合平均场，AROC/BS/BSS 走原始成员。"""
    forecast_root = tmp_path / "forecast_ens"
    station_root = tmp_path / "stations"
    station_root.mkdir()
    _write_mixed_ensemble(forecast_root)
    _write_split_observations(station_root)
    output_dir = tmp_path / "out"

    config = EvalConfig(
        name="synthetic_ensemble_prob",
        description="集合概率评分端到端",
        pipeline="weather_ts_ens",
        forecast_reader={
            "type": "fuxi_ens",
            "root_dir": str(forecast_root),
            "variable": "tp",
            "step_hours": STEP_HOURS,
        },
        observation_reader={
            "type": "station",
            "root_dir": str(station_root),
            "variable": "precipitation",
        },
        transform_options={"time_window_accumulator": {"window_hours": WINDOW_HOURS}},
        metric_options={
            "ts_score": {"thresholds": [0.1]},
            "ensemble_probability": {"thresholds": [0.1]},
        },
        start_date=INIT_TIME.strftime("%Y%m%d"),
        end_date=INIT_TIME.strftime("%Y%m%d"),
        output_dir=str(output_dir),
        writers=["csv_long", "categorical_wide", "probability_wide"],
    )

    assert run_evaluation(config) == 0

    scores = pd.read_csv(output_dir / "scores.csv")
    ts_row = scores[scores["metric"] == "ts"].iloc[0]
    assert ts_row["value"] == pytest.approx(1.0)  # 集合平均场：A 正确否定，B 命中
    assert ts_row["n_valid"] == len(STATIONS)

    prob = scores[scores["product_kind"] == "probabilistic"].set_index("metric")
    assert {"aroc", "bs", "bss"}.issubset(set(prob.index))
    assert prob.loc["aroc", "value"] == pytest.approx(1.0)
    assert prob.loc["bs", "value"] == pytest.approx(0.125)
    assert prob.loc["bss", "value"] == pytest.approx(0.5)
    assert prob.loc["aroc", "n_valid"] == len(STATIONS)

    wide = pd.read_csv(output_dir / "diagnostics" / "probability_wide.csv")
    assert {"grade", "threshold_mm", "AROC", "BS", "BS_ref", "BSS", "base_rate", "n_points"}.issubset(
        set(wide.columns)
    )
    first = wide.iloc[0]
    assert first["AROC"] == pytest.approx(1.0)
    assert first["BSS"] == pytest.approx(0.5)
    assert first["base_rate"] == pytest.approx(0.5)
    assert first["n_points"] == len(STATIONS)
