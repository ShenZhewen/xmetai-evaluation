"""端到端：分块执行不改变结果口径。

三天起报、两个时效窗（24h / 48h）的站点 TS 评测，分别用一块跑完
（= 旧的单段行为）、逐日切块串行、逐日切块线程并发、再按时效切窗，
所有执行的 scores.csv 必须逐行一致；resume 重跑（块状态复用）也一致。
这是分块改造最重要的回归底线：**块怎么切、用什么并发形态，都不许改变分数。**

时效维是切块的两个维度之一，所以 fixture 必须给出**两个时效窗**：
只有 4 个时效（一个 24h 窗）时，切时效这条路径根本没被走到。
"""

import json
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
import xarray as xr

from xmetai_evaluation.cli import run_evaluation
from xmetai_evaluation.configs.base import EvalConfig

GRID_LATS = np.array([-10.0, 0.0, 10.0])
GRID_LONS = np.array([100.0, 105.0, 110.0])
STATIONS = [(45004, 100.0, 0.0, 10.0), (45005, 110.0, 10.0, 20.0)]

START_DAY = datetime(2025, 1, 1)
INIT_DAYS = 3
STEP_HOURS = 6
WINDOW_HOURS = 24
MAX_LEAD_HOURS = 48  # 两个 24h 时效窗：采样时效 24 与 48
FORECAST_INTERVAL_MM = 2.5  # 4 个时效之和 = 10mm（每天都报 10mm）

SAMPLE_LEADS = [24, 48]
INIT_TIMES = [START_DAY + timedelta(days=d) for d in range(INIT_DAYS)]


def _write_forecast(root):
    """三天 FuXi 起报：root/YYYYMMDD/NNN.nc，每个 24h 窗口共 10mm。"""
    for init_time in INIT_TIMES:
        init_dir = root / init_time.strftime("%Y%m%d")
        init_dir.mkdir(parents=True)
        for index in range(MAX_LEAD_HOURS // STEP_HOURS):
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
    """逐小时 Diamond 文件，铺满观测窗。

    按 24h 窗口分日：第 2 天无雨，其余每天 24h 累积 10mm。起报日 d 的
    lead 24 落在第 d 天、lead 48 落在第 d+1 天 —— 两个时效的 TS 都是
    4/(4+0+2)=2/3，且对"漏掉某个时效窗"敏感（漏掉一个时效的命中/空报
    配对就会变），这正是切时效最容易错的地方。
    """
    obs_start = INIT_TIMES[0] + timedelta(hours=8) + timedelta(hours=1 - WINDOW_HOURS)
    obs_end = INIT_TIMES[-1] + timedelta(hours=8) + timedelta(hours=MAX_LEAD_HOURS)
    # 一个小时属于哪个起报日的观测窗：窗口 d 从起报日 d 的 09 时（北京时）
    # 开始铺 24 小时，三天的窗口首尾相接，直接按 24h 整除即可。
    window_anchor = INIT_TIMES[0] + timedelta(hours=9)
    moment = obs_start
    while moment <= obs_end:
        day_index = int((moment - window_anchor).total_seconds() // 86400)
        hourly = 0.0 if day_index == 1 else 10.0 / WINDOW_HOURS
        lines = [
            f"diamond  3 {moment.strftime('%Y年%m月%d日%H时')}1小时降水(逐时)",
            "25 1 1 0 1000 0 0 0 0 1 77017",
        ]
        for station_id, lon, lat, altitude in STATIONS:
            lines.append(f"{station_id} {lon} {lat} {altitude} {hourly}")
        (root / f"{moment.strftime('%Y%m%d%H')}.000").write_text(
            "\n".join(lines) + "\n", encoding="gbk"
        )
        moment += timedelta(hours=1)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("chunked_equivalence")
    forecast_root = root / "forecast"
    station_root = root / "stations"
    station_root.mkdir()
    _write_forecast(forecast_root)
    _write_observations(station_root)
    return {"forecast": forecast_root, "stations": station_root, "root": root}


def _config(paths, execution, output_dir):
    return EvalConfig(
        name="chunked_equivalence",
        description="分块执行等价性测试",
        pipeline="weather_ts_det",
        forecast_reader={
            "type": "fuxi",
            "root_dir": str(paths["forecast"]),
            "variable": "tp",
            "step_hours": STEP_HOURS,
            # 显式钉死起报时刻：不写的话计划层按 catalog 现扫，删掉某天的
            # 预报文件重跑时计划会**静默缩水**（3 起报变 2 起报），resume
            # 测试就变成了在比"少了半天的分数"，而不是在验状态复用。
            "init_times": [moment.isoformat() for moment in INIT_TIMES],
        },
        observation_reader={
            "type": "station",
            "root_dir": str(paths["stations"]),
            "variable": "precipitation",
        },
        transform_options={"time_window_accumulator": {"window_hours": WINDOW_HOURS}},
        metric_options={"ts_score": {"thresholds": [0.1]}},
        start_date="20250101",
        end_date="20250103",
        output_dir=str(output_dir),
        writers=["csv_long"],
        execution=execution,
    )


def _run_and_read(paths, execution, output_dir):
    """跑一次并读回分数；顺带断言没有失败块（空时效窗会静默变成失败块）。"""
    assert run_evaluation(_config(paths, execution, output_dir)) == 0
    _assert_no_failed_chunk(output_dir)
    scores = pd.read_csv(output_dir / "scores.csv")
    return scores.sort_values(["metric", "threshold", "lead_h"]).reset_index(drop=True)


def _assert_no_failed_chunk(output_dir):
    """切块不许切出跑不出分的空块（时效窗按原始跨度切就会）。

    空块不改变 scores.csv（它什么都不产出），所以只比分数抓不到——必须显式看
    manifest。没有 ``failed_chunks`` 键就表示零失败块。
    """
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "failed_chunks" not in manifest, manifest.get("failed_chunks")


@pytest.fixture(scope="module")
def single_chunk_scores(synthetic_run):
    """一块跑完 = 改造前的单段行为，作为基准。"""
    scores = _run_and_read(
        synthetic_run,
        execution={"mode": "serial", "chunk_days": 90, "lead_chunk_days": 0},
        output_dir=synthetic_run["root"] / "out_single",
    )

    # 两个时效窗都出分，各三天起报 × 两站
    ts = scores[scores["metric"] == "ts"]
    assert sorted(ts["lead_h"]) == SAMPLE_LEADS
    for _, row in ts.iterrows():
        # ts = 命中 4 / (命中 4 + 漏报 0 + 空报 2) = 2/3
        assert row["value"] == pytest.approx(4.0 / 6.0)
        assert row["n_valid"] == len(STATIONS) * INIT_DAYS
    return scores


def test_daily_chunks_serial_matches_single_chunk(synthetic_run, single_chunk_scores):
    scores = _run_and_read(
        synthetic_run,
        execution={"mode": "serial", "chunk_days": 1},
        output_dir=synthetic_run["root"] / "out_serial",
    )
    assert_frame_equal(scores, single_chunk_scores)


def test_daily_chunks_threads_match_single_chunk(synthetic_run, single_chunk_scores):
    scores = _run_and_read(
        synthetic_run,
        execution={"mode": "threads", "n_workers": 2, "chunk_days": 1},
        output_dir=synthetic_run["root"] / "out_threads",
    )
    assert_frame_equal(scores, single_chunk_scores)


def test_daily_chunks_processes_match_single_chunk_if_fork(
    synthetic_run, single_chunk_scores
):
    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("无 fork 平台（spawn 下子进程直读，等价性由线程用例覆盖）")
    scores = _run_and_read(
        synthetic_run,
        execution={"mode": "processes", "n_workers": 2, "chunk_days": 1},
        output_dir=synthetic_run["root"] / "out_processes",
    )
    assert_frame_equal(scores, single_chunk_scores)


def test_lead_split_serial_matches_single_chunk(synthetic_run, single_chunk_scores):
    """时效维切窗：整段时效一块 -> 逐 24h 时效窗，分数必须一样。

    切窗时读取集要往前带累积预热（24h 窗 / 6h 步长 = 3 步），否则每个窗的
    首个样本会因为没有完整窗口被静默丢掉——那会直接改变 TS。
    """
    scores = _run_and_read(
        synthetic_run,
        execution={"mode": "serial", "chunk_days": 1, "lead_chunk_days": 1},
        output_dir=synthetic_run["root"] / "out_lead_serial",
    )
    assert_frame_equal(scores, single_chunk_scores)


def test_lead_split_threads_match_single_chunk(synthetic_run, single_chunk_scores):
    scores = _run_and_read(
        synthetic_run,
        execution={
            "mode": "threads",
            "n_workers": 2,
            "chunk_days": 1,
            "lead_chunk_days": 1,
        },
        output_dir=synthetic_run["root"] / "out_lead_threads",
    )
    assert_frame_equal(scores, single_chunk_scores)


def test_lead_chunk_days_zero_is_the_escape_hatch(synthetic_run, single_chunk_scores):
    """``lead_chunk_days=0`` = 不切时效（改造前的行为），结果逐位一致。"""
    scores = _run_and_read(
        synthetic_run,
        execution={"mode": "serial", "chunk_days": 1, "lead_chunk_days": 0},
        output_dir=synthetic_run["root"] / "out_lead_off",
    )
    assert_frame_equal(scores, single_chunk_scores)


def test_slice_load_policy_also_matches(synthetic_run, single_chunk_scores):
    """把站点观测改成逐块直读（不驻留），结果必须一样。"""
    scores = _run_and_read(
        synthetic_run,
        execution={"mode": "serial", "chunk_days": 1, "loads": {"observation": "slice"}},
        output_dir=synthetic_run["root"] / "out_slice",
    )
    assert_frame_equal(scores, single_chunk_scores)


def test_resume_reuses_chunk_states(synthetic_run, single_chunk_scores):
    """resume 开启：第一次跑出块状态，第二次全部复用，结果一致。

    第二次是**在首日起报目录被移走的前提下**跑的 —— 复用真生效就完全不会
    碰数据，分数逐位不变；复用没生效则该块读不到文件，直接进
    ``failed_chunks``，被 ``_run_and_read`` 里的断言抓住。靠"少读一次盘"
    是验不出来的，得让不读盘变成唯一能过的路径。
    """
    output_dir = synthetic_run["root"] / "out_resume"
    execution = {"mode": "serial", "chunk_days": 1, "resume": True}
    first = _run_and_read(synthetic_run, execution, output_dir)
    assert_frame_equal(first, single_chunk_scores)

    # 移走首日起报目录（不是单个文件）：该起报的全部时效都读不到了
    forecast_dir = synthetic_run["forecast"] / INIT_TIMES[0].strftime("%Y%m%d")
    backup = output_dir.parent / "forecast_day1_backup"
    shutil.move(str(forecast_dir), str(backup))
    try:
        second = _run_and_read(synthetic_run, execution, output_dir)
        assert_frame_equal(second, single_chunk_scores)
    finally:
        shutil.move(str(backup), str(forecast_dir))
