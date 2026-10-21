# -*- coding: utf-8 -*-
"""FDP 要素检验（Fengqing × CRA40）：集合平均场当确定性预报，仅评 z500。

时段与时效对齐旧实现 ensemble_verification_2026082000.csv，rmse 可直接对表。
"""
from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

VARS = ["z500"]

VAR_METRICS = {
    "z500": ["rmse", "bias", "acc", "activity", "spectrum"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# ACC 用 FDP / WeatherBench2 的 uncentered 口径。
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# 谱按纬度降序，对齐标准脚本。
METRIC_OPTIONS["spectrum"] = {**METRIC_OPTIONS["spectrum"], "lat_order": "descending"}

# 数据根：本机 tmp_data/ 即其镜像。
_DATA_ROOT = "/workspace/data/shenzw/fdp_test_data"
_FENGQING_ROOT = f"{_DATA_ROOT}/fengqing"
_CRA_ROOT = f"{_DATA_ROOT}/cra_root"

cfg = EvalConfig(
    name="fdp_rmse_single_fengqing",
    description="FDP 要素检验（Fengqing × CRA40）：RMSE / Bias / ACC / 活跃度 / 功率谱",
    pipeline=["fdp_field_scores", "fdp_activity_spectrum"],

    forecast_reader={
        "type": "fengqing",     # 集合；本流程先用 ensemble_mean 降成确定性场
        "root_dir": _FENGQING_ROOT,
        "variables": VARS,
        "step_hours": 6.0,      # 布局步长：时效取文件名里的 3 位 × 6h
        "lead_times": [6, 12, 18],
    },
    observation_reader={
        "type": "cra",
        "root_dir": _CRA_ROOT,
        "variables": VARS,
    },
    # 编造气候态，仅用于跑通链路；ACC / 活跃度数值无意义。
    reference_reader={
        "type": "climatology",
        "root_dir": f"{_DATA_ROOT}/cli_fake",
    },

    # cra_root 只有 20260820 一天。
    start_date="20260820",
    end_date="20260820",
    limit=None,             # 限起报数，直接改这里；None = 不限

    output_dir="/workspace/szwCode/evaluation_results/fdp_rmse_single_fengqing",
    writers=["csv_long", "spectrum", "json"],
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # ── 并发与数据加载 ──────────────────────────────────────────────────
    # 用 processes：活跃度/谱为单核 numpy FFT，threads 受 GIL 限制。
    execution={
        "mode": "processes",
        "n_workers": 24,        # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {
            "observation": "slice",     # CRA40 GRIB2 按请求跨度现读现弃
            "reference": "resident",    # 气候态整 run 驻留一份，fork 共享
        },
        "resume": True,         # 已完成块的状态落 .states/，重跑跳过
    },
)
