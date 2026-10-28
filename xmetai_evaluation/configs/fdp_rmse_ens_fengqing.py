# -*- coding: utf-8 -*-
"""FDP 集合场检验（Fengqing × CRA40）：字段评分 + CRPS + 活跃度/谱，仅评 z500。

时段与时效对齐旧实现 ensemble_verification_2026082000.csv，rmse/crps/spread 可直接对表。
"""
from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

VARS = ["z500"]

VAR_METRICS = {
    "z500": ["rmse", "bias", "acc", "activity", "spectrum", "crps", "spread_error"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# ACC 用 FDP / WeatherBench2 的 uncentered 口径。
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# 谱按纬度降序，对齐标准脚本。
METRIC_OPTIONS["spectrum"] = {**METRIC_OPTIONS["spectrum"], "lat_order": "descending"}
# spread_error 默认按 M−1 无偏；与老档案对账才加 ddof=0。

# 数据根：本机 tmp_data/ 即其镜像。
_DATA_ROOT = "/workspace/data/shenzw/fdp_test_data"
_FENGQING_ROOT = f"{_DATA_ROOT}/fengqing"
_CRA_ROOT = f"{_DATA_ROOT}/cra_root"

cfg = EvalConfig(
    name="fdp_rmse_ens_fengqing",
    description="FDP 集合场检验（Fengqing × CRA40）：RMSE / Bias / CRPS / Spread-Error / ACC / 活跃度 / 谱",
    pipeline=["fdp_field_scores", "fdp_ens_crps", "fdp_activity_spectrum"],

    forecast_reader={
        "type": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": VARS,
        "step_hours": 6.0,
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
    limit=None,

    output_dir="/workspace/data/shenzw/evaluation_results/fdp_rmse_ens_fengqing",
    writers=["csv_long", "spectrum", "json"],
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # ── 并发与数据加载 ──────────────────────────────────────────────────
    # 用 processes：CRPS 与谱为单核 numpy 计算，threads 受 GIL 限制。
    execution={
        "mode": "processes",
        "n_workers": 24,        # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {
            "observation": "slice",     # CRA 按时刻一个 grib2；本配置时效同一天，窗块命中不上
            "reference": "resident",    # 日序气候态整 run 驻留，fork 共享 1 份
        },
        "resume": True,         # 已完成块的状态落 .states/，重跑跳过
    },
)
