# -*- coding: utf-8 -*-
"""FDP 降水确定性检验（Fengqing tp × Diamond 站点）：集合平均场插值到站，评 TS / POD / FAR / 频率偏差。

起报 20260820，仅时效 6 有观测可配；格点 FSS 段默认不启用。
"""
from xmetai_evaluation.configs.base import EvalConfig

# 数据根：本机 tmp_data/ 即其镜像。
_DATA_ROOT = "/workspace/data/shenzw/fdp_test_data"
_FENGQING_ROOT = f"{_DATA_ROOT}/fengqing"
_CRA_ROOT = f"{_DATA_ROOT}/cra_root"
_STATION_ROOT = f"{_DATA_ROOT}/rain"
# 只有时效 6 配得上：观测只到 0820 06Z。
_LEAD_TIMES = [6]


def _execution():
    """两条配置各拿一份新的 execution —— 共用同一个 dict 会被就地改写串味。"""
    return {
        "mode": "auto",
        "n_workers": 4,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "resident",  # 站点观测整 run 驻留（模板推导的缺省也是这个）
        },
        "resume": True,
    }


# ── 站点段：集合平均 → 插值到站 → TS ──────────────────────────────────────
# 模板自带 ensemble_mean，且缺省 local_utc_offset_hours=0（站点文件名标签为 UTC）。
_station = EvalConfig(
    name="fdp_precip_single_fengqing",
    description="FDP 降水站点检验（确定性，Fengqing × Diamond 站点）：TS / POD / FAR / 频率偏差",
    pipeline="fdp_precip_ts",

    forecast_reader={
        "type": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": ["tp"],
        "step_hours": 6.0,
        "lead_times": _LEAD_TIMES,
    },
    observation_reader={
        "type": "station",          # 别名 diamond_station
        "root_dir": _STATION_ROOT,
        "variable": "precipitation",
    },

    start_date="20260820",   # 起报日；fengqing/ 里只有这一天
    end_date="20260820",
    limit=None,

    output_dir="/workspace/szwCode/evaluation_results/fdp_precip_single_fengqing",
    writers=["csv_long", "categorical_wide"],
    log_level="INFO",
    execution=_execution(),
)

# ── 格点段：FSS（默认不启用）─────────────────────────────────────────────
# cra_root 里的 tp 是编造的 fixture，数字无气象意义。
_grid = EvalConfig(
    name="fdp_precip_fss",
    description="FDP 降水空间检验（Fengqing × CRA40）：FSS",
    pipeline="fdp_precip_fss",

    forecast_reader={
        "type": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": ["tp"],
        "step_hours": 6.0,
        "lead_times": [6, 12, 18],
    },
    observation_reader={
        "type": "cra",
        "root_dir": _CRA_ROOT,
        "variables": ["tp"],
    },

    start_date="20260820",   # 起报日；编的那份 tp 只有这一天
    end_date="20260820",
    limit=None,

    output_dir="/workspace/szwCode/evaluation_results/fdp_precip_fss",
    writers=["csv_long"],
    log_level="INFO",
    execution=_execution(),
)

cfgs = [
    _station,
]
