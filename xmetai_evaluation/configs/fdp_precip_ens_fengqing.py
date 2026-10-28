# -*- coding: utf-8 -*-
"""FDP 降水集合概率检验（Fengqing tp × Diamond 站点）：AROC / BS / BSS，不做集合平均。

BSS 无外部参考，用样本气候频率 r(1−r)；时效 6 唯一可出分，12 仅为累积器凑步长。
"""
from xmetai_evaluation.configs.base import EvalConfig

# 数据根：本机 tmp_data/ 即其镜像。
_DATA_ROOT = "/workspace/data/shenzw/fdp_test_data"
_FENGQING_ROOT = f"{_DATA_ROOT}/fengqing"
_STATION_ROOT = f"{_DATA_ROOT}/rain"
# BSS 气候概率参考：ref/MMDDHH.000。
_REF_PROB_ROOT = "/workspace/data/worm/r0/ref"
# 时效 6 是唯一出得了分的；12 只为给累积器凑出步长。
_LEAD_TIMES = [6, 12]


def _execution():
    """``loads`` 里**不能**写 ``reference``：ref_probability 逐样本直读概率、
    没有「整包」可言，写了会被 ExecutionStrategy 直接判配置错
    （execution/strategy.py:219）。观测是站点，本来就默认整段驻留。"""
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


cfg = EvalConfig(
    name="fdp_precip_ens_fengqing",
    description="FDP 降水站点检验（集合，Fengqing × Diamond 站点）：AROC / BS / BSS",
    pipeline="weather_ts_ens_prob",

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
    options={
        # 站点文件名标签为 UTC 有效时刻。
        "local_utc_offset_hours": 0,
        # 对齐标准脚本：cos(lat) 权重、中国框、窗口末端打底且缺测补 0。
        "weights": "cos_lat",
        "region": {"lat": [15.0, 55.0], "lon": [70.0, 140.0], "name": "china"},
        "observation_window": "reference",
    },

    start_date="20260820",   # 起报日；fengqing/ 里只有这一天
    end_date="20260820",
    limit=None,

    output_dir="/workspace/data/shenzw/evaluation_results/fdp_precip_ens_fengqing",
    writers=["csv_long", "probability_wide"],
    log_level="INFO",
    execution=_execution(),
)
