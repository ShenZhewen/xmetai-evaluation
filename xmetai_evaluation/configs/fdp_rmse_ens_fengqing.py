# -*- coding: utf-8 -*-
"""FDP 集合场检验：Fengqing × CRA40。

流程 ``["fdp_field_scores", "fdp_ens_crps", "fdp_activity_spectrum"]`` 三段按顺序
跑、结果合并落同一个 output_dir。与 ``fdp_rmse_single_fengqing`` 的唯一区别是中间
多挂了 ``fdp_ens_crps``——同一批集合样本上既看确定性误差（集合均值 vs 实况），也看
集合分布本身的质量（CRPS 直接吃原始成员，不走均值）。

其余（数据源、要素表、时段、执行）与单场那份逐字一致，改路径记得两边都改。
"""
from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# 与 fdp_rmse_single_fengqing 同一张表，理由见那份的注释。
VARS = ["z500", "t2m", "u10", "v10"]

# 比单场那份多了 crps / spread_error。
#   crps         只路由给 z500（xu 也只对它算）
#   spread_error 逐变量都点名——它出 Spread / RMSE(集合平均) / 两者之比三个数，
#                对标老仓 regr_ens 无条件出的 spread_*.csv 与 spread_rmse_ratio_*.csv
VAR_METRICS = {
    "z500": ["rmse", "bias", "acc", "activity", "spectrum", "crps", "spread_error"],
    "t2m": ["rmse", "bias", "acc", "spread_error"],
    "u10": ["rmse", "bias", "acc", "spread_error"],
    "v10": ["rmse", "bias", "acc", "spread_error"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# ACC 口径：True = 经典皮尔逊；False = FDP / WeatherBench2 的 uncentered。
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# spread 的归一口径：不写 = 除以 M−1 无偏（FDP 口径）；对拍老仓 ref_result 才写
# ddof: 0（除以成员数 N）。要跟老档案对账时把这行打开。
# METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}

_FENGQING_ROOT = "/mnt/d/weather_main/chunyan_weather_test_bash/fengqing"
_CRA_ROOT = "/mnt/d/weather_main/chunyan_weather_test_bash/cra_root"

cfg = EvalConfig(
    name="fdp_rmse_ens_fengqing",
    description="FDP 集合场检验（Fengqing × CRA40）：RMSE / Bias / CRPS / Spread-Error / ACC / 活跃度 / 谱",
    pipeline=["fdp_field_scores", "fdp_ens_crps", "fdp_activity_spectrum"],

    forecast_reader={
        "type": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": VARS,
        "step_hours": 6.0,
        "lead_times": [24, 30, 36, 42],
    },
    observation_reader={
        "type": "cra",
        "root_dir": _CRA_ROOT,
        "variables": VARS,
    },
    # 同单场那份：必须是 CRA40 自己的气候态，别拿 ERA5 的顶上。
    reference_reader={
        "type": "daily_climatology",
        "root_dir": "/workspace/data/worm/cra_clim_phys_14.nc",
        "smooth_days": 15,
    },

    # 与单场那份同一段起报/时效（见那份的注释：本机 fixture 只有 0820 有观测）。
    start_date="20260819",
    end_date="20260819",
    limit=None,

    output_dir="/mnt/d/fdp_rmse_ens_fengqing",
    # 三段模板的输出都要写全：field_scores → csv_long；ens_crps → csv_long；
    # activity_spectrum → csv_long + spectrum。
    writers=["csv_long", "spectrum", "json"],
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # 比单场多了 crps，单块更重；其余与单场那份同理。
    execution={
        "mode": "auto",
        "n_workers": 4,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "slice",
            "reference": "resident",
        },
        "resume": True,
    },
)
