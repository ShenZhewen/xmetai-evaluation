# -*- coding: utf-8 -*-
"""FDP 要素检验（确定性口径）：Fengqing × CRA40。

流程 ``["fdp_field_scores", "fdp_activity_spectrum"]`` 两段按顺序跑、结果合并落
同一个 output_dir：前段出误差族 RMSE / Bias 与距平族 ACC，后段出活跃度比与功率谱。
两段模板都带 ``ensemble_mean``，所以评的是集合平均场当确定性预报。

数据源三件套就是本仓库 weather 系列换了个来源（预报 fengqing / 观测 cra /
参考气候态）。⚠ 预报侧要用 ``fengqing`` 而不是 ``fengqing_phys``——后者 z500 报
m²/s²，与 CRA 的 gh（米）对不上（前者按 1/g 换算成米）。
"""
from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# 预报（FENGQING_LAYOUT）与观测（CRA_LAYOUT）两边布局都有的要素。
# msl / tp 暂不列入：实测本 cra_root 的两个 grib 里都没有 tp，而 msl 落在
# ART_ATM、CRA_LAYOUT 却把它归到 SURFACE 组——读文件是按组选的，归错组就读不到。
# 换成服务器上完整的 CRA40 归档后可以把它们加回这个表。
VARS = ["z500", "t2m", "u10", "v10"]

# 逐变量指标表 -> metric_options。⚠ 两段用到的 5 个指标都必须点到某个变量下，
# 漏点名的会拿到整批多变量数据并直接报错（框架有意防静默出假数）。
VAR_METRICS = {
    "z500": ["rmse", "bias", "acc", "activity", "spectrum"],
    "t2m": ["rmse", "bias", "acc"],
    "u10": ["rmse", "bias", "acc"],
    "v10": ["rmse", "bias", "acc"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# ACC 口径：True = 经典皮尔逊（距平再减域加权均值）；False = FDP / WeatherBench2
# 的 uncentered 口径。对标 FDP 用 False。
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}

_FENGQING_ROOT = "/mnt/d/weather_main/chunyan_weather_test_bash/fengqing"
_CRA_ROOT = "/mnt/d/weather_main/chunyan_weather_test_bash/cra_root"

cfg = EvalConfig(
    name="fdp_rmse_single_fengqing",
    description="FDP 要素检验（Fengqing × CRA40）：RMSE / Bias / ACC / 活跃度 / 功率谱",
    pipeline=["fdp_field_scores", "fdp_activity_spectrum"],

    forecast_reader={
        "type": "fengqing",     # 集合；本流程先用 ensemble_mean 降成确定性场
        "root_dir": _FENGQING_ROOT,
        "variables": VARS,
        "step_hours": 6.0,      # 布局步长：时效取文件名里的 3 位 × 6h
        # 本流程没有累积窗（变换只有 ensemble_mean），每个时效都是独立样本，
        # 所以这里**可以**写稀疏集；降水那两份必须按 6h 密排，不一样。
        "lead_times": [24, 30, 36, 42],
    },
    observation_reader={
        "type": "cra",
        "root_dir": _CRA_ROOT,
        "variables": VARS,
    },
    # 气候态：ACC / 活跃度必需（RMSE / Bias 用不上，但给了不构建）。
    # ⚠ 必须是 **CRA40 的** 气候态——拿 ERA5 的配 CRA 实况不同源，ACC 会系统性偏，
    # 而框架不会拦（文件本身合法）。改成你服务器上的路径。
    reference_reader={
        "type": "daily_climatology",
        "root_dir": "/workspace/data/worm/cra_clim_phys_14.nc",
        "smooth_days": 15,      # ±7.5 天环形滑动、跨年首尾相接，抹平单日噪声
    },

    # 只管起报日范围。本机 fixture 的 cra_root 只有 20260820 一天，所以起报钉在
    # 0819、时效 24~42 → 有效时刻正好是 0820 的 00/06/12/18，四个样本全配得上。
    # 0820 起报的 6/12/18 也有观测，要一起评就把 end_date 改 "20260820"、并在
    # lead_times 里加上 6/12/18；那两组跨不上的组合会各报一次 "No cra files
    # found" 后跳过，不影响其余样本。
    start_date="20260819",
    end_date="20260819",
    limit=None,             # 限起报数，直接改这里；None = 不限

    output_dir="/mnt/d/fdp_rmse_single_fengqing",
    # ⚠ 配置级 writers 是**替换**模板的，所以必须把两段模板的输出都写全：
    # fdp_field_scores 给 csv_long，fdp_activity_spectrum 给 csv_long + spectrum。
    # 漏了 spectrum 就没有逐波数曲线（30 个点的曲线走长表会爆行数）。
    writers=["csv_long", "spectrum", "json"],
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # 可用键就 mode / n_workers / chunk_days / lead_chunk_days / loads / resume，
    # 只覆盖写到的键（不写 = 按指标族与数据源形态推导），写错键名直接报配置错。
    execution={
        "mode": "auto",         # 本流程是轻指标 + 少量块；块数上去会落 threads
        "n_workers": 4,
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效
        "loads": {
            "observation": "slice",     # CRA40 GRIB2 按请求跨度现读现弃
            "reference": "resident",    # 气候态整 run 驻留一份，fork 共享
        },
        "resume": True,         # 已完成块的状态落 .states/，重跑跳过
    },
)
