# -*- coding: utf-8 -*-
"""FDP 降水检验：Fengqing tp × 站点 / 格点实况。

一个文件里放两条配置（``cfgs``），因为两者的**观测读取器不一样**，而一份
``EvalConfig`` 只能有一个 ``observation_reader``：

  站点段  TS（``fdp_precip_ts``）+ 集合概率 BSS / AROC（``weather_ts_ens_prob``）
          观测 = rain/*.000（Diamond 3 站点，逐小时）。两条模板的窗口都是 6h，
          共用一套 options，所以能挂在同一个 pipeline 列表里顺序跑
  格点段  FSS（``fdp_precip_fss``）——要**格点**降水实况，当前没启用，理由见文件末尾

⚠ 站点段的时效被观测覆盖钉死了：本机 ``rain/`` 只有 20260820 的 00Z~06Z 七个文件，
所以只有有效时刻落在 T00 / T06 的样本配得上，即 0819 起报的时效 24 与 30。换到
服务器上站点数据齐了，把 ``_LEAD_TIMES`` 放开即可。

BSS 要外部气候概率参考。缺参考会降级成样本气候频率 r(1-r)——数字出得来，但基准
不是同一个口径；要正式数就把 ``_REF_PROB_ROOT`` 指对你服务器上的目录。
"""
from xmetai_evaluation.configs.base import EvalConfig

_FENGQING_ROOT = "/mnt/d/weather_main/chunyan_weather_test_bash/fengqing"
_CRA_ROOT = "/mnt/d/weather_main/chunyan_weather_test_bash/cra_root"
_STATION_ROOT = "/mnt/d/weather_main/chunyan_weather_test_bash/rain"
# BSS 的气候概率参考：ref/MMDDHH.000（站号 + 与阈值一一对应的概率列）
_REF_PROB_ROOT = "/workspace/data/worm/ref_prob"
_LEAD_TIMES = [24, 30]

def _execution():
    """每条配置各拿一份新的 execution——共用同一个 dict 会被就地改写串味。"""
    return {
        "mode": "auto",
        "n_workers": 4,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "resident",  # 站点观测整 run 驻留（模板推导的缺省也是这个）
            "reference": "resident",    # 气候概率同理
        },
        "resume": True,
    }

# ── 站点段：TS + 集合概率 ─────────────────────────────────────────────────
# 两条模板都走 station_valid_time、窗口都是 6h，且都**不做集合平均**（概率要逐成员
# 算，均值会把成员信息抹掉；TS 那条是拿插值到站的格点场直接做列联表）。
_station = EvalConfig(
    name="fdp_precip_fengqing_station",
    description="FDP 降水站点检验（Fengqing × Diamond 站点）：TS / 频率偏差 + 集合概率 BS/AROC/BSS",
    pipeline=["fdp_precip_ts", "weather_ts_ens_prob"],

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
        # 不写 station_list = 用文件里的全部站点；要只用白名单就填那个 .dat 的路径
    },
    # fdp_precip_ts 用不上参考（给了也不构建），weather_ts_ens_prob 的 BSS 用它。
    reference_reader={"type": "ref_probability", "root_dir": _REF_PROB_ROOT},

    start_date="20260819",
    end_date="20260819",
    limit=None,

    output_dir="/mnt/d/fdp_precip_fengqing_station",
    # ⚠ 配置级 writers 是替换模板的，两段模板的输出都要写全：
    # fdp_precip_ts → csv_long + categorical_wide；weather_ts_ens_prob →
    # csv_long + probability_wide。漏了对应那项，报告里就没有那张主表。
    writers=["csv_long", "categorical_wide", "probability_wide"],
    log_level="INFO",
    execution=_execution(),
)

# ── 格点段：FSS（默认不启用）─────────────────────────────────────────────
# FSS 是网格邻域检验，**不吃站点观测**，要格点降水实况。本机 cra_root 里没有降水：
# 实测 ART_ATM 与 CRA40LAND_SURFACE 两个 grib 的 shortName 都没有 tp / apcp，
# 跑起来必然 DiscoveryError。所以先不列进 cfgs，等服务器上 CRA 归档带 tp 之后
# 把下面 cfgs 里那行取消注释即可（也可以直接换成 CMPAS 的格点降水 reader）。
_grid = EvalConfig(
    name="fdp_precip_fengqing_fss",
    description="FDP 降水空间检验（Fengqing × CRA40）：FSS",
    pipeline="fdp_precip_fss",

    forecast_reader={
        "type": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": ["tp"],
        "step_hours": 6.0,
        "lead_times": [6, 12, 18, 24, 30, 36, 42],
    },
    observation_reader={
        "type": "cra",
        "root_dir": _CRA_ROOT,
        "variables": ["tp"],
    },

    start_date="20260819",
    end_date="20260819",
    limit=None,

    output_dir="/mnt/d/fdp_precip_fengqing_fss",
    # 模板给的是 ["csv_long"]；FSS 的窗口维在长表里靠 group 列区分，不用额外视图。
    writers=["csv_long"],
    log_level="INFO",
    execution=_execution(),
)

cfgs = [
    _station,
    # _grid,   # ← 取消注释即启用 FSS（前提：观测侧拿得到格点降水）
]
