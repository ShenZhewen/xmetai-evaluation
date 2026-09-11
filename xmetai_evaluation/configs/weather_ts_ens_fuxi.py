# -*- coding: utf-8 -*-
"""集合降水分类检验（FuXi 集合 × Diamond 站点）：一条命令跑两段。

口径由负责人给定，对标原版 ``run_categorical.py``（``--window-ts 24 --window-aroc 6``）：

    窗口 24  集合平均场   TS / POD / FAR / 频率偏差        -> weather_ts_ens
    窗口 6   逐成员        AROC / BS / BSS（要气候概率）    -> weather_ts_ens_prob

两段的窗口不同，而一条模板只有一个 ``window_hours``（它同时决定观测累积长度和
有效时效的筛选），所以配置里用列表声明两条模板，框架按顺序各跑一段、结果合并后
落盘一次：

    scores.csv                        两段的统一长表，靠 window_h 列区分（24 / 6）
    diagnostics/categorical_wide.csv  仅 24h 的 TS 宽表
    diagnostics/probability_wide.csv  仅 6h 的 AROC/BS/BSS 宽表
    manifest.json                     运行记录（segments 里有每段的窗口）

阈值不在这里配：两条模板自带的阈值常量（24h 的 [0.1,10,25,50,100]、
6h 的 [0.1,4,13,25]）已与原版输出逐格一致，且 6h 那组必须与参考文件的 4 列同集合。

数据：
    预报  {fuxi_ens_output}/YYYYMMDD/member_*/001.nc…（TP，逐 6h）
    观测  Diamond 站点降水（北京时，协议默认 +8 对齐）
    参考  气候概率目录 ref/MMDDHH.000（**必填**，见 BSS_REF_DEFAULT）
"""

from xmetai_evaluation.configs.base import EvalConfig

#: BSS 的外部气候概率目录（``ref/MMDDHH.000``：站号 + 4 个超越式概率）。
#: 负责人给定的固定输入，对应原版的 ``--ref``；换目录直接改这一行。
BSS_REF_DEFAULT = "/workspace/data/worm/r0/ref"

cfg = EvalConfig(
    name="weather_ts_ens_fuxi",
    description="FuXi 集合降水评估：24h 集合平均 TS + 6h 逐成员概率 AROC/BS/BSS",
    # 两段：先 24h 的 TS，再 6h 的概率评分（顺序即执行顺序，结果合并落盘）
    pipeline=["weather_ts_ens", "weather_ts_ens_prob"],

    forecast_reader={
        "type": "fuxi_ens",
        "root_dir": "/workspace/data/shenzw/fuxi_ens_output",
        "variable": "tp",
        "step_hours": 6.0,
        # 只读部分时效可显著降内存（如 [6, 12, 18, 24]）；
        # 注意 24h 段要求时效里有 24 的倍数，裁太短会让那一段一个样本都跑不出来
        "lead_times": None,
    },
    observation_reader={
        "type": "station",
        "root_dir": "/workspace/data/worm/r0/2025",
        "variable": "precipitation",
        "station_list": "/workspace/data/worm/r0/zd_sta_10285.dat",
    },
    # BSS 的气候概率参考。必须给：缺了会降级成样本气候频率 r(1-r)，
    # 那算出来的 BSS 与原版不可比。
    reference_reader={"type": "ref_probability", "root_dir": BSS_REF_DEFAULT},

    start_date="20250101",
    end_date="20251231",
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir="/workspace/szwCode/xmetai-evaluate/evaluation_results/ts_multi_fuxi_ens",
    log_level="INFO",
)
