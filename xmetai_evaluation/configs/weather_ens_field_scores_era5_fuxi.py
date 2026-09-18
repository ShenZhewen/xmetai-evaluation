# -*- coding: utf-8 -*-
"""集合场检验（FuXi 集合 × ERA5）：对标 xu ``scripts/ensemble_fuxi.sh``。

流程：``weather_ens_field_scores``（确定性误差 + 集合分布质量同跑）。
xu 这条脚本的五种指标横跨"确定性"与"概率"两类，所以既不能套
``weather_field_scores``（无 crps），也不能套 ``weather_ens_crps``（无谱/ACC）。

    xu --metrics rmse crps acc fa spectrum  ->  rmse / crps / acc / activity / zonal_spectrum
    xu --vars z500 msl u200 v200 ws200      ->  VARS
    xu --var-metrics z500:rmse,crps,acc,fa,spectrum ...  ->  VAR_METRICS

注意 crps 只路由给 z500（xu 也只对它算）：CRPS 直接吃原始成员，不走集合均值。

单位口径（与 xu 报告层一致，便于逐格对拍）：
    z500 报 m²/s²（不除 g）、msl 报 Pa、风报 m/s。
``ws200`` 由 ``sqrt(u200²+v200²)`` 现合成（预报、实况、气候态三侧都合成）。

内存：``grid_valid_time`` 会一次读入评测时段内全部有效时刻；集合还多一份成员场。
默认给的是冒烟级小区间，正式跑直接改下面的 ``start_date`` / ``end_date`` / ``limit``。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# # xu: --vars z500 msl u200 v200 ws200
# VARS = ["z500", "msl", "u200", "v200", "ws200"]
#
# # xu: --var-metrics（逐条转写；fa -> activity、spectrum -> zonal_spectrum）
# VAR_METRICS = {
#     "z500": ["rmse", "crps", "acc", "activity", "zonal_spectrum"],
#     "msl": ["rmse", "acc"],
#     "u200": ["rmse", "activity", "zonal_spectrum"],
#     "v200": ["rmse", "activity", "zonal_spectrum"],
#     "ws200": ["rmse", "activity", "zonal_spectrum"],
# }

VARS = ["z500"]

# xu: --var-metrics（逐条转写；fa -> activity、spectrum -> zonal_spectrum）
VAR_METRICS = {
    "z500": ["rmse", "crps", "acc", "activity", "zonal_spectrum"]
}


METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# xu 的 ACC 是 uncentered（FDP / WeatherBench2 口径），不是经典皮尔逊
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# xu 的谱取到 720 波（全球 0.25° 的 Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}

_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

cfg = EvalConfig(
    name="weather_ens_field_scores_era5_fuxi",
    description="集合场检验（FuXi 集合 × ERA5）：z500/msl/u200/v200/ws200 的 RMSE / CRPS / ACC / 活跃度 / 谱",
    pipeline="weather_ens_field_scores",

    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fuxi_ens_output"
        ),
        "variables": VARS,
    },
    observation_reader={
        "type": "era5_zarr",
        "stores": {
            "pl": os.environ.get(
                "ERA5_PL_STORE",
                f"{_ERA5_STORE_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
            ),
            "sfc": os.environ.get(
                "ERA5_SFC_STORE",
                f"{_ERA5_STORE_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
            ),
        },
        "variables": VARS,
    },
    reference_reader={
        "type": "daily_climatology",
        "root_dir": os.environ.get("ERA5_CLIMO", "/workspace/data/worm/era5_clim_phys_14.nc"),
        "window": 15,
    },

    start_date=os.environ.get("START_DATE", "20250102"),
    end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_ens_field_scores_era5_fuxi",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # 逐波数谱曲线（720 个点）不走 value——value 里每个分组键都会展开成明细行，
    # 2165 行/样本 × 20880 样本 ≈ 四千五百万行。改用 spectrum writer：出
    # diagnostics/spectrum_{var}.csv（全体均值）与 spectrum_by_init.csv（逐起报），
    # 对标参考实现的 det_summary_spectrum_{var}.csv / spectrum_{init}_{var}.csv。
    writers=["csv_long", "spectrum"],

    # 采样口径：每个起报各报满 15 天（60 个 step），逐个 step 出分。
    # 缺省是 valid_time——「一个有效时刻一个样本、最新起报获胜」的连续场口径；
    # 逐日起报配 15 天时效时，后一个起报的短时效会把前一个起报的长时效覆盖掉，
    # 每个起报只落得下头 4 个时效（348×4 + 末个起报 60 = 1448 行），
    # 后面 56 个 step 全在归并时被扔掉。init_lead 给每个 (起报, 时效) 一个样本。
    options={"sample_by": "init_lead"},

    # 并发与数据加载覆盖项（不写的键用推导值）：
    #   mode:       serial / threads / processes / auto（auto：单块=串行、
    #               重指标=进程、轻指标=线程；本流程含 crps+谱 → 重 → processes）
    #   n_workers:  并发数；不写时 processes=用满 CPU、threads=4
    #   chunk_days: 几天起报一块；不写 = 1（一天一块）
    #   lead_chunk_days: 一个块装几天时效；不写 = 1（一天，通常 4 个 6h 时效）。
    #               单位是天、只收整数，1 已是地板；0 = 不切（整段时效一块，最费内存）
    #   loads:      角色驻留覆盖；本流程推导值 observation=slice（era5_zarr）、
    #               reference=resident（日序气候态整 run 驻留）
    #   resume:     True 时已完成块落 output_dir/.states/，重跑跳过
    #
    # 规模：lead 到 360h、6h 步长 → 按 lead_chunk_days=1 切出 16 个时效窗；
    # 348 个起报 × 16 窗 = 5568 块，24 段各 232 块（段数 = min(n_workers, 待跑块数)）。
    #
    # 内存账：日序气候态 resident 预热实测 5.60G —— 这是 **z500 单要素**的量。
    # det 那份（weather_field_scores_era5_fuxi）评 5 个要素才是 67G，别把那个数
    # 搬过来。单 worker 工作集估算 ~1.7G（1 要素 × 4 时效 × 51 成员 = 204 个场，
    # _combine 拼接时分量与结果并存，峰值翻倍），没实测过，按估的数留余量。
    # 总量 ≈ 5.6 + 1.7 × n_workers —— n_workers=24 时约 47G。卡口是核数不是内存。
    # 驻留那 5.6G 是 fork 前预热、子进程 COW 共享的**固定份额**，与 worker 数无关，
    # 所以加 worker 只在工作集上线性花钱。
    # ⚠ 若把 VARS 放回上面注释里那 5 个要素：气候态 resident 涨到 ~67G、工作集 ×5，
    #   n_workers=24 就是 240G 量级 —— 那种配法必须把 n_workers 降到 8~12。
    execution={
        "mode": "processes",
        "n_workers": 24,       # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        # loads 这两行就是推导值，写出来是为了好改。代价：写死之后不再自动
        # 跟随数据源——换了观测/参考读取器要记得回来核对一遍。
        "loads": {
            "observation": "slice",      # era5_zarr 按请求跨度切，现读现弃
            "reference": "resident",     # 日序气候态整 run 驻留（z500 实测 5.6G，fork 共享 1 份）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
