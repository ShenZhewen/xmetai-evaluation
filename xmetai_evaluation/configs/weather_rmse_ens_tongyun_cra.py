# -*- coding: utf-8 -*-
"""集合场检验（Tongyun CRA 版 × CRA-40 c71 zarr）：RMSE / CRPS / spread / 谱。

与 ``weather_rmse_ens_fgvp``（FuXi 集合 × ERA5）同一条流程骨架，两处不同：

  * **实况源换成 ``cra_zarr``**：CRA 的 store 存的是归一化值、tp 在 ``log1p``
    空间，读取器（``CraZarrReader``）负责 ``x*std+mean`` 与 ``expm1``，
    配置侧写标准名即可。
  * **不算 ACC / 预报活跃度**：没有日气候态参考，所以走
    ``weather_ens_field_scores_noref``（同批指标去掉 ACC/活跃度），
    不给 ``reference_reader``。

单位：z500 报 m²/s²（不除 g）、msl 报 Pa、t2m 报 K、tp 报 mm（6h 累积）。
"""

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 路径（字面量，直接改这里）──
FORECAST_ROOT = "/workspace/data/shenzw/tongyun_cra"
CRA_STORE = "/workspace/data/zarr_refresh_v3/cra.19790101_20260714.c71.p25.h6.v1"
OUTPUT_DIR = "/workspace/data/shenzw/evaluation_results/weather_rmse_ens_tongyun_cra"

# ── 要素表 ──
VARS = ["z500", "msl", "t2m", "tp"]

# ── 逐变量指标表 ──
# 没有气候态，所以只有这五个指标可用；谱/球谐带默认只给 z500（与 fgvp 同口径）。
VAR_METRICS = {
    "z500": ["rmse", "crps", "spread_error", "zonal_spectrum", "spherical_bands"],
    "msl": ["rmse", "crps", "spread_error"],
    "t2m": ["rmse", "crps", "spread_error"],
    "tp": ["rmse", "crps", "spread_error"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# max_wavenumber=720（全球 0.25° Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}
# spherical_bands 用默认分带（老仓 (1,4)/(5,20)/(21,40)/(41,64)/(65,128)），对拍别动。
# ddof=0（除 N，对齐 ref_result 的 spread_*.csv）
METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}

# ── 时效 ──
# CRA store 的时间轴止于 2026-07-14；20260701 00Z 起报能精确对上的最长时效是
# +318h（60 步跑到 +360h 会越界，实况读取器会报"时间轴里没有这些有效时刻"）。
# 起报换成更晚的日子就把上限跟着收窄（上限 = store 末尾 − 起报时刻）。
LEADS = list(range(6, 319, 6))

cfg = EvalConfig(
    name="weather_rmse_ens_tongyun_cra",
    description=(
        "集合场检验（Tongyun CRA × CRA-40 c71 zarr）："
        "z500/msl/t2m/tp 的 RMSE / CRPS / spread / 谱（不做 ACC / 活跃度）"
    ),
    pipeline="weather_ens_field_scores_noref",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": FORECAST_ROOT,
        "variables": VARS,
        "step_hours": 6.0,
        "lead_times": LEADS,
    },
    observation_reader={
        "type": "cra_zarr",
        "path": CRA_STORE,
        "variables": VARS,
    },
    # 不给 reference_reader：本流程的指标都不需要气候态

    # ── 时段 ──
    periods=[
        ("20260701", "20260701")
    ],
    limit=None,  # 限起报数：冒烟改 2；None = 不限

    output_dir=OUTPUT_DIR,
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

    # ── 输出视图 ──
    writers=["csv_long", "spectrum"],

    # ── 协议参数 ──
    options={
        "sample_by": "init_lead",
        "regions": {},
    },

    # ── 执行 ──
    execution={
        "mode": "processes",
        "n_workers": 18,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "window:16",
        },
        "resume": True,
    },
)
