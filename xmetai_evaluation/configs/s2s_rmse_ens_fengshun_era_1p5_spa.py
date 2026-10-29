# -*- coding: utf-8 -*-
"""集合场检验：s2s_gssp_gdn2 + SPA 注入位置实验（多臂、单起报、只评 z500）。

评的是 ``tools/s2s_exp_job.yaml``（推理 → 评测一条龙）一次串起来跑出的全部臂：

    /workspace/data/shenzw/s2s_spa_exp/forecast/<arm>/20240901/member_XXX/0NN.nc

``001.nc`` → +24h（日步长，``step_hours=24``），60 步到 +1440h = 2024-10-31，
实况与推理输入是**同一个盘** ``s2s.19500101_20260417.c76.1p5.h24.v1``
（1.5°、每日 00Z，与预报同网格；读取器按名取 z500 并反归一化）。

**本档刻意只做最小可比集**（定注入锚点用，不出版面）：

  * 1 个起报（20240901）——单起报只能筛「有没有崩」，臂间判读要等多起报；
  * 只评 z500（推理侧 ``vars="z500"``，落盘也只有它）；
  * 指标保留 rmse / crps / spread_error / 谱，谱与球谐带是 SPA 的必看项。

**加臂只改 ``ARMS`` 一行**，且必须与 Job yaml 里的 ``ARMS`` 同名同序：这里定义
``cfgs``（复数）走批量入口，CLI 顺序逐臂评、单臂失败不中断。推理 config 的
``output_dir`` 末级必须等于臂名（``s00`` / ``s04`` / ``s08`` / ``s12`` / ``tau10`` /
``ls08`` / ``r32`` / ``pl32``）。

口径：z500 报 m²/s²（不除 g）、不配气候态（``weather_ens_field_scores_noref``，
无 ACC / 活跃度）、spread 用 ddof=0。
"""
from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 实验臂（字面量，改这里加臂；与推理 config 的 output_dir 末级同名）──
#   s00    σ=0 空转对照          s04    σ=0.04 基线（幅度轴）
#   s08    σ=0.08                s12    σ=0.12（幅度轴上端）
#   tau10  τ=10d（时间记忆轴）    ls08   length_scale=8（空间尺度轴，换基）
#   r32    rank=32（基实验，高斯基） pl32   rank=32 + 幂律谱 α=2.5（与 r32 配对）
ARMS = ("s00", "s04", "s08", "s12", "tau10", "ls08", "r32", "pl32")

# ── 路径 ──
# SPA 实验的推理与评测产物都收在同一个实验目录下：
#   .../s2s_spa_exp/forecast/<arm>/ = 推理落盘（forecast）
#   .../s2s_spa_exp/eval/<arm>/     = 评测产物（eval）
FORECAST_ROOT = "/workspace/data/shenzw/s2s_spa_exp/forecast"
# 实况用推理的同一个盘（自洽评测）；只评 z500，tp 的 expm1 口径不涉及。
TRUTH_STORE = "/workspace/data/zarr_refresh_v3/s2s.19500101_20260417.c76.1p5.h24.v1"
OUTPUT_ROOT = "/workspace/data/shenzw/s2s_spa_exp/eval"

# ── 要素表：本批只评 z500 ──
VARS = ["z500"]

VAR_METRICS = {
    "z500": ["rmse", "crps", "spread_error", "zonal_spectrum", "spherical_bands"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# max_wavenumber=120（全球 1.5° Nyquist = 240/2）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 120,
}
# ddof=0（除 N，对齐 ref_result 的 spread_*.csv）
METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}

# ── 时效 ──
LEADS = list(range(24, 1441, 24))


def make_cfg(arm: str) -> EvalConfig:
    """一臂一份 EvalConfig：只有 run 名与目录随臂变，口径完全相同。"""
    return EvalConfig(
        name=f"fengshun_era_1p5_spa_{arm}",
        description=(
            f"SPA 注入位置实验（s2s_gssp_gdn2 + 潜空间 AR(1) 强迫，臂 {arm}）："
            "1 个起报 20240901、51 成员、24…1440h 日步长，只评 z500 的 "
            "RMSE / CRPS / spread / 谱"
        ),
        pipeline="weather_ens_field_scores_noref",

        forecast_reader={
            "type": "fuxi_ens_phys",
            "root_dir": f"{FORECAST_ROOT}/{arm}",
            "variables": VARS,
            "step_hours": 24.0,
            "lead_times": LEADS,
        },
        observation_reader={
            "type": "cra_zarr",
            "path": TRUTH_STORE,
            "variables": VARS,
        },

        periods=[("20240901", "20240901")],
        limit=None,

        output_dir=f"{OUTPUT_ROOT}/{arm}",
        metric_options=METRIC_OPTIONS,
        log_level="INFO",

        writers=["csv_long", "spectrum"],

        options={
            "sample_by": "init_lead",
            "regions": {},
        },

        execution={
            "mode": "processes",
            "n_workers": 12,
            "chunk_days": 1,
            "lead_chunk_days": 1,
            "loads": {
                "observation": "window:16",
            },
            "resume": True,
        },
    )


cfgs = [make_cfg(arm) for arm in ARMS]
