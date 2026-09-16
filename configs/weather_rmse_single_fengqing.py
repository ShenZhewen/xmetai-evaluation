"""Single FengQing deterministic evaluation config.

Migrated from configs/weather_rmse_single_fgvp.py (FengQing 输出通道是其子集)。

与 fgvp 版的关键差异 —— **不写 pred_q_scale**：
  FengQing 推理框架输出 q = kg/kg（官方 upper_mean.npy q 块 mean≈0.0018
  实锤，与 FuXi/FGVP 的 g/kg 相反），regr_ens 默认 pred_q_scale=1.0 正确。
  抄 fgvp 的 0.001 会把预报 q 缩成 1e-6 量级，q700 RMSE 大到离谱
  （只有 q 异常、其他变量全正常就是这个错的特征）。
"""
from pathlib import Path

CONFIG = {
    "type": "batch",
    "label": "fengqing",
    "output_name": "weather_rmse_single_fengqing",
    "pred_root": "/workspace/data/fanpy/fengqing-output",
    "target_zarr": [
        "/workspace/data/liujunjie/era5_foundation_store2/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "/workspace/data/liujunjie/era5_foundation_store2/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ],
    "outdir_root": "/workspace/_XMETAI_test_results/single_fengqing",
    "periods": [
        ("20250101", "20250630"),
        # 20251217 起预报数据缺失/无效（q700 全 NaN、单日内存暴涨 10 倍），
        # 该窗口不参与评测，止于 20251216。
        ("20250701", "20251216"),
    ],
    "metrics": ["rmse", "spectrum", "acc", "fa"],
    # FengQing 推理落盘只有 10 个通道（configs/fengqing.py 的 vars 去掉
    # 不参与 rmse 评分的 tp），fgvp 配置里的 d2m/u200/v200 无预报数据。
    # ws10m/ws850 是派生要素（= sqrt(u^2+v^2)，评测时现算，不需要预报落盘），
    # 但显式传 --vars 时不会自动补入（regr_ens.py:310），所以这里要写进清单。
    "variables": [
        "z500", "q700", "t700", "t2m", "t850", "msl",
        "u850", "v850", "u10m", "v10m", "ws10m", "ws850",
    ],
    "var_metrics": {
        "z500": ["rmse", "spectrum", "acc", "fa"],
        "q700": ["rmse"],
        "t700": ["rmse", "spectrum"],
        "t2m": ["rmse", "spectrum"],
        "t850": ["rmse"],
        "msl": ["rmse", "spectrum"],
        "u850": ["rmse", "fa", "spectrum"],
        "v850": ["rmse", "fa", "spectrum"],
        "u10m": ["rmse", "fa"],
        "v10m": ["rmse", "fa"],
        "ws10m": ["rmse", "fa"],
        "ws850": ["rmse", "fa", "spectrum"],
    },
    "climo": "/workspace/data/worm/era5_clim_phys_14.nc",
    # pred_q_scale 故意不写：默认 1.0，FengQing 输出 q 已是 kg/kg。
    # 2026-09-15 日志实锤：48 workers 下普通日期（单 worker 内存 <10G）吞吐最高。
    # 原 20251217-20251228 坏数据窗口已在 periods 里排除；fallback 留 4 作纯保底。
    "n_workers": 48,
    "worker_fallback": [48, 4, 2],
    "resume": True,
    "resume_cache": True,
    "summarize_mode": "--summarize-det",
    "env_overrides": {
        "VFC_DATES_PER_CHILD": "1",
        "VFC_CLIMO_ROLLING": "1",
        "VFC_SINGLE_STREAM": "1",
        "VFC_OBS_BLOCK": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    },
}
