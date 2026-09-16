"""Single Pangu deterministic batch config.

对照 weather_rmse_single_fgvp.py 写；Pangu 无 d2m 输出（其余高空/地面通道齐），
变量清单去掉了 d2m。
"""
from pathlib import Path

CONFIG = {
    "type": "batch",
    "label": "pangu",
    "output_name": "weather_rmse_single_pangu",
    "pred_root": "/workspace/data/shenzw/pangu_output",
    "target_zarr": [
        "/workspace/data/liujunjie/era5_foundation_store2/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "/workspace/data/liujunjie/era5_foundation_store2/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ],
    "outdir_root": "/workspace/_XMETAI_test_results/single_pangu",
    "periods": [
        ("20250101", "20250630"),
        # 20251217 起预报数据缺失/无效，不参与评测。
        ("20250701", "20251216"),
    ],
    "metrics": ["rmse", "spectrum", "acc", "fa"],
    "variables": [
        "z500", "q700", "t700", "t2m", "t850", "msl",
        "u200", "v200", "u850", "v850", "u10m", "v10m",
        "ws10m", "ws850", "ws200",
    ],
    "var_metrics": {
        "z500": ["rmse", "spectrum", "acc", "fa"],
        "q700": ["rmse"],
        "t700": ["rmse", "spectrum"],
        "t2m": ["rmse", "spectrum"],
        "t850": ["rmse"],
        "msl": ["rmse", "spectrum"],
        "u200": ["rmse", "fa", "spectrum"],
        "v200": ["rmse", "fa", "spectrum"],
        "u850": ["rmse", "fa", "spectrum"],
        "v850": ["rmse", "fa", "spectrum"],
        "u10m": ["rmse", "fa"],
        "v10m": ["rmse", "fa"],
        "ws10m": ["rmse"],
        "ws850": ["fa", "rmse", "spectrum"],
        "ws200": ["rmse", "spectrum"],
    },
    "climo": "/workspace/data/worm/era5_clim_phys_14.nc",
    # Pangu ONNX 官方输出 q 是 g/kg（推理框架若按原样落盘则为 g/kg → 0.001）。
    # ⚠ 单位待实锤：若 q700 RMSE 大到离谱且只有 q 异常，先查这里——
    # 预报 q 值域 ~1-10 = g/kg（本配置正确），~0.001-0.01 = kg/kg（删掉这行）。
    "pred_q_scale": 0.001,
    # 普通日期单 worker <10G，48 并发吞吐最高；fallback 留 4 作纯保底。
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
