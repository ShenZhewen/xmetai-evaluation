"""Single FGVP (pinpu 输出版本) deterministic batch config.

Copy of weather_rmse_single_fgvp.py，仅换 pred_root/outdir/输出名，
变量、指标、q 单位（g/kg → pred_q_scale 0.001）与原 config 完全一致。
"""
from pathlib import Path

CONFIG = {
    "type": "batch",
    "label": "fgvp_pinpu",
    "output_name": "weather_rmse_single_fgvp_pinpu",
    "pred_root": "/workspace/data/shenzw/fgvp_output_pinpu",
    "target_zarr": [
        "/workspace/data/liujunjie/era5_foundation_store2/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "/workspace/data/liujunjie/era5_foundation_store2/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ],
    "outdir_root": "/workspace/_XMETAI_test_results/single_fgvp_pinpu",
    "periods": [
        ("20250101", "20250630"),
        # 20251217 起预报数据缺失/无效，不参与评测。
        ("20250701", "20251216"),
    ],
    "metrics": ["rmse", "spectrum", "acc", "fa"],
    "variables": [
        "z500", "q700", "t700", "t2m", "t850", "msl", "d2m",
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
        "d2m": ["rmse"],
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
    "pred_q_scale": 0.001,  # FGVP outputs q in g/kg, target is kg/kg
    # 2026-09-15 日志实锤：普通日期单 worker <10G，48 并发吞吐最高；
    # 中间档撞坏日期一样崩，fallback 留 4 作纯保底。
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
