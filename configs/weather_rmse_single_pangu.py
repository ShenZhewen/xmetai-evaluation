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
    # 只被打印，不决定落盘（真实归档 outputs/results/<output_name>/<配置指纹>/，
    # 见 core/batch_adapter.py:290 的 outdir_root=str(work_root)）。
    "outdir_root": "/workspace/szwCode/xmetai-eval_pro/outputs",
    "periods": [
        # 20251217 起预报数据缺失/无效，止于 20251216，单段跑满 349 天。
        # **不要**再拆回 (0101-0630)+(0701-1216) 两段：regr_ens 的收尾步骤只按
        # **当次 CLI 的 --dates** 重写 summary.csv / batch_meta.json（见 vfc/regr_ens.py
        # 的 done_dates 取 _all_candidates，不是扫描 outdir_root），后一段会把前一段
        # 整个盖掉——归档里躺着 349 个日期目录，summary 只剩 169 行。
        # 合并不改任何逐日结果，只改这份日期清单。
        ("20250101", "20251216"),
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
    # pred_q_scale 故意不写：默认 1.0，Pangu 落盘 q 已是 kg/kg。
    # 2026-09-16 实锤：pangu_output/20250102/001.nc 的 Q700 mean=0.00230、
    # max=0.0118（同期 Z500 mean=54089、MSL mean=100997，均为标准气象单位）。
    # 曾经抄 fgvp 的 0.001 是错的：会把 q 再缩 1000 倍压成 ~0，q700 RMSE 退化成
    # 「观测自身的 RMS」≈4.4 g/kg 且**不随时效增长**（其余变量全正常）。这个特征
    # 和 FengQing 那次一模一样，见 weather_rmse_single_fengqing.py 的注释。
    # 判据：raw q 值域 ~1-10 = g/kg（要写 0.001）；~0.001-0.01 = kg/kg（不写）。
    # 普通日期单 worker <10G，48 并发吞吐最高；fallback 留 4 作纯保底。
    "n_workers": 48,
    "worker_fallback": [48, 4, 2],
    "resume": True,
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
