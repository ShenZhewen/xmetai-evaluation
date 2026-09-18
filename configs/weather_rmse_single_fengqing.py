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
    # 只被打印，不决定落盘（真实归档 outputs/results/<output_name>/<配置指纹>/，
    # 见 core/batch_adapter.py:290 的 outdir_root=str(work_root)）。
    "outdir_root": "/workspace/szwCode/xmetai-eval_pro/outputs",
    "periods": [
        # 20251217 起预报数据缺失/无效（q700 全 NaN、单日内存暴涨 10 倍），止于 20251216。
        # **不要**再拆回 (0101-0630)+(0701-1216) 两段：regr_ens 的收尾步骤只按
        # **当次 CLI 的 --dates** 重写 summary.csv / batch_meta.json（见 vfc/regr_ens.py
        # 的 done_dates 取 _all_candidates，不是扫描 outdir_root），后一段会把前一段
        # 整个盖掉——归档里躺着 349 个日期目录，summary 只剩 169 行。
        # 合并不改任何逐日结果，只改这份日期清单。
        ("20250101", "20251216"),
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
