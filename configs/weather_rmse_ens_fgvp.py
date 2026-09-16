"""Ensemble FGVP (iwc_fgvp_mamba_ens) evaluation config.

对标 weather_rmse_ens_fuxi.py 写的——同一套集合批处理路径（--summarize-ens），
被评对象是 D:\\output 里新接的 iwc_mamba_260909.onnx（51 成员）。

本批只评两个起报（20250102、20250103），用途是**打通链路**，不是出结论：
两天样本量下 crps / fa / spectrum 都算得出来但没有统计意义，不要拿这批数
去和 weather_rmse_ens_fuxi 的 349 天归档比大小。要出结论把 periods 改宽重跑。
"""

CONFIG = {
    "type": "batch",
    "label": "fgvp-ens",
    "output_name": "weather_rmse_ens_fgvp",
    # 必须与推理侧 configs/fgvp_ens.py 的 output_dir 一致
    "pred_root": "/workspace/data/shenzw/fgvp_ens_output",
    "target_zarr": [
        "/workspace/data/liujunjie/era5_foundation_store2/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "/workspace/data/liujunjie/era5_foundation_store2/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ],
    # 这个键现在**只被打印，不决定任何东西落在哪**（runner.py:113→136→151 取出、
    # 打印、往下传；batch_adapter.py:260 只 print，真正传下去的是 :290 的
    # outdir_root=str(work_root)）。真实归档位置由 work_root 决定：
    #   <仓库根>/outputs/results/<output_name>/<配置指纹>/
    # 也就是下面这个值再往下两级。留这个键只是为了和其余 batch config 同构
    # （缺了会 KeyError），填一个与实际情况相符的路径，别填 _XMETAI_test_results
    # 那种 shell 时代的遗留假路径——它印出来会把人引到空目录去。
    "outdir_root": "/workspace/szwCode/xmetai-eval_pro/outputs",
    # periods 是**起报日期**（不是检验日期），_find_dates 按它扫 pred_root 下的
    # 日期目录。推理侧 times="2025010200..2025010300:24" 正好产出这两天。
    # 检验窗口会自然延伸到 20250103+360h≈20250118，ERA5 store 覆盖得到。
    "periods": [
        ("20250102", "20250103"),
    ],
    "metrics": ["rmse", "crps", "acc", "fa", "spectrum"],
    # 只能列推理侧真正落盘的变量：configs/fgvp_ens.py 的 vars 是
    # "z500,u200,v200,msl,tp"。ws200 由 u200/v200 派生，不是落盘通道。
    # tp 在集合路径下没有先例（weather_rmse_ens_fuxi 也没评），先不列；
    # 要评 tp 得先确认集合侧支持，别直接加。
    "variables": ["z500", "msl", "u200", "v200", "ws200"],
    "var_metrics": {
        "z500": ["rmse", "crps", "acc", "fa", "spectrum"],
        "msl": ["rmse", "acc"],
        "u200": ["rmse", "fa", "spectrum"],
        "v200": ["rmse", "fa", "spectrum"],
        "ws200": ["rmse", "fa", "spectrum"],
    },
    "climo": "/workspace/data/worm/era5_clim_phys_14.nc",
    # 没有 pred_q_scale —— 本批变量表里没有 q，这个键不生效，写了是死配置。
    # 但**将来要评 q700 时必须回来确认单位**：single 版 fgvp 是 g/kg
    # （weather_rmse_single_fgvp.py 设了 0.001），同族的 ens 版大概率也是，
    # 但这是推测不是实测。判据：单位错的症状是**只有 q 异常、且误差不随时效发散**
    # （见 PAPER.md 4.4 节 q700 那段）；若 q 的 RMSE 是别的变量中位数的数倍且逐
    # 时效比值单调收敛，先查单位再谈精度。
    # 本批只 2 个日期，起 2 个 worker 就够（多于日期数纯属浪费）。
    # 改宽 periods 时记得一起提上去，照 fuxi_ens 那份是 48。
    "n_workers": 2,
    "worker_fallback": [2, 1],
    "resume": True,
    "summarize_mode": "--summarize-ens",
    "env_overrides": {
        "VFC_ENS_ENSMEAN_ONLY": "1",
        "VFC_ENS_BLOCK": "4",
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
