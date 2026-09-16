"""s2s 确定性预测 — 生产版（服务器 /gpu 路径，对齐原 step1_5_流程编排.py 的 CONFIG）。

流水线：step1 CRA 观测距平 → step2 模式（Fengshun v2.0=my_model）距平
→ step3 周平均 → step4 TCC/RMSE 柱状图 → step5 TCC/p-value/RMSE 空间图。

气候态是一次性投资：step1 CLIM（CRA 2004-2023）和 step2 CLIM（模式
hindcast 2004-2023）算完落盘永久复用，脚本自带"已存在即跳过"；
日常只需 steps: ["step4", "step5"]。换变量（tp/ssr/ssrd）时克隆本 config，
改 var + 路径里的变量名即可。
"""
CONFIG = {
    "capability": "s2s_det",
    "output_name": "s2s_tcc_single_multi",
    "steps": ["step1", "step2", "step3", "step4", "step5"],

    "var": "t2m",
    "year": 2024,
    "clim_years": [2004, 2023],
    "diff_years": [2024, 2025],
    "lead_time": 60,

    # step1：CRA 原始观测 {obs_raw}/{year}/{yyyymmdd}.nc（0.25°）
    "obs_raw": "/gpu/zhouchg/zhouchg/CRA/output/t2m",
    "obs_clim": "/gpu/zhouchg/liuzh/step1_output/CRA_1.5/clim_1p5/t2m",
    "obs_diff": "/gpu/zhouchg/liuzh/step1_output/CRA_1.5/diff_1p5/t2m",
    "obs_anom": "/gpu/zhouchg/liuzh/step1_output/CRA_1.5/anom_1p5/t2m",
    "obs_combine": "/gpu/zhouchg/liuzh/step1_output/CRA_1.5/anom_combine_1p5/t2m",

    # step2：模式原始预报 {model_raw}/{yyyymmdd}.nc（member×60×121×240）
    "model_raw": "/gpu/zhouchg/wangchp/FDP/Fengshun_v2.0/t2m",
    "model_clim": "/gpu/zhouchg/liuzh/step2_output/clim_1p5/t2m",
    "model_anom": "/gpu/zhouchg/liuzh/step2_output/anom_1p5/t2m",
    "clim_start": 2004,
    "clim_end": 2023,
    "anom_year": 2024,

    # step3：周平均输出（= step4/5 的输入）
    "weekly_obs": "/gpu/zhouchg/liuzh/step3_output/CRA_1.5/anom_combine_1p5_week/t2m",
    "weekly_model": "/gpu/zhouchg/liuzh/step3_output/my_model/anom_combine_1p5_week/t2m",

    # step4/5：三套周平均数据模板 + 起报区间
    "model_sources": ["my_model", "other_model"],
    "reference_source": "CRA_v1.5",
    "reference_path": "/gpu/zhouchg/liuzh/step3_output/CRA_1.5/anom_combine_1p5_week/t2m/{yyyymmdd}.nc",
    "my_model_path": "/gpu/zhouchg/liuzh/step3_output/my_model/anom_combine_1p5_week/t2m/{yyyymmdd}.nc",
    "other_model_path": "/gpu/zhouchg/liuzh/module1_other_model/anom_combine_1p5_week/t2m/{yyyymmdd}.nc",
    "init_start": "0101",
    "init_end": "1231",
    "cartopy_dir": "/home/nmic/project/cartopy",

    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/s2s_tcc_single_multi",
}
