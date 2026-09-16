"""fdp 确定性场检验 config（multi_model_verifier_fix.py）。

z500 + t2m/msl/u10/v10 的 RMSE、Bias（全部）、ACC（仅 z500，气候态距平）。
实况 CMA-RA (CRA40) GRIB2，气候态 CLI_6HOUR，预报 {model_data_root}/{Model}/DF/{date}/。
输出：outputs/.temp/fdp_field_det/ 逐日 CSV+图 → outputs/results/fdp_field_det/fdp_field_det.csv 长表。
"""
CONFIG = {
    "capability": "fdp_field_det",
    "output_name": "fdp_field_det",
    # 逐日起报（00 起报），区间含首尾；改 init_hour 可换 12 起报
    "start_date": "20260801",
    "end_date": "20260822",
    "init_hour": "00",
    "models": ["Fengqing", "PuYun", "YJ-TianJi", "NJU-Earth", "W2S"],
    "forecast_hours": list(range(6, 361, 6)),  # 6-360 每 6h
    "model_data_root": "/gpu/zhaochy/fdp2/FCSTDATA",
    "cra_root": "/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026",
    "cli_root": "/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR",
    # 只跑地面 / 只跑高空（都为 False 时两类都跑）
    "surface_only": False,
    "pressure_only": False,
    # 逐日起报跑完写 <date>.done 标记，重跑跳过已完成日期
    "resume": True,
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/fdp_field_det",
}
