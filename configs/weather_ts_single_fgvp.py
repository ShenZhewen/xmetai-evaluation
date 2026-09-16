CONFIG = {
    "capability": "weather_ts_det",
    "pred": "/workspace/data/shenzw/fgvp_output",
    "station_dir": "/workspace/data/worm/r0/2025",
    "station_list": "/workspace/data/worm/r0/zd_sta_10285.dat",
    # 20251217 起预报数据缺失/无效，不参与评测。
    "start_date": "20250101", "end_date": "20251216",
    "windows": [24.0], "lead_step": 6.0, "tz_shift": 8.0,
    "interp": "bilinear", "tp_scale": 1.0, "workers": 8,
    # name 决定落盘文件名：ts_single_fgvp.csv + single_fgvp_meta.json。
    # 不写的话会从 pred 根目录名生成（以前是 ts_fgvp_output.csv 这种）。
    "name": "single_fgvp",
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/weather_ts_single_fgvp",
}
