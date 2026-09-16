CONFIG = {
    "capability": "weather_ts_ens",
    "pred": "/workspace/data/shenzw/fuxi_ens_output",
    "station_dir": "/workspace/data/worm/r0/2025",
    "station_list": "/workspace/data/worm/r0/zd_sta_10285.dat",
    # 20251217 起预报数据缺失/无效，不参与评测。
    "start_date": "20250101", "end_date": "20251216",
    "windows": [24.0], "lead_step": 6.0, "tz_shift": 8.0,
    "interp": "bilinear", "tp_scale": 1.0, "workers": 8,
    # name 决定落盘文件名：ts_ens_fuxi.csv + ens_fuxi_meta.json。
    "name": "ens_fuxi",
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/weather_ts_ens_fuxi",
}
