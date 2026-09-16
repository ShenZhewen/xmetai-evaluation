"""Single FGVP (pinpu 输出版本) TS config.

Copy of weather_ts_single_fgvp.py，仅换 pred/name/输出目录，
站点、阈值、窗口口径与原 config 完全一致。
"""
CONFIG = {
    "capability": "weather_ts_det",
    "pred": "/workspace/data/shenzw/fgvp_output_pinpu",
    "station_dir": "/workspace/data/worm/r0/2025",
    "station_list": "/workspace/data/worm/r0/zd_sta_10285.dat",
    # 20251217 起预报数据缺失/无效，不参与评测。
    "start_date": "20250101", "end_date": "20251216",
    "windows": [24.0], "lead_step": 6.0, "tz_shift": 8.0,
    "interp": "bilinear", "tp_scale": 1.0, "workers": 8,
    # name 决定落盘文件名：ts_single_fgvp_pinpu.csv + single_fgvp_pinpu_meta.json。
    "name": "single_fgvp_pinpu",
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/weather_ts_single_fgvp_pinpu",
}
