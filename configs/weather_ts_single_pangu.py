"""Single Pangu TS config.

Copy of weather_ts_single_fgvp.py，仅换 pred/name/输出目录。
⚠ 前提：pangu_output 里必须有 tp（24h 累积降水）通道——Pangu-Weather
原版 69 通道不含降水，如果推理框架没额外落盘 tp，这个 config 跑不了。
"""
CONFIG = {
    "capability": "weather_ts_det",
    "pred": "/workspace/data/shenzw/pangu_output",
    "station_dir": "/workspace/data/worm/r0/2025",
    "station_list": "/workspace/data/worm/r0/zd_sta_10285.dat",
    # 20251217 起预报数据缺失/无效，不参与评测。
    "start_date": "20250101", "end_date": "20251216",
    "windows": [24.0], "lead_step": 6.0, "tz_shift": 8.0,
    "interp": "bilinear", "tp_scale": 1.0, "workers": 8,
    # name 决定落盘文件名：ts_single_pangu.csv + single_pangu_meta.json。
    "name": "single_pangu",
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/weather_ts_single_pangu",
}
