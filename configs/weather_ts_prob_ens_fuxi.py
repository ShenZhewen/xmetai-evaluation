"""Example config: ensemble probability TS scores (AROC/BSS).

BLOCKER: This capability requires categorical probability modules
(reader_categorical, station_categorical, metric_categorical) which are
missing from the reference source. Cannot be implemented without these modules.
"""

CONFIG = {
    "capability": "weather_ts_ens_prob",
    "root": "/workspace/data/shenzw/fuxi_ens_output",
    "station_dir": "/workspace/data/worm/r0/2025",
    "station_list": "/workspace/data/worm/r0/zd_sta_10285.dat",
    "window_ts": [6.0, 24.0],
    "window_aroc": [6.0],  # AROC/BSS windows
    "step": 6.0,
    "var": "TP",
    "ref_result": "/workspace/data/worm/r0/ref_result",  # External reference probabilities
    "workers": 32,
    # name 决定落盘文件名：aroc_bss_prob_ens_fuxi.csv + prob_ens_fuxi_meta.json。
    "name": "prob_ens_fuxi",
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/weather_ts_prob_ens_fuxi",
}
