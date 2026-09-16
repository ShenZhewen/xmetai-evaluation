"""fdp 确定性场检验 — 本地测试数据版（Windows 数据在 D:/fdp_weather_test_bash，WSL 里跑、路径是 /mnt/d 视角）。

数据实况（2026-09-16 盘点）：
- 预报只有 ENS_FCST（无 DF），multi_model 会 DF→ENS 回退后取集合平均，
  评的是"集合平均场"的 RMSE/Bias（member 处理在 read_model_data 里已实现）；
- CRA 实况只有 20260820 一天（00-18 时），所以时效只到 42h——
  更长时效的 valid time 落在 0820 之后，会逐时效报"未找到实况"跳过；
- 无 CLI_6HOUR 气候态，ACC（仅 z500 需要）会被自动跳过，CSV 里没有 acc 列；
- FCSTDATA/Fengqing/ENS/{date} 是指向 fengqing/{date} 的 junction
  （multi_model 固定要求 {root}/{Model}/{DF|ENS}/{date} 布局）。
"""
CONFIG = {
    "capability": "fdp_field_det",
    "output_name": "fdp_rmse_single_fengqing_test",
    "start_date": "20260819",
    "end_date": "20260820",
    "init_hour": "00",
    "models": ["Fengqing"],
    # CRA 只有 20260820 一天，6-42h 之外全部会对不上实况
    "forecast_hours": list(range(6, 43, 6)),
    "model_data_root": "/mnt/d/fdp_weather_test_bash/FCSTDATA",
    "cra_root": "/mnt/d/fdp_weather_test_bash/cra_root",
    "cli_root": "/mnt/d/fdp_weather_test_bash/CLI_6HOUR",  # 不存在 → ACC 自动跳过
    "surface_only": False,
    "pressure_only": False,
    "resume": True,
    "output_dir": "/mnt/d/xmetai-evalation/outputs/results/fdp_rmse_single_fengqing_test",
}
