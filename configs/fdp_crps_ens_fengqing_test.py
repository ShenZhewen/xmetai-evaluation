"""fdp 集合检验 — 本地测试数据版（Windows 数据在 D:/fdp_weather_test_bash，WSL 里跑、路径是 /mnt/d 视角）。

数据实况（2026-09-16 盘点）：
- 预报是 21 成员 ENS_FCST，走 fcstdata_root 的标准路径
  {root}/{Model}/ENS/{date}/（junction 指向 fengqing/{date}）；
- CRA 实况只有 20260820 一天 → CRPS/SSR 只有 valid 落在 0820 的时效有值；
- 站点降水只有 20260820 00-06 时（.000 逐小时）→ BSS/AROC 基本只有
  init 20260820 lead 6h 一档能出数，其余时效 NaN；
- 时效只配到 42h，更长会对不上实况。
"""
CONFIG = {
    "capability": "fdp_ens",
    "output_name": "fdp_crps_ens_fengqing_test",
    "start_date": "20260819",
    "end_date": "20260820",
    "init_hour": "00",
    "models": ["Fengqing"],
    "forecast_type": "ens",
    "forecast_hours": list(range(6, 43, 6)),
    "fcstdata_root": "/mnt/d/fdp_weather_test_bash/FCSTDATA",
    "cra_root": "/mnt/d/fdp_weather_test_bash/cra_root",
    "obs_rain_root": "/mnt/d/fdp_weather_test_bash/降水",
    "accum_hours": 6,
    "forecast_root": "",
    "resume": True,
    "output_dir": "/mnt/d/xmetai-evalation/outputs/results/fdp_crps_ens_fengqing_test",
}
