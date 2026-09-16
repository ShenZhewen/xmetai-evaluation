"""fdp 集合检验 config（ensemble_verifier.py）。

CRPS、Spread-Error Ratio、集合平均 RMSE（全球 z500）+ BSS、AROC
（中国区 6h 降水，0.1/4/13/25mm 四档）。
预报 {fcstdata_root}/{Model}/ENS/{date}/（21 成员）；forecast_type 改 "df"
可评确定性预报。默认时效只到 42h（W2S 支持 360h，按需加长）。
输出：outputs/.temp/fdp_crps_ens_multi/ → outputs/results/fdp_crps_ens_multi/fdp_crps_ens_multi.csv 长表。
"""
CONFIG = {
    "capability": "fdp_ens",
    "output_name": "fdp_crps_ens_multi",
    # 逐日起报（00 起报），区间含首尾；改 init_hour 可换 12 起报
    "start_date": "20260801",
    "end_date": "20260822",
    "init_hour": "00",
    "models": ["Fengqing", "PuYun", "YJ-TianJi", "NJU-Earth", "W2S"],
    "forecast_type": "ens",  # "ens" 集合预报 / "df" 确定性预报
    "forecast_hours": list(range(6, 43, 6)),  # 6-42 每 6h
    "fcstdata_root": "/gpu/zhaochy/fdp2/FCSTDATA",
    "cra_root": "/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026",
    "obs_rain_root": "/gpu/zhaochy/fdp2/RDATA/rain",  # 站点降水实况 (diamond 3 *.000)
    "accum_hours": 6,
    "forecast_root": "",  # 本地预报根目录（回退路径，不用留空）
    # 逐日起报跑完写 <date>.done 标记，重跑跳过已完成日期
    "resume": True,
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/fdp_crps_ens_multi",
}
