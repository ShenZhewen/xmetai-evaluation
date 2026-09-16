"""fdp 确定性降水检验 config（tp_deterministic_verifier.py）。

6h 档：TS/Bias 三档(≥0.1/13/25mm) + FSS（单阈值 13mm，窗口 1/3/5/15/31/63）
      + 加权综合评分；24h 档：TS/Bias 五档(≥0.1/10/25/50/100mm) + 综合。
实况：站点 diamond 3（TS/Bias）+ CMPAS 网格（FSS 优先，CRA 回退）。
输出：outputs/.temp/fdp_ts_single_multi/ → outputs/results/fdp_ts_single_multi/fdp_ts_single_multi.csv
长表（6h/24h 两个文件合并，accum_hours 列区分）。
"""
CONFIG = {
    "capability": "fdp_tp_det",
    "output_name": "fdp_ts_single_multi",
    # 逐日起报（00 起报），区间含首尾；改 init_hour 可换 12 起报
    "start_date": "20260801",
    "end_date": "20260822",
    "init_hour": "00",
    "models": ["Fengqing", "PuYun", "YJ-TianJi", "NJU-Earth", "W2S"],
    "hours_6h": [6, 12, 18, 24, 30, 36, 42, 48],
    "hours_24h": [24, 48, 72, 96, 120, 144, 168, 192, 216, 240],
    "obs_root": "/gpu/zhaochy/fdp2/RDATA/rain",  # 站点观测根目录 (*.000)
    "fcstdata_root": "/gpu/zhaochy/fdp2/FCSTDATA",
    "cra_root": "/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026",  # FSS 回退实况
    "forecast_root": "",  # 本地预报根目录（回退路径，不用留空）
    "fss_windows": [1, 3, 5, 15, 31, 63],
    "fss_threshold": 13,
    "skip_6h": False,
    "skip_24h": False,
    # 逐日起报跑完写 <date>.done 标记，重跑跳过已完成日期
    "resume": True,
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/fdp_ts_single_multi",
}
