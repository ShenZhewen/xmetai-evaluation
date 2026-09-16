"""fdp Z500 活跃度+功率谱 config（activity_spectrum_verifier.py）。

activity_ratio = std(预报距平)/std(实况距平)（CLI 气候态距平）；
功率谱 2D-FFT 逐波数 P(k) 预报 vs 实况。
输出：outputs/.temp/fdp_activity_single_multi/ → outputs/results/ 下两个长表：
fdp_activity_single_multi.csv（活跃度）+ fdp_activity_single_multi_power_spectrum.csv。
"""
CONFIG = {
    "capability": "fdp_activity_spectrum",
    "output_name": "fdp_activity_single_multi",
    # 逐日起报（00 起报），区间含首尾；改 init_hour 可换 12 起报
    "start_date": "20260801",
    "end_date": "20260822",
    "init_hour": "00",
    "models": ["Fengqing", "PuYun", "YJ-TianJi", "NJU-Earth", "W2S"],
    "forecast_hours": list(range(6, 361, 6)),  # 6-360 每 6h
    "model_data_root": "/gpu/zhaochy/fdp2/FCSTDATA",
    "cra_root": "/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/0P25/2026",
    "cli_root": "/gpu/COMMONDATA/CRA/ORIG/CMA-RA1.5/CLI_6HOUR",
    "spectrum_hours": [],  # 画功率谱对比图的时效（空 = 原脚本默认 72/360h）
    "debug": False,
    # 逐日起报跑完写 <date>.done 标记，重跑跳过已完成日期
    "resume": True,
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/fdp_activity_single_multi",
}
