"""FuXi 确定性台风路径/强度检验。

pred 指到哪一层决定跑多少场次：
  * 指到 <YYYYMMDD> 起报日目录 → 只跑那一场（如 .../fuxi_single_output/20250610）；
  * 指到上一层的预报根目录        → 批量，babj 里每个对得上的起报日各跑一场。
babj 同理：指到单个 babj<编号>.dat 只跑那一号，指到目录就跑目录下全部编号。

批量时起报时刻由 babj 的实况时刻反推（实况 - tz_shift），该日没有预报目录就跳过。
想只跑某一号台风就填上 tcid，留空/删掉就跑全部。

启动方式与其它 capability 一致：
    python runner.py --config configs/weather_typhoon_single_fuxi.py
"""
CONFIG = {
    "capability": "typhoon",
    "pred": "/workspace/data/shenzw/fuxi_single_output",
    "babj": "/workspace/data/LiuJs/babj",
    "lead_step": 6.0,
    "forecast_type": "det",
    "tz_shift": 8.0,
    "workers": 8,
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/weather_typhoon_single_fuxi",
}
