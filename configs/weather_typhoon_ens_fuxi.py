"""Ensemble FuXi typhoon track/intensity evaluation config.

与 weather_typhoon_single_fuxi.py 的唯一区别是集合口径。

``pred`` 必须指到**具体那一次起报的日期目录**，不是 ``fuxi_ens_output`` 根：

    /workspace/data/shenzw/fuxi_ens_output/20220901/member_000/001.nc …
                                            /member_001/001.nc …
                                            /member_002/…

core/tc_ref.py 的集合分支用的是**平铺扫描**——``os.scandir(pred)`` 找
``startswith("member_")`` 的目录，**不递归**。所以 ``pred`` 若指到根
（``fuxi_ens_output``），一个成员都找不到，会掉进确定性分支，
再由 ``FieldFile`` 把一个没有 ``NN.nc`` 的目录当成员目录读，
报 ``DataFileError: 集合成员目录 … 下未找到 .nc 文件``。
（``vfc/ensemble_io.py`` 能自动向上找一层，但台风这条链路不走它。）

``ens_agg`` 选聚合方案：
  ``"A"`` = 先算各成员自己的轨迹误差，再对成员求平均；
  ``"B"`` = 先把各成员位置平均成集合平均位置，再与实况求误差
（默认，与 recompute_ens_track_error.py 同口径）。

``workers`` 在集合分支里**才真正生效**：成员间用 ProcessPoolExecutor 并行，
不给则 ``min(cpu_count, 成员数)``。确定性分支里这个键是被忽略的。

注意 ``start_date`` / ``end_date`` 只是照搬单成员配置的占位：
core/api.py 的 run_tc 会把它们 pop 掉再调参考实现，**不参与任何筛选**。
真正决定评哪一场的是 ``pred``（哪次起报）与 ``babj``/``tcid``（哪颗台风）。

跑法::

    python runner.py -config configs/weather_typhoon_ens_fuxi.py
"""

CONFIG = {
    "capability": "typhoon",
    # ↓ 改成你要评的那次起报的日期目录（根下面那一层）
    "pred": "/workspace/data/shenzw/fuxi_ens_output/20220901",
    "babj": "/workspace/data/worm/babj/babj2205.dat",
    "tcid": "2205",
    "start_date": "20250101",
    "end_date": "20251231",
    "lead_step": 6.0,
    "forecast_type": "ens",
    "ens_agg": "B",
    "tz_shift": 8.0,
    "workers": 8,
    "output_dir": "/workspace/szwCode/xmetai-eval_pro/outputs/results/weather_typhoon_ens_fuxi",
}
