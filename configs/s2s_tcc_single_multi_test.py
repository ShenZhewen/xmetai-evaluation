"""s2s 确定性预测 — 本地测试版（数据在 D:/确定性预测/数据样例，WSL 里跑、路径是 /mnt/d 视角）。

数据实况（2026-09-16 盘点）：
- step3_output（CRA + my_model）和 module1_other_model 都有 t2m 2024 全年
  365 个周平均文件 → 直接跑 step4/step5 检验步，端到端出 TCC/RMSE 表；
- step1/2/3 的输入只有几个小样本文件，且中间产物已齐全，不在本 config 里跑
  （要冒烟预处理链的话把 steps 改全、路径指到各 step 的 *_input 目录）；
- step5 空间图依赖 cartopy 离线 shapefile，指 D:/确定性预测/cartopy；
- 日常重跑零成本：step4/5 输出已存在会直接覆盖重算（检验步本身很快）。
"""
CONFIG = {
    "capability": "s2s_det",
    "output_name": "s2s_tcc_single_multi_test",
    "steps": ["step4", "step5"],
    "var": "t2m",
    "year": 2024,
    # step4 起报日期区间（原脚本硬编码 9/23-12/31，这里用全年）
    "init_start": "0101",
    "init_end": "1231",
    "model_sources": ["my_model", "other_model"],
    "reference_source": "CRA_v1.5",
    "reference_path": "/mnt/d/确定性预测/数据样例/step3_output/CRA_1.5/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",
    "my_model_path": "/mnt/d/确定性预测/数据样例/step3_output/my_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",
    "other_model_path": "/mnt/d/确定性预测/数据样例/module1_other_model/anom_combine_1p5_week/{var}/{yyyymmdd}.nc",
    # step5 空间图离线国界数据（不含则自动跳过该目录用 cartopy 默认）
    "cartopy_dir": "/mnt/d/确定性预测/cartopy",
    "output_dir": "/mnt/d/xmetai-evalation/outputs/results/s2s_tcc_single_multi_test",
}
