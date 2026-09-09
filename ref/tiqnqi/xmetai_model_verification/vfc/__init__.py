# -*- coding: utf-8 -*-
"""vfc —— 分类检验核心库（run_categorical / run_pangu 专用）。

保留模块：
  station_categorical.py  站点层（diamond3 观测 + RefProb 气候概率参考）
  metric_categorical.py   分类指标层（插值 + TSContingency + ProbEventHistogram）
  reader_categorical.py   读取层（ClassDataset：确定性 / 集合布局）
  reader_pangu.py            读取层（PanguDataset：盘古单文件多成员）
"""
