# -*- coding: utf-8 -*-
"""评估配置包：一份 config = 一次评估任务（或一份 cfgs = 批量多个任务）。

换模型/换评估方法 = 在本包 configs 下新增一个 <name>.py，定义全局 `cfg`；
批量评多个模型就定义 `cfgs` 列表（cli 顺序逐个执行）。代码主体不动。
"""
from .base import EvalConfig, load_config

__all__ = ["EvalConfig", "load_config"]
