# -*- coding: utf-8 -*-
"""评估配置基础设施：EvalConfig + load_config()"""
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# 项目根目录
ROOT = Path(__file__).parent.parent.parent.resolve()


@dataclass
class EvalConfig:
    """一次评估的完整配置（由 cli.py 加载并执行）"""

    name: str  # 配置名称
    description: str  # 配置描述

    # 数据源
    forecast_reader: Dict[str, Any]  # 预报数据读取器配置
    observation_reader: Dict[str, Any]  # 观测数据读取器配置
    # 可选参考场（气候态等）：ACC / 活跃度 / 功率谱等距平类指标需要
    reference_reader: Optional[Dict[str, Any]] = None

    # 时间范围
    start_date: Optional[str] = None  # YYYYMMDD or YYYYMMDDHH
    end_date: Optional[str] = None  # YYYYMMDD or YYYYMMDDHH
    limit: Optional[int] = None  # 限制处理的初始化时间数（用于调试）

    # 输出
    output_dir: str = "evaluation_results"  # 输出目录

    # 结果输出视图：统一长表（csv_long）始终写出；
    # None = 用流程模板的视图，显式给列表则覆盖模板
    writers: Optional[List[str]] = None

    # 流程名（pipeline/pipelines.py 里的模板）；给了就用它的协议/变换/指标。
    # 写成列表就是按顺序跑多段（如集合降水检验两次窗口），结果合并落同一个 output_dir
    pipeline: Union[str, List[str]] = ""

    # 协议口径覆盖（如观测为北京时：local_utc_offset_hours=8）
    options: Dict[str, Any] = field(default_factory=dict)

    # 流程模板里"变换参数"的覆盖：{"time_window_accumulator": {"window_hours": 6}}
    transform_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 流程模板里"指标参数"的覆盖：{"ts_score": {"thresholds": [0.1, 10, 25]}}
    metric_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 运行配置
    log_level: str = "INFO"  # 日志级别
    num_workers: int = 1  # 并行 worker 数（预留，暂未实现）

    # 内部字段
    _source_path: str = field(init=False, repr=False, default="")

    def resolve_path(self, value: str) -> str:
        """相对外部 config 所在目录解析本地路径；URL 和绝对路径保持不变"""
        if not value or os.path.isabs(value) or "://" in value:
            return value
        base = os.path.dirname(self._source_path) if self._source_path else ROOT
        return str(Path(base) / value)

    def parse_date_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """解析起止日期"""
        start = None
        end = None
        if self.start_date:
            if len(self.start_date) == 8:
                start = datetime.strptime(self.start_date, "%Y%m%d")
            elif len(self.start_date) == 10:
                start = datetime.strptime(self.start_date, "%Y%m%d%H")
            else:
                raise ValueError(f"无效的起始日期格式: {self.start_date}")

        if self.end_date:
            if len(self.end_date) == 8:
                end = datetime.strptime(self.end_date, "%Y%m%d")
            elif len(self.end_date) == 10:
                end = datetime.strptime(self.end_date, "%Y%m%d%H")
            else:
                raise ValueError(f"无效的结束日期格式: {self.end_date}")

        return start, end


def metric_options_from_var_metrics(
    var_metrics: Dict[str, List[str]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """把「变量 -> 指标」翻译成 ``EvalConfig.metric_options``。

    参考实现用 ``--var-metrics z500:rmse,acc`` 声明"哪个变量算哪些指标"；
    框架里的对应声明是 ``metric_options[指标名]["variables"] = [变量...]``。
    这里做一次反转，配置就能直接照抄参考实现的形状：

        VAR_METRICS = {"z500": ["rmse", "acc"], "t2m": ["rmse"]}
        metric_options = metric_options_from_var_metrics(VAR_METRICS)

    没被任何变量点名的指标不会出现在结果里；想要某个指标评全部变量，
    就在 ``var_metrics`` 里给每个变量都写上它。

    Args:
        var_metrics: 变量 -> 指标名列表（顺序即结果里的变量顺序）。
        metrics: 要输出哪些指标；默认取 ``var_metrics`` 里出现过的全部。
    """
    names = (
        list(metrics)
        if metrics
        else [name for items in var_metrics.values() for name in items]
    )
    ordered: List[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return {
        name: {
            "variables": [
                variable for variable, items in var_metrics.items() if name in items
            ]
        }
        for name in ordered
    }


def _resolve_config_path(value: str) -> Path:
    """解析外部配置路径或包内配置名"""
    direct = Path(value).resolve()
    if direct.is_file():
        return direct

    # 尝试在 configs/ 目录下查找
    name = value
    if name.startswith("configs/") or name.startswith("configs\\"):
        name = name.replace("\\", "/").rsplit("/", 1)[-1]

    if not name.endswith(".py"):
        name += ".py"

    packaged = Path(__file__).parent / name
    if packaged.is_file():
        return packaged

    raise FileNotFoundError(f"配置文件不存在: {value}")


def load_config(path: str) -> EvalConfig:
    """加载外部配置文件或包内配置名，返回其中的 EvalConfig"""
    config_path = _resolve_config_path(path)

    config_dir = str(config_path.parent)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    module_name = f"_xmetai_eval_config_{abs(hash(str(config_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    cfg = getattr(module, "cfg", None)
    if cfg is None:
        raise ValueError(f"配置文件 {config_path} 没有定义 `cfg`（应写 cfg = EvalConfig(...)）")

    if not isinstance(cfg, EvalConfig):
        raise TypeError(
            f"配置文件 {config_path} 的 cfg 类型是 {type(cfg).__name__}，应为 EvalConfig"
        )

    cfg._source_path = str(config_path)
    return cfg
