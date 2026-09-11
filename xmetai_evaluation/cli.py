#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一评估入口。

CLI 只做三件事：加载配置、配置日志、交给 run_evaluation。
数据读取、样本循环、配对、状态合并和落盘都在框架内部，入口不含任何评测逻辑。
数据在哪、评哪段时间、输出到哪 —— 全部写在配置里（配置里可用环境变量覆盖）。
"""

import argparse
import logging
import sys
from pathlib import Path

from xmetai_evaluation.configs.base import load_config
from xmetai_evaluation.logging_util import configure_logging
from xmetai_evaluation.pipeline.pipelines import list_pipelines
from xmetai_evaluation.pipeline.runner import Runner
from xmetai_evaluation.pipeline.spec import PipelineSpec


def run_evaluation(cfg) -> int:
    """执行一次评测（EvalConfig -> PipelineSpec -> Runner）。

    程序化入口：集成测试直接构造 EvalConfig 调用这里，绕过 argparse。
    """
    Runner(PipelineSpec.from_config(cfg)).run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xmetai-evaluation 气象模型评估框架")
    parser.add_argument("--config", default=None, help="配置名称或 Python 配置路径")
    parser.add_argument("--list-pipelines", action="store_true", help="列出内置流程")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None
    )
    parser.add_argument("--log-file", default=None)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_pipelines:
        for name, description in sorted(list_pipelines().items()):
            print(f"{name}\t{description}")
        return 0
    if not args.config:
        parser.error("必须提供 --config（流程、数据与时段都在配置里指定）")

    try:
        cfg = load_config(args.config)
        configure_logging(
            args.log_level or cfg.log_level,
            Path(args.log_file) if args.log_file else None,
        )
        return run_evaluation(cfg)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("用户中断")
        return 130
    except Exception:
        logging.getLogger(__name__).exception("评估失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
