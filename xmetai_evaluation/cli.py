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
from xmetai_evaluation.core.logging import configure_logging
from xmetai_evaluation.pipeline.pipelines import list_pipelines
from xmetai_evaluation.pipeline.runner import Runner
from xmetai_evaluation.pipeline.spec import specs_from_config


def run_evaluation(cfg) -> int:
    """执行一次评测（EvalConfig -> PipelineSpec -> Runner）。

    配置里的 ``pipeline`` 可以是一串流程模板，那样就是一条命令跑多段、
    结果合并落同一个 ``output_dir``。并发与数据加载由 ``execution`` 字典
    覆盖，不写就按指标族与数据源形态推导。

    程序化入口：集成测试直接构造 EvalConfig 调用这里，绕过 argparse。
    """
    Runner(
        specs_from_config(cfg),
        execution=dict(getattr(cfg, "execution", None) or {}),
        num_workers=int(getattr(cfg, "num_workers", 1) or 1),
    ).run()
    return 0


def run_evaluations(configs) -> int:
    """顺序执行一批评测（批量配置 ``cfgs`` 展开来的）。

    严格串行：每个 run 内部已经吃满 n_workers，两个模型并行只会超订。
    单个模型失败记日志、继续跑后面的——批量跑最怕一个挂了全白跑；
    结束时统一报成败，有失败则返回 1（CI 可感知）。KeyboardInterrupt
    不在本函数拦（不是 Exception），冒出去由 main 统一处理。
    """
    logger = logging.getLogger(__name__)
    failed = []
    for index, cfg in enumerate(configs, start=1):
        logger.info("批量评测 %d/%d：%s", index, len(configs), cfg.name)
        try:
            run_evaluation(cfg)
        except Exception:
            logger.exception("评测 %s 失败，继续下一个", cfg.name)
            failed.append(cfg.name)
    if failed:
        logger.error(
            "批量评测结束：%d/%d 失败（%s）",
            len(failed), len(configs), ", ".join(failed),
        )
        return 1
    logger.info("批量评测结束：%d 个全部成功", len(configs))
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
        loaded = load_config(args.config)
        configs = loaded if isinstance(loaded, list) else [loaded]
        configure_logging(
            args.log_level or configs[0].log_level,
            Path(args.log_file) if args.log_file else None,
        )
        return run_evaluations(configs)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("用户中断")
        return 130
    except Exception:
        logging.getLogger(__name__).exception("评估失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
