# -*- coding: utf-8 -*-
"""统一日志配置模块"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo


class BeijingFormatter(logging.Formatter):
    """使用 Asia/Shanghai 时区的日志格式化器。"""

    _timezone = ZoneInfo("Asia/Shanghai")

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone(self._timezone)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds")


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    rank: Optional[str] = None,
) -> None:
    """
    配置评估框架的日志系统

    Args:
        level: 日志级别（DEBUG, INFO, WARNING, ERROR）
        log_file: 日志文件路径（None = 不写文件）
        console: 是否输出到控制台
        rank: 多进程时的 rank 标识（如 "0/4"）
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    # 格式化器（北京时间）
    rank_prefix = f"[{rank}] " if rank else ""
    fmt = f"%(asctime)s %(levelname)-7s %(name)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = BeijingFormatter(fmt, datefmt=date_fmt)

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel("DEBUG")  # 文件记录全部 DEBUG
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger"""
    return logging.getLogger(name)
