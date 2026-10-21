# -*- coding: utf-8 -*-
"""入口脚本共用样板：仓库根、UTF-8 控制台、``NAME=目录`` 参数。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: scripts/ -> xmetai-evaluation/ -> skills/ -> 仓库根
REPO_ROOT = Path(__file__).resolve().parents[3]


def bootstrap() -> Path:
    """把仓库根插进 ``sys.path``，并把 stdout / stderr 切成 UTF-8。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    return REPO_ROOT


def parse_model(value: str):
    """``NAME=目录`` → ``(NAME, Path)``。"""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--model 要写成 NAME=目录，收到的是 {value!r}")
    name, directory = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--model 的模型名不能是空的: {value!r}")
    return name, Path(directory.strip())
