#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch RMSE/ACC/FA/spectrum evaluation entry point.

Vendored from xmetai_model_verification_xu/run_rmse.py to use local vfc imports.
This is the CLI entry point called by batch_adapter for migrated shell scripts.

Usage (called by batch_adapter, not directly):
    python run_batch_rmse.py --pred-root <root> --target-zarr <zarr> \\
        --dates <date1> <date2> ... --metrics rmse acc fa spectrum \\
        --vars z500 t850 ... --n-workers 48 --outdir-root <out> --resume
    python run_batch_rmse.py --summarize-det <outdir_root>
    python run_batch_rmse.py --summarize-ens <outdir_root>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 本文件在 core/ 下，vfc/ 在仓库根 —— 把根目录补进 sys.path 再 import。
# （原先本文件在根目录、靠脚本目录默认在 sys.path 生效，移进 core/ 后必须显式加。）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Use local vfc modules from eval_pro/vfc/
from vfc import regr_ens
from vfc import regr_summary


def main(argv=None) -> int:
    """Batch evaluation entry point: delegates to vfc.regr_ens.ens_main."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # Summarize modes
    if "--summarize-det" in argv:
        return regr_summary.main_det(argv)

    if "--summarize-ens" in argv:
        return regr_summary.main(argv)

    # Batch mode (--pred-root + --target-zarr)
    return regr_ens.ens_main(argv)


if __name__ == "__main__":
    import time as _t
    _t0 = _t.monotonic()
    _rc = 1
    try:
        _rc = main()
    finally:
        _el = _t.monotonic() - _t0
        _h, _rem = divmod(_el, 3600.0)
        _m, _s = divmod(_rem, 60.0)
        print("\n[run_batch_rmse] 运行时长: %d:%02d:%05.1f (%.1f s)"
              % (int(_h), int(_m), _s, _el), flush=True)
    sys.exit(_rc)
