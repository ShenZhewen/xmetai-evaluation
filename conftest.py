# -*- coding: utf-8 -*-
"""pytest 根配置：确保仓库根目录在 sys.path，便于 `import vfc`。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
