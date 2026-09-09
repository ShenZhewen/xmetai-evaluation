#!/usr/bin/env python
"""测试站点数据解析"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/szwCode/xmetai-evalation')

from xmetai_evaluation.io.station_reader import DiamondStationReader

# 测试单个文件
test_file = Path("/workspace/data/worm/r0/2025/2025010100.000")

reader = DiamondStationReader()
result = reader._parse_diamond_file(test_file)

if result:
    print(f"✓ 解析成功")
    print(f"  时间: {result['time']}")
    print(f"  站点数: {len(result['df'])}")
    print(f"  前5行:\n{result['df'].head()}")
else:
    print(f"✗ 解析失败")

# 打开文件看看内容
print(f"\n原始文件前10行:")
with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if i < 10:
            print(f"  {i+1}: {line.rstrip()}")
