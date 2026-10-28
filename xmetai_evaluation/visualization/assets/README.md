# visualization/assets

## `china_maritime_boundary.geojson`

中国海上国界/九段线（10 段断续线，含台湾海峡两段），`MultiLineString`，经纬度。

- 来源：[cnmaps-data](https://github.com/cnmetlab/cnmaps-data) 1.1.2 的
  `data/datasets/administrative/amap/maritime/100000.geojson`（MIT License，
  Copyright © 2026 Clarmy）。
- 原样拷贝，未做裁剪或简化，8.4 KB。
- 用途：`visualization/rainstorm_spatial.py` 的南海诸岛小图上按标准地图口径
  画出断续线。各段本身就是断开的，直接画实线即可，不要再加虚线样式。
- 如果以后要换官方底图，把同名文件覆盖掉即可，渲染代码不用动。
