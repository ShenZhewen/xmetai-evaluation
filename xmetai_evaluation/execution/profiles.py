# -*- coding: utf-8 -*-
"""指标能力画像：指标族 -> 资源需求。

评测本质是三个数据集做差（预报 / 观测 / 参考），但不同指标族对三者的依赖
完全不同：误差族只碰预报和观测，距平族还要气候态参考，集合族要保留成员维，
谱族 / 空间族拿到的是完整未插值场。执行策略（见 ``execution/strategy.py``）
不逐个看指标，而是看这一段的**联合画像**——这一段要参考源吗？要成员吗？
计算是轻是重？据此推导分块大小、数据驻留方式和并发形态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

#: 轻量指标：逐样本 O(n) 的配对统计，线程并行通常就够（GIL 释放点在 numpy 里）
_LIGHT_METRICS = frozenset(
    {"rmse", "bias", "acc", "acc_uncentered", "ts_score", "activity"}
)

#: 重指标：FFT / 邻域卷积 / 逐成员积分，进程并行（绕开 GIL）收益明显
_HEAVY_METRICS = frozenset(
    {"spectrum", "zonal_spectrum", "fss", "crps", "spread_error", "ensemble_probability"}
)

#: 需要完整空间场的指标（不能按站点子集 / 插值后再算）
_FULL_FIELD_METRICS = frozenset({"spectrum", "zonal_spectrum", "fss"})


@dataclass(frozen=True)
class ResourceProfile:
    """一段评测的联合资源画像。

    needs_reference / needs_members 以指标实例自己的钩子为准
    （``metric.needs_reference()`` / ``metric.needs_members()``），
    新增指标无需登记画像；needs_full_field / compute_class 是框架对
    指标族的先验知识，按注册名查表，查不到就取最保守的缺省（轻量、整场）。
    """

    needs_reference: bool = False
    needs_members: bool = False
    needs_full_field: bool = False
    compute_class: str = "light"  # "light" | "heavy"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "needs_reference": self.needs_reference,
            "needs_members": self.needs_members,
            "needs_full_field": self.needs_full_field,
            "compute_class": self.compute_class,
        }


def _profile_for_name(name: str) -> ResourceProfile:
    if name in _FULL_FIELD_METRICS:
        return ResourceProfile(
            needs_full_field=True, compute_class="heavy"
        )
    if name in _HEAVY_METRICS:
        return ResourceProfile(compute_class="heavy")
    return ResourceProfile(compute_class="light")


def needs_full_field(metric: Any) -> bool:
    """该指标是否必须拿完整未插值场。

    分纬度带评估的 region 样本对这类指标没有意义——谱/FSS 在掩掉的子区域
    上算出来的不是同一个物理量——执行层用这个钩子把它们跳过，只在全球
    样本上算。
    """
    name = str(getattr(metric, "name", "") or "")
    return _profile_for_name(name).needs_full_field


def union_profile(metrics: Iterable[Any]) -> ResourceProfile:
    """一段里全部指标的联合画像（按指标实例算，不猜配置）。

    Args:
        metrics: 已构建的指标实例列表（Runner 在计划阶段构建一次）。
    """

    def _hook(metric: Any, name: str) -> bool:
        method = getattr(metric, name, None)
        return bool(method()) if callable(method) else False

    needs_reference = any(_hook(metric, "needs_reference") for metric in metrics)
    needs_members = any(_hook(metric, "needs_members") for metric in metrics)
    needs_full_field = False
    compute_class = "light"
    for metric in metrics:
        name = str(getattr(metric, "name", "") or "")
        profile = _profile_for_name(name)
        needs_full_field = needs_full_field or profile.needs_full_field
        if profile.compute_class == "heavy":
            compute_class = "heavy"
    return ResourceProfile(
        needs_reference=needs_reference,
        needs_members=needs_members,
        needs_full_field=needs_full_field,
        compute_class=compute_class,
    )
