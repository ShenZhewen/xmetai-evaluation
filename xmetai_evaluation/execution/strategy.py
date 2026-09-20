# -*- coding: utf-8 -*-
"""执行策略：怎么切块、怎么驻留数据、用什么并发形态。

默认值全部由联合画像与数据源**推导**，配置里的 ``execution`` 字典只做
覆盖（不写 = 用推导值），键全部是字面量：

    execution = {
        "mode": "auto",       # serial / threads / processes / auto
        "n_workers": 4,       # 进程/线程数；不写就 threads=4、processes=可用核数
                              # （可用核数 = affinity mask，认 cgroup/cpuset 配额，
                              #  不是 os.cpu_count() 报的宿主机核数）
        "chunk_days": 1,      # 一个工作块装几个起报日
        "lead_chunk_days": 1, # 一个工作块装几天时效；0 = 不切（整段时效一块）
        "loads": {"observation": "resident", "reference": "window:30"},
        "resume": False,      # True 时已完成块的状态落 output_dir/.states/，重跑跳过
    }

数据驻留策略（``loads``，按角色 observation / reference 各选一个）：

    slice      每个工作块现读现弃（预报恒为此策略，era5_zarr 这类按请求读取
               的源也默认它）；
    resident   整个 run 读一次驻留内存（站点观测、单文件日序气候态这类小数据）；
    window:W   按 W 天块 LRU 滚动（大气候态 / 大实况，块间有重叠时省重复读）。

工作块有两个切分维：**起报日**（``chunk_days``）与**时效**（``lead_chunk_days``），
块 = 两者的叉积。时效维是内存的主要旋钮——集合预报（51 成员）不切时效时
单块要把整个时效跨度 × 成员数全部物化（15 天 × 51 成员约 25GB）；按
``lead_chunk_days`` 切开后单块只装一段时效。``lead_chunk_days: 0`` 表示
不切（整段时效一块 = 旧行为）。

时效窗落在**采样格点**上（由协议声明的 ``sample_leads`` 决定），不是原始
时效跨度——否则固定累积窗的流程（如 24h 降水）会切出空的时效窗。
读取集会比采样集往前多带一段**预热**时效（累积变换要窗口完整才出值），
所以读取集可能跨窗重叠，但采样集始终是 (起报, 时效) 的纯划分。

并发形态（``mode``）：工作块是唯一的并行单位，块内数据用完即弃、块间
零共享。``auto`` 在知道块数后落地：单块 -> serial；重指标 -> processes；
轻指标 -> threads。processes 下两种启动方式都能让加载缓存生效：

    fork   计划层把 resident 数据预热进子进程（COW 零拷贝共享）；
    spawn  子进程反序列化一份 loader 副本自建缓存（内存 ×段数，但段内
           相邻块仍复用窗块——原本这里完全没有缓存、每块重读）。

窗块预取（``RunLoader(prefetch=...)``）只在非 threads 形态开：一个 loader
一个使用者时，后台线程读下一个窗块能把读盘和计算叠起来；threads 本来就有
N 路并发读，各自再排一个预取只会互相顶掉。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.execution.profiles import ResourceProfile

MODES = ("serial", "threads", "processes", "auto")
LOAD_ROLES = ("observation", "reference")
#: execution 字典里允许出现的键（多余键直接报配置错，别让拼错的名字静默失效）
EXECUTION_KEYS = (
    "mode",
    "n_workers",
    "chunk_days",
    "lead_chunk_days",
    "loads",
    "resume",
)

#: 默认整段驻留的读取器：数据量小、且几乎每个块都要用
_RESIDENT_READERS = frozenset(
    {"station", "diamond_station", "climatology", "daily_climatology"}
)
#: 不走加载策略层的参考源：协议逐样本直读概率，没有"整包"可言
_UNMANAGED_READERS = frozenset({"ref_probability"})


def usable_cpu_count() -> int:
    """本进程**实际能用**的核数——不是宿主机的核数。

    ``os.cpu_count()`` 报的是宿主机的逻辑核数，它不看 cgroup / cpuset 配额：
    在有配额的环境（Slurm、容器、``taskset``）上会严重高估。实测在一台配额
    24 核的机器上返回 192，于是开了 192 个进程，每个都要物化一整块集合预报场
    （51 成员），几分钟内全被 OOM 打死——症状是满屏 ``BrokenProcessPool``，
    离真正的原因（核数读错了）隔着十万八千里。

    ``sched_getaffinity`` 读的是 affinity mask，也就是认配额的核数。Windows /
    macOS 没有这个接口，退回 ``os.cpu_count()``。
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def parse_load_policy(value: str) -> tuple:
    """把驻留策略声明解析成 (策略, 窗口天数)。

    ``"window"`` 不带数字表示**自动定窗**（返回 None，由计划层按时效与
    并发跨度算出具体天数）；``"window:30"`` 显式指定 30 天。
    """
    text = str(value).strip()
    if text == "resident":
        return "resident", 0
    if text == "slice":
        return "slice", 0
    if text.startswith("window"):
        parts = text.split(":", 1)
        if len(parts) == 1 or not str(parts[1]).strip():
            return "window", None  # 自动：计划层定值
        days = int(parts[1])
        if days < 1:
            raise ConfigError(f"加载窗口必须 >= 1 天: {value!r}")
        return "window", days
    raise ConfigError(
        f"无效的加载策略 {value!r}（应为 slice / resident / window[:天数]）"
    )


@dataclass
class ExecutionStrategy:
    """一段评测的执行策略（推导 + 配置覆盖后的结果）。"""

    mode: str = "auto"
    n_workers: Optional[int] = None
    chunk_days: int = 1
    #: 一个工作块装几天时效；0 = 不切（整段时效一块）
    lead_chunk_days: int = 1
    #: 角色 -> 策略字符串（"slice" / "resident" / "window:W"）
    loads: Dict[str, str] = field(default_factory=dict)
    resume: bool = False

    def resolve_mode(self, n_chunks: int, profile: ResourceProfile) -> str:
        """``auto`` 在知道块数后落地成具体形态；显式声明的 mode 原样返回。"""
        if self.mode != "auto":
            return self.mode
        if n_chunks <= 1:
            return "serial"
        return "processes" if profile.compute_class == "heavy" else "threads"

    def resolve_workers(self, mode: str) -> int:
        """worker 数缺省：processes 用满**可用**核数，threads 给 4。"""
        if self.n_workers and int(self.n_workers) > 0:
            return int(self.n_workers)
        if mode == "processes":
            return usable_cpu_count()
        return 4

    def load_policy(self, role: str) -> tuple:
        """角色的 (驻留策略, 窗口天数)。"""
        return parse_load_policy(self.loads.get(role, "slice"))


def derive_strategy(
    profile: ResourceProfile,
    observation_reader: str,
    reference_reader: Optional[str],
    execution: Optional[Dict[str, Any]] = None,
    num_workers: int = 1,
) -> ExecutionStrategy:
    """推导一段评测的执行策略。

    Args:
        profile: 这一段全部指标的联合画像。
        observation_reader: 观测源读取器注册名（决定缺省驻留策略）。
        reference_reader: 参考源读取器注册名；None 表示这段没有参考源。
        execution: 配置里的 ``execution`` 字典（只覆盖写到的键）。
        num_workers: 旧字段（``EvalConfig.num_workers``）：execution 没给
            n_workers 且它 > 1 时当作 n_workers 用，老配置不失效。
    """
    cfg = dict(execution or {})
    unknown = sorted(set(cfg) - set(EXECUTION_KEYS))
    if unknown:
        raise ConfigError(
            f"execution 里有未知键 {unknown}（可用键: {', '.join(EXECUTION_KEYS)}）"
        )

    mode = str(cfg.get("mode", "auto"))
    if mode not in MODES:
        raise ConfigError(f"无效的执行形态 {mode!r}（应为 {' / '.join(MODES)}）")

    raw_chunk_days = cfg.get("chunk_days")
    chunk_days = 1 if raw_chunk_days is None else int(raw_chunk_days)
    if chunk_days < 1:
        raise ConfigError(f"chunk_days 必须 >= 1: {chunk_days}")

    # 时效切分：正数 = 一个块装几天时效；0 = 不切（整段时效一块 = 旧行为）
    raw_lead_chunk = cfg.get("lead_chunk_days")
    lead_chunk_days = 1 if raw_lead_chunk is None else int(raw_lead_chunk)
    if lead_chunk_days < 0:
        raise ConfigError(
            f"lead_chunk_days 必须 >= 0（0 表示不切时效）: {lead_chunk_days}"
        )

    n_workers = cfg.get("n_workers")
    if n_workers is None and int(num_workers or 1) > 1:
        n_workers = int(num_workers)
    if n_workers is not None and int(n_workers) < 1:
        raise ConfigError(f"n_workers 必须 >= 1: {n_workers}")

    # 缺省驻留策略按读取器形态定：小数据驻留、按请求读取的大源现读现弃
    loads: Dict[str, str] = {
        role: "resident" if reader in _RESIDENT_READERS else "slice"
        for role, reader in (
            ("observation", observation_reader),
            ("reference", reference_reader),
        )
        if reader and reader not in _UNMANAGED_READERS
    }
    overrides = cfg.get("loads") or {}
    if not isinstance(overrides, dict):
        raise ConfigError("execution.loads 必须是 {角色: 策略} 字典")
    for role, value in overrides.items():
        if role not in LOAD_ROLES:
            raise ConfigError(
                f"loads 里的角色 {role!r} 不存在（可用: {', '.join(LOAD_ROLES)}）"
            )
        parse_load_policy(str(value))  # 顺带校验取值
        if reference_reader in _UNMANAGED_READERS and role == "reference":
            raise ConfigError(
                f"参考源 {reference_reader} 逐样本直读概率，不提供整包加载，不能配置 loads['reference']"
            )
        loads[role] = str(value)

    return ExecutionStrategy(
        mode=mode,
        n_workers=int(n_workers) if n_workers is not None else None,
        chunk_days=chunk_days,
        lead_chunk_days=lead_chunk_days,
        loads=loads,
        resume=bool(cfg.get("resume", False)),
    )
