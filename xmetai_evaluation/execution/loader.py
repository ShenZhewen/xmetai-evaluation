# -*- coding: utf-8 -*-
"""数据加载策略层：resident / window / slice 三种驻留方式的一个统一入口。

协议不再自己决定"观测读多大、气候态读几遍"，而是把手里的 ``DataRequest``
交给 ``RunLoader.materialize``，由策略层决定这次请求命中缓存还是现读：

    slice      直接调 builder 按请求跨度读，读完即弃；
    resident   整个 run 读一次（builder(None)），之后所有请求在内存里切跨度；
    window:W   按 W 天块缓存，请求只物化覆盖到的块。

线程安全：缓存读写有锁；块间零共享的铁律不受影响——每个块拿到的仍是
自己跨度内的独立视图。进程模式（Linux fork）下整个 loader 由计划层在
fork 前预热，子进程通过写时复制共享 resident 数据，不序列化、不重读。
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr

from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.logging_util import peak_rss_text

log = logging.getLogger(__name__)

#: 逐请求切片时认的时间维（station 载荷是 time，格点/气候态是 valid_time）
TIME_DIMS = ("valid_time", "time")


@dataclass
class RoleSpec:
    """一个数据角色（observation / reference）的加载声明。

    builder 按时间跨度物化数据：``builder(None)`` 表示读全量，
    ``builder((start, end))`` 表示读这个闭区间内的数据。times 是该角色的
    全量时刻列表（window 分块用），span 只是它的首尾。
    """

    role: str
    policy: str  # "slice" | "resident" | "window"
    window_days: int = 1
    builder: Optional[Callable[[Optional[Tuple[datetime, datetime]]], Any]] = None
    times: Optional[List[datetime]] = None


def _time_dim(payload: Any) -> Optional[str]:
    for dim in TIME_DIMS:
        if dim in getattr(payload, "dims", ()):
            return dim
    return None


def _slice_by_span(bundle: Any, span: Tuple[datetime, datetime]) -> Optional[Any]:
    """把驻留数据按时间跨度切片；没有时间维就原样返回（静态数据）。"""
    dim = _time_dim(bundle.payload)
    if dim is None:
        return bundle
    start, end = span
    sliced = bundle.payload.sel({dim: slice(pd.Timestamp(start), pd.Timestamp(end))})
    if int(sliced.sizes.get(dim, 0)) == 0:
        return None
    # provenance 保留全量清单：manifest 里要的是"这个 run 读过哪些文件"，
    # 缓存切片不该把它截短。
    return replace(bundle, payload=sliced)


def _all_blocks(
    times: List[datetime], window_days: int
) -> List[Tuple[int, Tuple[datetime, datetime]]]:
    """把全量时刻按 W 天切成 (块号, 块跨度)，按块号升序。"""
    anchor = times[0].date()
    all_blocks: List[Tuple[int, Tuple[datetime, datetime]]] = []
    current_index: Optional[int] = None
    block: List[datetime] = []
    for value in times:
        index = (value.date() - anchor).days // window_days
        if index != current_index:
            if block:
                all_blocks.append((current_index, (block[0], block[-1])))
            current_index = index
            block = []
        block.append(value)
    if block:
        all_blocks.append((current_index, (block[0], block[-1])))
    return all_blocks


def _blocks_for_span(
    times: List[datetime], window_days: int, span: Tuple[datetime, datetime]
) -> List[Tuple[int, Tuple[datetime, datetime]]]:
    """把全量时刻按 W 天切块，返回覆盖请求跨度的 (块号, 块跨度) 列表。"""
    return [item for item in _all_blocks(times, window_days) if _overlaps(item[1], span)]


def _overlaps(a: Tuple[datetime, datetime], b: Tuple[datetime, datetime]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _next_block_span(
    spec: RoleSpec, wanted: List[Tuple[int, Tuple[datetime, datetime]]]
) -> Optional[Tuple[int, Tuple[datetime, datetime]]]:
    """按日期顺序、紧跟在本次请求之后的那个块；没有下一块就 None。

    为什么「下一个块号」是个靠谱的预取预测：块按日期切，两个协议的请求跨度都
    随时间**单调向后延伸**（时效窗推进抬 ``lead_max``、起报组推进抬起报），
    所以下一次要读的块几乎总是当前最大块号 +1。
    """
    if not spec.times or not spec.window_days or not wanted:
        return None
    index = max(item[0] for item in wanted) + 1
    span = dict(_all_blocks(spec.times, spec.window_days)).get(index)
    return None if span is None else (index, span)


def _payload_gb(payload: Any) -> float:
    """载荷的字节数（GB）：resident 驻留量的口径，日志里报给用户看。"""
    data = getattr(payload, "data_vars", None)
    if not data:
        return 0.0
    try:
        return sum(var.nbytes for var in data.values()) / 1024**3
    except Exception:
        return 0.0


class RunLoader:
    """一次评测 run 的共享加载器：策略层唯一入口。

    ``prefetch`` 打开时，每次读满一个 window 块就在后台线程里把**按日期顺序
    的下一个块**读进来备用（见 ``_schedule_next_block``）。threads 形态下要
    关掉：多个块在并发读同一个 loader，各自排一个预取只会互相顶掉、白读。
    """

    def __init__(self, roles: Dict[str, RoleSpec], prefetch: bool = True):
        self._roles: Dict[str, RoleSpec] = dict(roles)
        self._lock = threading.Lock()
        self._resident: Dict[str, Any] = {}
        self._window_blocks: Dict[str, Dict[int, Any]] = {}
        self._stats: Dict[str, int] = {"builder_calls": 0, "cache_hits": 0}
        #: 角色 -> builder 累计墙钟（秒）。单独一把锁：_build 会在持有
        #: self._lock 时被调用（resident 预热那条路径），再抢它会把非重入的
        #: Lock 锁死。
        self._seconds: Dict[str, float] = {}
        self._io_lock = threading.Lock()
        self._prefetch = bool(prefetch)
        #: 角色 -> (块号, 已物化的块)，后台预取一次只备**一个**块
        self._prefetched: Dict[str, Tuple[int, Any]] = {}
        self._prefetch_queue: "queue.Queue" = queue.Queue()
        self._prefetch_worker: Optional[threading.Thread] = None

    # -- 序列化：spawn 的子进程要能拿到一份 loader 自建缓存 ------------------
    #
    # 锁和队列天生 pickle 不了（内部是 _thread.lock），线程对象更是没法跨进程。
    # 它们都是"本进程的运行期状态"，子进程本来就该重建一份；缓存与统计照传。

    _RUNTIME_ATTRS = ("_lock", "_io_lock", "_prefetch_queue", "_prefetch_worker")

    def __getstate__(self) -> Dict[str, Any]:
        state = {key: value for key, value in self.__dict__.items()
                 if key not in self._RUNTIME_ATTRS}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._prefetch_queue = queue.Queue()
        self._prefetch_worker = None

    def materialize(self, role: str, request: DataRequest) -> Any:
        """按请求物化一个角色的数据（返回 DataBundle）。

        请求必须带 ``init_times``（离散时刻列表）；只取它的首尾当跨度，
        切片按闭区间做——站点文件是逐小时、格点是逐 6h，都天然落在区间内。
        """
        spec = self._roles.get(role)
        if spec is None or spec.builder is None:
            raise ConfigError(
                f"加载策略层没有配置角色 {role!r}（这段评测不经过它读取该数据源？）"
            )
        times = list(request.init_times or [])
        if not times:
            raise ConfigError(f"角色 {role!r} 的请求必须给出时刻列表")
        span = (min(times), max(times))

        if spec.policy == "slice":
            return self._build(spec, span)

        if spec.policy == "resident":
            with self._lock:
                if role not in self._resident:
                    self._resident[role] = self._build(spec, None)
                else:
                    self._stats["cache_hits"] += 1
                bundle = self._resident[role]
        else:  # window
            bundle = self._window_bundle(role, spec, span)

        sliced = _slice_by_span(bundle, span)
        if sliced is None:
            # 驻留缓存里没有这个跨度（计划期对时效的推断漏了它）：
            # 退回直读，别让一个块的缺时次拖垮整个 run。
            log.warning(
                "角色 %s 的驻留缓存不含跨度 %s 到 %s，退回按跨度直读",
                role,
                span[0],
                span[1],
            )
            return self._build(spec, span)
        return sliced

    def warm(self) -> None:
        """把 resident 角色全部读进内存（进程池 fork 前调用一次）。

        逐角色打耗时与驻留字节数：预热可能要几分钟（大参考场读盘），
        没有这两行日志的话这段就是"卡住不动"。"""
        for role, spec in self._roles.items():
            if spec.policy != "resident":
                continue
            started = perf_counter()
            log.info("预热 resident 角色 %s ……", role)
            with self._lock:
                if role not in self._resident:
                    self._resident[role] = self._build(spec, None)
                bundle = self._resident[role]
            # 「驻留」是载荷口径（array.nbytes 之和），不含读取过程中的中间
            # 对象；「进程峰值」是进程实际到过的最高水位。两个数一起看才知道
            # 这个角色把内存推到了哪，以及中间态比最终态多花了多少。
            log.info(
                "角色 %s 预热完成：耗时 %.1fs，驻留 %.2f GB，进程峰值 RSS %s",
                role,
                perf_counter() - started,
                _payload_gb(bundle.payload),
                peak_rss_text(),
            )

    def release(self) -> float:
        """放掉缓存里已物化的数据，返回释放掉的载荷 GB。**终止操作**：调用后
        这个 loader 不再可用。

        resident 驻留只在 **fork 之前**有意义——子进程写时复制共享的是父进程
        那一份，进程池一关子进程就没了，父进程再揣着这几十 GB 纯是白占。而
        紧接着的归并 / 指标收尾 / 落盘**恰恰是在这同一个父进程里跑**：峰值内存
        从"按 worker 分担"变回"父进程一个人扛"。实测 reference 驻留 22.42 GB 的
        run 就是在收尾阶段被 OOM 打死的。

        统计计数器（物化次数 / 缓存命中 / 读盘秒数）不清：执行器要先报完读盘
        占比，且这几个数只在执行期内有意义。
        """
        with self._lock:
            freed = sum(_payload_gb(bundle.payload) for bundle in self._resident.values())
            freed += sum(
                _payload_gb(bundle.payload)
                for blocks in self._window_blocks.values()
                for bundle in blocks.values()
            )
            freed += sum(
                _payload_gb(bundle.payload) for _, bundle in self._prefetched.values()
            )
            self._resident.clear()
            self._window_blocks.clear()
            self._prefetched.clear()
            # 关掉预取：缓存都放掉了，后台线程再读进来的块没人认领、白占内存
            self._prefetch = False
        return freed

    def stats(self) -> Dict[str, Any]:
        """加载统计（写进日志，方便核对策略有没有生效）。

        ``builder_seconds`` 是**花在 builder 里的墙钟**（读盘 + 解析 + 换算），
        分母由执行器报的块墙钟给——两者一比就是读盘占比。这个数决定预取值
        不值得做：预取的上限收益约为 ``min(1, 读盘/计算)``，计算占九成时它
        只值一成，不值得为它多一份内存。
        """
        result: Dict[str, Any] = dict(self._stats)
        result["builder_seconds"] = sum(self._seconds.values())
        for role, seconds in self._seconds.items():
            result[f"builder_seconds:{role}"] = seconds
        return result

    def _build(self, spec: RoleSpec, span: Optional[Tuple[datetime, datetime]]) -> Any:
        self._stats["builder_calls"] += 1
        started = perf_counter()
        try:
            return spec.builder(span)
        finally:
            # 失败也要记：读盘报错花掉的时间同样是 IO 成本
            with self._io_lock:
                self._seconds[spec.role] = (
                    self._seconds.get(spec.role, 0.0) + perf_counter() - started
                )

    def _window_bundle(self, role: str, spec: RoleSpec, span: Tuple[datetime, datetime]):
        if not spec.window_days:
            # 计划层会把自动窗（"window" 不带数字）落成具体天数；走到这里
            # 说明 RoleSpec 是绕过计划层手工构造的，必须显式给宽度。
            raise ConfigError(
                f"角色 {role!r} 用 window 策略但没有窗口宽度（应为 window:30 这类）"
            )
        if not spec.times:
            raise ConfigError(f"角色 {role!r} 用 window 策略但没有全量时刻列表")
        blocks = self._blocks(role, spec, span)
        payloads = [block.payload for block in blocks]
        if len(payloads) == 1:
            return blocks[0]
        dim = _time_dim(payloads[0]) or "valid_time"
        # coords/compat 双双写死：xarray 的弃用预告成对出现，只写一个它会接着
        # 问下一个；两个值都是当前默认，行为不变。
        merged_payload = xr.concat(
            payloads, dim=dim, coords="different", compat="equals"
        )
        # 跨块合并的 bundle 只保留首块的元数据 + 全部输入文件
        provenance = replace(
            blocks[0].provenance,
            input_files=[
                path for block in blocks for path in block.provenance.input_files
            ],
        )
        return replace(blocks[0], payload=merged_payload, provenance=provenance)

    def _blocks(
        self, role: str, spec: RoleSpec, span: Tuple[datetime, datetime]
    ) -> List[Any]:
        wanted = _blocks_for_span(spec.times, spec.window_days, span)
        wanted_ids = {index for index, _ in wanted}
        blocks = self._window_blocks.setdefault(role, {})
        with self._lock:
            # 预取好的块直接搬进来，省掉一次同步读
            ready = self._prefetched.get(role)
            if ready is not None:
                if wanted_ids and ready[0] < min(wanted_ids):
                    self._prefetched.pop(role, None)  # 请求已经走过去了，用不上
                elif ready[0] in wanted_ids and ready[0] not in blocks:
                    blocks[ready[0]] = ready[1]
                    self._prefetched.pop(role, None)
            missing = [index for index, _ in wanted if index not in blocks]
            if not missing and wanted:
                self._stats["cache_hits"] += 1
            for index in missing:
                block_span = next(item[1] for item in wanted if item[0] == index)
                blocks[index] = self._build(spec, block_span)
            # 滚动窗口：留本次请求覆盖的块 + **前后各一块**。
            #
            # 为什么前后都要留：块序是「起报日外层、时效窗内层」，一个起报日组
            # 要从头扫到尾（观测日 D … D+L），下一组又从头开始（D+1 …），于是
            # 观测读是"向前扫 L 天、再回跳 L-2 天"。只要窗宽 W ≥ L，这一轮扫描
            # 的观测日跨度就装得进相邻两个块；留着 ±1 块，回跳时上一轮读过的块
            # 还在，一个观测日整 run 只读一次。
            #   只留当前块（原行为）：回跳必落空，读放大 ~L 倍；
            #   只留"前一块"：回跳的目标块会在下一组开头被挤掉，同样落空。
            # 代价：内存峰值从 1 个窗块变成最多 3 个（典型 2 个）。
            keep = set(wanted_ids)
            for index in wanted_ids:
                keep.add(index - 1)
                keep.add(index + 1)
            for index in list(blocks):
                if index not in keep:
                    del blocks[index]
        self._schedule_next_block(role, spec, wanted)
        return [blocks[index] for index, _ in wanted]

    # ------------------------------------------------------------------
    # 预取：把按日期顺序的下一个块提前读进来
    # ------------------------------------------------------------------

    def _schedule_next_block(self, role: str, spec: RoleSpec, wanted: List[Any]) -> None:
        """排一个后台任务，读「下一个窗块」（预测依据见 ``_next_block_span``）。

        代价与风险都封顶：待用槽只放**一个**块（内存 +1 个窗块），猜错也只是
        白读一次，不影响任何结果——真正的请求到了按正常路径物化。
        """
        if not self._prefetch or spec.policy != "window":
            return
        target = _next_block_span(spec, wanted)
        if target is None:
            return
        index, block_span = target
        if self._prefetched.get(role, (None,))[0] == index:
            return  # 已经备好了同一个块
        # 该块已经在窗块缓存里（``_blocks`` 的 ±1 保留正好把它留着）：再读一遍
        # 是纯白读，而且读来的副本没人认领、会一直占着 _prefetched 那个槽。
        # 没有这道闸，预取会在每个请求上重读一次当前块的下一个块。
        with self._lock:
            if index in self._window_blocks.get(role, {}):
                return
        self._start_prefetch_worker()
        self._prefetch_queue.put((role, spec, index, block_span))

    def _start_prefetch_worker(self) -> None:
        if self._prefetch_worker is not None:
            return
        # daemon：预取线程绝不能拖住进程退出（进程池子进程尤其不能被它卡住）
        self._prefetch_worker = threading.Thread(
            target=self._prefetch_loop, name="runloader-prefetch", daemon=True
        )
        self._prefetch_worker.start()

    def _prefetch_loop(self) -> None:
        while True:
            item = self._prefetch_queue.get()
            try:
                if item is None:
                    return
                role, spec, index, block_span = item
                try:
                    bundle = self._build(spec, block_span)
                except Exception as exc:
                    # 预取失败不是错误：真正的请求会按正常路径再读一次并抛出
                    log.debug("预取角色 %s 的窗块 %d 失败: %s", role, index, exc)
                    continue
                with self._lock:
                    if not self._prefetch:
                        continue  # 缓存已经被 release() 放掉了，这份没人认领
                    current = self._prefetched.get(role)
                    # 同时只备一个块：已有更新的就丢掉这次的结果
                    if current is None or current[0] <= index:
                        self._prefetched[role] = (index, bundle)
            finally:
                # 与 Queue.get 配对：调用方（含测试）可以 queue.join() 等预取落地
                self._prefetch_queue.task_done()
