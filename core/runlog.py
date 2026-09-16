"""把一次运行的输出同时写进控制台和 ``logs/`` 下的一份日志文件。

用法::

    from core.runlog import start_logging
    log_path = start_logging("runner", "weather_ts_det")

日志文件名是 ``logs/<启动的脚本名>_<config名>_<北京时间>.log``，
例如 ``logs/runner_weather_ts_det_20260915_143022.log``。

**为什么不用 ``logging.basicConfig(filename=...)``**：那只能抓住走 ``logging``
模块发出来的记录。这个仓库通篇是 ``print(..., flush=True)``，而且 ``runner.py``
还会 fork 出 ``run_batch_rmse.py`` 子进程和 ``ProcessPoolExecutor`` 的 worker——
它们写的是各自继承到的 fd 1，跟 Python 层的 ``sys.stdout`` 对象没关系，换成
带文件参数的 ``StreamHandler`` 一个都抓不到。所以只能在**文件描述符**这一层做：
把 fd 1/2 换成一根管子的写端，后台线程从读端读出来，一份写回真正的控制台、
一份写进日志文件。这样父进程的 print、子进程的 print、C 层报错、没被捕获的
traceback，一个都漏不掉。

日志写的是**字节**，不解码也不编码：日志里原样保留程序吐出来的东西，不会因为
控制台代码页是 GBK 而变成乱码或抛 ``UnicodeEncodeError``。而且每个 chunk 都是一次
不带缓冲的 ``os.write``，进程被 ``kill -9`` 掉日志也是完整的——不依赖退出钩子。
"""
from __future__ import annotations

import atexit
import os
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

#: 日志目录。默认是仓库根下的 ``logs/``（本模块和 runner.py 同级），要改就改这里。
# 本文件在 core/ 下，logs/ 在仓库根（=core 的上一级），不是本文件所在目录。
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

#: 北京时间的固定偏移。服务器多半跑在 UTC，直接取 ``datetime.now()`` 会差 8 小时，
#: 所以写死 +08:00，不依赖机器时区设置。
BEIJING = timezone(timedelta(hours=8))

#: 一次从管子里读多少字节。太小会被高频输出拖慢，64K 够用。
_CHUNK = 65536

_lock = threading.Lock()
_active: Optional["_Tee"] = None


def _slug(text: str) -> str:
    """把名字收拾成能安全当文件名的样子。"""
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text)).strip("_")
    return cleaned or "run"


def _open_unique(stem: str) -> Tuple[Path, int]:
    """在 ``LOG_DIR`` 下独占新建 ``<stem>.log``。

    同一秒内用同一个 config 起两次（重跑、或者两条命令一起发）时，退化成
    ``<stem>_2.log`` / ``_3``，而不是默认的 ``O_APPEND`` 把两次运行的输出
    混进同一个文件里。
    """
    for attempt in range(1, 100):
        path = LOG_DIR / (f"{stem}.log" if attempt == 1 else f"{stem}_{attempt}.log")
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        return path, fd
    raise FileExistsError(f"{LOG_DIR} 下 {stem}*.log 太多了，先清一清")


class _Tee:
    """把 fd 1/2 接到一根管子上，读出来一份给控制台、一份给日志文件。"""

    def __init__(self, path: Path, log_fd: int) -> None:
        self.path = path
        self._log_fd = log_fd

        # 先把 Python 缓冲区里攒着的内容吐到**原**控制台，再换管子；
        # 否则换完之后才 flush 的话，这些内容要多绕一圈管子才到屏幕上。
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (ValueError, OSError):
                pass

        # 存下真正的控制台 fd，稍后要写回去，退出时也要还回去。
        self._console = (os.dup(1), os.dup(2))

        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)

        # dup2 之后 fd 1 从终端变成了管子，Python 会把 stdout 从行缓冲切成块缓冲，
        # 交互运行时屏幕会一顿一顿地出。显式改回行缓冲，保持原来的观感。
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except (AttributeError, ValueError, OSError):
                pass

        self._read_fd = read_fd
        self._thread = threading.Thread(target=self._pump, name="runlog", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while True:
            try:
                chunk = os.read(self._read_fd, _CHUNK)
            except OSError:
                break
            if not chunk:
                break  # 写端全关了（我们自己 + 所有子进程都退出了）
            for fd in (self._log_fd,) + self._console:
                try:
                    os.write(fd, chunk)
                except OSError:
                    pass

    def close(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (ValueError, OSError):
                pass
        # 把控制台还回去：管子的写端一没了，_pump 就会读到 EOF 收工。
        try:
            os.dup2(self._console[0], 1)
            os.dup2(self._console[1], 2)
        except OSError:
            pass
        # 子进程可能还攥着写端（并行 worker 没退干净），所以给个上限，
        # 不能为了等 EOF 把退出卡住。
        self._thread.join(timeout=2.0)
        for fd in (self._read_fd, self._log_fd) + self._console:
            try:
                os.close(fd)
            except OSError:
                pass


def start_logging(script: str, config: Optional[str] = None) -> Path:
    """开始把 stdout/stderr 同时写进 ``logs/<script>_<config>_<北京时间>.log``。

    返回日志文件路径。**重复调用是空操作**，返回第一次那份的路径——叠两层 tee
    会让输出在控制台上翻倍，而且第二个日志文件里只会有后半段。

    ``script`` 是启动的脚本名，``config`` 是跑的那份 config（不给就只留脚本名）。
    """
    global _active
    with _lock:
        if _active is not None:
            return _active.path

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(BEIJING).strftime("%Y%m%d_%H%M%S")
        parts = [_slug(script)]
        if config:
            parts.append(_slug(config))
        parts.append(stamp)
        path, log_fd = _open_unique("_".join(parts))

        try:
            tee = _Tee(path, log_fd)
        except OSError as exc:
            os.close(log_fd)
            print(f"[runlog] 接不上 stdout/stderr（{exc}），这次只写控制台",
                  file=sys.stderr, flush=True)
            return path

        _active = tee
        atexit.register(tee.close)

        started = datetime.now(BEIJING).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[runlog] 北京时间 {started} 启动: {' '.join(sys.argv)}", flush=True)
        print(f"[runlog] 日志文件: {path}", flush=True)
        return path
