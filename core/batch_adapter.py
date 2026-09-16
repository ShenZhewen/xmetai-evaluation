"""Adapt batch configs to the vendored reference CLI."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _find_dates(root: str, start: str, end: str) -> list[str]:
    """Return existing date directories in the inclusive date range."""
    print(f"[batch_adapter] Scanning dates in {root} from {start} to {end}", flush=True)
    root_p = Path(root)
    if not root_p.is_dir():
        print(f"[batch_adapter] Root directory not found: {root}", flush=True)
        return []
    start_norm, end_norm = start.replace("-", ""), end.replace("-", "")
    result = []
    for entry in root_p.iterdir():
        if not entry.is_dir():
            continue
        match = re.fullmatch(r"(\d{4})-?(\d{2})-?(\d{2})", entry.name)
        if match:
            date_norm = "".join(match.groups())
            if start_norm <= date_norm <= end_norm:
                result.append(entry.name)
    print(f"[batch_adapter] Found {len(result)} dates in range", flush=True)
    return sorted(result)


def _fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _run_command(cmd: list[str], env: dict[str, str], workers: list[int]) -> int:
    for worker_count in workers:
        try:
            idx = cmd.index("--n-workers")
            cmd[idx + 1] = str(worker_count)
        except (ValueError, IndexError):
            pass
        print(f"[batch_adapter] Attempting with n_workers={worker_count}", flush=True)
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            print(f"[batch_adapter] Success with n_workers={worker_count}", flush=True)
            return 0
        print(f"[batch_adapter] Failed with n_workers={worker_count}, exit={result.returncode}",
              file=sys.stderr, flush=True)
    return 1


def _count_done_dates(root_path: Path, only: set[str] | None = None) -> int:
    """已完成（写下了 ``<date>_meta.json``）的日期数。

    **为什么不能光看日期目录在不在**：日期目录是在这个日期**刚开始**处理时就
    ``makedirs`` 的（``vfc/regr_ens.py:357``，加载完预报和气候态、真正算指标之前）。
    只数目录的话，N 个 worker 一起启动的瞬间就凑出 N 个"已完成"，进度直接跳到
    ``N/总数``、ETA 被算成几十秒；之后计数还卡在 N 上不动，要等某个 worker 开跑
    第二个日期才会 +1——中间那段静默看起来就跟卡死一样。

    ``<date>_meta.json`` 是这个仓库自己对"这一天跑完了"的定义（``regr_ens.py:1572``
    的 resume 判定、``_resume_ok`` 用的也是它），照它数才对得上真实进度。

    并行跑时日期目录先落在 ``<outdir_root>/.parts/part_XXX/`` 下，等所有 worker
    结束才合并回 ``<outdir_root>/``——只数根目录的话，整个并行阶段恒为 0。
    """
    def _finished(parent: Path) -> set:
        return {d.name for d in parent.iterdir()
                if d.is_dir() and re.fullmatch(r"\d{8}", d.name)
                and (d / (d.name + "_meta.json")).is_file()}

    # only: 只数本 period 的日期。outdir_root 是各 period 共享的，不按本批日期
    # 过滤的话，上一个 period 已合并的日期会被一起数进来——出现过 "324/181
    # (179.0%)、ETA 负数" 的荒唐进度。


    names = _finished(root_path)
    parts = root_path / ".parts"
    if parts.is_dir():
        for part in parts.iterdir():
            if part.is_dir():
                names |= _finished(part)
    if only is not None:
        names &= only
    return len(names)


def _progress_monitor(outdir_root: str, total_dates: int, stop_event: threading.Event,
                      date_names: set[str] | None = None):
    """Background thread to monitor and report progress."""
    start_time = time.time()
    last_count = 0

    while not stop_event.is_set():
        time.sleep(10)  # Check every 10 seconds
        if stop_event.is_set():
            break

        # Count completed date directories
        root_path = Path(outdir_root)
        if not root_path.exists():
            continue

        try:
            completed = _count_done_dates(root_path, date_names)
        except OSError:
            continue  # 正赶上合并阶段在建/删 .parts，下一轮再看

        if completed > last_count:
            elapsed = time.time() - start_time
            if completed > 0:
                avg_time_per_date = elapsed / completed
                remaining = total_dates - completed
                eta_seconds = avg_time_per_date * remaining
                eta_minutes = eta_seconds / 60

                # 带时间戳：这行每完成一个日期才打一次，中间可能静默好几分钟。
                # 不带时间的话，分不清它是刚打的还是十几分钟前打的——那就是
                # "看起来卡住了"的来源。
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][progress] "
                      f"Completed {completed}/{total_dates} dates "
                      f"({100*completed/total_dates:.1f}%) - "
                      f"ETA: {eta_minutes:.1f}m", flush=True)
            last_count = completed


def run_batch_job(
    entry_script: Path,
    pred_root: str,
    target_zarr: list[str],
    outdir_root: str,
    dates: list[str],
    metrics: list[str],
    variables: list[str],
    var_metrics: dict[str, list[str]] | None = None,
    climo: str | None = None,
    pred_q_scale: float = 1.0,
    n_workers: int = 48,
    resume: bool = False,
    worker_fallback: list[int] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> int:
    """Run one date batch using the reference entry point."""
    if not dates:
        print(f"[batch_adapter] No dates to process", flush=True)
        return 0

    print(f"[batch_adapter] Running batch job with {len(dates)} dates", flush=True)
    print(f"[batch_adapter]   Dates: {dates[0]} ... {dates[-1]}", flush=True)

    # Start progress monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(outdir_root, len(dates), stop_event, set(dates)),
        daemon=True
    )
    monitor_thread.start()

    try:
        cmd = [sys.executable, str(entry_script), "--pred-root", pred_root,
               "--dates", *dates, "--target-zarr", *target_zarr,
               "--metrics", *metrics, "--vars", *variables,
               "--n-workers", str(n_workers), "--outdir-root", outdir_root]
        if var_metrics:
            for var, metric_names in var_metrics.items():
                cmd.extend(["--var-metrics", f"{var}:{','.join(metric_names)}"])
        if climo:
            cmd.extend(["--climo", climo])
        if pred_q_scale != 1.0:
            cmd.extend(["--pred-q-scale", str(pred_q_scale)])
        if resume:
            cmd.append("--resume")

        env = os.environ.copy()
        if env_overrides:
            env.update({str(k): str(v) for k, v in env_overrides.items()})
        workers = list(worker_fallback or [n_workers])
        if n_workers not in workers:
            workers.insert(0, n_workers)

        return _run_command(cmd, env, workers)
    finally:
        # Stop monitoring thread
        stop_event.set()
        monitor_thread.join(timeout=1)


def _summary_long_form(summary_dir: Path, label: str) -> pd.DataFrame:
    """Normalize reference summary tables without changing their values."""
    frames = []
    for path in sorted(summary_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        stem = path.stem
        if "var" in df.columns and len(df.columns) > 1:
            long = df.melt(id_vars=["var"], var_name="statistic", value_name="value")
            long["lead_h"] = pd.NA
            long["wavenumber"] = pd.NA
        else:
            index_name = df.columns[0]
            long = df.melt(id_vars=[index_name], var_name="var", value_name="value")
            long = long.rename(columns={index_name: "wavenumber" if "spectrum" in stem else "lead_h"})
            long["statistic"] = stem
            if "spectrum" in stem:
                long["lead_h"] = pd.NA
            else:
                # 非 spectrum 表（如 mean_rmse/acc/fa，首列=lead_h）没有 wavenumber 列，
                # 必须补上 NA，否则下面统一取列时 KeyError。
                long["wavenumber"] = pd.NA
        long.insert(0, "metric", stem)
        long.insert(0, "capability", label)
        frames.append(long[["capability", "metric", "var", "lead_h", "wavenumber",
                            "statistic", "value"]])
    if not frames:
        raise ValueError(f"{summary_dir} 下没有可发布的 summary CSV")
    return pd.concat(frames, ignore_index=True)


def run_batch_with_periods(
    entry_script: Path,
    pred_root: str,
    target_zarr: list[str],
    outdir_root: str,
    periods: list[tuple[str, str]],
    summarize_mode: str | None = None,
    output_name: str = "batch",
    resume_cache: bool = False,
    **kwargs: Any,
) -> int:
    """Run periods, summarize, and publish one capability-level CSV.

    Reference date directories are temporary by default.  Setting resume_cache=True
    retains a fingerprinted hidden cache so the reference --resume contract remains
    available across invocations.
    """
    print(f"[batch_adapter] Starting batch with periods", flush=True)
    print(f"[batch_adapter]   Output root: {outdir_root}", flush=True)
    print(f"[batch_adapter]   Periods: {len(periods)}", flush=True)
    print(f"[batch_adapter]   Resume cache: {resume_cache}", flush=True)

    # Use unified outputs directory structure
    # /workspace/szwCode/xmetai-eval_pro/outputs/
    #   ├── .temp/          <- Temporary work directories
    #   └── results/        <- Final CSV results
    # 本文件在 core/ 下，outputs/ 在仓库根（=core 的上一级），不是本文件所在目录。
    eval_pro_root = Path(__file__).resolve().parent.parent
    unified_outputs = eval_pro_root / "outputs"
    temp_base = unified_outputs / ".temp"
    results_base = unified_outputs / "results"

    temp_base.mkdir(parents=True, exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)

    config_fingerprint = _fingerprint({
        "pred_root": pred_root, "target_zarr": target_zarr, "periods": periods,
        "summarize_mode": summarize_mode, "kwargs": kwargs,
    })

    if resume_cache:
        # Use persistent cache under .temp
        work_root = temp_base / output_name / config_fingerprint
        work_root.mkdir(parents=True, exist_ok=True)
        cleanup = False
        print(f"[batch_adapter] Using persistent cache: {work_root}", flush=True)
    else:
        # Use timestamped temporary directory under .temp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        work_root = temp_base / f"{output_name}_{timestamp}_{config_fingerprint[:8]}"
        work_root.mkdir(parents=True, exist_ok=True)
        cleanup = True
        print(f"[batch_adapter] Using temporary work directory: {work_root}", flush=True)

    # Final results go to results/ directory
    final_output_dir = results_base / output_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[batch_adapter] Final results will be saved to: {final_output_dir}", flush=True)

    failed = 0
    try:
        for i, (start, end) in enumerate(periods, start=1):
            print(f"\n[batch_adapter] Period {i}/{len(periods)}: {start} to {end}", flush=True)
            dates = _find_dates(pred_root, start, end)
            if not dates:
                print(f"[batch_adapter] No dates found, skipping period", flush=True)
                continue
            print(f"[batch_adapter] Processing {len(dates)} dates...", flush=True)
            rc = run_batch_job(entry_script=entry_script, pred_root=pred_root,
                               target_zarr=target_zarr, outdir_root=str(work_root),
                               dates=dates, **kwargs)
            if rc != 0:
                print(f"[batch_adapter] Period {i} failed", file=sys.stderr, flush=True)
                failed = 1
                break
            print(f"[batch_adapter] Period {i} completed successfully", flush=True)

        summary_dir = work_root / ".summary"
        if not failed and summarize_mode and work_root.is_dir():
            print(f"\n[batch_adapter] Running summary: {summarize_mode}", flush=True)
            summary_dir.mkdir(parents=True, exist_ok=True)
            sum_cmd = [sys.executable, str(entry_script), summarize_mode, str(work_root),
                       "--out", str(summary_dir), "--no-plot", "--quiet"]
            print(f"[batch_adapter] Summary command: {' '.join(sum_cmd)}", flush=True)
            result = subprocess.run(sum_cmd, env=os.environ.copy())
            if result.returncode != 0:
                print(f"[batch_adapter] Summary failed with exit={result.returncode}",
                      file=sys.stderr, flush=True)
                failed = 1
            else:
                print(f"[batch_adapter] Summary completed, normalizing to long-form CSV...", flush=True)
                try:
                    table = _summary_long_form(summary_dir, output_name)
                    csv_path = final_output_dir / f"{output_name}.csv"
                    table.to_csv(csv_path, index=False, float_format="%.6g")
                    print(f"[batch_adapter] Published: {csv_path}", flush=True)
                    print(f"[batch_adapter] CSV has {len(table)} rows", flush=True)
                except (OSError, ValueError, pd.errors.ParserError) as exc:
                    print(f"[batch_adapter] Publish failed: {exc}", file=sys.stderr, flush=True)
                    failed = 1
        if resume_cache and not failed:
            print(f"[batch_adapter] Writing cache metadata", flush=True)
            (work_root / "cache_meta.json").write_text(
                json.dumps({"fingerprint": config_fingerprint, "output_name": output_name},
                           indent=2), encoding="utf-8")
    finally:
        if cleanup:
            print(f"[batch_adapter] Cleaning up temporary directory: {work_root}", flush=True)
            shutil.rmtree(work_root, ignore_errors=True)
        else:
            print(f"[batch_adapter] Preserving cache directory: {work_root}", flush=True)

    if failed:
        print(f"[batch_adapter] Batch failed", file=sys.stderr, flush=True)
    else:
        print(f"[batch_adapter] Batch completed successfully", flush=True)
    return failed
