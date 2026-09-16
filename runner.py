"""Lightweight evaluation runner: config-driven capability execution."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from core.runlog import start_logging

CAPABILITIES = {
    "weather_ts_det": ("core.api", "run_ts_det"),
    "weather_ts_ens": ("core.api", "run_ts_ens"),
    "weather_ts_ens_prob": ("core.api", "run_ts_ens_prob"),
    "weather_field_scores": ("core.api", "run_field"),
    "weather_ens_crps": ("core.api", "run_ens_crps"),
    "weather_ens_field_scores": ("core.api", "run_ens_field"),
    "typhoon": ("core.api", "run_tc"),
    # fdp 参考实现（core/fdp_adapter.py → runpy 启动 fdp/ 原脚本）
    "fdp_field_det": ("core.fdp_adapter", "run_field_det"),
    "fdp_ens": ("core.fdp_adapter", "run_ens"),
    "fdp_activity_spectrum": ("core.fdp_adapter", "run_activity_spectrum"),
    "fdp_tp_det": ("core.fdp_adapter", "run_tp_det"),
}


def run_single_config(path: Path) -> int:
    """Run a single configuration file."""
    t0 = time.time()
    print(f"[runner] Loading config: {path}", flush=True)

    ns = {"__file__": str(path)}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    cfg = dict(ns.get("CONFIG", ns.get("config", {})))

    # Check if this is a batch config (migrated shell script)
    if cfg.get("type") == "batch":
        print(f"[runner] Config type: batch", flush=True)
        rc = run_batch_config(path, cfg)
        elapsed = time.time() - t0
        print(f"[runner] Batch completed in {elapsed:.1f}s ({elapsed/60:.2f}m)", flush=True)
        return rc

    # Regular capability config
    print(f"[runner] Config type: regular capability", flush=True)
    name = cfg.pop("capability")
    print(f"[runner] Capability: {name}", flush=True)

    outdir = Path(cfg.pop("output_dir", "results"))
    if not outdir.is_absolute():
        outdir = path.parent / outdir
        print(f"[runner] Resolved output_dir: {outdir}", flush=True)
    else:
        print(f"[runner] Using absolute output_dir: {outdir}", flush=True)
    cfg["outdir"] = str(outdir)

    if name not in CAPABILITIES:
        print(f"[runner] ERROR: Unsupported capability: {name}. Available: {sorted(CAPABILITIES.keys())}",
              file=sys.stderr, flush=True)
        return 1

    mod, func = CAPABILITIES[name]
    print(f"[runner] Starting capability execution: {mod}.{func}", flush=True)

    try:
        result = getattr(importlib.import_module(mod), func)(cfg)
    except NotImplementedError as e:
        print(f"[runner] BLOCKER: {name}: {e}", file=sys.stderr, flush=True)
        return 1

    if isinstance(result, tuple):
        result = result[0]

    print(f"[runner] Creating output directory: {outdir}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "to_csv"):
        csv_path = outdir / f"{name}.csv"
        result.to_csv(csv_path, index=False)
        print(f"[runner] Wrote CSV: {csv_path}", flush=True)
    else:
        json_path = outdir / f"{name}.json"
        json_path.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        print(f"[runner] Wrote JSON: {json_path}", flush=True)

    elapsed = time.time() - t0
    print(f"[runner] {name} completed in {elapsed:.1f}s ({elapsed/60:.2f}m)", flush=True)
    return 0


def run_batch_config(path: Path, cfg: dict) -> int:
    """Run a batch configuration (migrated shell script)."""
    from core.batch_adapter import run_batch_with_periods

    # Use vendored entry script with local vfc imports
    entry_script = Path(__file__).parent / "core" / "run_batch_rmse.py"

    if not entry_script.exists():
        print(f"[runner] ERROR: vendored entry script not found: {entry_script}",
              file=sys.stderr, flush=True)
        return 1

    # Extract batch parameters
    label = cfg.pop("label", "batch")
    pred_root = cfg.pop("pred_root")
    target_zarr = cfg.pop("target_zarr")
    outdir_root = cfg.pop("outdir_root")
    periods = cfg.pop("periods")
    metrics = cfg.pop("metrics")
    variables = cfg.pop("variables")

    # Optional parameters
    var_metrics = cfg.pop("var_metrics", None)
    climo = cfg.pop("climo", None)
    pred_q_scale = cfg.pop("pred_q_scale", 1.0)
    n_workers = cfg.pop("n_workers", 48)
    resume = cfg.pop("resume", False)
    worker_fallback = cfg.pop("worker_fallback", None)
    summarize_mode = cfg.pop("summarize_mode", None)
    env_overrides = cfg.pop("env_overrides", None)
    output_name = cfg.pop("output_name", label)
    resume_cache = cfg.pop("resume_cache", False)

    print(f"[runner] Batch label: {label}", flush=True)
    print(f"[runner] Pred root: {pred_root}", flush=True)
    print(f"[runner] Output root: {outdir_root}", flush=True)
    print(f"[runner] Periods: {len(periods)}", flush=True)
    print(f"[runner] Metrics: {metrics}", flush=True)
    print(f"[runner] Variables: {len(variables)} vars", flush=True)
    print(f"[runner] Workers: {n_workers} (fallback: {worker_fallback})", flush=True)

    return run_batch_with_periods(
        entry_script=entry_script,
        pred_root=pred_root,
        target_zarr=target_zarr,
        outdir_root=outdir_root,
        periods=periods,
        metrics=metrics,
        variables=variables,
        var_metrics=var_metrics,
        climo=climo,
        pred_q_scale=pred_q_scale,
        n_workers=n_workers,
        resume=resume,
        worker_fallback=worker_fallback,
        summarize_mode=summarize_mode,
        env_overrides=env_overrides,
        output_name=output_name,
        resume_cache=resume_cache,
    )


def run_multi_config(path: Path) -> int:
    """Run multiple configurations listed in a file (like run_all.py)."""
    t0 = time.time()
    print(f"[runner] Loading multi-config: {path}", flush=True)

    ns = {"__file__": str(path)}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)

    configs = ns.get("CONFIGS", [])
    if not configs:
        print(f"[runner] ERROR: No CONFIGS list found in {path}", file=sys.stderr, flush=True)
        return 1

    print(f"[runner] Found {len(configs)} configs to execute", flush=True)
    failed = 0
    config_dir = path.parent

    for i, config_name in enumerate(configs, start=1):
        config_path = config_dir / config_name
        if not config_path.exists():
            print(f"[runner] ERROR: Config not found: {config_path}",
                  file=sys.stderr, flush=True)
            failed = 1
            continue

        print(f"\n{'='*80}", flush=True)
        print(f"[runner] [{i}/{len(configs)}] Running: {config_name}", flush=True)
        print(f"{'='*80}\n", flush=True)

        rc = run_single_config(config_path)
        if rc != 0:
            failed = 1
            print(f"[runner] ERROR: {config_name} failed with exit code {rc}",
                  file=sys.stderr, flush=True)
        else:
            print(f"[runner] {config_name} succeeded", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    if failed == 0:
        print(f"[runner] All {len(configs)} configs completed successfully in {elapsed:.1f}s ({elapsed/60:.2f}m)", flush=True)
    else:
        print(f"[runner] Completed with errors in {elapsed:.1f}s ({elapsed/60:.2f}m)", file=sys.stderr, flush=True)
    print(f"{'='*80}", flush=True)

    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Lightweight evaluation framework runner")
    ap.add_argument("-config", "--config", required=True,
                    help="Python config file path")
    args = ap.parse_args(argv)

    path = Path(args.config).resolve()
    # 越早接管 stdout/stderr 越好：下面这条 config 不存在的报错也要进日志。
    # 日志名 = logs/runner_<config名>_<北京时间>.log
    start_logging(Path(__file__).stem, path.stem)
    print(f"[runner] Starting runner with config: {path}", flush=True)

    if not path.exists():
        print(f"[runner] ERROR: Config not found: {path}", file=sys.stderr, flush=True)
        return 1

    # Check if this is a multi-config file (has CONFIGS list)
    ns = {"__file__": str(path)}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)

    if "CONFIGS" in ns and isinstance(ns["CONFIGS"], list):
        return run_multi_config(path)
    else:
        return run_single_config(path)


if __name__ == "__main__":
    raise SystemExit(main())
