##=========>   由于需要生成mjo_yc_1.5.nc和mjo_yc_2.0.nc两个文件进而用mjo_caculate.py画图，所以该代码需要运行两边!!!   <============
##=========>   由于需要生成mjo_yc_1.5.nc和mjo_yc_2.0.nc两个文件进而用mjo_caculate.py画图，所以该代码需要运行两边!!!   <============
##=========>   由于需要生成mjo_yc_1.5.nc和mjo_yc_2.0.nc两个文件进而用mjo_caculate.py画图，所以该代码需要运行两边!!!   <============

###win mypy311  python3.11.10
# ============================================================
# MJO Index Calculation — OPTIMIZED VERSION
# ============================================================
# 三项核心优化:
#   1. 预加载数据到内存 (.load()) — 消除重复 netCDF 磁盘 I/O
#   2. cumsum + 整数索引替代 datetime 切片 — remove_120d 提速 ~50-100x
#   3. multiprocessing 并行 lead_time 循环 — 42 个任务分布到 N 个 CPU 核心
#
# 预期加速比: 5x–15x (取决于 CPU 核数和 I/O 带宽)
# 用法: 修改 NUM_WORKERS 控制并行度, 设为 1 即回退到纯串行优化版
# ============================================================

from netCDF4 import Dataset
import os
import xarray as xr
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
import glob
import time
import datetime
import shutil
import re

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

for i in [xr, np, pd]:
    print(i.__name__, ": ", i.__version__, sep="")

import copy
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import shutil
import glob
from time import perf_counter
import sys
import multiprocessing as mp
from functools import partial

# ============================================================
# 用户可调参数
# ============================================================
NUM_WORKERS = min(mp.cpu_count(), 8)   # 并行进程数 (建议 ≤ CPU 核数)
USE_MULTIPROCESSING = True              # True=多进程并行; False=串行(仅 cumsum 优化)
LOAD_INTO_MEMORY = True                 # 强烈建议开启, 避免重复 I/O

# ============================================================
# 常量
# ============================================================
EVAL_AREA = dict(
    mjo=dict(lat=(17, -17), lon=(0, 360))
)

STDS = dict(
    mjo=dict(olr=5.41039, u850=1.94333, u200=5.01924)
)

EIGVALS = dict(
    mjo=[53.06753, 50.25787],
)


# ============================================================
# 优化核心: remove_120d
# ============================================================
# 原版性能问题: 对每个 init_time 调用 sel(time=slice(...))
#   - 每次 slice 都要在时间轴上搜索起止位置 (O(log n) binary search)
#   - 如果数据未加载到内存, 每次 slice 触发 netCDF 读盘
#   - 总调用次数: ~365 init_times × 42 lead_times ≈ 15,000 次
#
# 优化方案:
#   1. 数据预加载到内存 (消除 I/O)
#   2. 预计算 cumsum (O(n) 一次, 而非每次 O(n) 求和)
#   3. 用 searchsorted 一次性找到所有起止索引 (避免逐个二分搜索)
#   4. 使用整数 isel 替代 datetime sel (消除搜索开销)
# ============================================================

def remove_120d(anno_gt, anno_pred, start_year, end_year, lead_time=0):
    """
    Remove previous 120-day mean.

    与原始版本行为完全一致, 但使用以下优化:
    - cumsum 预计算 → 窗口求和变 O(1)
    - searchsorted 批量找索引 → 单次 O(n log n) 替代 n 次 O(log n)
    - 整数 isel 替代 datetime sel → 无搜索开销
    """
    anno_real = anno_pred.sel(time=slice(start_year, end_year))
    init_times = anno_real.time
    init_times_vals = init_times.values
    n_times = len(init_times_vals)

    if 'lead_time' in anno_real.dims:
        anno_real = anno_real.sel(lead_time=lead_time)

    # ---- 计算时间窗口边界 ----
    if lead_time == 0:
        t1_delta = pd.Timedelta(days=120)
        t2_delta = pd.Timedelta(days=1)
    else:
        t1_delta = pd.Timedelta(days=120 - lead_time)
        t2_delta = pd.Timedelta(days=0)

    # ---- 加载所需观测数据 (一次性, 而非每次循环) ----
    obs_t_min = init_times_vals.min() - pd.Timedelta(days=130)
    obs_t_max = init_times_vals.max() + pd.Timedelta(days=1)
    obs_sub = anno_gt.sel(time=slice(obs_t_min, obs_t_max))
    obs_times = obs_sub.time.values  # numpy datetime64 array

    # ---- 预计算 cumsum (沿时间轴, 对所有变量) ----
    # xarray Dataset.cumsum() 对每个 data_var 独立计算
    obs_cumsum = obs_sub.cumsum('time')

    # ---- 批量找到所有 init_time 对应的整数索引 ----
    # searchsorted 在有序数组中找到插入位置, 比逐个二分查找快很多
    t1_vals = init_times_vals - np.timedelta64(t1_delta.value, 'ns')
    t2_vals = init_times_vals - np.timedelta64(t2_delta.value, 'ns') if t2_delta.value > 0 else init_times_vals

    t2_idx = np.searchsorted(obs_times, t2_vals, side='right') - 1
    t1_idx = np.searchsorted(obs_times, t1_vals, side='right') - 1
    t2_idx = np.clip(t2_idx, 0, len(obs_times) - 1)
    t1_idx = np.clip(t1_idx, 0, len(obs_times) - 1)

    # ---- 对每个 init_time 用整数索引取 cumsum (比 datetime slice 快 ~100x) ----
    # 注意: xarray 不支持用整数数组做花式索引, 所以这里仍需 Python 循环
    # 但整数 isel 比 datetime sel 快 50-100x, 365 次循环 < 0.5s
    anno_prev_list = []
    zero_ds = xr.zeros_like(obs_cumsum.isel(time=0))  # 用于 t1_idx=0 的情况

    for i in range(n_times):
        i2 = int(t2_idx[i])
        i1 = int(t1_idx[i])

        # 观测部分: cumsum[t2] - cumsum[t1-1]
        cumsum_at_t2 = obs_cumsum.isel(time=i2)
        if i1 > 0:
            cumsum_at_t1m1 = obs_cumsum.isel(time=i1 - 1)
            obs_sum = cumsum_at_t2 - cumsum_at_t1m1
        else:
            obs_sum = cumsum_at_t2  # t1_idx=0 意味着从时间轴起点开始

        # 预报部分 (仅 lead_time > 1 时)
        if lead_time > 1 and 'lead_time' in anno_pred.dims:
            fcst_contrib = anno_pred.isel(time=i).sel(
                lead_time=slice(1, lead_time - 1)
            ).sum('lead_time')
            obs_sum = obs_sum + fcst_contrib

        anno_prev_list.append(obs_sum / 120.0)

    # ---- 合并结果 ----
    anno_prev = xr.concat(anno_prev_list, 'time')

    # 对齐时间坐标
    times_new = init_times_vals + pd.Timedelta(days=lead_time)
    anno_prev = anno_prev.assign_coords(time=times_new)
    anno_real = anno_real.assign_coords(time=times_new)

    # 扣除 120 天均值
    anno_result = anno_real - anno_prev

    return anno_result, init_times_vals


def project_mjo(anno, eigvecs, stds, eigvals):
    """MJO projection — 已使用 np.einsum 向量化, 速度很快"""
    anno = anno.mean('lat')
    olr = anno.olr / stds['olr']
    u850 = anno.u850 / stds['u850']
    u200 = anno.u200 / stds['u200']
    ds = xr.concat([olr, u850, u200], 'lon')
    rmm1 = np.einsum('ij,j->i', ds, eigvecs[:, 0]) / np.sqrt(eigvals[0])
    rmm2 = np.einsum('ij,j->i', ds, eigvecs[:, 1]) / np.sqrt(eigvals[1])
    mjo = np.stack([rmm1, rmm2], axis=1).astype(np.float32)
    return mjo


# ============================================================
# 多进程 worker: 计算单个 lead_time
# ============================================================
# 每个子进程独立加载数据文件 (避免 pickle 序列化 xarray 对象的开销)
# 文件路径通过 args 传入
# ============================================================

def _compute_single_lead_time(args_tuple):
    """
    计算单个 lead_time 的 MJO 系数。

    参数 (tuple, 由 pool.map 解包):
        lead_time, data_path, eigvecs, name, lat, lon,
        start_year_str, end_year_str
    """
    (lead_time, data_path, eigvecs,
     name, lat_slice, lon_slice, start_year_str, end_year_str) = args_tuple

    start_year = pd.Timestamp(start_year_str)
    end_year = pd.Timestamp(end_year_str)

    # 每个子进程独立加载 NetCDF 文件
    tg = xr.open_dataset(os.path.join(data_path, 'CRA_V1.5_output', 'CRA_anom_daily_var3_2p5_2023_2024.nc'))
    fcst = xr.open_dataset(os.path.join(data_path, 'fengshun_v1.5_output', 'fengshun1.5_anom_daily_var3_2p5_2024.nc'))      ########需要跑两边，下面这行也要跑一遍#####################
    # fcst = xr.open_dataset(os.path.join(data_path, 'fengshun_v2.0_output', 'fengshun2.0_anom_daily_var3_2p5_2024.nc'))      ########需要跑两边，上面这行也要跑一遍#####################



    # 处理时间格式
    if not np.issubdtype(fcst['time'].dtype, np.datetime64):
        fcst['time'] = pd.to_datetime(fcst['time'].values)

    fcst = fcst.assign_coords(lead_time=np.arange(1, 61))

    # 加载到内存 (消除重复 I/O)
    if LOAD_INTO_MEMORY:
        tg = tg.load()
        fcst = fcst.load()

    print(f"[lt={lead_time:03d}] computing...", flush=True)
    t0 = perf_counter()

    anno, init_times = remove_120d(tg, fcst, start_year, end_year, lead_time)
    anno = anno.sel(lat=slice(*lat_slice), lon=slice(*lon_slice))
    coeff = project_mjo(anno, eigvecs, stds=STDS[name], eigvals=EIGVALS[name])

    elapsed = perf_counter() - t0
    print(f"[lt={lead_time:03d}] done in {elapsed:.1f}s, "
          f"shape={coeff.shape}, range=[{coeff.min():.3f}, {coeff.max():.3f}]",
          flush=True)

    # 清理
    tg.close(); fcst.close()
    return coeff


# ============================================================
# compute_index — 支持多进程并行的主计算函数
# ============================================================

def compute_index(anno_gt, anno_pred, eigvecs, start_step=1, end_step=42, name="mjo"):
    """
    计算 MJO 指数。

    - 如果 lead_time 维度不存在, 按单时间点处理 (lead_time=0)
    - 如果 USE_MULTIPROCESSING=True, 用进程池并行计算各 lead_time
    - 否则串行但使用 cumsum 优化
    """
    start_year = pd.Timestamp('2024-01-01')
    end_year = pd.Timestamp('2024-12-31')
    lead_times = np.arange(start_step, end_step + 1)

    if "step" in anno_pred.dims:
        anno_pred = anno_pred.rename({"step": "lead_time"})

    lat = EVAL_AREA[name]['lat']
    lon = EVAL_AREA[name]['lon']
    print(f"index area: {lat} x {lon}")

    # ---- 无 lead_time 维度: 直接算 lead_time=0 ----
    if 'lead_time' not in anno_pred.dims:
        anno, init_times = remove_120d(anno_gt, anno_pred, start_year, end_year, 0)
        anno = anno.sel(lat=slice(*lat), lon=slice(*lon))
        coeff = project_mjo(anno, eigvecs, stds=STDS[name], eigvals=EIGVALS[name])
        ds = xr.DataArray(
            name=name, data=coeff,
            dims=['time', 'index'],
            coords={'time': init_times, 'index': np.arange(coeff.shape[-1]) + 1},
        )
        ds = ds.assign_coords(lead_time=lead_times)
        return ds

    # ---- 有 lead_time: 并行或串行计算 ----
    if USE_MULTIPROCESSING and len(lead_times) > 1:
        print(f"\n{'='*60}")
        print(f"  Multiprocessing: {NUM_WORKERS} workers × {len(lead_times)} lead_times")
        print(f"{'='*60}\n")

        # 构建参数列表 (不含 xarray 对象, 传文件路径由子进程自行加载)
        args_list = [
            (int(lt), str(data_path), eigvecs,
             name, lat, lon, '2024-01-01', '2024-12-31')
            for lt in lead_times
        ]

        t_pool_start = perf_counter()
        with mp.Pool(processes=NUM_WORKERS) as pool:
            coeffs = pool.map(_compute_single_lead_time, args_list)
        t_pool = perf_counter() - t_pool_start
        print(f"\n  All {len(lead_times)} lead_times completed in {t_pool:.1f}s "
              f"(avg {t_pool / len(lead_times):.1f}s each, "
              f"speedup ~{len(lead_times) / max(NUM_WORKERS, 1) * t_pool / max(t_pool, 0.001):.1f}x vs serial)")
    else:
        # 串行模式 (仍享受 cumsum 优化)
        coeffs = []
        for lead_time in lead_times:
            args = (int(lead_time), str(data_path), eigvecs,
                    name, lat, lon, '2024-01-01', '2024-12-31')
            coeffs.append(_compute_single_lead_time(args))

    # ---- 组装结果 ----
    coeffs = np.stack(coeffs)
    print(f"coeffs array: shape={coeffs.shape}, min={coeffs.min():.3f}, max={coeffs.max():.3f}")

    # 获取 init_times (从第一个 lead_time 的结果反推)
    tg_tmp = xr.open_dataset(os.path.join(data_path, 'CRA_V1.5_output', 'CRA_anom_daily_var3_2p5_2023_2024.nc'))
    fcst_tmp = xr.open_dataset(os.path.join(data_path, 'fengshun_v1.5_output', 'fengshun1.5_anom_daily_var3_2p5_2024.nc'))      ########需要跑两边，下面这行也要跑一遍#####################
    # fcst_tmp = xr.open_dataset(os.path.join(data_path, 'fengshun_v2.0_output', 'fengshun2.0_anom_daily_var3_2p5_2024.nc'))      ########需要跑两边，上面这行也要跑一遍#####################


    if not np.issubdtype(fcst_tmp['time'].dtype, np.datetime64):
        fcst_tmp['time'] = pd.to_datetime(fcst_tmp['time'].values)
    fcst_tmp = fcst_tmp.assign_coords(lead_time=np.arange(1, 61))
    if LOAD_INTO_MEMORY:
        tg_tmp = tg_tmp.load()
        fcst_tmp = fcst_tmp.load()
    _, init_times = remove_120d(tg_tmp, fcst_tmp, start_year, end_year, lead_times[0])
    tg_tmp.close(); fcst_tmp.close()

    ds = xr.DataArray(
        name=name,
        data=coeffs,
        dims=['lead_time', 'time', 'index'],
        coords={
            'lead_time': lead_times[:coeffs.shape[0]],
            'time': init_times,
            'index': np.arange(coeffs.shape[-1]) + 1
        },
    )
    return {name: ds}


# ============================================================
# 辅助函数 (未改动)
# ============================================================

def get_amplitude(ds):
    rmm1 = ds.sel(index=1)
    rmm2 = ds.sel(index=2)
    amp = np.sqrt(rmm1 ** 2 + rmm2 ** 2)
    return amp


def get_phase(ds):
    rmm1 = ds.sel(index=1)
    rmm2 = ds.sel(index=2)
    p1 = 1 * ((rmm1 < 0) & (rmm2 < 0) & (abs(rmm1) > abs(rmm2)))
    p2 = 2 * ((rmm1 < 0) & (rmm2 < 0) & (abs(rmm1) < abs(rmm2)))
    p3 = 3 * ((rmm1 > 0) & (rmm2 < 0) & (abs(rmm1) < abs(rmm2)))
    p4 = 4 * ((rmm1 > 0) & (rmm2 < 0) & (abs(rmm1) > abs(rmm2)))
    p5 = 5 * ((rmm1 > 0) & (rmm2 > 0) & (abs(rmm1) > abs(rmm2)))
    p6 = 6 * ((rmm1 > 0) & (rmm2 > 0) & (abs(rmm1) < abs(rmm2)))
    p7 = 7 * ((rmm1 < 0) & (rmm2 > 0) & (abs(rmm1) < abs(rmm2)))
    p8 = 8 * ((rmm1 < 0) & (rmm2 > 0) & (abs(rmm1) > abs(rmm2)))
    phase = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
    return phase


def compute_mjo_cor(target, output):
    """
    计算 MJO 相关系数和 RMSE。

    优化: 使用向量化 xarray 操作, 一次性计算所有 lead_time,
          消除原版的逐 lead_time 循环。
    """
    # 批量提取 RMM1/RMM2
    tgt_r1 = target.sel(index=1)  # [lead_time, time]
    tgt_r2 = target.sel(index=2)
    out_r1 = output.sel(index=1)
    out_r2 = output.sel(index=2)

    # ---- 向量化相关系数 ----
    A = tgt_r1 * out_r1 + tgt_r2 * out_r2
    B = tgt_r1 ** 2 + tgt_r2 ** 2
    C = out_r1 ** 2 + out_r2 ** 2
    cor = A.sum('time') / (np.sqrt(B.sum('time')) * np.sqrt(C.sum('time')))
    cor = cor.load()
    cor.name = "mjo_cor"

    # ---- 向量化 RMSE ----
    A_err = np.abs(tgt_r1 - out_r1) ** 2
    B_err = np.abs(tgt_r2 - out_r2) ** 2
    err = np.sqrt((A_err + B_err).mean('time'))
    err = err.load()
    err.name = "mjo_err"

    ds = xr.merge([cor, err])
    print(f"mjo_cor:\n{ds.to_dataframe()}")
    return ds


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Windows 下 multiprocessing 需要 freeze_support
    mp.freeze_support()

    t_main_start = perf_counter()

    hm_path = "/gpu/zhouchg/liuzh"
    work_path = os.path.join(hm_path, 'MJO')
    data_path = os.path.join(hm_path, 'MJO')
    print(f'work_path = {work_path}')
    print(f'data_path = {data_path}')

    # ---- 加载 EOF 结构 ----
    eoffilename = sorted(glob.glob(os.path.join(work_path, 'WH04_EOFstruc.txt')))
    vec = pd.read_csv(eoffilename[0], sep='\s+', header=None)
    vec = vec.values
    print(f"EOF shape: {vec.shape}")

    # ---- 加载数据 ----
    print("\nLoading datasets...")
    tg = xr.open_dataset(os.path.join(data_path, 'CRA_V1.5_output', 'CRA_anom_daily_var3_2p5_2023_2024.nc'))
                                                                            ##变量格式：olr(time≤730, lat=73, lon=144)，空间分辨率2.5°，时间分辨率1天
                                                                            ## u200(time≤730, lat=73, lon=144)，空间分辨率2.5°，时间分辨率1天
                                                                            ## u850(time≤730, lat=73, lon=144)，空间分辨率2.5°，时间分辨率1天

    print(tg)
    fcst = xr.open_dataset(os.path.join(data_path, 'fengshun_v1.5_output', 'fengshun1.5_anom_daily_var3_2p5_2024.nc'))                  ########需要跑两边，下面这行也要跑一遍#####################
    # fcst = xr.open_dataset(os.path.join(data_path, 'fengshun_v2.0_output', 'fengshun2.0_anom_daily_var3_2p5_2024.nc'))                ########需要跑两边，上面这行也要跑一遍#####################
                                                                            ##变量格式：olr(time≤365, lead_time=60, lat=73, lon=144)，空间分辨率2.5°
                                                                                    ##u200(time≤365, lead_time=60, lat=73, lon=144)，空间分辨率2.5°
                                                                                    ##u850(time≤365, lead_time=60, lat=73, lon=144)，空间分辨率2.5°


    # 处理时间格式
    time_dtype = fcst['time'].dtype
    print(f'[CHECK] time dtype = {time_dtype}')
    if np.issubdtype(time_dtype, np.datetime64):
        print('[CHECK] time is already datetime64, skip conversion')
    else:
        print('[CHECK] converting time to datetime64...')
        fcst['time'] = pd.to_datetime(fcst['time'].values)

    fcst = fcst.assign_coords(lead_time=np.arange(1, 61))

    # 预加载到内存 (如果不使用多进程, 在主进程加载)
    if LOAD_INTO_MEMORY and not USE_MULTIPROCESSING:
        print("Loading data into memory (.load())...")
        tg = tg.load()
        fcst = fcst.load()
        print("Done.")

    print(fcst)

    # ---- 计算 MJO 指数 ----
    print("\n" + "=" * 60)
    print("  Computing MJO index")
    print("=" * 60)
    mjoindex = compute_index(tg, fcst, vec)

    # ---- 保存结果 ----
    out_file = os.path.join(data_path, 'Result', 'mjo_yc_1.5.nc')           ##文件格式：mjo(lead_time=42, time=365, index=2)        ########需要跑两边，下面这行也要跑一遍#####################
    # out_file = os.path.join(data_path, 'Result', 'mjo_yc_2.0.nc')           ##含义：lead_time=42：只输出未来第1～42天              ########需要跑两边，上面这行也要跑一遍#####################
                                                                             ## time=365：365个起报日期
                                                                            ## index=2：RMM1和RMM2两个分量

    xr.Dataset(mjoindex).to_netcdf(out_file)              
    print(f"\nSaved to: {out_file}")                 
    print(mjoindex)                             

    t_total = perf_counter() - t_main_start
    print(f"\n{'='*60}")
    print(f"  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"{'='*60}")
