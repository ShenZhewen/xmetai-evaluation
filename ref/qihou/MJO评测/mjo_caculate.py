import os
import copy
import numpy as np
import pandas as pd
import xarray as xr
from time import perf_counter

EVAL_AREA = dict(
    mjo=dict(lat=(17, -17), lon=(0, 360))

)

STDS = dict(
    mjo=dict(olr=5.41039, u850=1.94333, u200=5.01924)

)

EIGVALS = dict(
    mjo=[53.06753, 50.25787], 

)


def remove_120d(anno_gt, anno_pred, start_year, end_year, lead_time=0):
    anno_real = anno_pred.sel(time=slice(start_year, end_year))
    init_times = anno_real.time
    print(len(init_times))

    if 'lead_time' in anno_real.dims:
        anno_real = anno_real.sel(lead_time=lead_time)

    # remove previous 120 day mean
    anno_prev = []
    for init_time in init_times:
        #print(init_time)

        t2 = copy.deepcopy(init_time)
        t1 = t2 - pd.Timedelta(days=120-lead_time)  

        if lead_time == 0:
            t2 -= pd.Timedelta(days=1)            

        v = anno_gt.sel(time=slice(t1, t2)).sum('time')
        #print(v)
        #sys.exit()
        if "lead_time" in anno_pred.dims and lead_time > 1:
            v += anno_pred.sel(time=init_time, lead_time=slice(1, lead_time-1)).sum('lead_time')
            #print(v)
        anno_prev.append(v / 120)


    anno_prev = xr.concat(anno_prev, 'time')  
    #print(anno_prev)
    times = init_times + pd.Timedelta(days=lead_time)
    anno_real = anno_real.assign_coords(time=times)
    anno_prev = anno_prev.assign_coords(time=times)
    anno_real = anno_real - anno_prev
    return anno_real, init_times.values


def project_mjo(anno, eigvecs, stds, eigvals):
    # from IPython import embed; embed()
    anno = anno.mean('lat')
    olr = anno.olr / stds['olr']
    u850 = anno.u850 / stds['u850']
    u200 = anno.u200 / stds['u200']
    # print_field(anno.olr, "anno_avg -> ")
    # print_field(olr, "anno_norm -> ")
    ds = xr.concat([olr, u850, u200], 'lon')
    #print(eigvecs)
    rmm1 = np.einsum('ij,j->i', ds, eigvecs[:,0]) / np.sqrt(eigvals[0])
    rmm2 = np.einsum('ij,j->i', ds, eigvecs[:,1]) / np.sqrt(eigvals[1])
    mjo = np.stack([rmm1, rmm2], axis=1).astype(np.float32)
    return mjo


def compute_index(anno_gt, anno_pred, eigvecs, start_step=1, end_step=42, name="mjo"):
    start_year = pd.Timestamp(f'2021-01-01')
    end_year = pd.Timestamp(f'2021-12-31')
    lead_times =  np.arange(start_step, end_step+1)

    if "step" in anno_pred.dims:
        anno_pred = anno_pred.rename({"step": "lead_time"})

    proj_func = dict(
        mjo=project_mjo,
    )[name]


    lat = EVAL_AREA[name]['lat']    
    lon = EVAL_AREA[name]['lon']    
    print(f"index area: {lat} x {lon}")
    
    if 'lead_time' not in anno_pred.dims:
        anno, init_times = remove_120d(anno_gt, anno_pred, start_year, end_year, 0)
        anno = anno.sel(lat=slice(*lat), lon=slice(*lon))
        print(anno)
        coeff = proj_func(anno, eigvecs, stds=STDS[name], eigvals=EIGVALS[name])
        print(coeff)
        ds = xr.DataArray(
            name=name,
            data=coeff,
            dims=['time', 'index'],
            coords={
                'time': init_times, 
                'index': np.arange(coeff.shape[-1])+1
            },
        )   

        ds = aggregate(ds, start_step=start_step, end_step=end_step, dim="lead_time")
        ds = ds.assign_coords(lead_time=lead_times)
        return ds
    

    coeffs = []
    for lead_time in lead_times:
        print(f"Compute {name} for {lead_time:03d} days ...")
        anno, init_times = remove_120d(anno_gt, anno_pred, start_year, end_year, lead_time)
        print('这一步都没过去呀鸽鸽')
        anno = anno.sel(lat=slice(*lat), lon=slice(*lon))
        coeff = proj_func(anno, eigvecs, stds=STDS[name], eigvals=EIGVALS[name])
        print(f"{lead_time}: {coeff.shape}, {coeff.min():.3f} ~ {coeff.max():.3f}")
        coeffs.append(coeff)

    coeffs = np.stack(coeffs)
    print(f"coeffs: {coeffs.shape}, {coeffs.min()} ~ {coeffs.max()}")

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


def get_amplitude(ds):
    rmm1 = ds.sel(index=1)
    rmm2 = ds.sel(index=2)
    amp = np.sqrt(rmm1 ** 2 + rmm2 **2)
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
    lead_times = target.lead_time.values
    cors = []
    errs = []
    for d, lead_time in enumerate(lead_times):
        tgt = target.sel(lead_time=lead_time)
        out = output.sel(lead_time=lead_time)  
        print(f"day: {d+1}, num_tgt: {len(tgt.time)}, num_out: {len(out.time)}, lead_time: {lead_time}")

        A = tgt.sel(index=1) * out.sel(index=1) + tgt.sel(index=2) * out.sel(index=2)
        B = tgt.sel(index=1) ** 2 + tgt.sel(index=2) ** 2
        C = out.sel(index=1) ** 2 + out.sel(index=2) ** 2
        cor = A.sum('time') / (np.sqrt(B.sum('time')) * np.sqrt(C.sum('time')))
        cors.append(cor.load())

        A = np.abs(tgt.sel(index=1) - out.sel(index=1)) ** 2
        B = np.abs(tgt.sel(index=2) - out.sel(index=2)) ** 2
        err = np.sqrt((A + B).mean('time'))
        errs.append(err.load())

    lead_times = lead_times[:len(cors)]
    cors = xr.concat(cors, 'lead_time')
    cors = cors.assign_coords(lead_time=lead_times)
    cors.name = "mjo_cor"

    errs = xr.concat(errs, 'lead_time')
    errs = errs.assign_coords(lead_time=lead_times)
    errs.name = "mjo_err"

    ds = xr.merge([cors, errs])
    df = ds.to_dataframe()
    print(f"mjo_cor: {df}")
    return ds

def build_tgmjo_from_txt():
    """
    从两个txt文件构建tgmjo.nc:
    1. MJO_RMM-CRA-OBS-Oyear-20241231.txt (2024年数据)
    2. MJO_RMM-CRA-OBS-Oyear-20251231.txt (2025年数据)

    输出格式: (lead_time: 42, time: 365, index: 2)
    - time: 2024年全年365天
    - lead_time: 1~42天
    - index: 1=rmm1, 2=rmm2
    """
    base_dir = r'/gpu/zhouchg/liuzh/MJO'

    # 读取2024年数据 (366行，闰年)
    print("读取2024年数据...")
    df2024 = pd.read_csv(
        os.path.join(base_dir, 'MJO_RMM-CRA-OBS-Oyear-20241231.txt'),
        sep=r'\s+', header=None,
        names=['date', 'rmm1', 'rmm2']
    )
    print(f"  2024年: {len(df2024)} 条记录, {df2024.date.iloc[0]} ~ {df2024.date.iloc[-1]}")

    # 读取2025年数据 (365行)
    print("读取2025年数据...")
    df2025 = pd.read_csv(
        os.path.join(base_dir, 'MJO_RMM-CRA-OBS-Oyear-20251231.txt'),
        sep=r'\s+', header=None,
        names=['date', 'rmm1', 'rmm2']
    )
    print(f"  2025年: {len(df2025)} 条记录, {df2025.date.iloc[0]} ~ {df2025.date.iloc[-1]}")

    # 合并为统一查找表
    df_all = pd.concat([df2024, df2025], ignore_index=True)
    df_all['datetime'] = pd.to_datetime(df_all['date'], format='%Y%m%d')
    df_all = df_all.set_index('datetime')
    print(f"  合并后: {len(df_all)} 条记录")

    # 时间维度: 2024年全年365天 (参考mjo_yc.nc格式, 虽然2024是闰年但取365天)
    start_date = pd.Timestamp('2024-01-01')
    times_2024 = pd.date_range(start=start_date, periods=365, freq='D')
    print(f"时间维度: {times_2024[0].strftime('%Y-%m-%d')} ~ {times_2024[-1].strftime('%Y-%m-%d')} ({len(times_2024)}天)")

    lead_times = np.arange(1, 43)  # 1~42 天

    # 构建输出数组 (lead_time, time, index)
    n_time = len(times_2024)
    n_lead = len(lead_times)
    data = np.full((n_lead, n_time, 2), np.nan, dtype=np.float32)

    print(f"\n开始构建数据, 形状: ({n_lead}, {n_time}, 2)...")
    t_start = perf_counter()

    missing_dates = []
    for i, t in enumerate(times_2024):
        for j, lt in enumerate(lead_times):
            target_date = t + pd.Timedelta(days=int(lt))
            if target_date in df_all.index:
                data[j, i, 0] = df_all.loc[target_date, 'rmm1']
                data[j, i, 1] = df_all.loc[target_date, 'rmm2']
            else:
                missing_dates.append(target_date.strftime('%Y%m%d'))

    t_elapsed = perf_counter() - t_start
    print(f"构建完成, 耗时: {t_elapsed:.2f}s")

    if missing_dates:
        print(f"警告: {len(missing_dates)} 个日期缺少数据: {missing_dates[:10]}...")

    # 检查是否有NaN
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        print(f"警告: 数据中有 {nan_count} 个NaN值")

    # 创建 xarray Dataset
    ds = xr.Dataset(
        {
            'mjo': xr.DataArray(
                data=data,
                dims=['lead_time', 'time', 'index'],
                coords={
                    'lead_time': lead_times,
                    'time': times_2024,
                    'index': [1, 2]
                },
                name='mjo',
                attrs={
                    'long_name': 'MJO RMM index from CRA observation',
                    'source': 'MJO_RMM-CRA-OBS data'
                }
            )
        }
    )

    # 设置time编码属性 (units/calendar属于encoding, 不是attrs)
    ds.time.encoding['units'] = 'days since 2024-01-01 00:00:00'
    ds.time.encoding['calendar'] = 'proleptic_gregorian'

    # 保存
    output_path = os.path.join(base_dir,'Calculate_Result', 'tgmjo.nc')
    ds.to_netcdf(output_path)
    print(f"\n已保存到: {output_path}")
    print(f"数据集概要:\n{ds}")

    return ds


def verify_tgmjo():
    """验证生成的tgmjo.nc文件"""
    print("\n" + "=" * 60)
    print("验证 tgmjo.nc")
    print("=" * 60)

    ds = xr.open_dataset(r'/gpu/zhouchg/liuzh/MJO/Calculate_Result/tgmjo.nc')       ##用两个.txt文件画出来一个.nc文件
    print(f"\n数据集信息:")
    print(ds)
    print(f"\n维度: {dict(ds.sizes)}")
    print(f"mjo shape: {ds.mjo.shape}")
    print(f"mjo dtype: {ds.mjo.dtype}")
    print(f"time units: {ds.time.encoding.get('units', 'N/A')}")
    print(f"time calendar: {ds.time.encoding.get('calendar', 'N/A')}")

    # 检查是否有NaN
    nan_count = np.isnan(ds.mjo.values).sum()
    print(f"\nNaN数量: {nan_count}")

    # 验证几个具体值
    print("\n--- 抽样验证 ---")
    # time=2024-01-01, lead_time=1 -> 应该是2024-01-02的值
    v = ds.mjo.sel(time='2024-01-01', lead_time=1).values
    print(f"2024-01-01, lead_time=1: rmm1={v[0]:.3f}, rmm2={v[1]:.3f}")
    print(f"  (应为2024-01-02的观测值)")

    # time=2024-01-01, lead_time=42 -> 应该是2024-02-12的值
    v = ds.mjo.sel(time='2024-01-01', lead_time=42).values
    print(f"2024-01-01, lead_time=42: rmm1={v[0]:.3f}, rmm2={v[1]:.3f}")
    print(f"  (应为2024-02-12的观测值)")

    # time=2024-12-30, lead_time=1 -> 应该是2024-12-31的值(2024文件)
    v = ds.mjo.sel(time='2024-12-30', lead_time=1).values
    print(f"2024-12-30, lead_time=1: rmm1={v[0]:.3f}, rmm2={v[1]:.3f}")
    print(f"  (应为2024-12-31的观测值, 来自2024文件)")

    # time=2024-12-30, lead_time=2 -> 应该是2025-01-01的值(2025文件)
    v = ds.mjo.sel(time='2024-12-30', lead_time=2).values
    print(f"2024-12-30, lead_time=2: rmm1={v[0]:.3f}, rmm2={v[1]:.3f}")
    print(f"  (应为2025-01-01的观测值, 来自2025文件)")

    # time=2024-12-20, lead_time=42 -> 应该是2025-01-31的值(跨年到2025)
    v = ds.mjo.sel(time='2024-12-20', lead_time=42).values
    print(f"2024-12-20, lead_time=42: rmm1={v[0]:.3f}, rmm2={v[1]:.3f}")
    print(f"  (应为2025-01-31的观测值, 来自2025文件)")

    # 检查第一个和最后一个时间
    print(f"\n第一个时间: {ds.time.values[0]}")
    print(f"最后一个时间: {ds.time.values[-1]}")
    print(f"lead_time范围: {ds.lead_time.values[0]} ~ {ds.lead_time.values[-1]}")

    # 打印前3个时间点、全部lead_time的数据
    print("\n--- 前3天, 前5个lead_time的数据 ---")
    for t_idx in range(3):
        t = ds.time.values[t_idx]
        for lt in range(1, 6):
            v = ds.mjo.sel(time=t, lead_time=lt).values
            print(f"  time={pd.Timestamp(t).strftime('%Y%m%d')}, lt={lt:2d}: rmm1={v[0]:8.4f}, rmm2={v[1]:8.4f}")

    ds.close()
    print("\n验证完成!")
    return True

tg = xr.open_dataset('/gpu/zhouchg/liuzh/MJO/Calculate_Result/tgmjo.nc')
# out = xr.open_dataset('M:\FDP\mjo/mjo_yc_1.5.nc')
# print(out.mjo)

if __name__ == '__main__':
    # build_tgmjo_from_txt()
    # verify_tgmjo()

    import math
    import matplotlib.pyplot as plt

    # 读取两个预报数据
    out1 = xr.open_dataset(r'/gpu/zhouchg/liuzh/MJO/Result/mjo_yc_1.5.nc')
    out2 = xr.open_dataset(r'/gpu/zhouchg/liuzh/MJO/Result/mjo_yc_2.0.nc')

    # 分别计算 MJO 相关系数
    print("=" * 50)
    print("计算 fengshun1.5 ...")
    mjocor1 = compute_mjo_cor(tg.mjo, out1.mjo)
    print("=" * 50)
    print("计算 fengshun2.0 ...")
    mjocor2 = compute_mjo_cor(tg.mjo, out2.mjo)

    lead_times = mjocor1['mjo_cor'].lead_time.values  # 1~42

    def find_cross(lead_times, cor_vals):
        """找到曲线与 y=0.5 的交点, 返回 (cross_lead, cross_lead_floor) 或 None"""
        for i in range(len(cor_vals) - 1):
            y1, y2 = cor_vals[i], cor_vals[i + 1]
            if (y1 >= 0.5 and y2 <= 0.5) or (y1 <= 0.5 and y2 >= 0.5):
                cross_lead = lead_times[i] + (0.5 - y1) / (y2 - y1) * (lead_times[i + 1] - lead_times[i])
                return cross_lead, math.floor(cross_lead)
        return None, None

    # 画图
    plt.figure(figsize=(10, 6))

    # fengshun1.5 — 蓝色
    cor_vals1 = mjocor1['mjo_cor'].values
    plt.plot(lead_times, cor_vals1, 'b-', linewidth=1.5, label='fengshun1.5')
    cross1 = find_cross(lead_times, cor_vals1)
    if cross1[0] is not None:
        plt.axvline(x=cross1[0], color='blue', linestyle=':', linewidth=1)
        plt.plot(cross1[0], 0.5, 'bo', markersize=5)
        print(f"fengshun1.5: lead_time = {cross1[0]:.1f} d 处与 y=0.5 相交, 向下取整 {cross1[1]} d")

    # fengshun2.0 — 红色
    cor_vals2 = mjocor2['mjo_cor'].values
    plt.plot(lead_times, cor_vals2, 'r-', linewidth=1.5, label='fengshun2.0')
    cross2 = find_cross(lead_times, cor_vals2)
    if cross2[0] is not None:
        plt.axvline(x=cross2[0], color='red', linestyle=':', linewidth=1)
        plt.plot(cross2[0], 0.5, 'ro', markersize=5)
        print(f"fengshun2.0: lead_time = {cross2[0]:.1f} d 处与 y=0.5 相交, 向下取整 {cross2[1]} d")

    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)

    plt.xlabel('Lead Time (days)')
    plt.ylabel('MJO Correlation')
    plt.title('MJO Forecast Skill (bivariate correlation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'M/gpu/zhouchg/liuzh/MJO/Calculate_Result/mjocor.png', dpi=150)


