# -*- coding: utf-8 -*-
"""
Test whether print statements appear progressively when using dask
with chunking along the non-core (time) dimension.

- Expands to 12 time slices to have enough tasks
- Uses stricter tolerance to increase iteration count per slice
- Opens multi-core via dask threaded scheduler

Run:  python -m tests.test_dask_print
"""
import time
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client

from xinvert import invert_Poisson


def test_dask_progressive_print():
    # 启动多线程 dask client（n_workers=1, 多线程，共享同一进程的 stderr）
    client = Client(n_workers=1, threads_per_worker=4)
    print(f'dask client: {client}')
    print(f'dask dashboard: {client.dashboard_link}')

    try:
        ds = xr.open_dataset('./Data/Helmholtz_atmos.nc')

        vor = ds.vor.rename('vorticity')

        # 扩展到 12 个时间步（复制原始 2 个 step 6 次）
        vor = xr.concat([vor] * 6, dim='time')
        # 重新赋 time 坐标，避免重复
        vor['time'] = np.arange(vor.sizes['time'])

        # 沿 time chunk，每个 time slice 是一个独立的 dask task
        vor_chunked = vor.chunk({'time': 1})

        print(f'\nvor shape: {vor.shape}')
        print(f'vor chunks: {vor_chunked.chunks}')
        print(f'number of time slices: {vor.sizes["time"]}')
        print(f'dask scheduler threads: {client.nthreads}')
        print('--- starting compute (prints should appear one-by-one on stderr) ---\n')

        iParams = {
            'BCs'      : ['extend', 'periodic'],
            'undef'    : np.nan,
            'mxLoop'   : 100000,    # 放大最大迭代次数
            'tolerance': 1e-16,     # 极严格容差，迫使每个 slice 迭代更多次
            'printInfo': True,
        }

        t0 = time.time()

        sf = invert_Poisson(vor_chunked, dims=['lat', 'lon'], iParams=iParams)

        # .compute() 触发计算，4 个线程并行处理 12 个 slice
        # 每个 slice 完成后应立即输出一条 print 到 stderr
        sf = sf.compute()

        elapsed = time.time() - t0
        print(f'\n--- done in {elapsed:.1f}s ---')
        print(f'result shape: {sf.shape}')

        # assert 计算结果形状与输入一致
        assert sf.shape == vor.shape, f'shape mismatch: {sf.shape} != {vor.shape}'
        assert np.isfinite(sf.values).any(), 'result contains no finite values'
    finally:
        # dask distributed 新版 + bokeh/tornado 在 Python 3.12 上，
        # client.close() 在 event loop 运行时同步调用会抛 RuntimeError。
        # 这是已知的 teardown 兼容性问题，不影响测试逻辑本身；
        # client 会在进程退出时自动清理。
        try:
            client.close()
        except RuntimeError:
            pass


if __name__ == '__main__':
    test_dask_progressive_print()
