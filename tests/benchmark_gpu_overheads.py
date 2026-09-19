# -*- coding: utf-8 -*-
"""GPU per-solve overhead micro-benchmark + dask submission strategies.

Saves results to tests/results/gpu_overheads.json.

1. Micro-benchmark of fixed GPU-side costs (512x512 float64 = 2 MB):
   - cuda.to_device (device alloc + H2D copy)
   - copy into an existing device buffer (H2D copy only)
   - copy_to_host (D2H)
   - kernel launch (noop kernel)

2. Dask submission strategies for GPU solves (tol=0, fixed iterations),
   covering grids 64..4096 (time-step count scaled down for large grids):
   - 'single-threaded': tasks run one at a time (one in-flight solve)
   - 'threads'        : default threaded scheduler (concurrent solves)
   Concurrent GPU solves gain nothing (kernels serialise on the default
   stream; blocking convergence-check syncs create a convoy effect) while
   each in-flight solve holds ~5 device buffers in VRAM.

Run:  python tests/benchmark_gpu_overheads.py
"""
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', message='.*under-utilization.*')
warnings.filterwarnings('ignore', message='.*low occupancy.*')

import numpy as np
import xarray as xr
import dask
from numba import cuda
from xinvert import invert_Poisson

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def micro_bench():
    a = np.zeros((512, 512))
    d = cuda.to_device(a)

    def bench(fn, n):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e6  # us

    d2 = cuda.to_device(np.zeros(4))

    @cuda.jit
    def k_noop(x):
        pass

    k_noop[1, 1](d2)

    return {
        'to_device_alloc_us': bench(lambda: cuda.to_device(a), 200),
        'copy_to_device_us': bench(lambda: d.copy_to_device(a), 200),
        'copy_to_host_us': bench(lambda: d.copy_to_host(a), 200),
        'kernel_launch_us': bench(lambda: k_noop[1, 1](d2), 2000),
    }


def make(nt, n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    f = -2.0 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    vor = np.stack([f] * nt)
    return xr.DataArray(vor, dims=['time', 'y', 'x'],
                        coords={'time': np.arange(nt), 'y': y, 'x': x}
                        ).chunk({'time': 1})


def dask_submission(ip):
    # cover grids 64..4096; scale the time-step count down for large grids
    cases = [(max(4, min(48, 4000000 // (n * n))), n)
             for n in (64, 128, 256, 512, 1024, 2048, 4096)]
    out = []
    for nt, n in cases:
        da = make(nt, n)
        sf = invert_Poisson(da, dims=['y', 'x'], coords='cartesian',
                            iParams=ip)
        sf.compute()  # JIT warmup
        row = {'nt': nt, 'grid': n, 'schedulers': {}}
        for sched in ('single-threaded', 'threads'):
            with dask.config.set(scheduler=sched):
                t0 = time.perf_counter()
                sf.compute()
                dt = time.perf_counter() - t0
            row['schedulers'][sched] = dt
        row['speedup_threads_vs_single'] = (row['schedulers']['single-threaded']
                                            / row['schedulers']['threads'])
        out.append(row)
        print(f"{nt} slices {n}x{n}: "
              f"single={row['schedulers']['single-threaded']:.3f}s "
              f"threads={row['schedulers']['threads']:.3f}s "
              f"({row['speedup_threads_vs_single']:.2f}x)")
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cuda.get_current_device()  # init CUDA context in the main thread

    print('=== micro-benchmark (512x512 float64) ===')
    micro = micro_bench()
    for k, v in micro.items():
        print(f'  {k:22}: {v:8.1f} us')

    ip = {'BCs': ['fixed', 'fixed'], 'undef': np.nan, 'mxLoop': 200,
          'tolerance': 0.0, 'printInfo': False, 'architect': 'gpu'}

    print('=== dask submission strategies (GPU) ===')
    sub = dask_submission(ip)

    with open(os.path.join(RESULTS_DIR, 'gpu_overheads.json'), 'w') as f:
        json.dump({'micro_bench_us': micro, 'dask_submission': sub}, f, indent=1)
    print('saved tests/results/gpu_overheads.json')


if __name__ == '__main__':
    main()
