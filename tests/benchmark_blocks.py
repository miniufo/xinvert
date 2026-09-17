# -*- coding: utf-8 -*-
"""
Sweep thread-block shapes for the GPU Red-Black SOR kernel.

Tests several (bx, by) block configs at grid sizes 128..4096 to find the
best coalescing/occupancy trade-off for the target GPU.

Block configs tested:
  - (16,16): old default (x-dim 16 < warp 32 → uncoalesced, 2 transactions/warp)
  - (32, 8): 256 threads, full-warp x → coalesced
  - (32,16): 512 threads, coalesced
  - (32,32): 1024 threads, coalesced (max block size, low occupancy)
  - (64, 4): 256 threads, 2 warps wide in x → coalesced
  - (16,32): 512 threads, uncoalesced

The block shape is read from the XINVERT_GPU_BLOCK2D env var at *call time*
(see gpus._block_2d), so no module reload is needed — each config reuses the
same compiled kernels and only changes the launch grid.

Run:  python tests/benchmark_blocks.py
      python tests/benchmark_blocks.py 128 256 512 1024 2048 4096
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# robustly suppress numba performance warnings (numba_cuda redefines the class)
warnings.filterwarnings('ignore', message='.*under-utilization.*')
warnings.filterwarnings('ignore', message='.*low occupancy.*')
for _mod in ('numba.core.errors', 'numba_cuda.errors',
             'numba_cuda.numba.core.errors',
             'numba_cuda.numba.cuda.errors'):
    try:
        _err = __import__(_mod, fromlist=['NumbaPerformanceWarning'])
        warnings.filterwarnings('ignore', category=_err.NumbaPerformanceWarning)
    except Exception:
        pass

import numpy as np
import xarray as xr
from xinvert import invert_Poisson


def _make_poisson_problem(n):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    psi_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    F = -2.0 * np.pi**2 * psi_true
    da_F = xr.DataArray(F, dims=['y', 'x'], coords={'y': y, 'x': x})
    return da_F, psi_true


def _solve(da_F, architect):
    ip = {'BCs': ['fixed', 'fixed'], 'undef': np.nan,
          'mxLoop': 1000, 'tolerance': 0.0, 'printInfo': False,
          'debug': False, 'architect': architect}
    return invert_Poisson(da_F, dims=['y', 'x'], coords='cartesian', iParams=ip)


def _time_gpu(da_F, block2d, repeat=3, warmup=1):
    """Time GPU solve with a given block shape (set via env var, no reload)."""
    n = da_F.shape[0]
    x = np.linspace(0, 1, n); y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    psi_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    os.environ['XINVERT_GPU_BLOCK2D'] = f'{block2d[0]},{block2d[1]}'
    for _ in range(warmup):
        _solve(da_F, 'gpu')
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        S = _solve(da_F, 'gpu')
        times.append(time.perf_counter() - t0)
    err = float(np.nanmax(np.abs(S.values - psi_true)))
    return float(np.median(times)), err


def _time_cpu(da_F, repeat=2, warmup=1):
    n = da_F.shape[0]
    x = np.linspace(0, 1, n); y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    psi_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    for _ in range(warmup):
        _solve(da_F, 'cpu')
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _solve(da_F, 'cpu')
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


CONFIGS = [(16, 16), (32, 8), (32, 16), (32, 32), (64, 4), (16, 32)]
SIZES = [int(s) for s in sys.argv[1:]] or [128, 256, 512, 1024, 2048, 4096]


def main():
    W = 12
    print('=' * 72)
    print('Thread-block shape sweep for GPU Red-Black SOR (1000 iters, tol=0)')
    print('=' * 72)
    import platform
    print(f'CPU: {platform.processor() or platform.machine()}')
    try:
        from numba import cuda
        d = cuda.get_current_device()
        nm = d.name.decode() if isinstance(d.name, bytes) else d.name
        nsm = getattr(d, 'MULTIPROCESSOR_COUNT', '?')
        print(f'GPU: {nm}  ({nsm} SMs, CC {d.compute_capability})')
    except Exception:
        print('GPU: not available')
    print(f'grid sizes: {SIZES}')
    print(f'configs:    {CONFIGS}')
    print('=' * 72)

    # CPU baseline per size (skip for large grids — 4096^2 CPU is ~100s/run)
    cpu_times = {}
    for n in SIZES:
        if n > 1024:
            continue
        da, _ = _make_poisson_problem(n)
        cpu_times[n] = _time_cpu(da, repeat=2, warmup=1)

    results = {}  # (n, cfg) -> (t, err)
    for n in SIZES:
        da, _ = _make_poisson_problem(n)
        for cfg in CONFIGS:
            t, err = _time_gpu(da, cfg, repeat=3, warmup=1)
            results[(n, cfg)] = (t, err)

    # print table: rows = sizes, cols = configs
    hdr = f'{"grid":>{W}}  {"cpu(s)":>{W}}  ' + '  '.join(
        f'{f"{c[0]}x{c[1]}":>{W}}' for c in CONFIGS)
    print(hdr)
    print('-' * len(hdr))
    for n in SIZES:
        ct = cpu_times.get(n)
        cells = [f'{ct:>{W}.4f}' if ct is not None else f'{"-":>{W}}']
        for c in CONFIGS:
            t = results[(n, c)][0]
            cells.append(f'{t:>{W}.4f}')
        print(f'{"%dx%d" % (n, n):>{W}}  ' + '  '.join(cells))
    print()

    # speedup table
    hdr2 = f'{"grid":>{W}}  ' + '  '.join(
        f'{f"{c[0]}x{c[1]}":>{W}}' for c in CONFIGS)
    print('Speedup vs CPU (n/a for large grids where CPU not timed):')
    print(hdr2)
    print('-' * len(hdr2))
    for n in SIZES:
        cells = []
        for c in CONFIGS:
            t = results[(n, c)][0]
            ct = cpu_times.get(n)
            if ct and t > 0:
                cells.append(f'{f"{ct / t:.1f}x":>{W}}')
            else:
                cells.append(f'{"-":>{W}}')
        print(f'{"%dx%d" % (n, n):>{W}}  ' + '  '.join(cells))
    print()

    # best config per size
    print('Best config per size:')
    for n in SIZES:
        best = min(CONFIGS, key=lambda c: results[(n, c)][0])
        t = results[(n, best)][0]
        ct = cpu_times.get(n)
        sp_str = f'{ct / t:.1f}x vs cpu' if ct else '(no cpu baseline)'
        print(f'  {n}x{n}: {best[0]}x{best[1]}  ({t:.4f}s, {sp_str})')


if __name__ == '__main__':
    main()
