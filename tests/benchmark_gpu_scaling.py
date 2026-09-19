# -*- coding: utf-8 -*-
"""GPU per-slice time decomposition & scaling for the SODA Poisson case.

Mirrors Parallel_inversions.ipynb (12 steps, 300x720, mxLoop=20000,
tolerance=1e-15) and answers: where does the GPU time go?

Results: tests/results/gpu_scaling.json
Run:  python tests/benchmark_gpu_scaling.py
"""
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
from numba import cuda

import xinvert.gpus as gpus
from xinvert import invert_Poisson

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
DATA = os.path.join(HERE, '..', 'Data', 'SODA_curl.nc')


def load_data(chunks=None):
    return xr.open_dataset(DATA, chunks=chunks)


def capture_params():
    """Run one real inversion; capture the exact per-slice solver args."""
    ds = load_data({'time': 1})
    ip = {'BCs': ['extend', 'periodic'], 'undef': np.nan, 'mxLoop': 100,
          'tolerance': 0.0, 'printInfo': False, 'debug': False,
          'architect': 'gpu'}
    captured = {}
    orig = gpus._solve_standard_2D_gpu

    def spy(*a, **kw):
        if not captured:
            names = ['S', 'A', 'B', 'C', 'F', 'info', 'yc', 'xc', 'BCy',
                     'BCx', 'delxSqr', 'ratioQtr', 'ratioSqr', 'optArg',
                     'undef', 'flags', 'mxLoop', 'tolerance']
            captured.update(zip(names, a))
        return orig(*a, **kw)

    gpus._solve_standard_2D_gpu = spy
    try:
        invert_Poisson(ds.curl, dims=['lat', 'lon'], coords='lat-lon',
                       iParams=ip).compute()
    finally:
        gpus._solve_standard_2D_gpu = orig
    return captured


def decompose(p, mxLoop, tol):
    """Instrumented per-slice GPU solve: where does the time go?"""
    S, A, B, C, F = (np.ascontiguousarray(p[k].copy())
                     for k in ('S', 'A', 'B', 'C', 'F'))
    yc, xc = p['yc'], p['xc']
    t0 = time.perf_counter()
    d_S = cuda.to_device(S)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    d_F = cuda.to_device(F)
    t_upload = time.perf_counter() - t0

    from xinvert.gpus import (_sor_2d_rb, _extend_y_boundary,
                              _abs_norm_2d, _auto_bsize_1d,
                              _compute_check_interval, _evaluate_gpu_norm)
    blocks = (16, 16)
    bs1d = _auto_bsize_1d(xc)
    bcx = max((xc + 15) // 16, 1)
    d_norm = cuda.device_array(2, dtype=np.float64)
    check_interval = _compute_check_interval(mxLoop, tol)

    t0 = time.perf_counter()
    t_sync = 0.0
    n_sync = 0
    loop = 0
    norm_prev = np.finfo(np.float64).max
    while True:
        _extend_y_boundary[bcx, bs1d](d_S, yc, xc, p['undef'])
        for color in (0, 1):
            _sor_2d_rb[blocks, blocks](d_S, d_A, d_B, d_C, d_F, yc, xc,
                                       True, p['delxSqr'], p['ratioQtr'],
                                       p['ratioSqr'], p['optArg'],
                                       p['undef'], color)
        loop += 1
        if loop % check_interval == 0 or loop >= mxLoop:
            ts = time.perf_counter()
            d_norm[0] = 0.0
            d_norm[1] = 0.0
            _abs_norm_2d[blocks, blocks](d_S, p['undef'], d_norm)
            norm_h = d_norm.copy_to_host()
            norm, error, overflow = _evaluate_gpu_norm(
                norm_h[0], norm_h[1], norm_prev, True, tol)
            t_sync += time.perf_counter() - ts
            n_sync += 1
            if overflow or error < tol or loop >= mxLoop or norm == 0:
                break
            norm_prev = norm
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_S.copy_to_host(S)
    t_download = time.perf_counter() - t0

    return {'loops_done': loop, 'n_syncs': n_sync, 't_upload_s': t_upload,
            't_loop_s': t_loop, 't_sync_s': t_sync,
            't_download_s': t_download,
            't_total_s': t_upload + t_loop + t_download,
            'per_iter_us': t_loop / loop * 1e6,
            'n_launches': 3 * loop + n_sync}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    cuda.get_current_device()

    print('=== 1. capture per-slice parameters ===')
    p = capture_params()
    print(f"  grid {p['yc']}x{p['xc']}, optArg={p['optArg']:.4f}")

    print('=== 2. per-slice decomposition (mxLoop=20000, tol=1e-15) ===')
    decomp = decompose(p, 20000, 1e-15)
    for k, v in decomp.items():
        print(f'  {k:14}: {v}')
    share = 100 * decomp['n_launches'] * 9.4e-6 / decomp['t_loop_s']
    print(f'  -> launch overhead ~{share:.0f}% of the SOR loop')

    print('=== 3. scaling: per-solve time vs mxLoop ===')
    sweep = []
    for mx in (100, 400, 1600, 6400, 20000):
        d = decompose(p, mxLoop=mx, tol=1e-15)
        sweep.append({'mxLoop': mx, 't_total_s': d['t_total_s']})
        print(f"  mxLoop={mx:6d}: {d['t_total_s']:6.3f} s "
              f"({d['per_iter_us']:5.1f} us/iter)")
    slope, intercept = np.polyfit([s['mxLoop'] for s in sweep],
                                  [s['t_total_s'] for s in sweep], 1)
    print(f'  fit: {slope*1e6:.1f} us/iter, intercept {intercept*1e3:.2f} ms')

    with open(os.path.join(RESULTS, 'gpu_scaling.json'), 'w') as f:
        json.dump({'decomposition': decomp, 'scaling': sweep,
                   'slope_s_per_iter': slope, 'intercept_s': intercept},
                  f, indent=1)
    print('saved tests/results/gpu_scaling.json')


if __name__ == '__main__':
    main()
