# -*- coding: utf-8 -*-
"""
Demo 2: Large-scale Poisson solve on GPU.

Solves the Poisson equation (wind-stress curl -> streamfunction) on a
single large grid using the GPU (Red-Black SOR).  The GPU parallelises
the SOR update *within* a single time step (fine-grained parallelism),
complementary to the dask demo which parallelises *across* time steps.

Uses real SODA wind-stress curl as the forcing, upsampled to larger
grids (600 x 1440, 1200 x 2880) to demonstrate GPU scaling.

Run from project root::

    python tests/demo_gpu_large.py

If no GPU is available, only CPU timings are reported.
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr

from xinvert import invert_Poisson

# silence GPU occupancy warnings for small grids during warmup
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

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data', 'SODA_curl.nc')


def _gpu_available():
    try:
        from numba import cuda
        if cuda.is_available():
            return True
        # real probe
        @cuda.jit
        def _t(x):
            x[0] = 1.0
        a = cuda.device_array(1, dtype=np.float64)
        _t[1, 1](a)
        return a.copy_to_host()[0] == 1.0
    except Exception:
        return False


def _gpu_name():
    try:
        from numba import cuda
        d = cuda.get_current_device()
        name = d.name
        if isinstance(name, bytes):
            name = name.decode()
        return name
    except Exception:
        return 'unknown'


def load_curl():
    ds = xr.open_dataset(DATA)
    curl = ds.curl.isel(time=0)  # single time step
    return curl


def upsample(curl, factor):
    """Upsample the forcing by integer factor via nearest-neighbour.

    This keeps the spatial structure of the real data while creating a
    larger grid to stress-test the GPU.
    """
    lat = curl.lat.values
    lon = curl.lon.values

    lat_new = np.linspace(lat[0], lat[-1], len(lat) * factor)
    lon_new = np.linspace(lon[0], lon[-1], len(lon) * factor)

    curl_up = curl.interp(lat=lat_new, lon=lon_new, method='linear')
    return curl_up


def solve(curl, architect, mxLoop=5000, tolerance=0.0):
    iParams = {
        'BCs'      : ['extend', 'periodic'],
        'undef'    : np.nan,
        'mxLoop'   : mxLoop,
        'tolerance': tolerance,  # fixed iterations: CPU & GPU do identical work
        'printInfo': False,
        'debug'    : False,
        'architect': architect,
    }
    return invert_Poisson(curl, dims=['lat', 'lon'], coords='lat-lon',
                          iParams=iParams)


def time_solve(curl, architect, mxLoop=5000, tolerance=0.0,
               repeat=3, warmup=1):
    # warmup (JIT compile / GPU context init)
    for _ in range(warmup):
        solve(curl, architect, mxLoop=mxLoop, tolerance=tolerance)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        sf = solve(curl, architect, mxLoop=mxLoop, tolerance=tolerance)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times)), sf


def main():
    gpu_ok = _gpu_available()

    print('=' * 72)
    print('Demo 2: Large-scale Poisson solve on GPU')
    print('  Problem: Poisson equation  (wind-stress curl -> streamfunction)')
    print('  Data:    SODA_curl.nc  (real wind-stress curl, upsampled)')
    print('=' * 72)
    print(f'  GPU: {"available" if gpu_ok else "NOT available (CPU-only)"}')
    if gpu_ok:
        print(f'  GPU model: {_gpu_name()}')
    print(f'  CPU cores: {os.cpu_count()}')
    print()

    curl_base = load_curl()
    nlat0, nlon0 = curl_base.shape
    print(f'  base grid: {nlat0} x {nlon0}  '
          f'({nlat0 * nlon0:,} points)')

    # grid sizes: base, x2, x4
    factors = [1, 2, 4]
    grids = []
    for f in factors:
        if f == 1:
            grids.append((curl_base, f))
        else:
            grids.append((upsample(curl_base, f), f))

    print(f'  tested grids: ' +
          ', '.join(f'{g.shape[0]}x{g.shape[1]}' for g, _ in grids))
    print()

    print(f'  {"grid":>14}  {"arch":>6}  {"time(s)":>10}  '
          f'{"speedup":>10}  {"max|psi|":>12}')
    print(f'  {"-" * 14}  {"-" * 6}  {"-" * 10}  {"-" * 10}  {"-" * 12}')

    for curl, factor in grids:
        nlat, nlon = curl.shape
        label = f'{nlat}x{nlon}'

        # warmup on this grid
        time_solve(curl, 'cpu', mxLoop=50, tolerance=0.0, repeat=1, warmup=0)

        t_cpu, sf_cpu = time_solve(curl, 'cpu', repeat=2, warmup=0)
        maxpsi_cpu = float(np.nanmax(np.abs(sf_cpu.values)))

        if gpu_ok:
            time_solve(curl, 'gpu', mxLoop=50, tolerance=0.0, repeat=1, warmup=0)
            t_gpu, sf_gpu = time_solve(curl, 'gpu', repeat=2, warmup=0)
            speedup = t_cpu / t_gpu if t_gpu > 0 else float('inf')
            maxdiff = float(np.nanmax(
                np.abs(sf_gpu.values - sf_cpu.values)))
            print(f'  {label:>14}  {"cpu":>6}  {t_cpu:>10.3f}  '
                  f'{"":>10}  {maxpsi_cpu:>12.4f}')
            print(f'  {label:>14}  {"gpu":>6}  {t_gpu:>10.3f}  '
                  f'{f"{speedup:.1f}x":>10}  {maxpsi_cpu:>12.4f}'
                  f'  (diff {maxdiff:.1e})')
        else:
            print(f'  {label:>14}  {"cpu":>6}  {t_cpu:>10.3f}  '
                  f'{"-":>10}  {maxpsi_cpu:>12.4f}')
        print()

    print('=' * 72)
    print('Summary')
    print('=' * 72)
    if gpu_ok:
        print('  GPU parallelises the SOR update *within* a single grid')
        print('  (fine-grained, per-point parallelism via Red-Black SOR).')
        print('  Larger grids => more GPU cores utilised => bigger speedup.')
        print()
        print('  Combine with the dask demo for *both* levels of parallelism:')
        print('    dask across time steps (coarse) + GPU within each step (fine).')
    else:
        print('  GPU not available.  Install numba-cuda on a CUDA-capable')
        print('  machine to see GPU speedup numbers.')


if __name__ == '__main__':
    main()
