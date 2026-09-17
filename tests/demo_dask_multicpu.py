# -*- coding: utf-8 -*-
"""
Demo 1: Coarse-grained parallelism with dask on multi-core CPU.

Solves the Poisson equation (wind-stress curl -> streamfunction) for
multiple time steps in parallel.  Each time step is an independent
SOR iteration, so dask chunks along the time dimension and dispatches
one chunk per worker thread — classic coarse-grained (task-level)
parallelism.

Data: Data/SODA_curl.nc  (12 time steps, 300 x 720, var = curl)

Run from project root::

    python tests/demo_dask_multicpu.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import dask
from dask.distributed import Client

from xinvert import invert_Poisson

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data',
                    'SODA_curl.nc')


def load_curl():
    """Load wind-stress curl and return as xarray DataArray."""
    ds = xr.open_dataset(DATA)
    curl = ds.curl
    return curl


def solve_serial(curl, mxLoop=4000):
    """Solve all time steps serially (no dask, single thread)."""
    iParams = {
        'BCs'      : ['extend', 'periodic'],
        'undef'    : np.nan,
        'mxLoop'   : mxLoop,
        'tolerance': 1e-12,
        'printInfo': False,
        'debug'    : False,
    }
    sf = invert_Poisson(curl, dims=['lat', 'lon'], coords='lat-lon',
                        iParams=iParams)
    return sf.compute()


def solve_dask(curl, mxLoop=4000, n_workers=1, threads_per_worker=None):
    """Solve all time steps in parallel via dask (chunked along time).

    Parameters
    ----------
    n_workers, threads_per_worker : dask cluster config.
        Coarse-grained parallelism: one dask task = one time-step's SOR solve.
    """
    if threads_per_worker is None:
        threads_per_worker = os.cpu_count() or 4

    client = Client(n_workers=n_workers,
                    threads_per_worker=threads_per_worker)

    try:
        curl_chunked = curl.chunk({'time': 1})

        iParams = {
            'BCs'      : ['extend', 'periodic'],
            'undef'    : np.nan,
            'mxLoop'   : mxLoop,
            'tolerance': 1e-12,
            'printInfo': False,
            'debug'    : False,
        }
        sf = invert_Poisson(curl_chunked, dims=['lat', 'lon'],
                            coords='lat-lon', iParams=iParams)

        t0 = time.perf_counter()
        sf = sf.compute()
        elapsed = time.perf_counter() - t0
        return sf, elapsed, client
    finally:
        try:
            client.close()
        except RuntimeError:
            pass


def main():
    print('=' * 72)
    print('Demo 1: Coarse-grained dask parallelism (multi-core CPU)')
    print('  Problem: Poisson equation  (wind-stress curl -> streamfunction)')
    print('  Data:    SODA_curl.nc')
    print('=' * 72)

    curl = load_curl()
    ntime, nlat, nlon = curl.shape
    print(f'\n  grid:  {nlat} x {nlon}  ({nlat * nlon:,} points per time step)')
    print(f'  time steps:  {ntime}')
    print(f'  total grid points: {ntime * nlat * nlon:,}')
    print(f'  CPU cores detected: {os.cpu_count()}')

    # ---- warmup (JIT compile) ----
    print('\n  [warmup] compiling JIT kernels on a single time step...')
    _ = solve_serial(curl.isel(time=0), mxLoop=10)

    # ---- serial baseline ----
    print('\n  [serial] solving all time steps on a single thread...')
    t0 = time.perf_counter()
    sf_serial = solve_serial(curl)
    t_serial = time.perf_counter() - t0
    print(f'  serial time:  {t_serial:.2f} s')

    # ---- dask parallel ----
    ncores = os.cpu_count() or 4
    configs = [
        (1, 1),       # 1 thread  (sanity check, ~ serial)
        (1, ncores),  # ncores threads in one worker
    ]

    print(f'\n  [dask] solving with dask distributed (chunk time=1)...')
    results = []
    for nw, tw in configs:
        label = (f'{nw} worker x {tw} threads'
                 if nw > 1 else f'{tw} thread{"s" if tw > 1 else ""}')
        print(f'\n  -> {label} ...', end=' ', flush=True)
        sf_dask, t_dask, _ = solve_dask(curl, n_workers=nw,
                                        threads_per_worker=tw)
        speedup = t_serial / t_dask if t_dask > 0 else float('inf')
        print(f'{t_dask:.2f} s   (speedup {speedup:.2f}x)')

        # verify same result
        maxdiff = float(np.nanmax(np.abs(sf_dask.values - sf_serial.values)))
        print(f'     max diff vs serial: {maxdiff:.2e}')
        results.append((label, t_dask, speedup))

    # ---- summary ----
    print('\n' + '=' * 72)
    print('Summary')
    print('=' * 72)
    print(f'  {"config":<28}  {"time(s)":>10}  {"speedup":>10}')
    print(f'  {"-" * 28}  {"-" * 10}  {"-" * 10}')
    print(f'  {"serial (1 thread)":<28}  {t_serial:>10.2f}  {"1.00x":>10}')
    for label, t, sp in results:
        print(f'  {label:<28}  {t:>10.2f}  {f"{sp:.2f}x":>10}')
    print()
    print('  Note: coarse-grained parallelism = one independent Poisson')
    print('        solve per dask task.  Speedup is bounded by the number')
    print('        of time steps and CPU cores.')
    print('  Note: each time step is a full SOR iteration on a 300x720 grid;')
    print(f'        with {ntime} time steps and {ncores} cores, ideal speedup')
    print(f'        ~ min({ntime}, {ncores}) = {min(ntime, ncores)}x.')


if __name__ == '__main__':
    main()
