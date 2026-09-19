# -*- coding: utf-8 -*-
"""CPU (lexicographic SOR) vs GPU (red-black SOR) convergence comparison.

True error vs the analytic solution (psi = sin(pi x) sin(pi y)) at fixed
iteration checkpoints, swept over grid sizes and iteration counts.

Defaults cover grids 64..4096 and iterations 100..25600 (doubling).
Runtime guards (the full sweep takes tens of minutes):

  - plateau break: a (grid, arch) series stops early once the true error
    stops improving (<1% relative change between checkpoints) -- converged
    to the discretisation floor.
  - --cpu-budget SECONDS: skip CPU checkpoint runs whose estimated wall
    time (extrapolated from the previous checkpoint) exceeds the budget.
    Use a large value (e.g. 1e9) for a complete overnight sweep.

Results: tests/results/convergence.json
Run:  python tests/benchmark_convergence.py [--grids 64 ... 4096]
          [--iters 100 ... 25600] [--cpu-budget 120]
"""
import argparse
import io
import json
import os
import re
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', message='.*under-utilization.*')
warnings.filterwarnings('ignore', message='.*low occupancy.*')

import numpy as np
import xarray as xr
import contextlib
from xinvert import invert_Poisson

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def solve(arch, n, mxLoop, tol, optArg):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    psi_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    F = xr.DataArray(-2.0 * np.pi**2 * psi_true, dims=['y', 'x'],
                     coords={'y': y, 'x': x})
    ip = {'BCs': ['fixed', 'fixed'], 'undef': np.nan, 'mxLoop': mxLoop,
          'tolerance': tol, 'optArg': optArg, 'printInfo': True,
          'architect': arch}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = invert_Poisson(F, dims=['y', 'x'], coords='cartesian', iParams=ip)
    m = re.search(r'loops\s+(\d+)', buf.getvalue())
    loop = int(m.group(1)) if m else -1
    err = float(np.nanmax(np.abs(r.values - psi_true)))
    return loop, err


def run_series(arch, n, iters, optarg, budget):
    """True error/loops at each checkpoint; None where skipped.

    Early-exits on an error plateau (converged to the discretisation
    floor); CPU runs estimated above *budget* seconds are skipped.
    """
    errs, lps = [], []
    prev = None
    t_prev = None
    k_prev = None
    for k in iters:
        if arch == 'cpu' and t_prev is not None and budget is not None:
            if t_prev * k / k_prev > budget:
                errs.append(None)
                lps.append(None)
                continue
        t0 = time.perf_counter()
        loop, err = solve(arch, n, k, 0.0, optarg)
        t_prev = time.perf_counter() - t0
        k_prev = k
        errs.append(err)
        lps.append(loop)
        if prev is not None and abs(err - prev) / max(prev, 1e-300) < 0.01:
            break  # plateau: reached the discretisation floor
        prev = err
    return errs, lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', nargs='+', type=int,
                    default=[64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument('--iters', nargs='+', type=int,
                    default=[100, 200, 400, 800, 1600, 3200, 6400,
                             12800, 25600])
    ap.add_argument('--cpu-budget', type=float, default=120.0,
                    help='skip CPU runs estimated above this wall time in '
                         'seconds (use 1e9 for a complete sweep)')
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    solve('gpu', args.grids[-1], args.iters[0], 0.0, None)  # JIT warmup

    errors = {'cpu': {}, 'gpu': {}}
    loops = {'cpu': {}, 'gpu': {}}
    for n in args.grids:
        for arch in ('gpu', 'cpu'):
            errs, lps = run_series(arch, n, args.iters, None,
                                   args.cpu_budget)
            errors[arch][str(n)] = errs
            loops[arch][str(n)] = lps
            row = '  '.join('skip' if e is None else f'{e:.2e}'
                            for e in errs)
            print(f'{arch} {n}x{n}: {row}')

    with open(os.path.join(RESULTS_DIR, 'convergence.json'), 'w') as f:
        json.dump({'grids': args.grids, 'iters': args.iters,
                   'errors': errors, 'loops': loops}, f, indent=1)
    print('saved tests/results/convergence.json')


if __name__ == '__main__':
    main()
