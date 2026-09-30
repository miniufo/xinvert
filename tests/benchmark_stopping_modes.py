# -*- coding: utf-8 -*-
"""Benchmark legacy norm-change and residual stopping modes.

This benchmark measures an end-to-end converged solve, unlike
``benchmark_cpu_gpu.py`` which deliberately performs a fixed number of
iterations.  It records wall time, iteration count, reported stopping error,
and true error against a manufactured Poisson solution.

Run from the project root::

    python tests/benchmark_stopping_modes.py
    python tests/benchmark_stopping_modes.py --grids 128 256 512

Results are written to ``tests/results/stopping_modes.json``.  Run only on an
otherwise idle machine: competing CPU or GPU jobs invalidate timing results.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr

from xinvert import invert_Poisson


RESULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results',
    'stopping_modes.json')


def make_problem(n, dtype):
    x = np.linspace(0.0, 1.0, n, dtype=dtype)
    y = np.linspace(0.0, 1.0, n, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    forcing = xr.DataArray(
        -2.0 * np.pi ** 2 * exact,
        dims=['y', 'x'], coords={'y': y, 'x': x})
    return forcing, exact


def solve(forcing, architect, convergence, dtype, tolerance, max_iterations):
    return invert_Poisson(
        forcing, dims=['y', 'x'], coords='cartesian',
        iParams={
            'BCs': ['fixed', 'fixed'],
            'architect': architect,
            'convergence': convergence,
            'dtype': dtype,
            'mxLoop': max_iterations,
            'tolerance': tolerance,
            'printInfo': False,
            'return_diagnostics': True,
        })


def benchmark_case(forcing, exact, architect, convergence, dtype, tolerance,
                   max_iterations, repeat):
    # Compile kernels and initialise the CUDA context outside timed samples.
    solve(forcing, architect, convergence, dtype, tolerance, max_iterations)

    samples = []
    result = diagnostics = None
    for _ in range(repeat):
        start = time.perf_counter()
        result, diagnostics = solve(
            forcing, architect, convergence, dtype, tolerance,
            max_iterations)
        samples.append(time.perf_counter() - start)

    return {
        'seconds_median': statistics.median(samples),
        'seconds_samples': samples,
        'iterations': int(diagnostics.iterations.item()),
        'reported_error': float(diagnostics.error.item()),
        'true_max_error': float(np.nanmax(np.abs(result.values - exact))),
        'converged': bool(diagnostics.converged.item()),
        'stop_reason': str(diagnostics.stop_reason.item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grids', nargs='+', type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument('--architectures', nargs='+', choices=['cpu', 'gpu'],
                        default=['cpu', 'gpu'])
    parser.add_argument('--dtype', choices=['float32', 'float64'],
                        default='float64')
    parser.add_argument('--tolerance', type=float, default=1e-8)
    parser.add_argument('--max-iterations', type=int, default=20000)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', default=RESULT_PATH,
                        help='JSON output path')
    args = parser.parse_args()

    records = []
    for n in args.grids:
        forcing, exact = make_problem(n, args.dtype)
        for architect in args.architectures:
            for convergence in ('norm', 'residual'):
                record = benchmark_case(
                    forcing, exact, architect, convergence, args.dtype,
                    args.tolerance, args.max_iterations, args.repeat)
                record.update({
                    'grid': n,
                    'architect': architect,
                    'convergence': convergence,
                })
                records.append(record)
                print(
                    f"{n:4d} {architect:3s} {convergence:8s} "
                    f"{record['seconds_median']:9.4f}s "
                    f"{record['iterations']:6d} loops "
                    f"true_err={record['true_max_error']:.3e} "
                    f"{record['stop_reason']}", flush=True)

    payload = {
        'metadata': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'dtype': args.dtype,
            'tolerance': args.tolerance,
            'max_iterations': args.max_iterations,
            'repeat': args.repeat,
        },
        'records': records,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    print(f'saved {output}')


if __name__ == '__main__':
    main()
