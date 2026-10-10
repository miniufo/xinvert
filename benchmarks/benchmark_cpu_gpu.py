"""Benchmark xinvert backends against the single-core CPU solver.

Two modes are intentionally kept separate: ``converged`` measures time to a
common residual tolerance, while ``fixed`` measures throughput for exactly the
same number of SOR iterations. Correctness is covered by pytest. Run on an
otherwise idle machine from the repository root::

    python benchmarks/benchmark_cpu_gpu.py
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

# Set these before importing NumPy, Numba, or xinvert so the CPU reference is
# unambiguously single-threaded.
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numba
import numpy as np
import xarray as xr

from xinvert import invert_Poisson

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')


def make_problem(n, dtype):
    """Return a smooth Cartesian Poisson forcing on an ``n x n`` grid."""
    dtype = np.dtype(dtype)
    x = np.linspace(0.0, 1.0, n, dtype=dtype)
    y = np.linspace(0.0, 1.0, n, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    psi = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    forcing = (-2.0 * np.pi**2 * psi).astype(dtype, copy=False)
    return xr.DataArray(
        forcing, dims=('y', 'x'), coords={'y': y, 'x': x})


def solve(forcing, architect, dtype, tolerance, max_iterations,
          diag_interval):
    """Run one solve and return its solution and residual diagnostics."""
    return invert_Poisson(
        forcing, dims=['y', 'x'], coords='cartesian',
        iParams={
            'BCs': ['fixed', 'fixed'],
            'architect': architect,
            'dtype': dtype,
            'mxLoop': max_iterations,
            'tolerance': tolerance,
            'diag_interval': diag_interval,
            'printInfo': False,
        })


def set_cpu_threads(architect, cpu_mc_threads):
    """Select one thread for CPU or the requested count for CPU multi-color."""
    if architect == 'cpu':
        numba.set_num_threads(1)
    elif architect == 'cpu-mc':
        numba.set_num_threads(cpu_mc_threads)


def warm_up(forcing, architect, dtype, cpu_mc_threads):
    """Compile kernels and initialize the backend outside timed samples."""
    set_cpu_threads(architect, cpu_mc_threads)
    solve(forcing, architect, dtype, 1e-30, 2, 1)


def benchmark_solve(forcing, architect, dtype, tolerance, max_iterations,
                    diag_interval, repeat, cpu_mc_threads):
    """Measure repeated end-to-end solves to a common residual tolerance."""
    warm_up(forcing, architect, dtype, cpu_mc_threads)
    samples = []
    diagnostics = None
    for _ in range(repeat):
        start = time.perf_counter()
        _, diagnostics = solve(
            forcing, architect, dtype, tolerance, max_iterations,
            diag_interval)
        samples.append(time.perf_counter() - start)
    return {
        'seconds_median': statistics.median(samples),
        'seconds_samples': samples,
        'iterations': int(diagnostics.iterations.item()),
        'final_residual': float(diagnostics.final_error.item()),
        'converged': bool(diagnostics.converged.item()),
    }


def repeat_count(grid, repeat, large_repeat, large_grid):
    """Return the sample count selected for *grid*."""
    return large_repeat if grid >= large_grid else repeat


def benchmark_trace(forcing, architect, dtype, checkpoints, repeat,
                    cpu_mc_threads):
    """Measure residual and cumulative time from identical initial states."""
    warm_up(forcing, architect, dtype, cpu_mc_threads)
    records = []
    for iterations in checkpoints:
        samples = []
        residual = None
        for _ in range(repeat):
            start = time.perf_counter()
            _, diagnostics = solve(
                forcing, architect, dtype, 1e-30, iterations, iterations)
            samples.append(time.perf_counter() - start)
            residual = float(diagnostics.final_error.item())
        records.append({
            'iterations': iterations,
            'seconds_median': statistics.median(samples),
            'seconds_samples': samples,
            'residual': residual,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['converged', 'fixed'],
                        default='converged')
    parser.add_argument('--grids', nargs='+', type=int,
                        default=[128, 256, 512, 1024])
    parser.add_argument(
        '--architectures', nargs='+',
        choices=['cpu', 'cpu-mc', 'gpu'],
        default=['cpu', 'cpu-mc', 'gpu'])
    parser.add_argument(
        '--cpu-mc-threads', type=int,
        default=min(8, numba.get_num_threads()),
        help='Numba threads used by cpu-mc (default: min(8, available))')
    parser.add_argument('--trace-grid', type=int, default=512)
    parser.add_argument('--checkpoints', nargs='+', type=int,
                        default=[1, 25, 50, 100, 200, 400, 800, 1600, 3200])
    parser.add_argument('--dtype', choices=['float32', 'float64'],
                        default='float64')
    parser.add_argument('--tolerance', type=float, default=1e-9,
                        help='target residual in converged mode (default: 1e-9)')
    parser.add_argument('--max-iterations', type=int, default=50000)
    parser.add_argument('--fixed-iterations', type=int, default=1000,
                        help='iterations per solve in fixed mode')
    parser.add_argument('--diag-interval', type=int, default=50)
    parser.add_argument('--repeat', type=int, default=5,
                        help='samples per grid below --large-grid')
    parser.add_argument('--large-repeat', type=int, default=2,
                        help='samples per grid at or above --large-grid')
    parser.add_argument('--large-grid', type=int, default=2048)
    parser.add_argument('--skip-trace', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()

    if args.mode == 'converged' and args.tolerance <= 0:
        parser.error('--tolerance must be positive')
    if args.diag_interval <= 0:
        parser.error('--diag-interval must be positive')
    if args.repeat < 1 or args.large_repeat < 1:
        parser.error('--repeat and --large-repeat must be at least 1')
    if args.fixed_iterations < 1:
        parser.error('--fixed-iterations must be at least 1')
    if not 1 <= args.cpu_mc_threads <= numba.config.NUMBA_NUM_THREADS:
        parser.error(
            '--cpu-mc-threads must be between 1 and the available Numba '
            f'thread count ({numba.config.NUMBA_NUM_THREADS})')
    if any(value < 1 for value in args.checkpoints):
        parser.error('--checkpoints must contain positive iteration counts')

    solves = []
    for n in args.grids:
        forcing = make_problem(n, args.dtype)
        by_arch = {}
        repeats = repeat_count(
            n, args.repeat, args.large_repeat, args.large_grid)
        if args.mode == 'fixed':
            tolerance = 0.0
            max_iterations = args.fixed_iterations
            diag_interval = args.fixed_iterations
        else:
            tolerance = args.tolerance
            max_iterations = args.max_iterations
            diag_interval = args.diag_interval
        for architect in args.architectures:
            record = benchmark_solve(
                forcing, architect, args.dtype, tolerance,
                max_iterations, diag_interval, repeats,
                args.cpu_mc_threads)
            record.update({
                'grid': n, 'architect': architect, 'repeat': repeats})
            solves.append(record)
            by_arch[architect] = record
        summary = [f"{n:4d}x{n:<4d}"]
        for architect in args.architectures:
            summary.append(
                f"{architect}={by_arch[architect]['seconds_median']:.4f}s")
        if 'cpu' in by_arch:
            for architect in args.architectures:
                if architect != 'cpu':
                    speedup = (by_arch['cpu']['seconds_median'] /
                               by_arch[architect]['seconds_median'])
                    summary.append(f"{architect}/cpu={speedup:.2f}x")
        print('  '.join(summary), flush=True)

    traces = {}
    if args.mode == 'converged' and not args.skip_trace:
        trace_forcing = make_problem(args.trace_grid, args.dtype)
        for architect in args.architectures:
            traces[architect] = benchmark_trace(
                trace_forcing, architect, args.dtype,
                sorted(set(args.checkpoints)), args.repeat,
                args.cpu_mc_threads)

    payload = {
        'metadata': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'cpu': platform.processor() or platform.machine(),
            'cpu_threads': 1,
            'cpu_mc_threads': args.cpu_mc_threads,
            'architectures': args.architectures,
            'mode': args.mode,
            'dtype': args.dtype,
            'tolerance': tolerance,
            'max_iterations': max_iterations,
            'diag_interval': diag_interval,
            'repeat_small': args.repeat,
            'repeat_large': args.large_repeat,
            'large_grid': args.large_grid,
            'trace_grid': args.trace_grid,
        },
        'solves': solves,
        'residual_traces': traces,
    }
    output_name = ('time_to_tolerance.json' if args.mode == 'converged'
                   else 'fixed_work.json')
    output = os.path.abspath(
        args.output or os.path.join(RESULTS, output_name))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    print(f'saved {output}')


if __name__ == '__main__':
    main()
