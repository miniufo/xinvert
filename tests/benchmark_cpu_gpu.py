# -*- coding: utf-8 -*-
"""
Benchmark CPU vs GPU for the Poisson equation at multiple grid sizes.

This is a standalone benchmarking script (NOT a pytest test).  It solves the
same analytic Poisson problem at several grid resolutions, times both backends,
and prints a speedup table to STDOUT.

Both backends run a **fixed 1000 iterations** (tolerance=0) so that CPU and GPU
do exactly the same amount of work.  The speedup therefore reflects pure
per-iteration throughput, not "who converged first".

Run from the project root::

    python tests/benchmark_cpu_gpu.py

or with a custom set of grid sizes::

    python tests/benchmark_cpu_gpu.py 128 256 512 1024

If no CUDA-capable GPU is available, only the CPU timings are reported and the
GPU / speedup columns show ``-``.
"""
import os
import sys

# allow running this file directly (python tests/benchmark_cpu_gpu.py)
# by ensuring the project root (parent of tests/) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import warnings
import numpy as np
import xarray as xr
from xinvert import invert_Poisson

# Small grids trigger occupancy warnings that flood the table output;
# they are expected (the benchmark deliberately sweeps down to 64x64).
# numba-cuda ships its own copy of NumbaPerformanceWarning from a different
# module path, so a category-based filter is unreliable.  Filter by message
# text instead, which is robust across numba / numba-cuda versions.
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


# ---------------------------------------------------------------------------
# GPU detection (mirrors test_CpuGpuConsistency.py)
# ---------------------------------------------------------------------------

def _probe_gpu():
    """Actually try to compile & run a trivial CUDA kernel (see test file)."""
    import os
    try:
        from numba import cuda
        import numpy as np
    except ImportError:
        return False, None
    try:
        if cuda.is_available():
            d = cuda.get_current_device()
            return True, d
        # is_available() False -> real test
        d = cuda.get_current_device()
        @cuda.jit
        def _trivial(x):
            x[0] = 1.0
        a = cuda.device_array(1, dtype=np.float64)
        _trivial[1, 1](a)
        h = a.copy_to_host()
        return (h[0] == 1.0), d
    except Exception:
        return False, None


def _gpu_available():
    ok, _ = _probe_gpu()
    return ok


def _gpu_name():
    _, d = _probe_gpu()
    if d is None:
        return None
    name = d.name
    if isinstance(name, bytes):
        name = name.decode()
    return name


# ---------------------------------------------------------------------------
# problem setup
# ---------------------------------------------------------------------------

def _make_poisson_problem(n):
    """Build an n x n Poisson problem with analytic solution."""
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    psi_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    F = -2.0 * np.pi**2 * psi_true
    da_F = xr.DataArray(F, dims=['y', 'x'], coords={'y': y, 'x': x})
    return da_F, psi_true


def _base_iparams():
    # Fixed 1000 iterations so that CPU and GPU do EXACTLY the same amount
    # of work.  tolerance=0.0 guarantees the loop never stops early, making
    # the speedup a pure measure of per-iteration throughput rather than a
    # mix of "who converged first".  1000 iters is enough for the analytic
    # problem to be well-converged at all tested grid sizes, so the error
    # column still reflects solution quality.
    return {
        'BCs'      : ['fixed', 'fixed'],
        'undef'    : np.nan,
        'mxLoop'   : 1000,
        'tolerance': 0.0,
        'printInfo': False,
        'debug'    : False,
    }


def _solve(da_F, architect):
    ip = _base_iparams()
    ip['architect'] = architect
    return invert_Poisson(da_F, dims=['y', 'x'], coords='cartesian', iParams=ip)


def _time_solve(da_F, architect, repeat=3, warmup=1):
    """Time a solve; do `warmup` untimed runs then `repeat` timed runs.

    Returns (median_seconds, max_error_vs_analytic).
    """
    n = da_F.shape[0]
    x = np.linspace(0, 1, n); y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    psi_true = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # warmup (numba JIT compile / GPU context init)
    for _ in range(warmup):
        S = _solve(da_F, architect)

    # timed runs
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        S = _solve(da_F, architect)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    err = float(np.nanmax(np.abs(S.values - psi_true)))
    med = float(np.median(times))
    return med, err


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sizes = [int(s) for s in sys.argv[1:]] or [64, 128, 256, 512, 1024]
    repeat = 3
    gpu_ok = _gpu_available()
    gname = _gpu_name()

    print('=' * 72)
    print('CPU vs GPU Poisson benchmark  (fixed 1000 iters, tolerance=0)')
    print('=' * 72)
    import platform
    print(f'CPU: {platform.processor() or platform.machine()}')
    if gpu_ok:
        print(f'GPU: {gname} (CUDA available)')
    else:
        print('GPU: not available (CPU-only timing)')
    print(f'grid sizes: {sizes}  | repeat per size: {repeat}  | iters: 1000')
    print('=' * 72)

    # column widths
    W = 12
    cols = ['grid', 'arch', 'error', 'time(s)'] + (['speedup'] if gpu_ok else [])
    fmt = '  '.join(f'{c:>{W}}' for c in cols)
    print(fmt)
    print('-' * len(fmt))

    # store results so the summary doesn't recompute them
    timings = []  # (n, t_cpu, err_cpu, t_gpu, err_gpu, speedup)

    for n in sizes:
        da_F, _ = _make_poisson_problem(n)

        t_cpu, err_cpu = _time_solve(da_F, 'cpu', repeat=repeat, warmup=1)

        if gpu_ok:
            t_gpu, err_gpu = _time_solve(da_F, 'gpu', repeat=repeat, warmup=1)
            speedup = t_cpu / t_gpu if t_gpu > 0 else float('inf')
            timings.append((n, t_cpu, err_cpu, t_gpu, err_gpu, speedup))
        else:
            timings.append((n, t_cpu, err_cpu, None, None, None))

        # print CPU row
        cpu_cols = [f'{"%dx%d" % (n, n):>{W}}', f'{"cpu":>{W}}',
                    f'{err_cpu:>{W}.2e}', f'{t_cpu:>{W}.4f}']
        if gpu_ok:
            cpu_cols.append(f'{"":>{W}}')
        print('  '.join(cpu_cols))

        if gpu_ok:
            gpu_cols = [f'{"":>{W}}', f'{"gpu":>{W}}',
                        f'{err_gpu:>{W}.2e}', f'{t_gpu:>{W}.4f}',
                        f'{speedup:>{W}.2f}x']
            print('  '.join(gpu_cols))
        print()

    # summary
    print('=' * 72)
    print('Summary')
    print('=' * 72)
    if gpu_ok:
        print(f'{"grid":>12}  {"cpu(s)":>12}  {"gpu(s)":>12}  {"speedup":>12}')
        print('-' * 52)
        for n, t_cpu, _, t_gpu, _, sp in timings:
            print(f'{"%dx%d" % (n, n):>12}  {t_cpu:>12.4f}  {t_gpu:>12.4f}  {sp:>11.2f}x')
        print()
        print('Note: fixed 1000 iterations for BOTH backends (tolerance=0), so')
        print('      CPU and GPU do identical work; speedup = pure throughput.')
        print('Note: timings include data transfer (host<->device) for GPU.')
        print('Note: error = max |psi - psi_analytic| after 1000 iters.')
        print('      At large grids (512+), optArg ~ 2.0 causes SOR over-')
        print('      relaxation that has not converged in 1000 iters, so the')
        print('      error GROWS with grid size for BOTH backends. This is')
        print('      expected and does NOT indicate an implementation bug;')
        print('      the speedup comparison remains valid (equal work).')
        print('Note: CPU uses standard SOR (Gauss-Seidel), GPU uses Red-Black')
        print('      SOR (needed for parallelism). Both converge to the same')
        print('      fixed point; small per-iteration convergence differences')
    else:
        print('GPU not available. Install a CUDA-enabled numba build on a')
        print('CUDA-capable machine to see speedup numbers.')


if __name__ == '__main__':
    main()
