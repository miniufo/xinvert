# -*- coding: utf-8 -*-
"""
Test CPU/GPU consistency for the Poisson equation.

Solves a synthetic Poisson problem with a known analytic solution and
verifies that the CPU (numba) and GPU (cuda) backends produce consistent
results.  The GPU tests are skipped automatically when CUDA is not
available.

Run from the project root::

    pytest tests/test_CpuGpuConsistency.py -v

or run directly::

    python tests/test_CpuGpuConsistency.py
"""
import os
import sys

# allow running this file directly (python tests/test_CpuGpuConsistency.py)
# by ensuring the project root (parent of tests/) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import pytest
import warnings
from xinvert import invert_Poisson

# The 51x41 test grid underutilizes the GPU; suppress the expected warning
# so the consistency report stays readable.  numba-cuda ships its own copy
# of NumbaPerformanceWarning, so catch both.
for _mod in ('numba.core.errors', 'numba_cuda.errors', 'numba_cuda.numba.core.errors'):
    try:
        _err = __import__(_mod, fromlist=['NumbaPerformanceWarning'])
        warnings.filterwarnings('ignore', category=_err.NumbaPerformanceWarning)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _probe_gpu():
    """Actually try to compile & run a trivial CUDA kernel.

    ``cuda.is_available()`` is too conservative in recent numba — it can
    return False even when a GPU is present and usable (e.g. deprecated
    built-in CUDA target, missing CUDA_HOME, etc.).  This function does the
    real test: JIT-compile a one-line kernel and run it.

    Returns
    -------
    (ok, detail) : (bool, str)
        ok is True iff a kernel actually ran on the GPU.
        detail is a human-readable status string.
    """
    import os
    try:
        from numba import cuda
        import numpy as np
    except ImportError:
        return False, 'no   (numba not installed)'

    # --- fast path: is_available() is True ---
    try:
        if cuda.is_available():
            d = cuda.get_current_device()
            name = d.name.decode() if isinstance(d.name, bytes) else d.name
            return True, (f'yes  (device {d.id}: {name}, '
                          f'compute capability {d.compute_capability})')
    except Exception as e:
        is_avail_err = f'{type(e).__name__}: {e}'
    else:
        is_avail_err = 'cuda.is_available() returned False'

    # --- is_available() said False: find out WHY ---
    detail_parts = [f'no   ({is_avail_err})']
    detail_parts.append(f'numba={__import__("numba").__version__}')
    detail_parts.append(f'CUDA_HOME={os.environ.get("CUDA_HOME", "(not set)")}')

    # can the driver at least enumerate a device?
    try:
        d = cuda.get_current_device()
        name = d.name.decode() if isinstance(d.name, bytes) else d.name
        detail_parts.append(f'driver sees: {name} (CC {d.compute_capability})')
    except Exception as e:
        detail_parts.append(f'driver: {type(e).__name__}: {e}')
        return False, '  |  '.join(detail_parts)

    # --- real test: actually compile & run a trivial kernel ---
    try:
        @cuda.jit
        def _trivial(x):
            x[0] = 1.0
        a = cuda.device_array(1, dtype=np.float64)
        _trivial[1, 1](a)
        h = a.copy_to_host()
        if h[0] == 1.0:
            return True, (f'yes  (kernel ran, {name} CC {d.compute_capability}; '
                          f'note: is_available() was False but kernels work)')
        else:
            detail_parts.append('kernel ran but returned wrong value')
    except Exception as e:
        detail_parts.append(f'kernel compile/run FAILED: {type(e).__name__}: {e}')
        detail_parts.append('=> likely missing CUDA toolkit (nvcc/NVVM) or need: pip install numba-cuda')

    return False, '  |  '.join(detail_parts)


# module-level cache so the probe only runs once
_gpu_probe_cache = None


def _gpu_available():
    """Return True when a CUDA kernel can actually be compiled & run."""
    global _gpu_probe_cache
    if _gpu_probe_cache is None:
        _gpu_probe_cache = _probe_gpu()
    return _gpu_probe_cache[0]


def _gpu_info():
    """Return a human-readable GPU status string for diagnostics."""
    global _gpu_probe_cache
    if _gpu_probe_cache is None:
        _gpu_probe_cache = _probe_gpu()
    return _gpu_probe_cache[1]


def _make_poisson_problem(nx=51, ny=41):
    r"""Build a 2-D Poisson problem with known analytic solution.

    Solution:  :math:`\psi = \sin(\pi x)\sin(\pi y)`  on :math:`[0,1]^2`

    Forcing:    :math:`F = \nabla^2 \psi = -2\pi^2 \sin(\pi x)\sin(\pi y)`

    Boundary:  :math:`\psi = 0` on all sides (Dirichlet / ``'fixed'`` BC).

    Returns
    -------
    da_F : xarray.DataArray
        Forcing field with dims ``['y', 'x']`` and cartesian coords.
    psi_true : numpy.ndarray
        Analytic solution, shape ``(ny, nx)``.
    """
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    psi_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    F = -2.0 * np.pi**2 * psi_true
    da_F = xr.DataArray(F, dims=['y', 'x'], coords={'y': y, 'x': x})
    return da_F, psi_true


_BASE_IPARAMS = {
    'BCs'      : ['fixed', 'fixed'],
    'undef'    : np.nan,
    'mxLoop'   : 20000,
    'tolerance': 1e-10,
    'printInfo': False,
    'debug'    : False,
}


def _solve(da_F, dims, architect):
    """Solve the Poisson equation with the given architect."""
    ip = dict(_BASE_IPARAMS)
    ip['architect'] = architect
    return invert_Poisson(da_F, dims=dims, coords='cartesian', iParams=ip)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestPoissonCpuGpuConsistency:
    """Verify CPU and GPU Poisson solvers agree."""

    def test_cpu_vs_analytic(self):
        """CPU solver should converge close to the analytic solution."""
        da_F, psi_true = _make_poisson_problem()
        S = _solve(da_F, ['y', 'x'], 'cpu')
        err = float(np.nanmax(np.abs(S.values - psi_true)))
        # 2nd-order discretization error on 51x41 grid is ~1e-3
        assert err < 1e-2, f'CPU error vs analytic too large: {err:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_gpu_vs_analytic(self):
        """GPU solver should converge close to the analytic solution."""
        da_F, psi_true = _make_poisson_problem()
        S = _solve(da_F, ['y', 'x'], 'gpu')
        err = float(np.nanmax(np.abs(S.values - psi_true)))
        assert err < 1e-2, f'GPU error vs analytic too large: {err:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu(self):
        """CPU and GPU results must be consistent."""
        da_F, _ = _make_poisson_problem()
        S_cpu = _solve(da_F, ['y', 'x'], 'cpu')
        S_gpu = _solve(da_F, ['y', 'x'], 'gpu')
        diff = float(np.nanmax(np.abs(S_cpu.values - S_gpu.values)))
        # Red-Black SOR and standard SOR converge to the same fixed point;
        # differences arise from different convergence paths / float roundoff.
        assert diff < 1e-3, f'CPU-GPU diff too large: {diff:.4e}'

    def test_invalid_architect_raises(self):
        """Unknown architect should raise ValueError."""
        da_F, _ = _make_poisson_problem()
        with pytest.raises(ValueError, match='unsupported architect'):
            _solve(da_F, ['y', 'x'], 'tpu')


# ---------------------------------------------------------------------------
# allow running this file directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== CPU/GPU consistency tests ===')
    gpu_ok = _gpu_available()
    print(f'GPU available: {gpu_ok}')
    print(f'GPU status   : {_gpu_info()}')
    print()
    # NOTE: if GPU is unavailable, the GPU-related assertions below are SKIPPED,
    # not passed.  Only the CPU path is actually exercised.
    print('Tests that run:')
    print('  - test_cpu_vs_analytic       (always)')
    print('  - test_gpu_vs_analytic       ' + ('(RUN)' if gpu_ok else '(SKIPPED, no CUDA)'))
    print('  - test_cpu_vs_gpu            ' + ('(RUN)' if gpu_ok else '(SKIPPED, no CUDA)'))
    print('  - test_invalid_architect     (always)')
    print()

    da_F, psi_true = _make_poisson_problem()

    # CPU
    S_cpu = _solve(da_F, ['y', 'x'], 'cpu')
    err_cpu = float(np.nanmax(np.abs(S_cpu.values - psi_true)))
    print(f'CPU  max error vs analytic: {err_cpu:.6e}')

    if gpu_ok:
        S_gpu = _solve(da_F, ['y', 'x'], 'gpu')
        err_gpu = float(np.nanmax(np.abs(S_gpu.values - psi_true)))
        diff = float(np.nanmax(np.abs(S_cpu.values - S_gpu.values)))
        print(f'GPU  max error vs analytic: {err_gpu:.6e}')
        print(f'CPU-GPU max diff:          {diff:.6e}')

        ok = err_cpu < 1e-2 and err_gpu < 1e-2 and diff < 1e-3
        print('RESULT:', 'PASS' if ok else 'FAIL')
        print()
        print('See tests/benchmark_cpu_gpu.py for multi-scale speedup numbers.')
    else:
        ok = err_cpu < 1e-2
        print('GPU path SKIPPED (no CUDA). Only CPU correctness was verified.')
        print('RESULT:', 'PASS (CPU-only)' if ok else 'FAIL')
        print()
        print('To actually test the GPU, run on a machine with a CUDA-capable GPU')
        print('and a CUDA-enabled numba install, then:')
        print('    python tests/test_CpuGpuConsistency.py')
        print('    python tests/benchmark_cpu_gpu.py')
