# -*- coding: utf-8 -*-
"""
Test CPU/GPU consistency for invert_standard_1D.

Two layers of verification:

1. Direct kernel-level tests with analytic solutions for all three
   boundary conditions:

   - fixed:    psi = sin(pi x),  A = 1, B = -pi^2,  F = -2 pi^2 sin(pi x)
   - extend:   psi = cos(pi x),  A = 1, B = -pi^2,  F = -2 pi^2 cos(pi x)
   - periodic: psi = cos(2 pi x), A = 1, B = -4 pi^2, F = -8 pi^2 cos(2 pi x)

   (equation solved: (A psi_x)/x + B psi = F)

2. Application-level CPU/GPU consistency through invert_GeoAdjustment
   (which dispatches to invert_standard_1D).  invert_RefStateSWM uses the
   same kernel, so it is covered by the kernel-level tests plus its own
   CPU regression test.

The GPU tests are skipped automatically when CUDA is not available.

Run from the project root::

    pytest tests/test_GpuStandard1D.py -v

or run directly::

    python tests/test_GpuStandard1D.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import pytest
import warnings
from xinvert import invert_GeoAdjustment

try:
    from tests.test_CpuGpuConsistency import _gpu_available, _gpu_info
except ImportError:  # running directly: tests/ is on sys.path
    from test_CpuGpuConsistency import _gpu_available, _gpu_info

for _mod in ('numba.core.errors', 'numba_cuda.errors'):
    try:
        _err = __import__(_mod, fromlist=['NumbaPerformanceWarning'])
        warnings.filterwarnings('ignore', category=_err.NumbaPerformanceWarning)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_NX = 100  # even so red-black coloring is valid across a periodic seam
_UNDEF = -9.99e8
_OPTARG = 1.85
_MXLOOP = 200000
_TOL = 1e-12


def _solve_kernel(A, B, F, BCx, S0, architect):
    """Call the CPU or GPU 1D kernel directly with the shared signature."""
    x = np.linspace(0.0, 1.0, _NX, endpoint=(BCx != 'periodic'))
    dx = x[1] - x[0]
    delxSqr = dx * dx
    flags = np.zeros(3)

    if architect == 'cpu':
        from xinvert.cpus import invert_standard_1D
        func = invert_standard_1D
    else:
        from xinvert.gpus import invert_standard_1D_gpu
        func = invert_standard_1D_gpu

    S = np.array(S0, dtype=np.float64)
    func(S, A, B, F, 'test', _NX, BCx, delxSqr,
         _OPTARG, _UNDEF, flags, _MXLOOP, _TOL)
    return S, flags


def _kernel_problem(psi_true, bcx):
    """Build A=1, B=-k^2 and F = psi'' + B*psi for the given psi.

    For 'extend' a PURE Poisson (B=0) is used with
    psi = cos(2 pi x) - cos(4 pi x)/4, which has psi' = 0 at BOTH ends,
    a zero mean, AND psi'' = 0 at the ends.  The last property matters:
    the extend (zero-gradient) closure drops the boundary points from
    the discrete system, so the discrete compatibility condition is a
    sum over INTERIOR points only -- forcing that vanishes at the
    boundary keeps that sum O(h^2) close to zero.  A strong Helmholtz
    term would additionally make extend + gauge-anchor ill-posed
    (exponential homogeneous solutions).
    """
    x = np.linspace(0.0, 1.0, _NX, endpoint=(bcx != 'periodic'))
    X = np.meshgrid(x)[0]
    A = np.ones(_NX)
    if bcx == 'extend':
        # psi = cos(2 pi x) - cos(4 pi x)/4  =>  psi'' as below
        B = np.zeros(_NX)
        F = (-4.0 * np.pi ** 2 * np.cos(2.0 * np.pi * X)
             + 4.0 * np.pi ** 2 * np.cos(4.0 * np.pi * X))
    else:
        k2 = (np.pi if bcx == 'fixed' else 2.0 * np.pi) ** 2
        B = -k2 * np.ones(_NX)
        F = -(k2 + k2) * psi_true          # psi'' = -k2*psi  =>  F = -2 k2 psi
    S0 = np.zeros(_NX)
    return A, B, F, X[0], S0


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestStandard1DCpuGpuConsistency:
    """Verify CPU and GPU solvers agree for the standard 1D form."""

    @pytest.mark.parametrize('bcx', ['fixed', 'extend', 'periodic'])
    def test_cpu_vs_analytic(self, bcx):
        """CPU kernel should converge close to the analytic solution."""
        if bcx == 'fixed':
            x = np.linspace(0, 1, _NX)
            psi = np.sin(np.pi * x)
        elif bcx == 'extend':
            # see _kernel_problem: psi' = 0 and psi'' = 0 at both ends
            x = np.linspace(0, 1, _NX)
            psi = np.cos(2.0 * np.pi * x) - np.cos(4.0 * np.pi * x) / 4.0
        else:
            x = np.linspace(0, 1, _NX, endpoint=False)
            psi = np.cos(2.0 * np.pi * x)
        A, B, F, _, S0 = _kernel_problem(psi, bcx)
        S, flags = _solve_kernel(A, B, F, bcx, S0, 'cpu')
        assert not flags[0], 'CPU overflow'
        if bcx == 'extend':
            # extend (Neumann at both ends) makes the system singular; the
            # gauge anchor pins the first interior point to its initial
            # guess (0), so the solution agrees with the analytic one up
            # to a constant offset
            S = S - S[1] + psi[1]
        err = float(np.max(np.abs(S - psi)))
        # 'extend' BC is a first-order (zero-gradient) approximation, so its
        # discretization error is O(dx) ~ 1.5e-2 rather than O(dx^2)
        tol = {'fixed': 1e-4, 'extend': 3e-2, 'periodic': 1e-3}[bcx]
        assert err < tol, f'CPU error vs analytic ({bcx}): {err:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    @pytest.mark.parametrize('bcx', ['fixed', 'extend', 'periodic'])
    def test_gpu_vs_analytic(self, bcx):
        """GPU kernel should converge close to the analytic solution."""
        if bcx == 'fixed':
            x = np.linspace(0, 1, _NX)
            psi = np.sin(np.pi * x)
        elif bcx == 'extend':
            # see _kernel_problem: psi' = 0 and psi'' = 0 at both ends
            x = np.linspace(0, 1, _NX)
            psi = np.cos(2.0 * np.pi * x) - np.cos(4.0 * np.pi * x) / 4.0
        else:
            x = np.linspace(0, 1, _NX, endpoint=False)
            psi = np.cos(2.0 * np.pi * x)
        A, B, F, _, S0 = _kernel_problem(psi, bcx)
        S, flags = _solve_kernel(A, B, F, bcx, S0, 'gpu')
        assert not flags[0], 'GPU overflow'
        if bcx == 'extend':
            # see the CPU test: solution agrees up to a constant offset
            S = S - S[1] + psi[1]
        err = float(np.max(np.abs(S - psi)))
        tol = {'fixed': 1e-4, 'extend': 3e-2, 'periodic': 1e-3}[bcx]
        assert err < tol, f'GPU error vs analytic ({bcx}): {err:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    @pytest.mark.parametrize('bcx', ['fixed', 'extend', 'periodic'])
    def test_cpu_vs_gpu(self, bcx):
        """CPU and GPU results must be consistent."""
        if bcx == 'fixed':
            x = np.linspace(0, 1, _NX)
            psi = np.sin(np.pi * x)
        elif bcx == 'extend':
            # see _kernel_problem: psi' = 0 and psi'' = 0 at both ends
            x = np.linspace(0, 1, _NX)
            psi = np.cos(2.0 * np.pi * x) - np.cos(4.0 * np.pi * x) / 4.0
        else:
            x = np.linspace(0, 1, _NX, endpoint=False)
            psi = np.cos(2.0 * np.pi * x)
        A, B, F, _, S0 = _kernel_problem(psi, bcx)
        S_cpu, _ = _solve_kernel(A, B, F, bcx, S0, 'cpu')
        S_gpu, _ = _solve_kernel(A, B, F, bcx, S0, 'gpu')
        diff = float(np.max(np.abs(S_cpu - S_gpu)))
        assert diff < 1e-5, f'CPU-GPU diff ({bcx}) too large: {diff:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu_geoadjustment(self):
        """Application-level consistency through invert_GeoAdjustment.

        NOTE: latitudes stay in the NH (10~80) because A = cosH/fH has a
        sign-changing singularity at the equator (fH -> 0), where the SOR
        iteration diverges for BOTH architects.
        """
        lat = np.linspace(10.0, 80.0, 57)
        PV0 = xr.DataArray(
            2e-4 * np.exp(-((lat - 40.0) / 15.0) ** 2),
            dims=['lat'], coords={'lat': lat})
        ip = {'BCs': ['extend'], 'undef': np.nan, 'mxLoop': 20000,
              'tolerance': 1e-11, 'printInfo': False}
        mp = {'g': 9.80665, 'Omega': 7.292e-5}

        ip_cpu = dict(ip, architect='cpu')
        ip_gpu = dict(ip, architect='gpu')
        r_cpu = invert_GeoAdjustment(PV0, dims=['lat'], coords='lat',
                                     mParams=mp, iParams=ip_cpu)
        r_gpu = invert_GeoAdjustment(PV0, dims=['lat'], coords='lat',
                                     mParams=mp, iParams=ip_gpu)
        scale = float(np.nanmax(np.abs(r_cpu.values)))
        diff = float(np.nanmax(np.abs(r_cpu.values - r_gpu.values)))
        rel = diff / max(scale, 1e-300)
        assert rel < 1e-4, \
            f'GeoAdjustment CPU-GPU relative diff too large: {rel:.4e}'


# ---------------------------------------------------------------------------
# allow running this file directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== CPU/GPU consistency tests: standard_1D ===')
    gpu_ok = _gpu_available()
    print(f'GPU available: {gpu_ok}')
    print(f'GPU status   : {_gpu_info()}')
    print()

    ok = True
    for bcx, x, psi in [
            ('fixed',    np.linspace(0, 1, _NX),                np.sin(np.pi * np.linspace(0, 1, _NX))),
            ('extend',   np.linspace(0, 1, _NX),                np.cos(2 * np.pi * np.linspace(0, 1, _NX)) - np.cos(4 * np.pi * np.linspace(0, 1, _NX)) / 4.0),
            ('periodic', np.linspace(0, 1, _NX, endpoint=False), np.cos(2 * np.pi * np.linspace(0, 1, _NX, endpoint=False)))]:
        A, B, F, _, S0 = _kernel_problem(psi, bcx)
        S_cpu, _ = _solve_kernel(A, B, F, bcx, S0, 'cpu')
        err_cpu = float(np.max(np.abs(S_cpu - psi)))
        line = f'[{bcx:8s}] CPU err: {err_cpu:.3e}'
        ok &= err_cpu < {'fixed': 1e-4, 'extend': 3e-2, 'periodic': 1e-3}[bcx]
        if gpu_ok:
            S_gpu, _ = _solve_kernel(A, B, F, bcx, S0, 'gpu')
            err_gpu = float(np.max(np.abs(S_gpu - psi)))
            diff = float(np.max(np.abs(S_cpu - S_gpu)))
            line += f' | GPU err: {err_gpu:.3e} | diff: {diff:.3e}'
            ok &= err_gpu < {'fixed': 1e-4, 'extend': 3e-2, 'periodic': 1e-3}[bcx] and diff < 1e-5
        print(line)

    if gpu_ok:
        lat = np.linspace(10.0, 80.0, 57)
        PV0 = xr.DataArray(2e-4 * np.exp(-((lat - 40.0) / 15.0) ** 2),
                           dims=['lat'], coords={'lat': lat})
        ip = {'BCs': ['extend'], 'undef': np.nan, 'mxLoop': 20000,
              'tolerance': 1e-11, 'printInfo': False}
        mp = {'g': 9.80665, 'Omega': 7.292e-5}
        r_cpu = invert_GeoAdjustment(PV0, dims=['lat'], coords='lat',
                                     mParams=mp, iParams=dict(ip, architect='cpu'))
        r_gpu = invert_GeoAdjustment(PV0, dims=['lat'], coords='lat',
                                     mParams=mp, iParams=dict(ip, architect='gpu'))
        diff = float(np.nanmax(np.abs(r_cpu.values - r_gpu.values)))
        rel = diff / max(float(np.nanmax(np.abs(r_cpu.values))), 1e-300)
        print(f'GeoAdjustment CPU-GPU relative diff: {rel:.3e}')
        ok &= rel < 1e-4

    print('RESULT:', 'PASS' if ok else 'FAIL')
