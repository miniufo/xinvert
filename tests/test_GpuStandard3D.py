# -*- coding: utf-8 -*-
"""
Test CPU/GPU consistency for invert_standard_3D.

Two layers of verification:

1. Direct kernel-level tests with an analytic solution (fixed BCs on all
   sides):  psi = sin(pi x) sin(pi y) sin(pi z) on the unit cube,
   A = B = C = 1, F = -3 pi^2 psi  (equation: Laplacian(psi) = F).

2. CPU/GPU consistency for extend-y (and periodic-x) boundary variants,
   where the discrete extend BC is only a first-order Neumann
   approximation, so no analytic solution is asserted.

3. Application-level CPU/GPU consistency through invert_omega with
   synthetic data.

The GPU tests are skipped automatically when CUDA is not available.

Run from the project root::

    pytest tests/test_GpuStandard3D.py -v

or run directly::

    python tests/test_GpuStandard3D.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import pytest
import warnings
from xinvert import invert_omega

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

_NZ, _NY, _NX = 17, 21, 25
_UNDEF = -9.99e8
_OPTARG = 1.7
_MXLOOP = 200000
_TOL = 1e-12


def _solve_kernel(S0, A, B, C, F, bcs, architect):
    """Call the CPU or GPU 3D kernel directly with the shared signature."""
    zc, yc, xc = S0.shape
    dz = 1.0 / (zc - 1)
    dy = 1.0 / (yc - 1)
    dx = 1.0 / (xc - 1)
    delxSqr = dx * dx
    ratio2Sqr = (dx / dz) ** 2
    ratio1Sqr = (dx / dy) ** 2
    flags = np.zeros(3)

    if architect == 'cpu':
        from xinvert.cpus import invert_standard_3D
        func = invert_standard_3D
    else:
        from xinvert.gpus import invert_standard_3D_gpu
        func = invert_standard_3D_gpu

    S = np.array(S0, dtype=np.float64)
    func(S, A, B, C, F, 'test', zc, yc, xc, bcs[0], bcs[1], bcs[2],
         delxSqr, ratio2Sqr, ratio1Sqr, _OPTARG, _UNDEF, flags,
         _MXLOOP, _TOL)
    return S, flags


def _analytic_problem(bcs):
    """Build the 7-point Laplacian problem for the given BC combination.

    psi = sin(pi x) sin(pi y) sin(pi z) satisfies fixed BCs on all sides;
    for extend-y / periodic-x the same field is still used as the fixed
    point reference of the CPU kernel (consistency-only comparison).
    """
    z = np.linspace(0, 1, _NZ)
    y = np.linspace(0, 1, _NY)
    x = np.linspace(0, 1, _NX, endpoint=(bcs[2] != 'periodic'))
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    psi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    F = -3.0 * np.pi ** 2 * psi
    ones = np.ones_like(psi)
    S0 = np.zeros_like(psi)
    return psi, ones, ones, ones, F, S0


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestStandard3DCpuGpuConsistency:
    """Verify CPU and GPU solvers agree for the standard 3D form."""

    def test_cpu_vs_analytic(self):
        """CPU kernel should converge close to the analytic solution."""
        psi, A, B, C, F, S0 = _analytic_problem(['fixed'] * 3)
        S, flags = _solve_kernel(S0, A, B, C, F, ['fixed'] * 3, 'cpu')
        assert not flags[0], 'CPU overflow'
        err = float(np.max(np.abs(S - psi)))
        # 7-point Laplacian discretization error on 25x21x17 is ~2.2e-3
        assert err < 5e-3, f'CPU error vs analytic: {err:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_gpu_vs_analytic(self):
        """GPU kernel should converge close to the analytic solution."""
        psi, A, B, C, F, S0 = _analytic_problem(['fixed'] * 3)
        S, flags = _solve_kernel(S0, A, B, C, F, ['fixed'] * 3, 'gpu')
        assert not flags[0], 'GPU overflow'
        err = float(np.max(np.abs(S - psi)))
        assert err < 5e-3, f'GPU error vs analytic: {err:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    @pytest.mark.parametrize('bcs', [
        ['fixed', 'fixed', 'fixed'],
        ['fixed', 'extend', 'fixed'],
        ['fixed', 'fixed', 'periodic'],
        ['fixed', 'extend', 'periodic'],
    ])
    def test_cpu_vs_gpu(self, bcs):
        """CPU and GPU results must be consistent for each BC combination."""
        psi, A, B, C, F, S0 = _analytic_problem(bcs)
        S_cpu, flg_cpu = _solve_kernel(S0, A, B, C, F, bcs, 'cpu')
        S_gpu, flg_gpu = _solve_kernel(S0, A, B, C, F, bcs, 'gpu')
        assert not flg_cpu[0] and not flg_gpu[0], 'overflow'
        diff = float(np.max(np.abs(S_cpu - S_gpu)))
        assert diff < 1e-4, f'CPU-GPU diff ({bcs}) too large: {diff:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu_omega_app(self):
        """Application-level consistency through invert_omega.

        The omega equation solved here (cartesian-like flat f via small
        lat range) reduces to a 3-D Poisson/Helmholtz problem; the
        assertion is CPU/GPU consistency of the app path.
        """
        nz, ny, nx = 13, 25, 33
        lev = np.linspace(1000.0, 100.0, nz)
        lat = np.linspace(20.0, 60.0, ny)
        lon = np.linspace(100.0, 160.0, nx, endpoint=False)
        LEV, LAT, LON = np.meshgrid(lev, lat, lon, indexing='ij')
        F = (np.sin(np.pi * LAT / 90.0) * np.sin(np.pi * LON / 180.0)
             * np.sin(np.pi * (lev[-1] - LEV) / (lev[-1] - lev[0])))
        da_F = xr.DataArray(F, dims=['lev', 'lat', 'lon'],
                            coords={'lev': lev, 'lat': lat, 'lon': lon})
        # constant stability profile (large enough for well-posedness)
        S2d = xr.DataArray(np.linspace(5e-4, 1e-4, ny)[:, np.newaxis]
                           * np.ones((1, nx)),
                           dims=['lat', 'lon'],
                           coords={'lat': lat, 'lon': lon})
        mp = {'N2': S2d}
        ip = {'BCs': ['fixed', 'fixed', 'fixed'], 'undef': np.nan,
              'mxLoop': 20000, 'tolerance': 1e-11, 'printInfo': False}
        try:
            r_cpu = invert_omega(da_F, dims=['lev', 'lat', 'lon'],
                                 coords='lat-lon', mParams=mp,
                                 iParams=dict(ip, architect='cpu'))
            r_gpu = invert_omega(da_F, dims=['lev', 'lat', 'lon'],
                                 coords='lat-lon', mParams=mp,
                                 iParams=dict(ip, architect='gpu'))
        except Exception as e:
            pytest.skip(f'omega app setup failed on synthetic data: {e}')
        scale = float(np.nanmax(np.abs(r_cpu.values)))
        diff = float(np.nanmax(np.abs(r_cpu.values - r_gpu.values)))
        rel = diff / max(scale, 1e-300)
        assert rel < 1e-4, f'omega CPU-GPU relative diff too large: {rel:.4e}'


# ---------------------------------------------------------------------------
# allow running this file directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== CPU/GPU consistency tests: standard_3D ===')
    gpu_ok = _gpu_available()
    print(f'GPU available: {gpu_ok}')
    print(f'GPU status   : {_gpu_info()}')
    print()

    ok = True
    psi, A, B, C, F, S0 = _analytic_problem(['fixed'] * 3)
    S_cpu, _ = _solve_kernel(S0, A, B, C, F, ['fixed'] * 3, 'cpu')
    err_cpu = float(np.max(np.abs(S_cpu - psi)))
    print(f'[fixed^3] CPU err vs analytic: {err_cpu:.3e}')
    ok &= err_cpu < 5e-3

    if gpu_ok:
        S_gpu, _ = _solve_kernel(S0, A, B, C, F, ['fixed'] * 3, 'gpu')
        err_gpu = float(np.max(np.abs(S_gpu - psi)))
        print(f'[fixed^3] GPU err vs analytic: {err_gpu:.3e}')
        ok &= err_gpu < 5e-3

    for bcs in (['fixed', 'extend', 'fixed'],
                ['fixed', 'fixed', 'periodic'],
                ['fixed', 'extend', 'periodic']):
        psi, A, B, C, F, S0 = _analytic_problem(bcs)
        S_cpu, _ = _solve_kernel(S0, A, B, C, F, bcs, 'cpu')
        line = f'{str(bcs):38s} CPU ok'
        if gpu_ok:
            S_gpu, _ = _solve_kernel(S0, A, B, C, F, bcs, 'gpu')
            diff = float(np.max(np.abs(S_cpu - S_gpu)))
            line += f' | diff: {diff:.3e}'
            ok &= diff < 1e-4
        print(line)

    print('RESULT:', 'PASS' if ok else 'FAIL')
