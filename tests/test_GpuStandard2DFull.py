# -*- coding: utf-8 -*-
"""
Test CPU/GPU consistency for invert_standard_2D_full (the full divergence
form including the Helmholtz term E*psi).

Exercised through two real applications that use this kernel:

- Bretherton-Haidvogel:  grad^2(psi) - lambda*D*psi = h
  (cartesian coeffs: A = D = 1, B = C = 0, E = -lambda*D)
  Analytic solution: psi = sin(pi x) sin(pi y) on [0,1]^2 with fixed BCs,
  forcing  h = (-2 pi^2 - lambda*D) * psi.

- Fofonoff: same kernel with a different coefficient set (c0, c1);
  verified by CPU/GPU consistency.

The GPU tests are skipped automatically when CUDA is not available.

Run from the project root::

    pytest tests/test_GpuStandard2DFull.py -v

or run directly::

    python tests/test_GpuStandard2DFull.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import pytest
import warnings
from xinvert import invert_BrethertonHaidvogel, invert_Fofonoff

# reuse the robust GPU probe from the Poisson consistency test
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

_NY, _NX = 41, 50  # periodic x requires an even extent for parity coloring

_IPARAMS = {
    'BCs'      : ['fixed', 'fixed'],
    'undef'    : np.nan,
    'mxLoop'   : 20000,
    'tolerance': 1e-10,
    'printInfo': False,
    'debug'    : False,
}

_MPARAMS_BRETH = {'f0': 1e-4, 'beta': 0.0, 'D': 1000.0, 'lambda': 1e-4,
                  'Omega': 7.292e-5}

_MPARAMS_FOFONOFF = {'f0': 1e-4, 'beta': 0.0, 'c0': 1e-3, 'c1': 1e-1,
                     'Omega': 7.292e-5}


def _grid():
    x = np.linspace(0.0, 1.0, _NX)
    y = np.linspace(0.0, 1.0, _NY)
    X, Y = np.meshgrid(x, y)
    return X, Y


def _make_bretherton_problem():
    r"""Bretherton-Haidvogel problem with known analytic solution.

    Equation actually solved (cartesian coeffs A = D = 1, B = C = 0,
    E = -lambda*D, kernel RHS F = -f0/D * h with beta = 0):

        grad^2(psi) - lambda*D*psi = -(f0/D) * h

    Solution:  psi = sin(pi x) sin(pi y)   (fixed BCs: psi = 0 on boundary)
    Forcing :  h = (2 pi^2 + lambda*D) * (D/f0) * psi
    """
    X, Y = _grid()
    psi_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    lam_D = _MPARAMS_BRETH['lambda'] * _MPARAMS_BRETH['D']
    h = (2.0 * np.pi**2 + lam_D) * (_MPARAMS_BRETH['D'] / _MPARAMS_BRETH['f0']) \
        * psi_true
    da_h = xr.DataArray(h, dims=['y', 'x'],
                        coords={'y': np.linspace(0, 1, _NY),
                                'x': np.linspace(0, 1, _NX)})
    return da_h, psi_true


def _make_fofonoff_problem():
    """Fofonoff forcing on the same grid (no closed analytic form needed;
    the CPU/GPU consistency check is the real assertion)."""
    X, Y = _grid()
    f = np.cos(np.pi * X) * np.cos(np.pi * Y)
    da_f = xr.DataArray(f, dims=['y', 'x'],
                        coords={'y': np.linspace(0, 1, _NY),
                                'x': np.linspace(0, 1, _NX)})
    return da_f


def _solve(app, da, mParams, architect):
    ip = dict(_IPARAMS)
    ip['architect'] = architect
    return app(da, dims=['y', 'x'], coords='cartesian',
               mParams=mParams, iParams=ip)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestStandard2DFullCpuGpuConsistency:
    """Verify CPU and GPU solvers agree for the full divergence form."""

    def test_cpu_bretherton_vs_analytic(self):
        """CPU solver should converge close to the analytic solution."""
        da_h, psi_true = _make_bretherton_problem()
        S = _solve(invert_BrethertonHaidvogel, da_h, _MPARAMS_BRETH, 'cpu')
        err = float(np.nanmax(np.abs(S.values - psi_true)))
        assert err < 1e-2, f'CPU error vs analytic too large: {err:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_gpu_bretherton_vs_analytic(self):
        """GPU solver should converge close to the analytic solution."""
        da_h, psi_true = _make_bretherton_problem()
        S = _solve(invert_BrethertonHaidvogel, da_h, _MPARAMS_BRETH, 'gpu')
        err = float(np.nanmax(np.abs(S.values - psi_true)))
        assert err < 1e-2, f'GPU error vs analytic too large: {err:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu_bretherton(self):
        """CPU and GPU results must be consistent (Bretherton)."""
        da_h, _ = _make_bretherton_problem()
        S_cpu = _solve(invert_BrethertonHaidvogel, da_h, _MPARAMS_BRETH, 'cpu')
        S_gpu = _solve(invert_BrethertonHaidvogel, da_h, _MPARAMS_BRETH, 'gpu')
        diff = float(np.nanmax(np.abs(S_cpu.values - S_gpu.values)))
        assert diff < 1e-3, f'CPU-GPU diff too large: {diff:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu_fofonoff(self):
        """CPU and GPU results must be consistent (Fofonoff)."""
        da_f = _make_fofonoff_problem()
        S_cpu = _solve(invert_Fofonoff, da_f, _MPARAMS_FOFONOFF, 'cpu')
        S_gpu = _solve(invert_Fofonoff, da_f, _MPARAMS_FOFONOFF, 'gpu')
        diff = float(np.nanmax(np.abs(S_cpu.values - S_gpu.values)))
        assert diff < 1e-3, f'CPU-GPU diff too large: {diff:.4e}'

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_gpu_periodic_x(self):
        """Periodic x-boundary path of the full kernel must work too."""
        # periodic x: the coordinate must NOT include the duplicated endpoint
        x = np.linspace(0.0, 1.0, _NX, endpoint=False)
        y = np.linspace(0.0, 1.0, _NY)
        X, Y = np.meshgrid(x, y)
        psi_true = np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y)
        # laplacian(psi) = -(4 pi^2 + pi^2) psi = -5 pi^2 psi
        lam_D = _MPARAMS_BRETH['lambda'] * _MPARAMS_BRETH['D']
        h = (5.0 * np.pi**2 + lam_D) * (_MPARAMS_BRETH['D'] / _MPARAMS_BRETH['f0']) \
            * psi_true
        da_h = xr.DataArray(h, dims=['y', 'x'], coords={'y': y, 'x': x})
        ip = dict(_IPARAMS)
        ip['architect'] = 'gpu'
        ip['BCs'] = ['fixed', 'periodic']
        S = invert_BrethertonHaidvogel(da_h, dims=['y', 'x'],
                                       coords='cartesian',
                                       mParams=_MPARAMS_BRETH, iParams=ip)
        err = float(np.nanmax(np.abs(S.values - psi_true)))
        # CPU cross-check: the GPU must agree with the CPU on the same
        # (well-posed fixed-y + periodic-x) problem
        ip_cpu = dict(_IPARAMS, architect='cpu', BCs=['fixed', 'periodic'])
        S_cpu = invert_BrethertonHaidvogel(da_h, dims=['y', 'x'],
                                           coords='cartesian',
                                           mParams=_MPARAMS_BRETH,
                                           iParams=ip_cpu)
        diff = float(np.nanmax(np.abs(S.values - S_cpu.values)))
        assert diff < 1e-3, f'GPU periodic-x CPU-GPU diff too large: {diff:.4e}'
        assert err < 1e-2, f'GPU periodic-x error too large: {err:.4e}'


# ---------------------------------------------------------------------------
# allow running this file directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== CPU/GPU consistency tests: standard_2D_full ===')
    gpu_ok = _gpu_available()
    print(f'GPU available: {gpu_ok}')
    print(f'GPU status   : {_gpu_info()}')
    print()

    da_h, psi_true = _make_bretherton_problem()

    S_cpu = _solve(invert_BrethertonHaidvogel, da_h, _MPARAMS_BRETH, 'cpu')
    err_cpu = float(np.nanmax(np.abs(S_cpu.values - psi_true)))
    print(f'CPU  max error vs analytic: {err_cpu:.6e}')

    ok = err_cpu < 1e-2

    if gpu_ok:
        S_gpu = _solve(invert_BrethertonHaidvogel, da_h, _MPARAMS_BRETH, 'gpu')
        err_gpu = float(np.nanmax(np.abs(S_gpu.values - psi_true)))
        diff = float(np.nanmax(np.abs(S_cpu.values - S_gpu.values)))
        print(f'GPU  max error vs analytic: {err_gpu:.6e}')
        print(f'CPU-GPU max diff:          {diff:.6e}')

        da_f = _make_fofonoff_problem()
        S_cpu2 = _solve(invert_Fofonoff, da_f, _MPARAMS_FOFONOFF, 'cpu')
        S_gpu2 = _solve(invert_Fofonoff, da_f, _MPARAMS_FOFONOFF, 'gpu')
        diff2 = float(np.nanmax(np.abs(S_cpu2.values - S_gpu2.values)))
        print(f'Fofonoff CPU-GPU max diff: {diff2:.6e}')

        ok = ok and err_gpu < 1e-2 and diff < 1e-3 and diff2 < 1e-3
    else:
        print('GPU path SKIPPED (no CUDA). Only CPU correctness was verified.')

    print('RESULT:', 'PASS' if ok else 'FAIL')
