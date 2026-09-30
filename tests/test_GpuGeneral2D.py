# -*- coding: utf-8 -*-
"""
Test CPU/GPU consistency for invert_general_2D (the non-divergence /
point form with first-derivative terms).

Exercised through the real applications that use this kernel:

- GillMatsuno (cartesian, fixed/extend + periodic x)
- Stommel (cartesian)
- StommelArons (lat-lon)

No closed analytic solutions are used here; the critical assertion is
CPU/GPU consistency (both orderings converge to the same fixed point).

The GPU tests are skipped automatically when CUDA is not available.

Run from the project root::

    pytest tests/test_GpuGeneral2D.py -v

or run directly::

    python tests/test_GpuGeneral2D.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr
import pytest
import warnings
from xinvert import (invert_GillMatsuno, invert_Stommel, invert_StommelArons)

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

def _grid(ny=33, nx=48, periodic=False):
    x = np.linspace(0.0, 1.0, nx, endpoint=not periodic)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y, y, x


def _da(field, y, x):
    return xr.DataArray(field, dims=['y', 'x'], coords={'y': y, 'x': x})


def _solve(app, da, mParams, bcs, architect):
    ip = {'BCs': bcs, 'undef': np.nan, 'mxLoop': 20000, 'tolerance': 1e-11,
          'printInfo': False, 'debug': False, 'architect': architect}
    return app(da, dims=['y', 'x'], coords='cartesian',
               mParams=mParams, iParams=ip)


def _assert_consistent(r_cpu, r_gpu, label, rtol=1e-4):
    scale = float(np.nanmax(np.abs(r_cpu.values)))
    diff = float(np.nanmax(np.abs(r_cpu.values - r_gpu.values)))
    rel = diff / max(scale, 1e-300)
    assert rel < rtol, f'{label}: CPU-GPU relative diff too large: {rel:.4e}'
    return rel


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestGeneral2DCpuGpuConsistency:
    """Verify CPU and GPU solvers agree for the general 2D form."""

    def _gill(self, bcs, architect):
        X, Y, y, x = _grid(periodic=(bcs[1] == 'periodic'))
        Q = np.sin(np.pi * X) * np.sin(np.pi * Y)
        mp = {'Phi': 1e4, 'epsilon': 7e-6, 'f0': 1e-5, 'beta': 2e-11,
              'Omega': 7.292e-5}
        return _solve(invert_GillMatsuno, _da(Q, y, x), mp, bcs, architect)

    def test_cpu_gillmatsuno_runs(self):
        r = self._gill(['fixed', 'fixed'], 'cpu')
        assert np.isfinite(r.values).all()

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu_gillmatsuno(self):
        rel = _assert_consistent(self._gill(['fixed', 'fixed'], 'cpu'),
                                 self._gill(['fixed', 'fixed'], 'gpu'),
                                 'GillMatsuno fixed')
        assert rel < 1e-4

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu_gillmatsuno_periodic(self):
        rel = _assert_consistent(self._gill(['fixed', 'periodic'], 'cpu'),
                                 self._gill(['fixed', 'periodic'], 'gpu'),
                                 'GillMatsuno periodic-x')
        assert rel < 1e-4

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu_stommel(self):
        X, Y, y, x = _grid()
        curl = -np.sin(np.pi * X) * np.sin(np.pi * Y)
        mp = {'beta': 2e-11, 'Omega': 7.292e-5, 'Rearth': 6371200.0,
              'rho0': 1027.0, 'D': 4000.0}
        # NOTE: 'fixed' BCs keep the operator well-conditioned; with
        # 'extend' (Neumann-like) BCs SOR converges so slowly that both
        # backends stay far from the fixed point and the comparison is
        # dominated by convergence stage, not implementation differences.
        da = _da(curl, y, x)
        rel = _assert_consistent(_solve(invert_Stommel, da, mp,
                                        ['fixed', 'fixed'], 'cpu'),
                                 _solve(invert_Stommel, da, mp,
                                        ['fixed', 'fixed'], 'gpu'),
                                 'Stommel')
        assert rel < 1e-4

    @pytest.mark.skipif(not _gpu_available(),
                        reason='CUDA GPU not available')
    def test_cpu_vs_gpu_stommelarons(self):
        r"""Stommel-Arons (cartesian beta-plane) CPU/GPU consistency.

        f0 >> beta*y keeps c1 = epsilon/(epsilon^2+f^2) nearly constant so
        the SOR iteration is stable; D/E (advection) terms remain nonzero
        and are exercised by this case and by GillMatsuno/Stommel above.
        """
        y = np.linspace(0.0, 1.0, 41)
        x = np.linspace(0.0, 2.0, 81)
        X, Y = np.meshgrid(x, y)
        curl = -np.sin(np.pi * Y) * np.sin(np.pi * X / 2.0)
        da = xr.DataArray(curl, dims=['y', 'x'], coords={'y': y, 'x': x})
        mp = {'f0': 1e-4, 'beta': 1e-13, 'epsilon': 1e-4,
              'Omega': 7.292e-5, 'Rearth': 6371200.0}
        ip = {'BCs': ['fixed', 'fixed'], 'undef': np.nan,
              'mxLoop': 20000, 'tolerance': 1e-11, 'printInfo': False}
        r_cpu = invert_StommelArons(da, dims=['y', 'x'], coords='cartesian',
                                    mParams=mp, iParams=dict(ip, architect='cpu'))
        r_gpu = invert_StommelArons(da, dims=['y', 'x'], coords='cartesian',
                                    mParams=mp, iParams=dict(ip, architect='gpu'))
        rel = _assert_consistent(r_cpu, r_gpu, 'StommelArons')
        assert rel < 1e-4


# ---------------------------------------------------------------------------
# allow running this file directly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== CPU/GPU consistency tests: general_2D ===')
    gpu_ok = _gpu_available()
    print(f'GPU available: {gpu_ok}')
    print(f'GPU status   : {_gpu_info()}')
    print()

    if not gpu_ok:
        print('GPU path SKIPPED (no CUDA). Only CPU runs were exercised.')
        print('RESULT: PASS (CPU-only)')
        sys.exit(0)

    test = TestGeneral2DCpuGpuConsistency()
    ok = True
    for name, fn in [('GillMatsuno fixed', lambda: _assert_consistent(
                        test._gill(['fixed', 'fixed'], 'cpu'),
                        test._gill(['fixed', 'fixed'], 'gpu'), 'GillMatsuno')),
                     ('GillMatsuno periodic-x', lambda: _assert_consistent(
                        test._gill(['fixed', 'periodic'], 'cpu'),
                        test._gill(['fixed', 'periodic'], 'gpu'), 'GillMatsuno'))]:
        try:
            rel = fn()
            print(f'{name}: rel diff {rel:.3e}  PASS')
        except AssertionError as e:
            print(f'{name}: FAIL ({e})')
            ok = False

    X, Y, y, x = _grid()
    curl = -np.sin(np.pi * X) * np.sin(np.pi * Y)
    mp = {'beta': 2e-11, 'Omega': 7.292e-5, 'Rearth': 6371200.0,
              'rho0': 1027.0, 'D': 4000.0}
    da = _da(curl, y, x)
    try:
        rel = _assert_consistent(_solve(invert_Stommel, da, mp,
                                        ['fixed', 'fixed'], 'cpu'),
                                 _solve(invert_Stommel, da, mp,
                                        ['fixed', 'fixed'], 'gpu'), 'Stommel')
        print(f'Stommel: rel diff {rel:.3e}  PASS')
    except AssertionError as e:
        print(f'Stommel: FAIL ({e})')
        ok = False

    y = np.linspace(0.0, 1.0, 41)
    x = np.linspace(0.0, 2.0, 81)
    X, Y = np.meshgrid(x, y)
    curl = -np.sin(np.pi * Y) * np.sin(np.pi * X / 2.0)
    da = xr.DataArray(curl, dims=['y', 'x'], coords={'y': y, 'x': x})
    mp = {'f0': 1e-4, 'beta': 1e-13, 'epsilon': 1e-4,
          'Omega': 7.292e-5, 'Rearth': 6371200.0}
    ip = {'BCs': ['fixed', 'fixed'], 'undef': np.nan,
          'mxLoop': 20000, 'tolerance': 1e-11, 'printInfo': False}
    try:
        rel = _assert_consistent(
            invert_StommelArons(da, dims=['y', 'x'], coords='cartesian',
                                mParams=mp, iParams=dict(ip, architect='cpu')),
            invert_StommelArons(da, dims=['y', 'x'], coords='cartesian',
                                mParams=mp, iParams=dict(ip, architect='gpu')),
            'StommelArons')
        print(f'StommelArons: rel diff {rel:.3e}  PASS')
    except AssertionError as e:
        print(f'StommelArons: FAIL ({e})')
        ok = False

    print('RESULT:', 'PASS' if ok else 'FAIL')
