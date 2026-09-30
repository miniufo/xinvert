"""Regression tests for conflict-free GPU coloring of a 9-point stencil."""

import numpy as np
import pytest
import xarray as xr

from xinvert import invert_Eliassen
from tests.test_CpuGpuConsistency import _gpu_available


def _manufactured_problem(n=41, mixed=0.2):
    r"""Return ``A psi_yy + 2 B psi_yx + C psi_xx = forcing``.

    All coefficients are constant and ``psi = sin(pi*x) sin(pi*y)``.  A
    non-zero B is essential here: it activates the diagonal entries of the
    nine-point stencil that cannot safely use ordinary red-black coloring.
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    psi = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    forcing = (-2.0 * np.pi**2 * psi
               + 2.0 * mixed * np.pi**2
               * np.cos(np.pi * xx) * np.cos(np.pi * yy))
    da = xr.DataArray(forcing, dims=['y', 'x'], coords={'y': y, 'x': x})
    return da, psi


def _solve(architect):
    forcing, truth = _manufactured_problem()
    solution = invert_Eliassen(
        forcing,
        dims=['y', 'x'],
        coords='cartesian',
        mParams={'A': 1.0, 'B': 0.2, 'C': 1.0},
        iParams={
            'architect': architect,
            'BCs': ['fixed', 'fixed'],
            'dtype': np.float64,
            'mxLoop': 30000,
            'tolerance': 1e-12,
            'printInfo': False,
        },
    )
    return solution.values, truth


def test_mixed_derivative_cpu_vs_manufactured_solution():
    solution, truth = _solve('cpu')
    assert np.max(np.abs(solution - truth)) < 5e-3


@pytest.mark.skipif(not _gpu_available(), reason='CUDA GPU not available')
def test_mixed_derivative_gpu_is_accurate_and_deterministic():
    cpu, truth = _solve('cpu')
    gpu_runs = [_solve('gpu')[0] for _ in range(3)]

    assert np.max(np.abs(gpu_runs[0] - truth)) < 5e-3
    assert np.max(np.abs(gpu_runs[0] - cpu)) < 1e-5
    for repeat in gpu_runs[1:]:
        np.testing.assert_allclose(repeat, gpu_runs[0], rtol=0.0, atol=1e-12)
