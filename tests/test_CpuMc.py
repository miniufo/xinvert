"""Tests for the experimental multi-core, multi-color CPU backend."""
import numba
import numpy as np
import pytest
import xarray as xr

from xinvert import invert_BrethertonHaidvogel, invert_Eliassen, invert_Poisson
from xinvert.cpu_mc import invert_standard_2D_full_mc


def test_full_cpu_mc_public_application():
    from tests.test_GpuStandard2DFull import (
        _make_bretherton_problem,
        _MPARAMS_BRETH,
    )

    forcing, truth = _make_bretherton_problem()
    solution = invert_BrethertonHaidvogel(
        forcing, dims=['y', 'x'], coords='cartesian',
        mParams=_MPARAMS_BRETH,
        iParams={'architect': 'cpu-mc', 'BCs': ['fixed', 'fixed'],
                 'mxLoop': 20000, 'tolerance': 1e-10, 'printInfo': False})
    np.testing.assert_allclose(solution.values, truth, atol=1e-2, rtol=0)


@pytest.mark.parametrize('cross', [(0.0, 0.0), (0.1, 0.0), (0.0, 0.2)])
@pytest.mark.parametrize('periodic', [False, True])
def test_full_cpu_mc_discrete_manufactured_solution(cross, periodic):
    ny, nx = 19, 20
    y = np.linspace(0, 1, ny)
    x = np.arange(nx) / nx if periodic else np.linspace(0, 1, nx)
    dx, dy = x[1] - x[0], y[1] - y[0]
    truth = np.sin(np.pi * y[:, None]) * np.cos(2 * np.pi * x[None, :])
    b, c = cross
    forcing = np.zeros_like(truth)
    for j in range(1, ny - 1):
        for i in range(0 if periodic else 1, nx if periodic else nx - 1):
            ip, im = (i + 1) % nx, (i - 1) % nx
            forcing[j, i] = (
                (truth[j+1, i] - 2*truth[j, i] + truth[j-1, i]) / dy**2
                + (truth[j, ip] - 2*truth[j, i] + truth[j, im]) / dx**2
                + (b+c) * (truth[j+1, ip] - truth[j+1, im]
                           - truth[j-1, ip] + truth[j-1, im]) / (4*dx*dy)
                - 0.5*truth[j, i])
    solution = truth.copy()
    solution[1:-1, :] = 0 if periodic else solution[1:-1, :]
    if not periodic:
        solution[1:-1, 1:-1] = 0
    coeffs = [np.full_like(truth, value) for value in (1, b, c, 1, -0.5)]
    flags = np.zeros(3)
    invert_standard_2D_full_mc(
        solution, *coeffs, forcing, '', ny, nx, 'fixed',
        'periodic' if periodic else 'fixed', dx**2,
        dx/dy/4, (dx/dy)**2, 1.5, -9.99e8, flags, 5000, 1e-12, 10)
    assert flags[2] < 5000
    np.testing.assert_allclose(solution, truth, atol=1e-9, rtol=1e-9)


def _forcing(ny=32, nx=None):
    nx = ny if nx is None else nx
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)
    return xr.DataArray(
        -2.0 * np.pi**2 * np.sin(np.pi * xx) * np.sin(np.pi * yy),
        dims=('y', 'x'), coords={'y': y, 'x': x})


def _solve(forcing, architect):
    return invert_Poisson(
        forcing, dims=['y', 'x'], coords='cartesian',
        iParams={
            'architect': architect,
            'BCs': ['fixed', 'fixed'],
            'dtype': 'float64',
            'mxLoop': 5000,
            'tolerance': 1e-9,
            'diag_interval': 20,
            'printInfo': False,
        })


def test_cpu_mc_matches_lexicographic_cpu():
    forcing = _forcing()
    cpu, cpu_diag = _solve(forcing, 'cpu')
    cpu_mc, mc_diag = _solve(forcing, 'cpu-mc')

    assert bool(cpu_diag.converged.item())
    assert bool(mc_diag.converged.item())
    np.testing.assert_allclose(cpu_mc, cpu, rtol=2e-7, atol=2e-7)


def test_cpu_mc_is_stable_across_thread_counts():
    forcing = _forcing(34)
    original = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one_thread, _ = _solve(forcing, 'cpu-mc')
        numba.set_num_threads(min(4, original))
        four_threads, _ = _solve(forcing, 'cpu-mc')
    finally:
        numba.set_num_threads(original)

    np.testing.assert_allclose(four_threads, one_thread, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('shape', [(33, 34), (34, 33), (35, 35)])
def test_cpu_mc_skip_scan_handles_odd_and_rectangular_grids(shape):
    forcing = _forcing(*shape)
    cpu, _ = _solve(forcing, 'cpu')
    cpu_mc, _ = _solve(forcing, 'cpu-mc')

    np.testing.assert_allclose(cpu_mc, cpu, rtol=2e-7, atol=2e-7)


def test_cpu_mc_skip_scan_preserves_four_color_mixed_derivative():
    forcing = _forcing(35)
    common = {
        'dims': ['y', 'x'],
        'coords': 'cartesian',
        'mParams': {'A': 1.0, 'B': 0.2, 'C': 1.0},
    }
    params = {
        'BCs': ['fixed', 'fixed'],
        'dtype': 'float64',
        'mxLoop': 10000,
        'tolerance': 1e-9,
        'diag_interval': 20,
        'printInfo': False,
    }
    cpu, _ = invert_Eliassen(
        forcing, iParams={**params, 'architect': 'cpu'}, **common)
    cpu_mc, _ = invert_Eliassen(
        forcing, iParams={**params, 'architect': 'cpu-mc'}, **common)

    np.testing.assert_allclose(cpu_mc, cpu, rtol=2e-6, atol=2e-6)


def test_cpu_mc_rejects_odd_periodic_x_grid():
    forcing = _forcing(31)
    with pytest.raises(ValueError, match='requires an even x grid length'):
        invert_Poisson(
            forcing, dims=['y', 'x'], coords='cartesian',
            iParams={
                'architect': 'cpu-mc',
                'BCs': ['fixed', 'periodic'],
                'mxLoop': 10,
                'tolerance': 1e-6,
                'printInfo': False,
            })
