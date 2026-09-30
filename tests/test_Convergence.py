# -*- coding: utf-8 -*-
"""Tests for solver convergence metrics and backward compatibility."""
import numpy as np
import pytest
import xarray as xr

from xinvert import invert_Poisson
from xinvert.core import _convergence_code

from .gpu_utils import gpu_available


def _poisson_problem(n=17):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    forcing = xr.DataArray(
        -2.0 * np.pi ** 2 * exact,
        coords={'y': y, 'x': x},
        dims=['y', 'x'],
    )
    return forcing, exact


def _solve(forcing, convergence=None, architect='cpu', **overrides):
    params = {
        'BCs': ['fixed', 'fixed'],
        'dtype': 'float64',
        'mxLoop': 5000,
        'tolerance': 1e-8,
        'printInfo': False,
        'architect': architect,
    }
    if convergence is not None:
        params['convergence'] = convergence
    params.update(overrides)
    return invert_Poisson(
        forcing, dims=['y', 'x'], coords='cartesian', iParams=params)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [('norm', 0), ('solution-norm', 0), ('legacy', 0),
     ('residual', 1), ('PRECONDITIONED_RESIDUAL', 1)],
)
def test_convergence_mode_aliases(value, expected):
    assert _convergence_code(value) == expected


def test_invalid_convergence_mode_is_rejected():
    forcing, _ = _poisson_problem(9)
    with pytest.raises(ValueError, match='convergence mode'):
        _solve(forcing, convergence='not-a-metric', mxLoop=1)


def test_default_norm_mode_is_backward_compatible():
    forcing, _ = _poisson_problem(9)
    implicit = _solve(forcing, mxLoop=20, tolerance=0.0)
    explicit = _solve(forcing, convergence='norm', mxLoop=20, tolerance=0.0)
    np.testing.assert_array_equal(implicit.values, explicit.values)


def test_cpu_preconditioned_residual_converges():
    forcing, exact = _poisson_problem(33)
    solution = _solve(forcing, convergence='residual')
    assert np.max(np.abs(solution.values - exact)) < 1e-3


def test_structured_diagnostics_report_convergence():
    forcing, _ = _poisson_problem(17)
    solution, diagnostics = _solve(
        forcing, convergence='residual', return_diagnostics=True)

    assert isinstance(solution, xr.DataArray)
    assert isinstance(diagnostics, xr.Dataset)
    assert bool(diagnostics.converged.item())
    assert diagnostics.stop_reason.item() == 'converged'
    assert 0 < diagnostics.iterations.item() <= 5000
    assert diagnostics.error.item() < 1e-8
    assert diagnostics.attrs == {
        'convergence': 'residual',
        'tolerance': 1e-8,
        'max_iterations': 5000,
    }


def test_structured_diagnostics_report_iteration_limit():
    forcing, _ = _poisson_problem(9)
    _, diagnostics = _solve(
        forcing, convergence='residual', return_diagnostics=True,
        mxLoop=3, tolerance=0.0)

    assert not bool(diagnostics.converged.item())
    assert diagnostics.iterations.item() == 3
    assert diagnostics.stop_reason.item() == 'max_iterations'


@pytest.mark.parametrize('architect', ['cpu', 'gpu'])
def test_zero_solution_is_reported_as_converged(architect):
    if architect == 'gpu' and not gpu_available():
        pytest.skip('CUDA GPU not available')
    forcing, _ = _poisson_problem(9)
    _, diagnostics = _solve(
        xr.zeros_like(forcing), convergence='norm', architect=architect,
        return_diagnostics=True)

    assert bool(diagnostics.converged.item())
    assert diagnostics.error.item() == 0.0
    assert diagnostics.stop_reason.item() == 'converged'


def test_structured_diagnostics_preserve_batch_dimensions():
    forcing, _ = _poisson_problem(9)
    forcing = xr.concat([forcing, 2.0 * forcing], dim='time').assign_coords(
        time=[0, 1])
    solution, diagnostics = _solve(
        forcing, convergence='residual', return_diagnostics=True)

    assert solution.dims == ('time', 'y', 'x')
    assert set(diagnostics.data_vars) == {
        'converged', 'iterations', 'error', 'stop_reason'}
    assert diagnostics.sizes == {'time': 2}
    assert diagnostics.time.identical(forcing.time)
    assert diagnostics.converged.values.tolist() == [True, True]


def test_structured_diagnostics_support_dask_arrays():
    pytest.importorskip('dask.array')
    forcing, _ = _poisson_problem(9)
    forcing = xr.concat([forcing, forcing], dim='time').chunk({'time': 1})
    solution, diagnostics = _solve(
        forcing, convergence='residual', return_diagnostics=True)

    assert solution.chunks is not None
    computed = diagnostics.compute()
    assert computed.converged.values.tolist() == [True, True]
    assert computed.stop_reason.values.tolist() == ['converged', 'converged']


@pytest.mark.skipif(not gpu_available(), reason='CUDA GPU not available')
def test_gpu_preconditioned_residual_matches_cpu():
    forcing, _ = _poisson_problem(16)
    cpu = _solve(forcing, convergence='residual', architect='cpu')
    gpu = _solve(forcing, convergence='residual', architect='gpu')
    np.testing.assert_allclose(gpu.values, cpu.values, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not gpu_available(), reason='CUDA GPU not available')
def test_gpu_structured_diagnostics():
    forcing, _ = _poisson_problem(16)
    _, diagnostics = _solve(
        forcing, convergence='residual', architect='gpu',
        return_diagnostics=True)
    assert bool(diagnostics.converged.item())
    assert diagnostics.stop_reason.item() == 'converged'
    assert diagnostics.error.item() < 1e-8
