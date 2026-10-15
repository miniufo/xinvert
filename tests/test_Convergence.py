"""Tests for residual convergence and structured diagnostics."""
import numpy as np
import pytest
import xarray as xr

from xinvert import invert_Poisson

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


def _solve(forcing, architect='cpu', **overrides):
    params = {
        'BCs': ['fixed', 'fixed'],
        'dtype': 'float64',
        'mxLoop': 5000,
        'tolerance': 1e-8,
        'printInfo': False,
        'architect': architect,
    }
    params.update(overrides)
    return invert_Poisson(
        forcing, dims=['y', 'x'], coords='cartesian', iParams=params)


def test_cpu_preconditioned_residual_converges(architect):
    forcing, exact = _poisson_problem(33)
    solution = _solve(forcing, architect=architect)
    assert np.max(np.abs(solution.values - exact)) < 1e-3


def test_structured_diagnostics_report_convergence(architect):
    forcing, _ = _poisson_problem(17)
    solution, diagnostics = _solve(
        forcing, architect=architect, diag_interval=100)

    assert isinstance(solution, xr.DataArray)
    assert isinstance(diagnostics, xr.Dataset)
    assert bool(diagnostics.converged.item())
    assert diagnostics.stop_reason.item() == 'converged'
    assert 0 < diagnostics.iterations.item() <= 5000
    assert diagnostics.final_error.item() < 1e-8
    assert diagnostics.error.dropna('iteration').isel(iteration=-1).item() == (
        diagnostics.final_error.item())
    assert diagnostics.attrs == {
        'tolerance': 1e-8,
        'max_iterations': 5000,
        'diag_interval': 100,
    }


def test_diag_interval_controls_first_convergence_check(architect):
    forcing, _ = _poisson_problem(9)
    _, diagnostics = _solve(
        forcing, architect=architect, diag_interval=3,
        tolerance=2.0, mxLoop=20)

    assert diagnostics.iterations.item() == 3
    assert diagnostics.iteration.values.tolist() == [3, 6, 9, 12, 15, 18, 20]
    assert diagnostics.error.notnull().values.tolist() == [
        True, False, False, False, False, False, False]
    assert diagnostics.attrs['diag_interval'] == 3
    np.testing.assert_array_equal(
        diagnostics.solution_min.notnull(), diagnostics.error.notnull())
    np.testing.assert_array_equal(
        diagnostics.solution_max.notnull(), diagnostics.error.notnull())


def test_structured_diagnostics_report_iteration_limit(architect):
    forcing, _ = _poisson_problem(9)
    _, diagnostics = _solve(
        forcing, architect=architect, diag_interval=2,
        mxLoop=3, tolerance=0.0)

    assert not bool(diagnostics.converged.item())
    assert diagnostics.iterations.item() == 3
    assert diagnostics.stop_reason.item() == 'max_iterations'


def test_zero_solution_is_reported_as_converged(architect):
    if architect == 'gpu' and not gpu_available():
        pytest.skip('CUDA GPU not available')
    forcing, _ = _poisson_problem(9)
    _, diagnostics = _solve(
        xr.zeros_like(forcing), architect=architect,
        diag_interval=10)

    assert bool(diagnostics.converged.item())
    assert diagnostics.final_error.item() == 0.0
    assert diagnostics.error.sel(iteration=10).item() == 0.0
    assert diagnostics.stop_reason.item() == 'converged'


def test_structured_diagnostics_preserve_batch_dimensions(architect):
    forcing, _ = _poisson_problem(9)
    forcing = xr.concat([forcing, 2.0 * forcing], dim='time').assign_coords(
        time=[0, 1])
    solution, diagnostics = _solve(
        forcing, architect=architect, diag_interval=100)

    assert solution.dims == ('time', 'y', 'x')
    assert set(diagnostics.data_vars) == {
        'converged', 'iterations', 'error', 'final_error', 'stop_reason',
        'solution_min', 'solution_max'}
    assert diagnostics.sizes == {'time': 2, 'iteration': 50}
    assert diagnostics.time.identical(forcing.time)
    assert diagnostics.converged.values.tolist() == [True, True]


def test_structured_diagnostics_support_dask_arrays(architect):
    pytest.importorskip('dask.array')
    forcing, _ = _poisson_problem(9)
    forcing = xr.concat([forcing, forcing], dim='time').chunk({'time': 1})
    solution, diagnostics = _solve(
        forcing, architect=architect, diag_interval=100)

    assert solution.chunks is not None
    computed = diagnostics.compute()
    assert computed.converged.values.tolist() == [True, True]
    assert computed.stop_reason.values.tolist() == ['converged', 'converged']


@pytest.mark.skipif(not gpu_available(), reason='CUDA GPU not available')
def test_gpu_preconditioned_residual_matches_cpu():
    forcing, _ = _poisson_problem(16)
    cpu = _solve(forcing, architect='cpu')
    gpu = _solve(forcing, architect='gpu')
    np.testing.assert_allclose(gpu.values, cpu.values, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not gpu_available(), reason='CUDA GPU not available')
def test_gpu_structured_diagnostics():
    forcing, _ = _poisson_problem(16)
    _, diagnostics = _solve(
        forcing, architect='gpu',
        diag_interval=100)
    assert bool(diagnostics.converged.item())
    assert diagnostics.stop_reason.item() == 'converged'
    assert diagnostics.final_error.item() < 1e-8


@pytest.mark.parametrize('sign', [-1., 1.])
def test_solution_extrema_match_final_slice(architect, sign):
    forcing, _ = _poisson_problem(9)
    solution, diagnostics = _solve(
        sign*forcing, architect=architect, diag_interval=3,
        mxLoop=7, tolerance=0.)
    assert diagnostics.solution_min.dims == diagnostics.error.dims
    assert diagnostics.iteration.values.tolist() == [3, 6, 7]
    assert diagnostics.solution_min.sel(iteration=7).item() == float(solution.min())
    assert diagnostics.solution_max.sel(iteration=7).item() == float(solution.max())
    assert bool((diagnostics.solution_min <= diagnostics.solution_max).all())


def test_solution_extrema_exclude_masked_outlier(architect):
    forcing, _ = _poisson_problem(9)
    forcing[4, 4] = np.nan
    initial = xr.zeros_like(forcing)
    initial[4, 4] = 1e20
    solution, diagnostics = invert_Poisson(
        forcing, dims=['y', 'x'], coords='cartesian', icbc=initial,
        iParams={'architect': architect, 'dtype': 'float64',
                 'BCs': ['fixed', 'fixed'], 'diag_interval': 3,
                 'internal_bcs': 'neumann',
                 'mxLoop': 7, 'tolerance': 0., 'printInfo': False})
    assert diagnostics.solution_max.sel(iteration=7).item() == float(solution.max())
    assert diagnostics.solution_min.sel(iteration=7).item() == float(solution.min())
    assert float(diagnostics.solution_max.max()) < 1e10
