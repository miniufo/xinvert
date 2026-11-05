"""Wet Dirichlet boundaries checked against independently assembled equations."""
import numpy as np
import pytest
import xarray as xr

from xinvert import record_iteration, invert_Poisson
from xinvert.core import _fixed_internal_rhs, _undeftmp


def case(n=9):
    y, x = np.indices((n, n))
    wet = np.ones((n, n), bool)
    wet[n//2, n//2] = False
    wet[2:5, :2] = False
    outer = (y == 0) | (y == n-1) | (x == 0) | (x == n-1)
    coast = np.zeros_like(wet)
    # Enumerate physical neighbors without wrap-around.
    for j, i in np.argwhere(wet & ~outer):
        coast[j, i] = any(not wet[j+dj, i+di]
                          for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    f = xr.DataArray(np.where(wet, 1., np.nan), dims=['y', 'x'],
                     coords={'y': np.arange(n, dtype=float), 'x': np.arange(n, dtype=float)})
    initial = xr.where(xr.DataArray(outer, dims=f.dims), 1., 7.).where(f.notnull()).transpose('y', 'x')
    return f, initial, wet, outer, coast


def solve(f, initial, architect, **kw):
    return invert_Poisson(f, ['y', 'x'], coords='cartesian', icbc=initial,
                         iParams={'architect': architect, 'BCs': ['fixed', 'fixed'],
                                  'mxLoop': 10000, 'tolerance': 1e-12,
                                  'optArg': 1.2, 'printInfo': False, **kw})


def test_matches_dense_dirichlet_equations(architect):
    f, initial, wet, outer, coast = case()
    unknown = wet & ~outer & ~coast
    points = list(map(tuple, np.argwhere(unknown)))
    index = {point: k for k, point in enumerate(points)}
    matrix = np.zeros((len(points), len(points)))
    rhs = np.ones(len(points))
    for k, (j, i) in enumerate(points):
        matrix[k, k] = -4.
        for q in ((j-1, i), (j+1, i), (j, i-1), (j, i+1)):
            if q in index:
                matrix[k, index[q]] = 1.
            else:
                assert wet[q]  # masked land is never a value carrier
                rhs[k] -= initial.values[q]
    expected = initial.values.copy()
    expected[unknown] = np.linalg.solve(matrix, rhs)
    result = solve(f, initial, architect)
    np.testing.assert_allclose(result, expected, atol=2e-10)
    np.testing.assert_array_equal(result.values[coast], initial.values[coast])
    np.testing.assert_array_equal(np.isnan(result), ~wet)
    assert float(initial.where(coast).min()) == 7.  # input not mutated


def test_neumann_coast_is_iterated(architect):
    f, initial, wet, outer, coast = case()
    result = solve(f, initial, architect, internal_bcs='neumann')
    assert np.max(np.abs(result.values[coast] - 7.)) > 1.
    np.testing.assert_array_equal(result.values[outer & wet], 1.)


def test_internal_fixed_values_remove_extend_gauge_pin(architect):
    f, _, wet, _, coast = case()
    f = f*0.
    initial = xr.where(xr.DataArray(coast, dims=f.dims), 7., 0.).where(f.notnull())
    result = solve(f, initial, architect, BCs=['extend', 'extend'])
    np.testing.assert_allclose(result.values[wet], 7., atol=1e-8)


@pytest.mark.parametrize('which', ['finite_land', 'missing_wet'])
@pytest.mark.parametrize('lazy', [False, True])
def test_shared_mask_is_checked_including_lazy_batches(which, lazy):
    f, initial, *_ = case()
    initial = initial.copy()
    initial.values[(4, 4) if which == 'finite_land' else (1, 1)] = 0. if which == 'finite_land' else np.nan
    if lazy:
        f, initial = f.chunk({'y': 4}), initial.chunk({'y': 4})
    with pytest.raises(ValueError, match='same valid/invalid mask'):
        solve(f, initial, 'cpu').compute()


def test_numeric_undef_and_default_zero_boundary():
    f, _, wet, outer, coast = case()
    f = f.fillna(-999.)
    result = solve(f, None, 'cpu', undef=-999.)
    np.testing.assert_array_equal(result.values[coast | (outer & wet)], 0.)
    np.testing.assert_array_equal(result.values[~wet], -999.)


def test_periodic_seam_and_coefficient_masks_are_separate():
    wet = np.ones((7, 8), bool)
    wet[3, -1] = False
    rhs = np.where(wet, 0., _undeftmp)
    coeff = np.ones_like(rhs)
    coeff[1, 1] = _undeftmp  # extra coefficient mask does not expand land
    _, boundary = _fixed_internal_rhs([coeff, coeff*0, coeff, rhs],
                                      ['fixed', 'periodic'], 'invert_standard_2D')
    assert boundary[3, 0]
    assert not boundary[1, 2]
    assert not boundary[0].any()
    # Fully wet nonperiodic edges are not treated as internal land.
    _, fixed = _fixed_internal_rhs([coeff, coeff*0, coeff, np.zeros_like(rhs)],
                                   ['fixed', 'fixed'], 'invert_standard_2D')
    assert fixed is None


def test_mixed_and_fourth_order_stencils_prescribe_all_land_neighbors():
    rhs = np.zeros((11, 11)); rhs[5, 5] = _undeftmp
    one = np.ones_like(rhs)
    _, mixed = _fixed_internal_rhs([one]*6 + [rhs], ['fixed', 'fixed'], 'invert_general_2D')
    assert mixed[4, 4] and not mixed[3, 3]
    _, bih = _fixed_internal_rhs([one]*9 + [rhs], ['fixed', 'fixed'], 'invert_general_bih_2D')
    assert bih[3, 3] and bih[3, 5] and bih[4, 4]
    assert not bih[:2].any()


def test_animation_keeps_wet_boundary_and_diagnostic_extrema(architect, tmp_path):
    f, initial, wet, _, coast = case()
    path = tmp_path/'frames.nc'
    result = record_iteration('Poisson', f, ['y', 'x'], coords='cartesian',
                               icbc=initial, output=path, iter_per_frame=2, max_frames=2,
                               iParams={'architect': architect, 'BCs': ['fixed', 'fixed'],
                                        'diag_interval': 1, 'optArg': 1.2, 'printInfo': False})
    with xr.open_dataset(path) as saved:
        np.testing.assert_array_equal(saved.inverted.values[:, coast], 7.)
        assert np.isnan(saved.inverted.values[:, ~wet]).all()
    assert float(result.solution_max.min()) >= 7.


@pytest.mark.parametrize('kind', ['standard_1D', 'standard_2D', 'standard_2D_full',
                                  'standard_3D', 'general_2D', 'general_3D', 'general_bih_2D'])
def test_masked_templates_recover_manufactured_solution(architect, kind):
    import inspect

    from tests.test_backend_math import _operator, _problem
    from xinvert import core, cpus
    truth, active, values = _problem(kind)
    function = getattr(cpus, 'invert_' + kind)
    names = list(inspect.signature(function.py_func).parameters)
    coefficients = names[1:names.index('info')]
    grid = names[names.index('info')+1:names.index('optArg')]
    land = tuple(n//2 for n in truth.shape)
    rhs = values[coefficients[-1]].copy(); rhs[land] = _undeftmp
    arrays = [values[k] if k != coefficients[-1] else rhs for k in coefficients]
    bcs = ['fixed']*truth.ndim
    _, coast = _fixed_internal_rhs(arrays, bcs, function.__name__)
    unknown = active & ~coast
    unknown[land] = False
    initial = truth.copy(); initial[unknown] = 0.; initial[land] = 0.
    wrapper = core._make_kernel(function, [values[k] for k in grid],
                               {'architect': architect, 'optArg': 1., 'mxLoop': 10000,
                                'tolerance': 1e-12, 'diag_interval': 10, 'printInfo': False})
    solved, flags = wrapper(initial, *arrays, '')
    wet = rhs != _undeftmp
    np.testing.assert_allclose(solved[wet], truth[wet], atol=1e-8)
    assert flags[0] == 0 and flags[2] < 10000
    # The operator may reference sentinel land on prescribed coast points,
    # but never on a point where an equation is actually solved.
    residual = (_operator(kind, solved) - values[coefficients[-1]])[unknown]
    assert np.max(np.abs(residual)) < 1e-8
