"""Independent masked-Poisson mathematics across all three backends."""
import numpy as np
import pytest
import xarray as xr

from xinvert import invert_Poisson, invert_Stommel
from xinvert.domain import analyze_domain


def manufactured(mask, coords='cartesian'):
    """Return an independently evaluated physical Laplacian and known field."""
    ny, nx = mask.shape
    y = np.linspace(-5, 5, ny) if coords == 'lat-lon' else np.arange(ny)
    x = np.arange(nx, dtype=float)
    template = xr.DataArray(mask, dims=('y', 'x'), coords={'y': y, 'x': x})
    exact = np.sin(np.arange(ny)[:, None]*.7) + np.cos(x[None, :]*.5)
    weight = np.cos(np.deg2rad(y))[:, None] if coords == 'lat-lon' else np.ones((ny, 1))
    exact -= np.sum(exact*mask*weight)/np.sum(mask*weight)
    rhs = physical_operator(exact, mask, y, coords)
    return template, template.copy(data=np.where(mask, rhs, np.nan)), template.copy(data=exact)


def physical_operator(u, mask, y, coords='cartesian'):
    """Independent physical divergence of active face gradients."""
    ny, nx = mask.shape
    rhs = np.zeros(mask.shape)
    dx = 6371200*np.deg2rad(1.) if coords == 'lat-lon' else 1.
    dy = 6371200*np.deg2rad(y[1]-y[0]) if coords == 'lat-lon' else 1.
    for j, i in np.ndindex(mask.shape):
        if not mask[j, i] or j in (0, ny-1) or i in (0, nx-1):
            continue
        for jj, ii in ((j, i+1), (j, i-1), (j-1, i), (j+1, i)):
            if not mask[jj, ii]:
                continue
            g = 1.
            if coords == 'lat-lon':
                g = (1/np.cos(np.deg2rad(y[j]))/dx**2 if j == jj else
                     np.cos(np.deg2rad((y[j]+y[jj])/2))/dy**2)
                g /= np.cos(np.deg2rad(y[j]))
            rhs[j, i] += g*(u[jj, ii]-u[j, i])
    return rhs


def solve(forcing, architect='cpu', *, coords='cartesian', icbc=None, **controls):
    """Exercise string-only public API without user-created topology."""
    params = {'internal_bcs': 'neumann', 'architect': architect, 'dtype': 'float64',
              'optArg': 1.4, 'mxLoop': 20000, 'tolerance': 1e-11,
              'printInfo': False, 'diag_interval': 50}
    params.update(controls)
    return invert_Poisson(forcing, ['y', 'x'], coords=coords, icbc=icbc, iParams=params)


@pytest.mark.parametrize('coords', ['cartesian', 'lat-lon'])
@pytest.mark.parametrize('dtype,tol', [('float64', 2e-8), ('float32', 2e-4)])
def test_closed_component_manufactured_solution(architect, coords, dtype, tol):
    mask = np.zeros((11, 13), dtype=bool)
    mask[2:-2, 2:-2] = True
    mask[4:6, 5:7] = False
    _, forcing, exact = manufactured(mask, coords)
    result, diag = solve(forcing, architect, coords=coords, dtype=dtype,
                         tolerance=1e-12 if dtype == 'float64' else 8e-7)
    np.testing.assert_allclose(result.values[mask], exact.values[mask], atol=tol)
    assert np.isnan(result.values[~mask]).all()
    assert bool(diag.converged)
    residual = physical_operator(np.where(mask, result.values, 0), mask, result.y.values, coords)
    np.testing.assert_allclose(residual[mask], forcing.values[mask],
                               atol=tol*max(abs(forcing.values[mask]).max(), 1e-30))


def test_neumann_coast_with_fixed_outer_boundary(architect):
    mask = np.ones((9, 10), dtype=bool)
    mask[3:5, 4:6] = False
    _, forcing, exact = manufactured(mask)
    initial = xr.zeros_like(exact)
    initial.values[[0, -1], :] = exact.values[[0, -1], :]
    initial.values[:, [0, -1]] = exact.values[:, [0, -1]]
    result, diag = solve(forcing, architect, icbc=initial)
    np.testing.assert_allclose(result.values[mask], exact.values[mask], atol=1e-8)
    assert bool(diag.converged)


def test_incompatible_closed_basin(architect):
    mask = np.zeros((7, 8), dtype=bool)
    mask[2:5, 2:6] = True
    a, _, _ = manufactured(mask)
    with pytest.raises(ValueError, match='component 1.*incompatible forcing'):
        solve(xr.where(a, 1., np.nan), architect)


def test_isolated_cell(architect):
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    a, _, _ = manufactured(mask)
    forcing = xr.where(a, 0., np.nan)
    result, _ = solve(forcing, architect)
    assert result.values[2, 2] == 0
    forcing.values[2, 2] = 1
    with pytest.raises(ValueError, match='incompatible forcing'):
        solve(forcing, architect)


def test_every_unanchored_point_updates_including_former_pin(architect):
    """One independent sweep detects a skipped reference-point equation."""
    mask = np.zeros((7, 8), dtype=bool)
    mask[2:5, 2:6] = True
    _, forcing, _ = manufactured(mask)
    assert forcing.values[2, 2] != 0
    expected = np.zeros(mask.shape)
    for color in range(1 if architect == 'cpu' else 2):
        for j, i in zip(*np.where(mask)):
            if architect != 'cpu' and (j+i) % 2 != color:
                continue
            neighbours = [(jj, ii) for jj, ii in [(j-1, i), (j+1, i), (j, i-1), (j, i+1)]
                          if mask[jj, ii]]
            flux = sum(expected[jj, ii]-expected[j, i] for jj, ii in neighbours)
            expected[j, i] += 1.4*(flux-forcing.values[j, i])/len(neighbours)
    expected[mask] -= expected[mask].mean()
    result, _ = solve(forcing, architect, mxLoop=1, tolerance=0., diag_interval=1)
    np.testing.assert_allclose(result.values[mask], expected[mask], rtol=0, atol=1e-12)


def test_constant_initial_offset_does_not_change_closed_solution(architect):
    """The output gauge, not an initial pinned value, determines the constant."""
    mask = np.zeros((9, 10), dtype=bool)
    mask[2:-2, 2:-2] = True
    _, forcing, exact = manufactured(mask)
    result, _ = solve(forcing, architect, icbc=xr.full_like(forcing, 123.), tolerance=1e-12)
    np.testing.assert_allclose(result.values[mask], exact.values[mask], atol=1e-8)


@pytest.mark.parametrize('bcs', [('extend', 'extend'), ('extend', 'periodic'),
                               ('fixed', 'periodic'), ('fixed', 'extend')])
def test_independent_sparse_reference(architect, bcs):
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    mask = np.ones((8, 10), dtype=bool)
    mask[3:5, 4:6] = False
    ny, nx = mask.shape
    a = xr.DataArray(mask, dims=('y', 'x'), coords={'y': np.arange(ny), 'x': np.arange(nx)})
    unknown = mask.copy()
    unknown[[0, -1], :] = False
    if bcs[1] != 'periodic':
        unknown[:, [0, -1]] = False
    ids = {point: i for i, point in enumerate(zip(*np.where(unknown)))}
    matrix = lil_matrix((len(ids), len(ids)))
    for (j, i), p in ids.items():
        for jj, ii in ((j, i+1), (j, i-1), (j-1, i), (j+1, i)):
            if bcs[1] == 'periodic':
                ii %= nx
            if not mask[jj, ii]:
                continue
            if jj in (0, ny-1) and bcs[0] == 'extend':
                continue
            if ii in (0, nx-1) and bcs[1] == 'extend':
                continue
            matrix[p, p] -= 1
            if (jj, ii) in ids:
                matrix[p, ids[(jj, ii)]] += 1
    vector = matrix.tocsr() @ np.sin(np.arange(len(ids))*.3)
    forcing = xr.where(a, 0., np.nan)
    forcing.values[unknown] = vector
    if 'fixed' not in bcs:
        reference = matrix.copy().tolil()
        reference[0, :] = 0
        reference[0, 0] = 1
        compatible_rhs = vector.copy()
        compatible_rhs[0] = 0
        expected = spsolve(reference.tocsr(), compatible_rhs)
        expected -= expected.mean()
    else:
        expected = spsolve(matrix.tocsr(), vector)
    result, diag = solve(forcing, architect, BCs=bcs)
    assert bool(diag.converged)
    np.testing.assert_allclose(result.values[unknown], expected, atol=1e-8)
    np.testing.assert_allclose(matrix.tocsr() @ result.values[unknown], vector, atol=1e-9)
    last = int(diag.iterations)
    assert abs(float(diag.solution_min.sel(iteration=last))-float(result.min())) < 1e-12
    assert abs(float(diag.solution_max.sel(iteration=last))-float(result.max())) < 1e-12


def test_disconnected_component_gauges(architect):
    mask = np.zeros((14, 15), dtype=bool)
    mask[2:6, 2:6] = True
    mask[8:12, 8:13] = True
    a, forcing, _ = manufactured(mask)
    topology = analyze_domain(a)
    result, _ = solve(forcing, architect)
    for component in (1, 2):
        assert abs(result.values[topology.component_labels.values == component].mean()) < 1e-10
    forcing.values[9, 9] += 1
    with pytest.raises(ValueError, match='component 2.*incompatible forcing'):
        solve(forcing, architect)


def test_closed_lake_not_anchored_by_outer_boundary(architect):
    mask = np.ones((13, 14), dtype=bool)
    mask[3:10, 3:11] = False
    mask[5:8, 5:9] = True
    _, forcing, _ = manufactured(mask)
    forcing.values[6, 6] += 1
    with pytest.raises(ValueError, match='component 2.*incompatible forcing'):
        solve(forcing, architect)


def test_neumann_ignores_inactive_initial_values(architect):
    mask = np.ones((8, 9), dtype=bool)
    mask[3:5, 4] = False
    a, forcing, _ = manufactured(mask)
    initial = xr.zeros_like(forcing)
    one, _ = solve(forcing, architect, icbc=initial, mxLoop=30, tolerance=0)
    two, _ = solve(forcing, architect, icbc=initial.where(a, 1e30), mxLoop=30, tolerance=0)
    np.testing.assert_array_equal(one, two)


def test_varying_time_masks_stay_lazy(architect):
    first = np.ones((9, 10), dtype=bool)
    first[3:5, 4:6] = False
    second = first.copy()
    second[5, 6] = False
    _, f1, _ = manufactured(first)
    _, f2, _ = manufactured(second)
    forcing = xr.concat([f1, f2], dim=xr.IndexVariable('time', [0, 1])).chunk({'time': 1})
    result, diag = solve(forcing, architect)
    assert hasattr(result.data, 'dask')
    assert hasattr(diag.error.data, 'dask')
    result = result.compute()
    assert np.isfinite(result.values[0, 5, 6])
    assert np.isnan(result.values[1, 5, 6])
    for index, field in enumerate((f1, f2)):
        reference, _ = solve(field, architect)
        np.testing.assert_allclose(result.values[index], reference, atol=1e-10)


def test_fixed_default_is_unchanged(architect):
    _, forcing, _ = manufactured(np.ones((8, 9), dtype=bool))
    params = {'dtype': 'float64', 'optArg': 1., 'mxLoop': 10, 'tolerance': 0.,
              'printInfo': False, 'architect': architect}
    old = invert_Poisson(forcing, ['y', 'x'], coords='cartesian', iParams=params)
    new = invert_Poisson(forcing, ['y', 'x'], coords='cartesian',
                         iParams={**params, 'internal_bcs': ' FIXED '})
    np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize('architect', ['cpu-mc', 'gpu'])
def test_odd_periodic_x_fails_before_launch(architect):
    _, forcing, _ = manufactured(np.ones((5, 5), dtype=bool))
    with pytest.raises(ValueError, match='even periodic x'):
        solve(forcing, architect, BCs=['fixed', 'periodic'])


@pytest.mark.parametrize('value', [True, None, 1, 'dirichlet', 'neuman'])
def test_invalid_internal_bcs(value):
    _, forcing, _ = manufactured(np.ones((5, 6), dtype=bool))
    with pytest.raises(ValueError, match='internal_bcs'):
        solve(forcing, internal_bcs=value)


def test_other_equations_do_not_silently_ignore_neumann():
    _, forcing, _ = manufactured(np.ones((5, 6), dtype=bool))
    with pytest.raises(NotImplementedError, match='two-dimensional'):
        invert_Stommel(forcing, ['y', 'x'], coords='cartesian',
                       iParams={'internal_bcs': 'neumann'})


def test_numeric_undef_and_nonfinite_cells_are_inactive(architect):
    _, forcing, _ = manufactured(np.ones((7, 8), dtype=bool))
    forcing.values[3, 3] = -999.
    forcing.values[3, 4] = np.inf
    result, _ = solve(forcing, architect, undef=-999., mxLoop=20, tolerance=0)
    assert result.values[3, 3] == -999
    assert result.values[3, 4] == -999


def test_animation_rejects_neumann_before_writing(tmp_path):
    from xinvert import animate_iteration
    _, forcing, _ = manufactured(np.ones((7, 8), dtype=bool))
    output = tmp_path/'unsupported.nc'
    with pytest.raises(NotImplementedError, match='animate_iteration'):
        animate_iteration('poisson', forcing, ['y', 'x'], coords='cartesian',
                          output=output, iParams={'internal_bcs': 'neumann'})
    assert not output.exists()


def test_low_level_templates_reject_neumann():
    from xinvert.core import _make_kernel
    from xinvert.cpus import invert_standard_2D
    with pytest.raises(NotImplementedError, match='low-level generic'):
        _make_kernel(invert_standard_2D, [], {'internal_bcs': 'neumann'})


def test_dtype_overflow_does_not_silently_change_the_mask():
    _, forcing, _ = manufactured(np.ones((7, 8), dtype=bool))
    forcing.values[3, 3] = 1e100
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='finite on every active'):
        solve(forcing, dtype='float32')
