"""Independent sparse operators for new diagonal/Helmholtz masked apps."""
import numpy as np
import pytest
import xarray as xr
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from xinvert import (
    invert_BrethertonHaidvogel,
    invert_Eliassen,
    invert_Fofonoff,
    invert_geostrophic,
    invert_GillMatsunoFlux,
    invert_PV2D,
    invert_RefState,
    invert_StommelFlux,
)


def problem(model, closed=True):
    """Independently build physical coefficients, RHS and sparse reference."""
    ny, nx = 10, 12
    mask = np.ones((ny, nx), dtype=bool)
    mask[4:6, 5:7] = False
    if closed:
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False
    y, x = np.arange(ny, dtype=float), np.arange(nx, dtype=float)
    a = np.ones(mask.shape)
    c = np.ones(mask.shape)
    reaction = np.zeros(mask.shape)
    params = {}
    if model == 'pv':
        params = {'f0': 1., 'N2': .4}
        a[:] = 2.5
        app = invert_PV2D
    elif model == 'eliassen':
        params = {'A': 2., 'B': 0., 'C': 3.}
        a[:] = 2.
        c[:] = 3.
        app = invert_Eliassen
    elif model.startswith('geo'):
        sign = -1 if model == 'geo-negative' else 1
        params = {'f0': sign*2., 'beta': sign*.05}
        a[1:] = (params['f0'] + params['beta']*(y[1:]+y[:-1])/2)[:, None]
        c[:] = (params['f0'] + params['beta']*y)[:, None]
        app = invert_geostrophic
    elif model == 'bretherton':
        params = {'f0': 2., 'beta': 0., 'D': 1., 'lambda': .2}
        reaction[:] = -.2
        app = invert_BrethertonHaidvogel
    elif model == 'fofonoff':
        params = {'f0': 1., 'beta': .05, 'c0': .2, 'c1': .3}
        reaction[:] = -.2
        app = invert_Fofonoff
    elif model == 'gill-diagonal':
        params = {'f0': 0., 'beta': 0., 'epsilon': 1., 'Phi': 2.}
        a[:] = 2.
        c[:] = 2.
        reaction[:] = -1.
        app = invert_GillMatsunoFlux
    else:
        raise AssertionError(model)
    unknown = mask.copy()
    unknown[[0, -1], :] = False
    unknown[:, [0, -1]] = False
    points = list(zip(*np.where(unknown)))
    ids = {p: i for i, p in enumerate(points)}
    matrix = lil_matrix((len(points), len(points)))
    for (j, i), p in ids.items():
        matrix[p, p] = reaction[j, i]
        for jj, ii, conductance in ((j, i+1, c[j, i+1]), (j, i-1, c[j, i]),
                                    (j-1, i, a[j, i]), (j+1, i, a[j+1, i])):
            if not mask[jj, ii]:
                continue
            matrix[p, p] -= conductance
            if (jj, ii) in ids:
                matrix[p, ids[(jj, ii)]] += conductance
    matrix = matrix.tocsr()
    if model == 'fofonoff':
        physical_rhs = (params['c1']-params['f0']-params['beta']*y)[:, None]*np.ones(mask.shape)
        vector = physical_rhs[unknown]
    else:
        vector = matrix @ np.sin(np.arange(len(points))*.23)
    reference = matrix.tolil(copy=True)
    solve_rhs = vector.copy()
    if closed and not reaction.any():
        reference[0, :] = 0
        reference[0, 0] = 1
        solve_rhs[0] = 0
    expected = spsolve(reference.tocsr(), solve_rhs)
    if closed and not reaction.any():
        expected -= expected.mean()
    forcing = np.zeros(mask.shape)
    forcing[unknown] = vector
    if model == 'bretherton':
        forcing *= -params['D']/params['f0']
    elif model == 'fofonoff':
        forcing[:] = 1.  # application uses this input as validity, not its RHS
    field = xr.DataArray(np.where(mask, forcing, np.nan), dims=('y', 'x'),
                         coords={'y': y, 'x': x})
    return app, field, params, unknown, matrix, vector, expected


@pytest.mark.parametrize('model', ['pv', 'eliassen', 'geo', 'geo-negative', 'bretherton', 'fofonoff',
                                  'gill-diagonal'])
@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_applications_against_independent_sparse_operator(architect, model, closed, dtype):
    app, field, params, unknown, matrix, rhs, expected = problem(model, closed)
    result, diag = app(field, ['y', 'x'], coords='cartesian', mParams=params,
                       iParams={'internal_bcs': 'neumann', 'architect': architect,
                                'dtype': dtype, 'optArg': 1.3, 'mxLoop': 20000,
                                'tolerance': 1e-12 if dtype == 'float64' else 8e-7,
                                'diag_interval': 50, 'printInfo': False})
    assert bool(diag.converged)
    assert np.isnan(result.values[~np.isfinite(field.values)]).all()
    np.testing.assert_allclose(result.values[unknown], expected,
                               atol=2e-8 if dtype == 'float64' else 2e-4)
    np.testing.assert_allclose(matrix @ result.values[unknown], rhs,
                               atol=1e-9 if dtype == 'float64' else 2e-4*max(1., np.max(abs(rhs))))
    if closed and model == 'fofonoff':
        assert abs(result.values[unknown].mean()) > 1  # reaction removes gauge ambiguity


def test_nonzero_eliassen_cross_coefficient_is_not_ignored(architect):
    _, field, _, _, _, _, _ = problem('eliassen')
    with pytest.raises(NotImplementedError, match='mixed-derivative'):
        invert_Eliassen(field, ['y', 'x'], coords='cartesian',
                        mParams={'A': 2., 'B': .3, 'C': 3.},
                        iParams={'internal_bcs': 'neumann', 'architect': architect})


def test_gill_flux_cancellation_does_not_remove_boundary_tensor(architect):
    _, field, _, _, _, _, _ = problem('pv')
    with pytest.raises(NotImplementedError, match='mixed-derivative'):
        invert_GillMatsunoFlux(field, ['y', 'x'], coords='cartesian',
                               iParams={'internal_bcs': 'neumann', 'architect': architect})


def test_indefinite_reaction_requires_explicit_future_handling(architect):
    _, field, _, _, _, _, _ = problem('fofonoff')
    with pytest.raises(NotImplementedError, match='indefinite Helmholtz'):
        invert_Fofonoff(field, ['y', 'x'], coords='cartesian', mParams={'c0': -.2},
                        iParams={'internal_bcs': 'neumann', 'architect': architect})


def test_zero_face_conductance_does_not_merge_disconnected_unknowns(architect):
    _, field, _, _, _, _, _ = problem('pv')
    with pytest.raises(ValueError, match='finite and nonzero'):
        invert_PV2D(field, ['y', 'x'], coords='cartesian', mParams={'f0': 0.},
                     iParams={'internal_bcs': 'neumann', 'architect': architect})


@pytest.mark.parametrize('internal_bcs', ['fixed', 'neumann'])
def test_reference_state_zero_rhs_preserves_constant_solution(architect, internal_bcs):
    # Constant u satisfies div(A grad u)=0 independently of positive PV.
    field = xr.DataArray(np.linspace(1., 2., 90).reshape(9, 10),
                         dims=['y', 'x'],
                         coords={'y': np.arange(9.), 'x': np.arange(1., 11.)})
    field[4, 5] = np.nan
    initial = xr.full_like(field, 7.)
    result = invert_RefState(
        field, ['y', 'x'], coords='cartesian', icbc=initial,
        mParams={'Gamma': 1.},
        iParams={'internal_bcs': internal_bcs, 'architect': architect,
                 'dtype': 'float64', 'BCs': ['fixed', 'fixed'],
                 'mxLoop': 20, 'tolerance': 1e-12, 'printInfo': False})
    np.testing.assert_allclose(result.values[np.isfinite(field)], 7., atol=1e-12)
    assert np.isnan(result[4, 5])


@pytest.mark.parametrize('model', ['geo', 'bretherton', 'fofonoff', 'stommel-diagonal'])
def test_spherical_applications_with_independent_scaled_operator(architect, model):
    ny, nx = 9, 10
    y, x = np.linspace(25, 29, ny), np.linspace(110, 115, nx)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[2:-2, 2:-2] = True
    mask[4, 5] = False
    lat = np.deg2rad(y)
    w = np.cos(lat)
    ah = np.cos((lat[1:]+lat[:-1])/2)
    cg = 1/w
    dx2 = (6371200*np.deg2rad(x[1]-x[0]))**2
    dy2 = (6371200*np.deg2rad(y[1]-y[0]))**2
    f = 2*7.292e-5*np.sin(lat)
    reaction = np.zeros(ny)
    params = {}
    if model == 'geo':
        ah *= 2*7.292e-5*np.sin((lat[1:]+lat[:-1])/2)
        cg *= f
        app = invert_geostrophic
    elif model == 'bretherton':
        params = {'D': 1., 'lambda': .2}
        reaction = -.2*w
        app = invert_BrethertonHaidvogel
    elif model == 'fofonoff':
        params = {'c0': .2, 'c1': .3}
        reaction = -.2*w
        app = invert_Fofonoff
    else:
        params = {'Omega': 0., 'D': 1., 'R': 1., 'rho0': 1.}
        ah *= -1
        cg *= -1
        app = invert_StommelFlux
    ids = {p: i for i, p in enumerate(zip(*np.where(mask)))}
    matrix = lil_matrix((len(ids), len(ids)))
    for (j, i), p in ids.items():
        matrix[p, p] = reaction[j]
        for jj, ii in ((j, i-1), (j, i+1), (j-1, i), (j+1, i)):
            if not mask[jj, ii]:
                continue
            g = cg[j]/dx2 if j == jj else ah[min(j, jj)]/dy2
            matrix[p, p] -= g
            matrix[p, ids[(jj, ii)]] += g
    matrix = matrix.tocsr()
    if model == 'fofonoff':
        vector = ((.3-f)*w)[:, None]*np.ones(mask.shape)
        vector = vector[mask]
    else:
        vector = matrix @ np.sin(np.arange(len(ids))*.3)
    reference = matrix.tolil(copy=True)
    solve_rhs = vector.copy()
    if not reaction.any():
        reference[0, :] = 0
        reference[0, 0] = 1
        solve_rhs[0] = 0
    expected = spsolve(reference.tocsr(), solve_rhs)
    if not reaction.any():
        weights = np.broadcast_to(w[:, None], mask.shape)[mask]
        expected -= np.sum(expected*weights)/np.sum(weights)
    forcing = np.zeros(mask.shape)
    forcing[mask] = vector
    forcing /= w[:, None]
    if model == 'bretherton':
        forcing /= -f[:, None]
    elif model == 'fofonoff':
        forcing[:] = 1.
    elif model == 'stommel-diagonal':
        forcing *= -1
    field = xr.DataArray(np.where(mask, forcing, np.nan), dims=('y', 'x'), coords={'y': y, 'x': x})
    result, diag = app(field, ['y', 'x'], coords='lat-lon', mParams=params,
                       iParams={'internal_bcs': 'neumann', 'architect': architect,
                                'dtype': 'float64', 'optArg': 1., 'mxLoop': 20000,
                                'tolerance': 1e-12, 'diag_interval': 50, 'printInfo': False})
    assert bool(diag.converged)
    np.testing.assert_allclose(result.values[mask], expected, atol=2e-8)
    np.testing.assert_allclose(matrix @ result.values[mask], vector,
                               atol=max(np.max(abs(vector))*1e-8, 1e-30))


def test_variable_face_coefficients(architect):
    _, field, _, unknown, _, _, _ = problem('pv', closed=True)
    y, x = np.meshgrid(field.y.values, field.x.values, indexing='ij')
    a = 2 + .02*y + .01*x
    c = 3 + .01*y + .03*x
    ids = {p: i for i, p in enumerate(zip(*np.where(unknown)))}
    matrix = lil_matrix((len(ids), len(ids)))
    for (j, i), p in ids.items():
        for jj, ii, g in ((j, i+1, c[j, i+1]), (j, i-1, c[j, i]),
                          (j+1, i, a[j+1, i]), (j-1, i, a[j, i])):
            if (jj, ii) in ids:
                matrix[p, p] -= g
                matrix[p, ids[(jj, ii)]] += g
    matrix = matrix.tocsr()
    exact = np.sin(np.arange(len(ids))*.3)
    exact -= exact.mean()
    vector = matrix @ exact
    field.values[unknown] = vector
    result, diag = invert_Eliassen(field, ['y', 'x'], coords='cartesian',
        mParams={'A': field.copy(data=a), 'B': 0., 'C': field.copy(data=c)},
        iParams={'internal_bcs': 'neumann', 'architect': architect, 'dtype': 'float64',
                 'optArg': 1.3, 'mxLoop': 20000, 'tolerance': 1e-12,
                 'diag_interval': 50, 'printInfo': False})
    assert bool(diag.converged)
    np.testing.assert_allclose(result.values[unknown], exact, atol=1e-8)
    np.testing.assert_allclose(matrix @ result.values[unknown], vector, atol=1e-9)


def test_isolated_reactive_cell_keeps_nonzero_solution(architect):
    mask = np.zeros((5, 6), dtype=bool)
    mask[2, 2] = True
    field = xr.DataArray(np.where(mask, 1., np.nan), dims=('y', 'x'),
                         coords={'y': np.arange(5), 'x': np.arange(6)})
    result = invert_Fofonoff(field, ['y', 'x'], coords='cartesian',
                             mParams={'c0': .2, 'c1': .3, 'f0': 1., 'beta': 0.},
                             iParams={'internal_bcs': 'neumann', 'architect': architect,
                                      'dtype': 'float64', 'optArg': 1., 'printInfo': False})
    assert abs(float(result.values[2, 2])-3.5) < 1e-10


def test_z_lat_pv_coordinates_keep_physical_spacing(architect):
    app, field, params, unknown, matrix, rhs, expected = problem('pv', closed=True)
    spacing = 6371200*np.deg2rad(1.)
    field = field.assign_coords(y=np.arange(field.sizes['y'])*spacing)
    field = field.copy(data=field.values/spacing**2)
    result, diag = app(field, ['y', 'x'], coords='z-lat', mParams=params,
                       iParams={'internal_bcs': 'neumann', 'architect': architect,
                                'dtype': 'float64', 'optArg': 1.3, 'mxLoop': 20000,
                                'tolerance': 1e-12, 'diag_interval': 50, 'printInfo': False})
    assert bool(diag.converged)
    np.testing.assert_allclose(result.values[unknown], expected, atol=2e-8)
    np.testing.assert_allclose(matrix @ result.values[unknown], rhs, atol=1e-9)


def test_zero_reaction_and_nonzero_reaction_have_different_compatibility(architect):
    _, field, _, _, _, _, _ = problem('fofonoff')
    with pytest.raises(ValueError, match='incompatible forcing'):
        invert_Fofonoff(field, ['y', 'x'], coords='cartesian',
                        mParams={'c0': 0., 'c1': .3, 'f0': 1., 'beta': 0.},
                        iParams={'internal_bcs': 'neumann', 'architect': architect})
    result = invert_Fofonoff(field, ['y', 'x'], coords='cartesian',
                             mParams={'c0': .2, 'c1': .3, 'f0': 1., 'beta': 0.},
                             iParams={'internal_bcs': 'neumann', 'architect': architect,
                                      'dtype': 'float64', 'optArg': 1.3, 'printInfo': False})
    np.testing.assert_allclose(result.values[np.isfinite(field.values)], 3.5, atol=1e-5)


def test_reaction_nullspace_classification_is_per_time_slice(architect):
    _, field, _, unknown, matrix, _, _ = problem('pv', closed=True)
    # Reassemble an isotropic operator for Bretherton from independent faces.
    ids = {p: i for i, p in enumerate(zip(*np.where(unknown)))}
    matrix = lil_matrix((len(ids), len(ids)))
    for (j, i), p in ids.items():
        for jj, ii in ((j-1, i), (j+1, i), (j, i-1), (j, i+1)):
            if (jj, ii) in ids:
                matrix[p, p] -= 1
                matrix[p, ids[(jj, ii)]] += 1
    matrix = matrix.tocsr()
    exact = np.sin(np.arange(len(ids))*.3)
    exact -= exact.mean()
    first = field.copy()
    second = field.copy()
    first.values[unknown] = -.5*(matrix @ exact)
    second.values[unknown] = -.5*(matrix @ (exact+.7)-.2*(exact+.7))
    forcing = xr.concat([first, second], dim=xr.IndexVariable('time', [0, 1])).chunk({'time': 1})
    reaction = xr.DataArray([0., .2], dims='time', coords={'time': [0, 1]})
    result = invert_BrethertonHaidvogel(forcing, ['y', 'x'], coords='cartesian',
        mParams={'D': 1., 'f0': 2., 'beta': 0., 'lambda': reaction},
        iParams={'internal_bcs': 'neumann', 'architect': architect, 'dtype': 'float64',
                 'optArg': 1.3, 'mxLoop': 20000, 'tolerance': 1e-12, 'printInfo': False})
    assert hasattr(result.data, 'dask')
    result = result.compute()
    np.testing.assert_allclose(result.values[0][unknown], exact, atol=1e-8)
    np.testing.assert_allclose(result.values[1][unknown], exact+.7, atol=1e-8)
