"""Small public-application smoke checks, not substitutes for nonzero oracles."""
import numpy as np
import pytest
import xarray as xr

from xinvert import apps


@pytest.mark.parametrize('application,ndim,model', [
    ('invert_Poisson', 2, {}),
    ('invert_Eliassen', 2, {'A': 1., 'B': 0.1, 'C': 1.}),
    ('invert_PV2D', 2, {'N2': 0.01, 'f0': 0.1}),
    ('invert_BrethertonHaidvogel', 2, {'D': 100.}),
    ('invert_GillMatsuno', 2, {}),
    ('invert_GillMatsunoFlux', 2, {}),
    ('invert_Stommel', 2, {'D': 100.}),
    ('invert_StommelFlux', 2, {'D': 100.}),
    ('invert_StommelMunk', 2, {'A4': 1., 'D': 100.}),
    ('invert_StommelArons', 2, {}),
    ('invert_Fofonoff', 2, {'c0': 0.5, 'c1': 0.1, 'f0': 0.1, 'beta': 0.}),
    ('invert_omega', 3, {'N2': 0.01, 'f0': 0.1}),
    ('invert_3DOcean', 3, {'N2': 0.01, 'k': 1e-5, 'epsilon': 1e-4}),
])
def test_public_zero_forcing_dispatch(architect, application, ndim, model):
    dims = ['z', 'y', 'x'][-ndim:]
    forcing = xr.DataArray(
        np.zeros((9,)*ndim), dims=dims,
        coords={dim: np.linspace(1e3, 9e3, 9) for dim in dims})
    solution, diagnostics = getattr(apps, application)(
        forcing, dims=dims, coords='cartesian', mParams=model,
        iParams={'architect': architect, 'dtype': 'float64',
                 'BCs': ['fixed']*ndim, 'optArg': 1.,
                 'mxLoop': 20, 'tolerance': 1e-9,
                 'diag_interval': 5, 'printInfo': False})
    assert solution.dims == forcing.dims
    assert solution.shape == forcing.shape
    np.testing.assert_array_equal(solution.values, forcing.values)
    assert bool(diagnostics.converged.item())
    assert diagnostics.stop_reason.item() == 'converged'


def test_geoadjustment_constant_height(architect):
    """PV0=f admits height=1: the variable-coefficient flux derivative is zero."""
    lat = np.linspace(20., 60., 9)
    pv = xr.DataArray(2*7.292e-5*np.sin(np.deg2rad(lat)),
                      dims=['lat'], coords={'lat': lat})
    initial = xr.zeros_like(pv)
    initial[0] = initial[-1] = 1.
    solution, diagnostics = apps.invert_GeoAdjustment(
        pv, dims=['lat'], coords='lat', icbc=initial,
        iParams={'architect': architect, 'dtype': 'float64',
                 'BCs': ['fixed'], 'optArg': 1., 'mxLoop': 5000,
                 'tolerance': 1e-12, 'diag_interval': 10, 'printInfo': False})
    assert bool(diagnostics.converged.item())
    np.testing.assert_allclose(solution.values, 1., atol=1e-8, rtol=1e-8)
