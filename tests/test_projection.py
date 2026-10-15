"""Independent face-operator and decomposition tests for closed projection."""
import numpy as np
import pytest
import xarray as xr

from xinvert.projection import project_vector


def test_example_main_program(capsys):
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).parents[1] / 'examples' / 'projection_closed_basin.py'),
                   run_name='__main__')
    assert 'Max remaining divergence:' in capsys.readouterr().out


def fields():
    """Return anisotropic grid, island mask, and manufactured potential faces."""
    y, x = np.arange(9.)*2, np.arange(12.)*3
    wet = xr.DataArray(np.ones((9, 12), dtype=bool), dims=['y', 'x'],
                       coords={'y': y, 'x': x})
    wet[[0, -1], :] = False
    wet[:, [0, -1]] = False
    wet[4, 5] = False
    phi = xr.DataArray(np.sin(y[:, None]/5) + np.cos(x[None, :]/7),
                       dims=wet.dims, coords=wet.coords)
    east = wet & wet.shift(x=-1, fill_value=False)
    north = wet & wet.shift(y=-1, fill_value=False)
    fx = ((phi.shift(x=-1)-phi)/3).where(east, 0)
    fy = ((phi.shift(y=-1)-phi)/2).where(north, 0)
    return wet, phi, fx, fy


def test_manufactured_gradient_projection(architect):
    wet, phi, fx, fy = fields()
    result, diagnostic = project_vector(
        fx, fy, wet, ['y', 'x'],
        iParams={'architect': architect, 'dtype': 'float64', 'diag_interval': 20,
                 'mxLoop': 20000, 'tolerance': 1e-12, 'optArg': 1.5, 'printInfo': False})
    expected = phi.where(wet)-float(phi.where(wet).mean())
    np.testing.assert_allclose(result.potential.values[wet], expected.values[wet], atol=2e-9)
    assert float(abs(result.remaining_divergence).max()) < 2e-10
    assert float(abs(result.x_solenoidal).max()) < 2e-9
    assert float(abs(result.y_solenoidal).max()) < 2e-9
    xr.testing.assert_allclose((result.x_gradient+result.x_solenoidal).fillna(0), fx)
    xr.testing.assert_allclose((result.y_gradient+result.y_solenoidal).fillna(0), fy)
    assert diagnostic.error.size > 0


def test_arbitrary_field_operator_identity(architect):
    wet, _, fx, fy = fields()
    rng = np.random.default_rng(12)
    fx = xr.where(fx != 0, xr.DataArray(rng.normal(size=wet.shape), dims=wet.dims), 0)
    fy = xr.where(fy != 0, xr.DataArray(rng.normal(size=wet.shape), dims=wet.dims), 0)
    result = project_vector(fx, fy, wet, ['y', 'x'],
                            iParams={'architect': architect, 'mxLoop': 30000,
                                     'tolerance': 1e-12, 'optArg': 1.5, 'printInfo': False})
    p, mask = result.potential.values, wet.values
    # Independent loop, not the module's divergence/gradient helpers.
    for j, i in zip(*np.where(mask)):
        laplace = 0.
        for dj, di, spacing in [(1, 0, 2), (-1, 0, 2), (0, 1, 3), (0, -1, 3)]:
            if mask[j+dj, i+di]:
                laplace += (p[j+dj, i+di]-p[j, i])/spacing**2
        assert abs(laplace-result.divergence.values[j, i]) < 2e-10
    assert float(abs(result.remaining_divergence).max()) < 2e-10


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_nonzero_coastal_flux_rejected(axis):
    wet, _, fx, fy = fields()
    if axis == 'x':
        fx[4, 4] = 1.
    else:
        fy[3, 5] = 1.
    with pytest.raises(ValueError, match='nonzero coastal'):
        project_vector(fx, fy, wet, ['y', 'x'])


def test_collocated_domain_and_conflicting_controls_rejected():
    wet, _, fx, fy = fields()
    with pytest.raises(ValueError, match='internal Neumann'):
        project_vector(fx, fy, wet, ['y', 'x'], iParams={'internal_bcs': 'fixed'})
    wet[0, 2] = True
    with pytest.raises(ValueError, match='inactive outer ring'):
        project_vector(fx, fy, wet, ['y', 'x'])


def test_float32_storage_with_float64_projection():
    wet, _, fx, fy = fields()
    rng = np.random.default_rng(42)
    fx = fx.where(fx == 0, xr.DataArray(rng.normal(size=wet.shape), dims=wet.dims)).astype('float32')
    fy = fy.where(fy == 0, xr.DataArray(rng.normal(size=wet.shape), dims=wet.dims)).astype('float32')
    result = project_vector(fx, fy, wet, ['y', 'x'],
                            iParams={'mxLoop': 20000, 'optArg': 1.6,
                                     'tolerance': 1e-12, 'printInfo': False})
    assert float(abs(result.remaining_divergence).max()) < 2e-10


def test_projection_accepts_gauge_shifted_warm_start(architect):
    wet, phi, fx, fy = fields()
    result, diagnostic = project_vector(
        fx, fy, wet, ['y', 'x'], icbc=phi.where(wet)+123.,
        iParams={'architect': architect, 'mxLoop': 1, 'diag_interval': 1,
                 'tolerance': 0., 'printInfo': False})
    expected = phi.where(wet)-phi.where(wet).mean()
    np.testing.assert_allclose(result.potential.values[wet], expected.values[wet], atol=1e-11)
    assert float(abs(result.remaining_divergence).max()) < 1e-11
    assert int(diagnostic.iterations) == 1


def spherical_fields(periodic):
    """Construct physical face gradients without using projection helpers."""
    lat = np.linspace(20., 50., 9)
    lon = np.arange(12.)*30 if periodic else np.linspace(100., 120., 12)
    wet = xr.DataArray(np.ones((9, 12), dtype=bool), dims=['lat', 'lon'],
                       coords={'lat': lat, 'lon': lon})
    wet[[0, -1], :] = False
    if not periodic:
        wet[:, [0, -1]] = False
    wet[4, 5] = False
    phi = xr.DataArray(np.sin(np.deg2rad(lat))[:, None] +
                       np.cos(np.deg2rad(lon))[None, :], dims=wet.dims, coords=wet.coords)
    right_wet = wet.roll(lon=-1, roll_coords=False) if periodic else wet.shift(lon=-1, fill_value=False)
    right_phi = phi.roll(lon=-1, roll_coords=False) if periodic else phi.shift(lon=-1)
    radius = 10.
    dx, dy = radius*np.deg2rad(lon[1]-lon[0]), radius*np.deg2rad(lat[1]-lat[0])
    fx = ((right_phi-phi)/(dx*np.cos(np.deg2rad(wet.lat)))).where(wet & right_wet, 0)
    fy = ((phi.shift(lat=-1)-phi)/dy).where(wet & wet.shift(lat=-1, fill_value=False), 0)
    return wet, phi, fx, fy, radius, dx, dy


@pytest.mark.parametrize('periodic', [False, True])
def test_spherical_manufactured_projection(architect, periodic):
    wet, phi, fx, fy, radius, dx, dy = spherical_fields(periodic)
    result = project_vector(fx, fy, wet, ['lat', 'lon'], coords='lat-lon',
                            periodic_x=periodic, mParams={'Rearth': radius},
                            iParams={'architect': architect, 'tolerance': 1e-12,
                                     'mxLoop': 50000, 'optArg': 1.6, 'printInfo': False})
    weight = np.cos(np.deg2rad(wet.lat))
    expected = phi.where(wet)-phi.where(wet).weighted(weight).mean()
    np.testing.assert_allclose(result.potential.values[wet], expected.values[wet], atol=2e-8)
    assert float(abs(result.remaining_divergence).max()) < 3e-9
    assert float(abs(result.x_solenoidal).max()) < 3e-8
    p, mask = result.potential.values, wet.values
    lat = np.deg2rad(wet.lat.values)
    # Independent spherical stencil, including seam edges and a coastal hole.
    for j, i in zip(*np.where(mask)):
        laplace = 0.
        for dj, di in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            jj, ii = j+dj, (i+di) % mask.shape[1]
            if mask[jj, ii]:
                coefficient = (np.cos((lat[j]+lat[jj])/2)/(dy**2*np.cos(lat[j]))
                               if dj else 1/(dx**2*np.cos(lat[j])**2))
                laplace += coefficient*(p[jj, ii]-p[j, i])
        assert abs(laplace-result.divergence.values[j, i]) < 3e-9


@pytest.mark.parametrize('fault', ['pole', 'duplicate', 'regional', 'odd'])
def test_spherical_periodic_grid_validation(fault):
    wet, _, fx, fy, _, _, _ = spherical_fields(True)
    if fault == 'pole':
        latitude = wet.lat.values.copy()
        latitude[-1] = 90.
        # All rows must remain uniform for the pole-specific validation.
        latitude = np.linspace(10., 90., 9)
        wet, fx, fy = (a.assign_coords(lat=latitude) for a in (wet, fx, fy))
        message = 'pole'
    elif fault in ('duplicate', 'regional'):
        longitude = np.linspace(0., 360. if fault == 'duplicate' else 120., 12)
        wet, fx, fy = (a.assign_coords(lon=longitude) for a in (wet, fx, fy))
        message = '360 degrees'
    else:
        wet, fx, fy = (a.isel(lon=slice(0, 11)).assign_coords(lon=np.arange(11.)*360/11)
                       for a in (wet, fx, fy))
        # Avoid an artificial nonzero coast when changing the seam connectivity.
        fx = fx.where(wet & wet.roll(lon=-1, roll_coords=False), 0)
        message = 'even periodic'
    with pytest.raises(ValueError, match=message):
        project_vector(fx, fy, wet, ['lat', 'lon'], coords='lat-lon', periodic_x=True,
                       iParams={'architect': 'cpu-mc', 'printInfo': False})
