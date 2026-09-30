# -*- coding: utf-8 -*-
"""Clear validation errors for public inversion inputs."""
import numpy as np
import pytest
import xarray as xr
import warnings

import xinvert.apps as apps
from xinvert.core import _validate_inversion_dims, _validate_solver_controls


def _forcing():
    return xr.DataArray(
        np.zeros((3, 4, 5)),
        dims=['time', 'lat', 'lon'],
    )


def test_dimension_validation_accepts_tuple_and_1d_string():
    forcing = _forcing()
    assert _validate_inversion_dims(forcing, ('lat', 'lon'), 2) == [
        'lat', 'lon']
    assert _validate_inversion_dims(forcing, 'lat', 1) == ['lat']


@pytest.mark.parametrize(
    'dims,message',
    [
        ('lat', 'not one string'),
        (['lat'], '2 dimensions'),
        (['lat', 'lat'], 'duplicate'),
        (['lat', 'depth'], 'not present'),
        (['lat', 1], 'must be a string'),
    ],
)
def test_invalid_dimensions_have_actionable_errors(dims, message):
    with pytest.raises(ValueError, match=message):
        _validate_inversion_dims(_forcing(), dims, 2)


def test_forcing_must_be_dataarray():
    with pytest.raises(ValueError, match='xarray.DataArray'):
        _validate_inversion_dims(np.zeros((3, 4)), ['lat', 'lon'], 2)


@pytest.mark.parametrize('mx_loop', [0, -1, 1.5, True, '10'])
def test_mxloop_must_be_a_positive_integer(mx_loop):
    params = {'mxLoop': mx_loop, 'tolerance': 0.0, 'undef': np.nan}
    with pytest.raises(ValueError, match='positive integer'):
        _validate_solver_controls(params)


@pytest.mark.parametrize('tolerance', [np.nan, np.inf, True, '1e-8'])
def test_tolerance_must_be_finite_real(tolerance):
    params = {'mxLoop': 1, 'tolerance': tolerance, 'undef': np.nan}
    with pytest.raises(ValueError, match='finite real'):
        _validate_solver_controls(params)


def test_nonpositive_tolerance_remains_supported_to_disable_early_stop():
    _validate_solver_controls(
        {'mxLoop': np.int64(2), 'tolerance': -1e-8, 'undef': np.nan})


@pytest.mark.parametrize('undef', [True, 'nan', 1 + 2j])
def test_undef_must_be_a_real_scalar(undef):
    params = {'mxLoop': 1, 'tolerance': 0.0, 'undef': undef}
    with pytest.raises(ValueError, match='real scalar'):
        _validate_solver_controls(params)


def _mask_params():
    return {'BCs': ['fixed', 'fixed'], 'dtype': np.float64, 'undef': np.nan}


def test_spatial_icbc_broadcasts_over_noncore_dimensions():
    forcing = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=['time', 'lat', 'lon'],
        coords={'time': [0, 1], 'lat': [10, 20, 30], 'lon': [1, 2, 3, 4]},
    )
    icbc = xr.DataArray(
        np.ones((3, 4)),
        dims=['lat', 'lon'],
        coords={'lat': forcing.lat, 'lon': forcing.lon},
    )

    _, initialized, _ = apps.__mask_FS(
        forcing, ['lat', 'lon'], _mask_params(), icbc)
    assert initialized.dims == forcing.dims
    np.testing.assert_array_equal(initialized[:, 0, :], 1.0)
    np.testing.assert_array_equal(initialized[:, -1, :], 1.0)
    np.testing.assert_array_equal(initialized[:, 1, 1:-1], 1.0)


@pytest.mark.parametrize('with_icbc', [False, True])
def test_invalid_forcing_cells_ignore_icbc_values(with_icbc):
    forcing = xr.DataArray(
        [[0.0, np.nan, 0.0], [0.0, 0.0, 0.0]],
        dims=['lat', 'lon'])
    icbc = xr.full_like(forcing, 7.0) if with_icbc else None

    maskf, initialized, _ = apps.__mask_FS(
        forcing, ['lat', 'lon'], _mask_params(), icbc)

    assert maskf[0, 1] == apps._undeftmp
    assert initialized[0, 1] == 0.0
    if with_icbc:
        assert initialized[1, 1] == 7.0
    else:
        assert initialized[1, 1] == 0.0


def test_invalid_forcing_cells_are_restored_after_solve_with_icbc():
    forcing = xr.DataArray(
        np.zeros((5, 5)), dims=['y', 'x'],
        coords={'y': np.arange(5.0), 'x': np.arange(5.0)})
    forcing[2, 2] = np.nan
    icbc = xr.full_like(forcing, 7.0)

    result = apps.invert_Poisson(
        forcing, ['y', 'x'], coords='cartesian', icbc=icbc,
        iParams={
            'BCs': ['fixed', 'fixed'],
            'dtype': 'float64',
            'mxLoop': 1,
            'tolerance': 0.0,
            'printInfo': False,
        },
    )

    assert np.isnan(result[2, 2])


def test_icbc_requires_all_inversion_dimensions():
    forcing = _forcing()
    icbc = xr.DataArray(np.zeros(4), dims=['lat'])
    with pytest.raises(ValueError, match='missing inversion dimensions'):
        apps.__mask_FS(
            forcing, ['lat', 'lon'], _mask_params(), icbc)


def test_icbc_rejects_mismatched_coordinates():
    forcing = xr.DataArray(
        np.zeros((3, 4)), dims=['lat', 'lon'],
        coords={'lat': [10, 20, 30], 'lon': [1, 2, 3, 4]})
    icbc = xr.DataArray(
        np.zeros((3, 4)), dims=['lat', 'lon'],
        coords={'lat': [10, 25, 30], 'lon': [1, 2, 3, 4]})
    with pytest.raises(ValueError, match='coordinates must exactly match'):
        apps.__mask_FS(
            forcing, ['lat', 'lon'], _mask_params(), icbc)


def test_icbc_must_be_dataarray():
    with pytest.raises(ValueError, match='xarray.DataArray'):
        apps.__mask_FS(
            _forcing(), ['lat', 'lon'], _mask_params(), 0.0)


def _flow_scalar():
    return xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=['y', 'x'],
        coords={'y': np.arange(3.0), 'x': np.arange(4.0)},
    )


@pytest.mark.parametrize(
    'bcs,message',
    [
        ({'Y': 'fixed', 'X': 'periodic'}, 'list or tuple'),
        (['fixed'], 'has 1 entries'),
        (['fixed', {'E': 'fixed', 'W': 'extend'}], 'FiniteDiff only'),
        (['fixed', ('fixed', 'extend')], 'FiniteDiff only'),
        (['fixed', 'unknown'], 'expected one of'),
    ],
)
def test_cal_flow_rejects_non_scalar_or_invalid_bcs(bcs, message):
    with pytest.raises(ValueError, match=message):
        apps.cal_flow(_flow_scalar(), ['y', 'x'], coords='cartesian', BCs=bcs)


def test_cal_flow_normalizes_bcs_without_sor_pending_warning():
    scalar = _flow_scalar()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = apps.cal_flow(
            scalar, ['y', 'x'], coords=' cartesian ',
            BCs=[' Reflect ', ' PERIODIC '])
    expected = apps.cal_flow(
        scalar, ['y', 'x'], coords='cartesian',
        BCs=['reflect', 'periodic'])
    assert not caught
    xr.testing.assert_allclose(actual[0], expected[0])
    xr.testing.assert_allclose(actual[1], expected[1])


@pytest.mark.parametrize(
    'dims,message',
    [
        ('y', 'not one string'),
        (['y'], '2 dimensions'),
        (['y', 'y'], 'duplicate'),
        (['y', 'z'], 'not present'),
    ],
)
def test_cal_flow_validates_dimensions(dims, message):
    with pytest.raises(ValueError, match=message):
        apps.cal_flow(_flow_scalar(), dims, coords='cartesian')


@pytest.mark.parametrize('name,value', [('vtype', None), ('coords', None)])
def test_cal_flow_requires_string_modes(name, value):
    kwargs = {name: value}
    with pytest.raises(ValueError, match=f'{name} must be a string'):
        apps.cal_flow(_flow_scalar(), ['y', 'x'], **kwargs)
