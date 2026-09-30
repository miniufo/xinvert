# -*- coding: utf-8 -*-
"""Public parameter defaults must not leak mutable state between calls."""
import inspect

import numpy as np
import xarray as xr

import xinvert.apps as apps
from xinvert import cal_flow, invert_Poisson


def _field():
    lat = np.linspace(-30.0, 30.0, 5)
    lon = np.linspace(0.0, 40.0, 6)
    values = np.sin(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    return xr.DataArray(
        values, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})


def test_public_mapping_defaults_are_none():
    public = [
        value for name, value in vars(apps).items()
        if callable(value) and (name.startswith('invert_') or
                                name in {'animate_iteration', 'cal_flow'})
    ]
    for func in public:
        for parameter in inspect.signature(func).parameters.values():
            assert not isinstance(parameter.default, (dict, list)), (
                f'{func.__name__}.{parameter.name} has a mutable default')


def test_user_parameter_mapping_is_not_modified():
    field = _field()
    params = {
        'BCs': ['fixed', 'fixed'],
        'mxLoop': 1,
        'tolerance': 0.0,
        'printInfo': False,
    }
    original = {key: (value.copy() if isinstance(value, list) else value)
                for key, value in params.items()}
    invert_Poisson(
        field, dims=['lat', 'lon'], coords='lat-lon', iParams=params)
    assert params == original


def test_cal_flow_vtype_is_case_insensitive():
    field = _field()
    upper = cal_flow(
        field, ['lat', 'lon'], vtype='GillMatsuno')
    lower = cal_flow(
        field, ['lat', 'lon'], vtype='gillmatsuno')
    for actual, expected in zip(upper, lower):
        xr.testing.assert_identical(actual, expected)
