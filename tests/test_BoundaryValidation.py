# -*- coding: utf-8 -*-
"""Validation and compatibility warnings for inversion BC specifications."""
import warnings

import pytest

from xinvert.apps import _validate_inversion_bcs


def _validate(bcs, ndim):
    params = {'BCs': bcs}
    _validate_inversion_bcs(params, ndim)
    return params['BCs']


def test_inversion_bcs_require_one_scalar_string_per_dimension():
    with pytest.raises(ValueError, match='FiniteDiff only'):
        _validate(['fixed', {'E': 'fixed', 'W': 'extend'}], 2)

    with pytest.raises(ValueError, match='has 1 entries'):
        _validate(['fixed'], 2)

    with pytest.raises(ValueError, match='must be a string'):
        _validate(['fixed', ('extend', 'fixed')], 2)


def test_inversion_bcs_reject_unknown_names():
    with pytest.raises(ValueError, match='is invalid'):
        _validate(['fixed', 'open'], 2)


def test_inversion_bcs_are_normalized():
    assert _validate((' EXTEND ', 'Periodic'), 2) == ['extend', 'periodic']


def test_unimplemented_periodic_y_emits_warning_but_remains_allowed():
    with pytest.warns(RuntimeWarning, match='periodic.*y dimension'):
        result = _validate(['periodic', 'fixed'], 2)
    assert result == ['periodic', 'fixed']


def test_revalidation_does_not_duplicate_warning():
    params = {'BCs': ['periodic', 'fixed']}
    with pytest.warns(RuntimeWarning) as captured:
        _validate_inversion_bcs(params, 2)
        _validate_inversion_bcs(params, 2)
    assert len(captured) == 1


@pytest.mark.parametrize('bcz', ['extend', 'periodic'])
def test_unimplemented_z_boundary_emits_warning_but_remains_allowed(bcz):
    with pytest.warns(RuntimeWarning, match='z dimension'):
        result = _validate([bcz, 'extend', 'periodic'], 3)
    assert result == [bcz, 'extend', 'periodic']


def test_implemented_combinations_do_not_warn():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result = _validate(['fixed', 'extend', 'periodic'], 3)
    assert not captured
    assert result == ['fixed', 'extend', 'periodic']
