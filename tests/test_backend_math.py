"""Shared discrete mathematical oracles, independent of backend ordering."""
import inspect

import numpy as np
import pytest

from xinvert import core, cpus

from .backend_utils import _case, _first, _second

KINDS = (
    'standard_1D', 'standard_2D', 'standard_2D_full', 'standard_3D',
    'general_2D', 'general_3D', 'general_bih_2D',
)


def _problem(kind):
    if kind not in ('standard_2D', 'standard_2D_full'):
        return _case(kind, False)
    truth, active, args = _case('general_2D', False)
    args.update({key: np.full_like(truth, value)
                 for key, value in zip('ABCDE', [1, 0.2, 1, 1, -0.5])})
    if kind == 'standard_2D':
        args['F'] = (_second(truth, 0) + _second(truth, 1)
                     + 0.4*_first(_first(truth, 0), 1))
    else:
        args['C'][:] = 0.1
        args['F'] = (_second(truth, 0) + _second(truth, 1)
                     + 0.3*_first(_first(truth, 0), 1) - 0.5*truth)
    return truth, active, args


def _operator(kind, field):
    """Apply the constant-coefficient discrete operator, not a SOR sweep."""
    if kind == 'standard_1D':
        return _second(field, 0) - 0.5*field
    if kind == 'standard_3D':
        return sum(_second(field, axis) for axis in range(3))
    if kind == 'general_3D':
        return (sum(_second(field, axis) for axis in range(3))
                + sum((axis+1)*0.1*_first(field, axis) for axis in range(3))
                - 0.5*field)
    mixed = _first(_first(field, 0), 1)
    lap = _second(field, 0) + _second(field, 1)
    if kind == 'standard_2D':
        return lap + 0.4*mixed
    if kind == 'standard_2D_full':
        return lap + 0.3*mixed - 0.5*field
    if kind == 'general_2D':
        return (lap + 0.2*mixed + 0.1*_first(field, 0)
                + 0.2*_first(field, 1) - 0.5*field)
    return (_second(_second(field, 0), 0)
            + 2*_first(_first(_first(_first(field, 0), 0), 1), 1)
            + _second(_second(field, 1), 1)
            - 0.2*_second(field, 0) + 0.1*mixed - 0.3*_second(field, 1)
            + 0.1*_first(field, 0) + 0.2*_first(field, 1) + 2*field)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_backend_discrete_solution_and_equation_residual(architect, kind, dtype):
    truth, active, values = _problem(kind)
    function = getattr(cpus, f'invert_{kind}')
    names = list(inspect.signature(function.py_func).parameters)
    coefficients = names[1:names.index('info')]
    grid_names = names[names.index('info')+1:names.index('optArg')]
    tolerance = 1e-6 if dtype == 'float32' else 1e-12
    accuracy = 1e-4 if dtype == 'float32' else 1e-8
    wrapper = core._make_kernel(
        function, [values[key] for key in grid_names],
        {'architect': architect, 'optArg': 1., 'mxLoop': 10000,
         'tolerance': tolerance, 'diag_interval': 10, 'printInfo': False})
    solution = truth.astype(dtype)
    solution[active] = 0
    solved, diagnostics = wrapper(
        solution, *[values[key].astype(dtype) for key in coefficients], '')
    assert diagnostics[0] == 0
    assert 0 < diagnostics[2] < 10000
    np.testing.assert_allclose(solved, truth, atol=accuracy, rtol=accuracy)
    np.testing.assert_array_equal(solved[~active], truth.astype(dtype)[~active])
    forcing = values[coefficients[-1]]
    residual = (_operator(kind, solved.astype('float64')) - forcing)[active]
    assert np.max(np.abs(residual)) < accuracy * max(1., np.max(np.abs(forcing)))


def test_variable_coefficient_general_2d(architect):
    """A nonuniform positive row scaling preserves the manufactured solution."""
    truth, active, values = _problem('general_2D')
    scale = 1. + np.arange(truth.size).reshape(truth.shape) / truth.size
    function = cpus.invert_general_2D
    names = list(inspect.signature(function.py_func).parameters)
    coefficients = names[1:names.index('info')]
    grid_names = names[names.index('info')+1:names.index('optArg')]
    wrapper = core._make_kernel(
        function, [values[key] for key in grid_names],
        {'architect': architect, 'optArg': 1., 'mxLoop': 10000,
         'tolerance': 1e-12, 'diag_interval': 10, 'printInfo': False})
    solution = truth.copy()
    solution[active] = 0
    solved, diagnostics = wrapper(
        solution, *[values[key]*scale for key in coefficients], '')
    assert diagnostics[0] == 0
    assert 0 < diagnostics[2] < 10000
    np.testing.assert_allclose(solved, truth, atol=1e-8, rtol=1e-8)
    residual = scale * (_operator('general_2D', solved) - values['G'])
    assert np.max(np.abs(residual[active])) < 1e-8
