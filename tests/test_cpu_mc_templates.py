"""Discrete manufactured-solution tests for all CPU-mc templates."""
import inspect

import numba
import numpy as np
import pytest

from xinvert import core, cpu_mc, cpus

from .backend_utils import _case


@pytest.mark.parametrize('kind', [
    'standard_1D', 'standard_3D', 'general_2D', 'general_3D', 'general_bih_2D'])
@pytest.mark.parametrize('periodic', [False, True])
def test_remaining_templates_match_discrete_solution_and_threads(kind, periodic):
    truth, active, args = _case(kind, periodic)
    kernel = getattr(cpu_mc, f'invert_{kind}_mc')
    names = inspect.signature(kernel.py_func).parameters
    original = numba.get_num_threads()
    outputs = []
    try:
        for threads in (1, min(4, original)):
            numba.set_num_threads(threads)
            solution = truth.copy()
            solution[active] = 0
            flags = np.zeros(3)
            values = {**args, 'S': solution, 'flags': flags}
            kernel(**{key: values[key] for key in names})
            assert flags[0] == 0
            assert flags[2] < args['mxLoop']
            np.testing.assert_allclose(solution, truth, atol=2e-9, rtol=2e-9)
            outputs.append(solution)
    finally:
        numba.set_num_threads(original)
    np.testing.assert_array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize('kind', [
    'standard_1D', 'standard_3D', 'general_2D', 'general_3D', 'general_bih_2D'])
def test_remaining_templates_dispatch_and_diagnostics(kind):
    truth, active, values = _case(kind, False)
    function = getattr(cpus, f'invert_{kind}')
    names = list(inspect.signature(function.py_func).parameters)
    coefficients = names[1:names.index('info')]
    grid_names = names[names.index('info')+1:names.index('optArg')]
    wrapper = core._make_kernel(
        function, [values[key] for key in grid_names],
        {'architect': 'cpu-mc', 'optArg': 1., 'mxLoop': 10000,
         'tolerance': 1e-12, 'diag_interval': 10, 'printInfo': False})
    solution = truth.copy()
    solution[active] = 0
    solved, diagnostics = wrapper(
        solution, *[values[key] for key in coefficients], '')
    np.testing.assert_allclose(solved, truth, atol=2e-9, rtol=2e-9)
    assert diagnostics[0] == 0
    assert 0 < diagnostics[2] < 10000
    assert np.isfinite(diagnostics[3])


@pytest.mark.parametrize('kind', [
    'standard_1D', 'standard_3D', 'general_2D', 'general_3D', 'general_bih_2D'])
def test_remaining_templates_preserve_mask_and_fixed_boundaries(kind):
    truth, active, args = _case(kind, False)
    kernel = getattr(cpu_mc, f'invert_{kind}_mc')
    names = inspect.signature(kernel.py_func).parameters
    rhs = 'J' if 'bih' in kind else 'H' if kind == 'general_3D' else (
        'G' if kind == 'general_2D' else 'F')
    masked = tuple(n//2 for n in truth.shape)
    args[rhs][masked] = args['undef']
    solution = truth.copy()
    flags = np.zeros(3)
    args.update({'S': solution, 'flags': flags, 'mxLoop': 10, 'tolerance': 0.})
    before = solution.copy()
    kernel(**{key: args[key] for key in names})
    assert solution[masked] == before[masked]
    np.testing.assert_array_equal(solution[~active], before[~active])


def test_all_elliptic_cpu_templates_have_multicolor_dispatch():
    expected = {function for name, function in vars(cpus).items()
                if name.startswith('invert_')}
    assert set(core._cpu_mc_kernel_map) == expected


@pytest.mark.parametrize('kind', [
    'standard_1D', 'standard_3D', 'general_2D', 'general_3D', 'general_bih_2D'])
def test_remaining_templates_extend_boundaries(kind):
    truth, active, args = _case(kind, False)
    truth[:] = 1
    rhs = 'J' if 'bih' in kind else 'H' if kind == 'general_3D' else (
        'G' if kind == 'general_2D' else 'F')
    linear = 'I' if 'bih' in kind else 'G' if kind == 'general_3D' else (
        'F' if kind == 'general_2D' else 'B' if kind == 'standard_1D' else None)
    args[rhs][:] = args[linear] if linear else 0
    args['BCx'] = 'extend'
    args['BCy'] = 'extend'
    solution = truth.copy()
    solution[active] = 0
    args.update({'S': solution, 'flags': np.zeros(3)})
    kernel = getattr(cpu_mc, f'invert_{kind}_mc')
    kernel(**{key: args[key] for key in inspect.signature(kernel.py_func).parameters})
    assert args['flags'][0] == 0
    assert args['flags'][2] < args['mxLoop']
    np.testing.assert_allclose(solution, truth, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize('kind', [
    'standard_1D', 'standard_3D', 'general_2D', 'general_3D', 'general_bih_2D'])
def test_remaining_templates_float32(kind):
    truth, active, args = _case(kind, True)
    for key, value in args.items():
        if isinstance(value, np.ndarray):
            args[key] = value.astype(np.float32)
    solution = truth.astype(np.float32)
    solution[active] = 0
    args.update({'S': solution, 'flags': np.zeros(3), 'tolerance': 1e-6})
    kernel = getattr(cpu_mc, f'invert_{kind}_mc')
    kernel(**{key: args[key] for key in inspect.signature(kernel.py_func).parameters})
    assert args['flags'][0] == 0
    assert args['flags'][2] < args['mxLoop']
    np.testing.assert_allclose(solution, truth, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('kind', [
    'standard_1D', 'standard_3D', 'general_2D', 'general_3D', 'general_bih_2D'])
def test_remaining_templates_reject_incompatible_periodic_coloring(kind):
    truth, _, args = _case(kind, True)
    args['xc'] -= 1
    args.update({'S': truth, 'flags': np.zeros(3)})
    kernel = getattr(cpu_mc, f'invert_{kind}_mc')
    with pytest.raises(ValueError, match='divisible by'):
        kernel(**{key: args[key] for key in inspect.signature(kernel.py_func).parameters})
