"""Independent discrete manufactured problems shared by backend tests."""
import numpy as np


def _first(field, axis):
    return (np.roll(field, -1, axis) - np.roll(field, 1, axis)) / 2


def _second(field, axis):
    return np.roll(field, -1, axis) - 2*field + np.roll(field, 1, axis)


def _case(kind, periodic):
    ndim = 1 if kind == 'standard_1D' else 3 if kind.endswith('3D') else 2
    shape = (9, 10, 12)[-ndim:]
    truth = np.ones(shape)
    for axis, n in enumerate(shape):
        coordinate = np.arange(n)
        values = np.cos(2*np.pi*coordinate/n)
        reshape = [1]*ndim
        reshape[axis] = n
        truth *= values.reshape(reshape)
    radius = 2 if 'bih' in kind else 1
    active = np.ones(shape, dtype=bool)
    for axis in range(ndim):
        if axis == ndim-1 and periodic:
            continue
        index = [slice(None)]*ndim
        index[axis] = slice(0, radius)
        active[tuple(index)] = False
        index[axis] = slice(-radius, None)
        active[tuple(index)] = False
    coeff = {}
    if kind == 'standard_1D':
        coeff = {'A': 1, 'B': -0.5}
        forcing = _second(truth, 0) - 0.5*truth
        rhs = 'F'
    elif kind == 'standard_3D':
        coeff = {'A': 1, 'B': 1, 'C': 1}
        forcing = sum(_second(truth, axis) for axis in range(3))
        rhs = 'F'
    elif kind == 'general_3D':
        coeff = dict(zip('ABCDEFG', [1, 1, 1, 0.1, 0.2, 0.3, -0.5]))
        forcing = (sum(_second(truth, axis) for axis in range(3))
                   + sum((axis+1)*0.1*_first(truth, axis) for axis in range(3))
                   - 0.5*truth)
        rhs = 'H'
    elif kind == 'general_2D':
        coeff = dict(zip('ABCDEF', [1, 0.2, 1, 0.1, 0.2, -0.5]))
        forcing = (_second(truth, 0) + _second(truth, 1)
                   + 0.2*_first(_first(truth, 0), 1)
                   + 0.1*_first(truth, 0) + 0.2*_first(truth, 1)
                   - 0.5*truth)
        rhs = 'G'
    else:
        coeff = dict(zip('ABCDEFGHI', [1, 2, 1, -0.2, 0.1, -0.3, 0.1, 0.2, 2]))
        forcing = (_second(_second(truth, 0), 0)
                   + 2*_first(_first(_first(_first(truth, 0), 0), 1), 1)
                   + _second(_second(truth, 1), 1)
                   - 0.2*_second(truth, 0)
                   + 0.1*_first(_first(truth, 0), 1)
                   - 0.3*_second(truth, 1)
                   + 0.1*_first(truth, 0) + 0.2*_first(truth, 1) + 2*truth)
        rhs = 'J'
    args = {key: np.full(shape, value, dtype=float) for key, value in coeff.items()}
    args[rhs] = forcing
    args.update({
        'info': '', 'xc': shape[-1], 'yc': shape[-2] if ndim > 1 else 1,
        'zc': shape[0] if ndim == 3 else 1,
        'BCx': 'periodic' if periodic else 'fixed', 'BCy': 'fixed',
        'BCz': 'fixed', 'delx': 1., 'delxSqr': 1., 'delxTr': 1.,
        'delxSSr': 1., 'ratio': 1., 'ratioQtr': 0.25, 'ratioSqr': 1.,
        'ratioSSr': 1., 'ratio1': 1., 'ratio2': 1.,
        'ratio1Sqr': 1., 'ratio2Sqr': 1., 'optArg': 1.,
        'undef': -9.99e8, 'mxLoop': 10000, 'tolerance': 1e-12,
        'check_interval': 10,
    })
    return truth, active, args
