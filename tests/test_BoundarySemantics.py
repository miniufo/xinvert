# -*- coding: utf-8 -*-
"""Regression tests for the mathematical semantics of scalar BCs."""
import numpy as np
import pytest

from xinvert import cpus

from tests.gpu_utils import gpu_available


UNDEF = -9.99e8


def _field_2d(ny=6, nx=7):
    return np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)


def _expected_2d(source, bcy, bcx):
    out = source.copy()
    ny, nx = out.shape
    if bcy == 'extend':
        cols = range(nx) if bcx == 'periodic' else range(1, nx - 1)
        for i in cols:
            out[0, i] = source[1, i]
            out[-1, i] = source[-2, i]
    if bcx == 'extend':
        out[1:-1, 0] = source[1:-1, 1]
        out[1:-1, -1] = source[1:-1, -2]
    if bcy == 'extend' and bcx == 'extend':
        out[0, 0] = source[1, 1]
        out[0, -1] = source[1, -2]
        out[-1, 0] = source[-2, 1]
        out[-1, -1] = source[-2, -2]
    return out


@pytest.mark.parametrize(
    'bcy,bcx',
    [('fixed', 'extend'), ('extend', 'fixed'),
     ('extend', 'extend'), ('extend', 'periodic')],
)
def test_cpu_2d_boundary_dimensions_are_independent(bcy, bcx):
    source = _field_2d()
    actual = source.copy()
    cpus._apply_extend_boundary_2d(actual, bcy, bcx, UNDEF, -1, -1)
    np.testing.assert_array_equal(actual, _expected_2d(source, bcy, bcx))


def test_cpu_3d_boundary_dimensions_are_independent():
    source = np.arange(4 * 6 * 7, dtype=np.float64).reshape(4, 6, 7)
    actual = source.copy()
    cpus._apply_extend_boundary_3d(actual, 'fixed', 'extend', UNDEF)

    expected = source.copy()
    expected[1:-1, 1:-1, 0] = source[1:-1, 1:-1, 1]
    expected[1:-1, 1:-1, -1] = source[1:-1, 1:-1, -2]
    np.testing.assert_array_equal(actual, expected)


def test_cpu_biharmonic_fixed_y_preserves_two_boundary_rows():
    source = _field_2d(8, 9)
    actual = source.copy()
    cpus._apply_extend_boundary_bih_2d(actual, 'fixed', 'extend', UNDEF)

    expected = source.copy()
    expected[2:-2, 0] = source[2:-2, 2]
    expected[2:-2, 1] = source[2:-2, 2]
    expected[2:-2, -1] = source[2:-2, -3]
    expected[2:-2, -2] = source[2:-2, -3]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not gpu_available(), reason='CUDA GPU is unavailable')
@pytest.mark.parametrize(
    'bcy,bcx',
    [('fixed', 'extend'), ('extend', 'fixed'),
     ('extend', 'extend'), ('extend', 'periodic')],
)
def test_gpu_2d_boundary_dimensions_are_independent(bcy, bcx):
    from numba import cuda
    from xinvert import gpus

    source = _field_2d()
    device = cuda.to_device(source)
    ny, nx = source.shape

    if bcy == 'extend':
        gpus._extend_y_boundary[1, 32](
            device, ny, nx, UNDEF, bcx == 'periodic', -1, -1)
    if bcx == 'extend':
        gpus._extend_x_boundary[1, 32](
            device, ny, nx, UNDEF, bcy == 'extend', -1, -1)

    np.testing.assert_array_equal(
        device.copy_to_host(), _expected_2d(source, bcy, bcx))
