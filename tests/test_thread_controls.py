"""Observe thread masks at execution, including lazy workers and failures."""

from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import pytest
import xarray as xr

from xinvert import core, cpus, invert_Poisson, masked_poisson
from xinvert._threads import DEFAULT_THREADS, numba_threads, validate_threads
from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import streamfunction_potential


@pytest.fixture(autouse=True)
def restore_threads():
    previous = numba.get_num_threads()
    yield
    numba.set_num_threads(previous)


@pytest.mark.parametrize('value', [0, -1, True, np.bool_(True), 1.5, '2'])
def test_invalid_thread_controls(value):
    with pytest.raises(ValueError, match='threads'):
        validate_threads(value)


def test_limit_numpy_integer_and_exception_restore():
    with pytest.raises(ValueError, match='threads'):
        validate_threads(numba.config.NUMBA_NUM_THREADS + 1)
    validate_threads(np.int64(1))
    numba.set_num_threads(1)
    requested = min(2, numba.config.NUMBA_NUM_THREADS)
    with pytest.raises(RuntimeError, match='injected'), numba_threads('cpu-mc', requested):
        assert numba.get_num_threads() == requested
        raise RuntimeError('injected')
    assert numba.get_num_threads() == 1


def forcing():
    f = xr.DataArray(
        np.sin(np.arange(80).reshape(8, 10)), dims=['y', 'x'],
        coords={'y': np.arange(8.), 'x': np.arange(10.)},
    )
    return f


@pytest.mark.parametrize('mode', ['default', 'explicit', 'external'])
@pytest.mark.parametrize('method', ['sor', 'neumann', 'li'])
def test_public_solver_uses_requested_mask_and_restores(monkeypatch, method, mode):
    external = min(4, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(external)
    options = {'architect': 'cpu-mc', 'mxLoop': 3, 'tolerance': 1e-6, 'printInfo': False}
    expected = external
    if mode == 'explicit':
        options['threads'] = 1
        expected = 1
    elif mode == 'external':
        options['threads'] = None
        expected = external
    observed = []

    def spy(original):
        def call(*args, **kwargs):
            observed.append(numba.get_num_threads())
            return original(*args, **kwargs)
        return call

    f = forcing()
    if method == 'li':
        monkeypatch.setattr(li, '_closed_forward', spy(li._closed_forward))
        options.update(BCs=['closed', 'closed'], on_nonconvergence='ignore')
        streamfunction_potential(f, f * .3, coords='cartesian', dims=['y', 'x'],
                                 iParams=options)
    else:
        if method == 'neumann':
            monkeypatch.setattr(masked_poisson, '_sor', spy(masked_poisson._sor))
            options['internal_bcs'] = 'neumann'
            f = f.where(~((f.y == 3) & (f.x == 4)))
        else:
            key = cpus.invert_standard_2D
            monkeypatch.setitem(core._cpu_mc_kernel_map, key,
                                spy(core._cpu_mc_kernel_map[key]))
        invert_Poisson(f, ['y', 'x'], coords='cartesian', iParams=options)
    assert observed and set(observed) == {expected}
    assert numba.get_num_threads() == external


def test_sor_exception_restores_mask(monkeypatch):
    numba.set_num_threads(1)

    def fail(*args, **kwargs):
        assert numba.get_num_threads() == 1
        raise RuntimeError('injected kernel failure')

    monkeypatch.setitem(core._cpu_mc_kernel_map, cpus.invert_standard_2D, fail)
    with pytest.raises(RuntimeError, match='injected kernel failure'):
        invert_Poisson(forcing(), ['y', 'x'], coords='cartesian',
                       iParams={'architect': 'cpu-mc', 'printInfo': False})
    assert numba.get_num_threads() == 1


def test_worker_default_and_external_are_local():
    def worker():
        numba.set_num_threads(1)
        with numba_threads('cpu-mc'):
            default = numba.get_num_threads()
        with numba_threads('cpu-mc', None):
            external = numba.get_num_threads()
        return default, external, numba.get_num_threads()

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(worker).result() == (1, 1, 1)


def test_dask_applies_mask_at_compute_and_preserves_results(monkeypatch):
    pytest.importorskip('dask.array')
    key = cpus.invert_standard_2D
    original = core._cpu_mc_kernel_map[key]
    observed = []

    def spy(*args, **kwargs):
        observed.append(numba.get_num_threads())
        return original(*args, **kwargs)

    monkeypatch.setitem(core._cpu_mc_kernel_map, key, spy)
    f = xr.concat([forcing(), forcing() * .5], dim='time')
    options = {'architect': 'cpu-mc', 'threads': 1, 'mxLoop': 20, 'tolerance': 1e-6, 'printInfo': False}
    eager = invert_Poisson(f, ['y', 'x'], coords='cartesian', iParams=options)
    observed.clear()
    lazy = invert_Poisson(f.chunk({'time': 1}), ['y', 'x'], coords='cartesian',
                          iParams=options)
    assert observed == []
    computed = lazy.compute(scheduler='threads', num_workers=1)
    assert observed and set(observed) == {1}
    xr.testing.assert_equal(eager, computed)


def test_default_is_inherit():
    from xinvert.apps import default_iParams

    assert DEFAULT_THREADS is None
    assert default_iParams['threads'] is None
