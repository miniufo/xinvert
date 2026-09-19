# -*- coding: utf-8 -*-
"""Regression: invert_Poisson works under a dask.distributed Client.

Guards against the serialisation regression: with a distributed Client
the task graph (including the ``_kernel_`` closure) is pickled and sent
to worker processes; referencing unpicklable objects (e.g. a
``threading.Lock``) directly from the closure breaks every distributed
run.  Also verifies the result matches the analytic solution.

Run:  pytest tests/test_distributed.py -v
"""
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

np = pytest.importorskip('numpy')
xr = pytest.importorskip('xarray')
dask_dist = pytest.importorskip('dask.distributed')


def _make(nt, n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    vor = np.stack([-2.0 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)] * nt)
    da = xr.DataArray(vor, dims=['time', 'y', 'x'],
                      coords={'time': np.arange(nt), 'y': y, 'x': x}
                      ).chunk({'time': 1})
    return da


@pytest.mark.slow
def test_distributed_serialization():
    from xinvert import invert_Poisson

    da = _make(12, 101)
    ip = {'BCs': ['fixed', 'fixed'], 'undef': np.nan, 'mxLoop': 200,
          'tolerance': 0.0, 'printInfo': False, 'architect': 'cpu'}

    client = dask_dist.Client(n_workers=1, threads_per_worker=4,
                              dashboard_address=None)
    try:
        sf = invert_Poisson(da, dims=['y', 'x'], coords='cartesian',
                            iParams=ip)
        t0 = time.perf_counter()
        re = sf.compute()
        dt = time.perf_counter() - t0
    finally:
        client.close()

    psi_true = (np.sin(np.pi * da['y'].values)[:, None]
                * np.sin(np.pi * da['x'].values)[None, :])
    err = float(np.nanmax(np.abs(re.values - psi_true)))
    assert err < 1e-3, f'distributed result wrong: err={err:.2e}'
    print(f'distributed compute OK: {dt:.2f} s, max err = {err:.2e}')


if __name__ == '__main__':
    test_distributed_serialization()
    print('OK')
