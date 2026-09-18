# -*- coding: utf-8 -*-
"""Verify invert_Poisson works under dask.distributed Client (serialization)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xarray as xr


def main():
    from dask.distributed import Client
    from xinvert import invert_Poisson

    nt = 12
    x = np.linspace(0, 1, 101); y = np.linspace(0, 1, 81); t = np.arange(nt)
    X, Y = np.meshgrid(x, y)
    vor3d = np.stack([-2.0*np.pi**2*np.sin(np.pi*X)*np.sin(np.pi*Y)]*nt)
    psi_true = np.sin(np.pi*X)*np.sin(np.pi*Y)

    da = xr.DataArray(vor3d, dims=['time', 'y', 'x'],
                      coords={'time': t, 'y': y, 'x': x}).chunk({'time': 1})

    ip = {'BCs': ['fixed', 'fixed'], 'mxLoop': 200, 'tolerance': 0.0,
          'printInfo': False, 'architect': 'cpu'}

    print('--- starting distributed Client ---')
    client = Client(n_workers=1, threads_per_worker=4, dashboard_address=None)

    sf = invert_Poisson(da, dims=['y', 'x'], coords='cartesian', iParams=ip)
    t0 = time.perf_counter()
    re = sf.compute()
    dt = time.perf_counter() - t0

    err = max(float(np.nanmax(np.abs(re.isel(time=k).values - psi_true)))
              for k in range(nt))
    print(f'distributed compute OK: {dt:.2f} s, max err vs analytic = {err:.2e}')
    client.close()


if __name__ == '__main__':
    main()
