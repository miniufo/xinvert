"""Main-program example: decompose a face field through the public Helmholtz API."""
import numpy as np
import xarray as xr

from xinvert.helmholtz import decompose


def main():
    """Project a potential-generated vector field in a basin with an island."""
    y, x = np.arange(20.), np.arange(24.)
    wet = xr.DataArray(np.ones((20, 24), dtype=bool), dims=['y', 'x'],
                       coords={'y': y, 'x': x})
    wet[[0, -1], :] = False
    wet[:, [0, -1]] = False
    wet[8:12, 10:14] = False
    phi = xr.DataArray(np.cos(y[:, None]/5) + np.sin(x[None, :]/6),
                       dims=wet.dims, coords=wet.coords)
    east = wet & wet.shift(x=-1, fill_value=False)
    north = wet & wet.shift(y=-1, fill_value=False)
    fx = (phi.shift(x=-1) - phi).where(east, 0)
    fy = (phi.shift(y=-1) - phi).where(north, 0)
    result, diagnostics = decompose(
        fx, fy, dims=['y', 'x'], mask=wet, method='sor',
        grid='arakawa-c', coords='cartesian', return_diagnostics=True,
        iParams={'architect': 'cpu', 'dtype': 'float64', 'diag_interval': 100,
                 'tolerance': 1e-12, 'mxLoop': 30000, 'optArg': 1.6,
                 'printInfo': False})
    print(result)
    print('Remaining divergence RMS:', float(diagnostics.div_error))
    print('Relative velocity error:', float(diagnostics.vel_error))


if __name__ == '__main__':
    main()
