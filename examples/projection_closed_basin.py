"""Main-program example: manufacture a face field, then call projection."""
import numpy as np
import xarray as xr

from xinvert.projection import project_vector


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
    result, diagnostics = project_vector(
        fx, fy, wet, ['y', 'x'],
        iParams={'architect': 'cpu', 'dtype': 'float64', 'diag_interval': 100,
                 'tolerance': 1e-12, 'mxLoop': 30000, 'optArg': 1.6,
                 'printInfo': False})
    print(result)
    print('Max remaining divergence:', float(abs(result.remaining_divergence).max()))
    print('Last solver error:', float(diagnostics.error.dropna('iteration').isel(iteration=-1)))


if __name__ == '__main__':
    main()
