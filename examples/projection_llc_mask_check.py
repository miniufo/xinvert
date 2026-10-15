"""Validate manufactured spherical projection on a read-only LLC coastal mask.

This is a mask/geometry verification, not a decomposition of SSH or real
ocean transport. The cropped perimeter is explicitly closed for this test.
"""
import argparse
import json
import resource
import time

import numba
import numpy as np
import xarray as xr
from scipy.ndimage import label

from xinvert.projection import project_vector


def main():
    """Load one small mask, project a manufactured face field, and report checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architect', choices=['cpu', 'cpu-mc', 'gpu'], default='cpu')
    parser.add_argument('--input', default='/mnt/Data2/LLC4320/global/interp/ssh/ssh_interp_20110913.nc')
    args = parser.parse_args()
    if args.architect == 'cpu-mc':
        numba.set_num_threads(min(4, numba.get_num_threads()))
    with xr.open_dataset(args.input) as source:
        nominal = {}
        for name in ['lat', 'lon']:
            original = source[name].values
            step = float(original[-1]-original[0])/(len(original)-1)
            regular = float(original[0])+np.arange(len(original))*step
            deviation = float(np.max(np.abs(original-regular)))
            precision = np.finfo(original.dtype).eps
            assert deviation <= 4*precision*max(1., float(np.max(np.abs(original)))), name
            nominal[name] = regular
        ssh = source.ssh.isel(time=0, drop=True, lat=slice(4976, 5104),
                              lon=slice(13856, 13984)).load()
    # Explicitly reconstruct nominal uniform coordinates from full-grid
    # endpoints: source float32 labels carry quantization, not mesh stretching.
    changes = {
        'lat': float(np.max(np.abs(ssh.lat.values-nominal['lat'][4976:5104]))),
        'lon': float(np.max(np.abs(ssh.lon.values-nominal['lon'][13856:13984]))),
    }
    ssh = ssh.assign_coords(lat=nominal['lat'][4976:5104], lon=nominal['lon'][13856:13984])
    wet = ssh.notnull()
    original_wet = int(wet.sum())
    wet[[0, -1], :] = False
    wet[:, [0, -1]] = False
    ny, nx = wet.shape
    phi = xr.DataArray(np.sin(np.linspace(0, np.pi, ny))[:, None] *
                       np.cos(np.linspace(0, 2*np.pi, nx))[None, :],
                       dims=wet.dims, coords=wet.coords)
    radius = 6371200.
    lat = np.deg2rad(wet.lat.values)
    dx = radius*np.deg2rad(float(wet.lon[1]-wet.lon[0]))
    dy = radius*np.deg2rad(float(wet.lat[1]-wet.lat[0]))
    fx = ((phi.shift(lon=-1)-phi)/(dx*np.cos(np.deg2rad(wet.lat)))).where(
        wet & wet.shift(lon=-1, fill_value=False), 0)
    fy = ((phi.shift(lat=-1)-phi)/dy).where(wet & wet.shift(lat=-1, fill_value=False), 0)
    start = time.perf_counter()
    output, diag = project_vector(
        fx, fy, wet, ['lat', 'lon'], coords='lat-lon',
        iParams={'architect': args.architect, 'dtype': 'float64', 'mxLoop': 200000,
                 'tolerance': 1e-12, 'diag_interval': 200, 'optArg': 1.95, 'printInfo': False})
    elapsed = time.perf_counter()-start
    labels, count = label(wet.values)
    expected = phi.values.copy()
    weights = np.broadcast_to(np.cos(lat)[:, None], wet.shape)
    for component in range(1, count+1):
        cells = labels == component
        expected[cells] -= np.average(expected[cells], weights=weights[cells])
    mask, p = wet.values, output.potential.values
    defects = []
    for j, i in zip(*np.where(mask)):
        laplace = 0.
        for dj, di in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            jj, ii = j+dj, i+di
            if mask[jj, ii]:
                g = (np.cos((lat[j]+lat[jj])/2)/(dy**2*np.cos(lat[j]))
                     if dj else 1/(dx**2*np.cos(lat[j])**2))
                laplace += g*(p[jj, ii]-p[j, i])
        defects.append(laplace-output.divergence.values[j, i])
    maximum_rhs = float(abs(output.divergence).max())
    report = {
        'architect': args.architect, 'shape': list(wet.shape), 'components': count,
        'source_wet_cells': original_wet, 'closed_crop_wet_cells': int(wet.sum()),
        'explicit_coordinate_regularization_degrees': changes,
        'wall_seconds_including_JIT': elapsed,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'converged': bool(diag.converged), 'iterations': int(diag.iterations),
        'relative_operator_defect': float(np.max(np.abs(defects))/maximum_rhs),
        'potential_max_error': float(np.max(np.abs(p[mask]-expected[mask]))),
        'relative_remaining_divergence': float(abs(output.remaining_divergence).max()/maximum_rhs),
    }
    print(json.dumps(report, indent=2))
    assert report['converged'], report
    assert report['relative_operator_defect'] < 1e-7, report
    assert report['potential_max_error'] < 1e-6, report


if __name__ == '__main__':
    main()
