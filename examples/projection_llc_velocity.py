"""Illustrative LLC surface-velocity projection in an explicitly closed crop.

Input is collocated, interpolated velocity, not native LLC face transport.
Wet-wet face values are arithmetic neighbour averages. Wet-dry and cropped
perimeter faces are explicitly set to zero for this closed-domain experiment.
Consequently this example is not a physical open-region/global decomposition.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xinvert.projection import project_vector


def main():
    """Load one coastal surface slice, solve on GPU, and save figures/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/mnt/Data2/LLC4320/global/interp')
    parser.add_argument('--output', default='/tmp/xinvert-llc-velocity-projection')
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    fields, adjustments, metadata = [], {}, {}
    for name in ['ssu', 'ssv']:
        path = Path(args.root)/name/f'{name}_interp_20110913.nc'
        with xr.open_dataset(path) as ds:
            metadata[name] = {'dims': dict(ds.sizes), 'attributes': dict(ds[name].attrs),
                              'time': str(ds.time.values[0])}
            nominal = {}
            for coordinate in ['lat', 'lon']:
                original = ds[coordinate].values
                step = float(original[-1]-original[0])/(len(original)-1)
                regular = float(original[0])+np.arange(len(original))*step
                deviation = np.max(np.abs(original-regular))
                if deviation > 4*np.finfo(original.dtype).eps*max(1., np.max(np.abs(original))):
                    raise ValueError('coordinates are not a quantized uniform grid')
                nominal[coordinate] = regular
            field = ds[name].isel(time=0, drop=True, lat=slice(4976, 5104),
                                  lon=slice(13856, 13984)).load()
        adjustments[name] = {
            coordinate: float(np.max(np.abs(field[coordinate].values-values)))
            for coordinate, values in [('lat', nominal['lat'][4976:5104]),
                                       ('lon', nominal['lon'][13856:13984])]}
        field = field.assign_coords(lat=nominal['lat'][4976:5104],
                                     lon=nominal['lon'][13856:13984]).astype('float64')
        fields.append(field)
    u, v = xr.align(*fields, join='exact')
    if u.ndim != 2:
        raise ValueError('expected a single collocated surface slice')
    print(json.dumps({'input': metadata, 'coordinate_adjustments_degrees': adjustments},
                     indent=2, default=str), flush=True)
    wet = np.isfinite(u) & np.isfinite(v)
    original_wet = int(wet.sum())
    wet[[0, -1], :] = False
    wet[:, [0, -1]] = False
    east = wet & wet.shift(lon=-1, fill_value=False)
    north = wet & wet.shift(lat=-1, fill_value=False)
    # Explicit model choice in this example, not hidden in project_vector.
    fx = ((u+u.shift(lon=-1))/2).where(east, 0)
    fy = ((v+v.shift(lat=-1))/2).where(north, 0)
    start = time.perf_counter()
    result, diagnostic = project_vector(
        fx, fy, wet, ['lat', 'lon'], coords='lat-lon',
        iParams={'architect': 'gpu', 'dtype': 'float64', 'mxLoop': 200000,
                 'optArg': 1.95, 'tolerance': 1e-12, 'diag_interval': 200,
                 'printInfo': False})
    elapsed = time.perf_counter()-start
    maximum_divergence = float(abs(result.divergence).max())
    remaining = float(abs(result.remaining_divergence).max())
    reconstruction = max(float(abs((result.x_gradient+result.x_solenoidal)-fx).max()),
                         float(abs((result.y_gradient+result.y_solenoidal)-fy).max()))
    report = {
        'time': metadata['ssu']['time'], 'shape': list(wet.shape),
        'lat_range': [float(wet.lat[0]), float(wet.lat[-1])],
        'lon_range': [float(wet.lon[0]), float(wet.lon[-1])],
        'source_wet_cells': original_wet, 'closed_wet_cells': int(wet.sum()),
        'converged': bool(diagnostic.converged), 'iterations': int(diagnostic.iterations),
        'solver_error': float(diagnostic.final_error),
        'maximum_input_divergence_s-1': maximum_divergence,
        'maximum_remaining_divergence_s-1': remaining,
        'relative_remaining_divergence': remaining/maximum_divergence,
        'face_reconstruction_max_error_m_s-1': reconstruction,
        'wall_seconds_including_JIT': elapsed,
        'boundary': 'crop deliberately closed; not an open-region physical decomposition',
    }
    print(json.dumps(report, indent=2), flush=True)
    if not report['converged'] or report['relative_remaining_divergence'] > 1e-6:
        raise RuntimeError(f'projection did not pass equation-defect checks: {report}')
    result['input_x_face'] = fx.where(east)
    result['input_y_face'] = fy.where(north)
    result['source_u'] = u
    result['source_v'] = v
    result['wet'] = wet
    result.attrs.update(input_time=report['time'], conversion='arithmetic wet-wet face average',
                        crop_boundary=report['boundary'])
    result.to_netcdf(directory/'projection.nc')
    diagnostic.to_netcdf(directory/'diagnostics.nc')
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def centered(a, b):
        """Average adjacent faces only for drawing, not for discrete diagnostics."""
        return ((a.fillna(0)+a.fillna(0).shift(lon=1, fill_value=0))/2).where(wet), (
            (b.fillna(0)+b.fillna(0).shift(lat=1, fill_value=0))/2).where(wet)

    prepared = centered(fx, fy)
    gradient = centered(result.x_gradient, result.y_gradient)
    solenoidal = centered(result.x_solenoidal, result.y_solenoidal)
    speed_limit = max(float(np.hypot(*components).max()) for components in
                      [prepared, gradient, solenoidal])
    potential_limit = float(abs(result.potential).max())
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
    panels = [
        (axes[0, 0], np.hypot(*prepared), 'Prepared input speed + arrows', 'm/s', 0, speed_limit, 'viridis', prepared),
        (axes[0, 1], result.potential, 'Velocity potential', 'm²/s', -potential_limit, potential_limit, 'RdBu_r', None),
        (axes[0, 2], np.hypot(*gradient), 'Gradient component speed', 'm/s', 0, speed_limit, 'viridis', gradient),
        (axes[1, 0], np.hypot(*solenoidal), 'Remaining component speed', 'm/s', 0, speed_limit, 'viridis', solenoidal),
        (axes[1, 1], result.divergence, 'Input divergence', 's⁻¹', -maximum_divergence, maximum_divergence, 'RdBu_r', None),
        (axes[1, 2], result.remaining_divergence, 'Remaining divergence (same scale)', 's⁻¹', -maximum_divergence, maximum_divergence, 'RdBu_r', None),
    ]
    X, Y = np.meshgrid(wet.lon.values, wet.lat.values)
    quiver_scale = max(speed_limit*9, 1e-15)
    for ax, data, title, unit, low, high, cmap, vector in panels:
        ax.set_facecolor('#dddddd')
        mesh = ax.pcolormesh(X, Y, data, shading='auto', cmap=cmap, vmin=low, vmax=high)
        ax.contour(X, Y, wet.values, levels=[.5], colors='black', linewidths=.5)
        if vector is not None:
            ax.quiver(X[::8, ::8], Y[::8, ::8], vector[0].values[::8, ::8],
                      vector[1].values[::8, ::8], scale=quiver_scale,
                      scale_units='width', color='black', width=.003)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_aspect(1/np.cos(np.deg2rad(float(wet.lat.mean()))))
        fig.colorbar(mesh, ax=ax, shrink=.85, label=unit)
    fig.suptitle('LLC4320 surface velocity — '+report['time'][:19]+'\n'
                 'Closed-crop illustration: averaged faces; imposed zero coastal/perimeter flux',
                 fontsize=12)
    fig.savefig(directory/'projection.png', dpi=160)
    plt.close(fig)
    print('OUTPUT:', directory, flush=True)


if __name__ == '__main__':
    main()
