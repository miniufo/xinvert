# -*- coding: utf-8 -*-
"""Resolution scaling: GPU sequential vs CPU dask-parallel (12 slices).

Upsamples SODA curl 1x/2x/4x to study how the GPU launch-overhead share
shrinks as grids grow.  Fixed workload: mxLoop=5000, tolerance=0.

Results: tests/results/resolution_scaling.json
"""
import json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
from numba import cuda
from demo_gpu_large import upsample, _gpu_available
from benchmark_gpu_scaling import capture_params, decompose
from xinvert import invert_Poisson

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data', 'SODA_curl.nc')


def main():
    os.makedirs(RESULTS, exist_ok=True)
    gpu_ok = _gpu_available()
    cuda.get_current_device()
    curl12 = xr.open_dataset(DATA).curl
    results = []
    for factor in (1, 2, 4):
        curl = curl12 if factor == 1 else upsample(curl12, factor)
        nlat, nlon = curl.shape[1:]
        print(f'=== {nlat}x{nlon} (x{factor}) ===')
        out = {'factor': factor, 'nlat': nlat, 'nlon': nlon}
        if gpu_ok:
            d = decompose(capture_params(curl.isel(time=0)), 5000, 0.0)
            out['gpu_decomp'] = d
            share = 100 * d['n_launches'] * 9.4e-6 / d['t_loop_s']
            print(f"  GPU decomp: {d['t_total_s']:.3f}s, "
                  f"launch share {share:.0f}%")
        for arch in ('gpu', 'cpu') if gpu_ok else ('cpu',):
            ip = {'BCs': ['extend', 'periodic'], 'undef': np.nan,
                  'mxLoop': 5000, 'tolerance': 0.0, 'printInfo': False,
                  'architect': arch}
            t0 = time.perf_counter()
            invert_Poisson(curl, dims=['lat', 'lon'], coords='lat-lon',
                           iParams=ip).compute()
            key = f't_{arch}_12slices'
            out[key] = time.perf_counter() - t0
            print(f"  {arch} 12 slices: {out[key]:.2f} s")
        if gpu_ok:
            out['speedup'] = out['t_cpu_12slices'] / out['t_gpu_12slices']
            print(f"  speedup: {out['speedup']:.2f}x")
        results.append(out)
    json.dump(results, open(os.path.join(RESULTS,
              'resolution_scaling.json'), 'w'), indent=1)
    print('saved tests/results/resolution_scaling.json')


if __name__ == '__main__':
    main()
