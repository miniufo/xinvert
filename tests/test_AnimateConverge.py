# -*- coding: utf-8 -*-
"""
Created on 2020.12.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import xarray as xr
from xinvert import animate_iteration


def test_animate(tmp_path):
    ds = xr.open_dataset('./Data/Helmholtz_atmos.nc')
    
    assert ds.sizes == {'time': 2, 'lat': 73, 'lon': 144}
    
    vor = ds.vor[0].rename('vorticity')
    
    assert vor.dims == ('lat', 'lon')
    
    iParams = {
        'BCs': ['fixed','periodic']
    }
    
    output = tmp_path / 'atmos_animation.nc'
    animate_iteration('Poisson', vor, dims=['lat','lon'], iParams=iParams,
                      loop_per_frame=1, max_frames=40, output=output)
    with xr.open_dataset(output) as result:
        sf = result.inverted
        assert sf.dims == ('iter', 'lat', 'lon')
        assert len(sf.iter) == 40
        assert sf[0].dims == vor.dims
        assert result.attrs['completed_frames'] == 40

