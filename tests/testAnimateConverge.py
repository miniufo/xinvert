# -*- coding: utf-8 -*-
"""
Created on 2020.12.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import xarray as xr
import numpy as np
from xinvert.xinvert import animate_iteration

ds = xr.open_dataset('./xinvert/Data/Helmholtz_atmos.nc')

vor = ds.vor[0].rename('vorticity')


iParams = {
    'BCs': ['fixed','periodic']
}

sf = animate_iteration('Poisson', vor, dims=['lat','lon'], iParams=iParams,
                       loop_per_frame=1, max_frames=40)


#%% plot vector
import proplot as pplt

u = ds.u.where(ds.u!=0)[0].load()
v = ds.v.where(ds.v!=0)[0].load()
m = np.hypot(u, v)

lat, lon = xr.broadcast(u.lat, u.lon)

fig, axes = pplt.subplots(nrows=1, ncols=1, figsize=(11, 6), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

fontsize = 16

axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
            lonlines=60, latlines=30, lonlabels='b', latlabels='l',
            grid=True, labels=False)

ax = axes[0]
ax.contourf(lon, lat, sf[0], levels=31, cmap='jet')
p = ax.quiver(lon.values, lat.values, u.values, v.values,
              width=0.0006, headwidth=12., headlength=15.)
              # headwidth=1, headlength=3, width=0.002)
ax.set_title('wind field', fontsize=fontsize)
# ax.colorbar(p, loc='r', label='', ticks=0.25, length=0.83)
# ax.set_xticklabels([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
#                     fontsize=fontsize)
