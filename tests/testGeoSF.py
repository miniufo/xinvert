# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
import xarray as xr


ds = xr.open_dataset('./xinvert/Data/MJO.nc')

geo = ds.h.load()


#%% calculate force
import numpy as np
import xarray as xr
from xinvert.xinvert import FiniteDiff


fd = FiniteDiff(dim_mapping={'Y':'lat', 'X':'lon'},
                coords='lat-lon')

force = fd.Laplacian(geo, ['Y', 'X'], BCs={'Y': 'extend', 'X':'periodic'})

latNew = (geo.lat + geo.lat.diff('lat')[0]/2.0)[:-1]

ny, nx = force.shape

forceHalf = force.interp_like(xr.DataArray(np.zeros((ny-1, nx)),
                                           dims=['lat','lon'],
                                           coords={'lat':latNew,
                                                   'lon':geo.lon}))


#%% invert
from xinvert.xinvert import invert_geostreamfunction

params = {
    'BCs'      : ['fixed', 'periodic'],
    'cal_flow' : True
}

sf, u, v = invert_geostreamfunction(forceHalf, dims=['lat','lon'], params=params)


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np


u = ds.u.where(ds.u!=0).load()
v = ds.v.where(ds.v!=0).load()
m = np.hypot(u, v)

lat, lon = xr.broadcast(u.lat, u.lon)

fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(11, 12), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

fontsize = 16

axes.format(abc='(a)', coast=True,
            lonlines=60, latlines=30, lonlabels='b', latlabels='l',
            grid=False, labels=False)

ax = axes[0]
p = ax.contourf(force*1e9, cmap='seismic',
                levels=np.linspace(-1, 1, 21))
ax.set_title('Laplacian of geopotential', fontsize=fontsize)
ax.set_ylim([-85, 85])
ax.colorbar(p, loc='b', label='', ticks=0.2, length=0.895)

sf2 = sf.copy().load()
sf2[:36,:] =  sf2[:36,:] + (sf[36,:].mean().values - sf2[35,:].mean().values)

ax = axes[1]
p = ax.contourf(sf2, levels=21, cmap='bwr')
ax.quiver(lon.values, lat.values, u.values, v.values,
              width=0.0007, headwidth=12., headlength=15.)
              # headwidth=1, headlength=3, width=0.002)
ax.set_title('wind field and inverted streamfunction', fontsize=fontsize)
ax.set_ylim([-85, 85])
ax.colorbar(p, loc='b', label='', length=0.895)
# ax.set_xticklabels([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
#                     fontsize=fontsize)



