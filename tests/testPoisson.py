# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% test global
from xgrads.xgrads import open_CtlDataset


ds = open_CtlDataset('D:/SOR.ctl')

vor = ds.vor.rename('vorticity')
div = ds.div.rename('divergence')


#%% invert
import numpy as np
from xinvert.xinvert.core import invert_Poisson


sf = invert_Poisson(vor, dims=['lat','lon'], BCs=['extend', 'periodic'])
vp = invert_Poisson(div, dims=['lat','lon'], BCs=['extend', 'periodic'])

sf = sf.drop_vars('time').rename('sf')
sf['time'] = np.arange(len(sf.time))
vp = vp.drop_vars('time').rename('vp')
vp['time'] = np.arange(len(vp.time))


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np

u = ds.u.where(ds.u!=0)[0].load()
v = ds.v.where(ds.v!=0)[0].load()
m = np.hypot(u, v)

lat, lon = xr.broadcast(u.lat, u.lon)

fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(12, 13), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

fontsize = 16

axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
            lonlines=60, latlines=30, lonlabels='b', latlabels='l',
            grid=False, labels=False)

ax = axes[0]
p = ax.contourf(lon, lat, vor[0], cmap='seismic',
                levels=np.linspace(-1e-4, 1e-4, 21))
ax.set_title('relative vorticity', fontsize=fontsize)
ax.colorbar(p, loc='b', label='', ticks=1e-5, length=0.895)

ax = axes[1]
p = ax.contourf(lon, lat, sf[0], levels=31, cmap='jet')
ax.quiver(lon.values, lat.values, u.values, v.values,
              width=0.0007, headwidth=12., headlength=15.)
              # headwidth=1, headlength=3, width=0.002)
ax.set_title('wind field and inverted streamfunction', fontsize=fontsize)
ax.colorbar(p, loc='b', label='', length=0.895)
# ax.set_xticklabels([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
#                     fontsize=fontsize)



#%% test ocean with topo
import xarray as xr

ds = xr.open_dataset('D:/mitgcm.nc')

vor = ds.vor[0]

#%% invert
import numpy as np
from xinvert.xinvert.core import invert_Poisson


sf = invert_Poisson(vor, dims=['YG','XG'], BCs=['extend', 'periodic'],
                    undef=0, tolerance=1e-9)


#%% plot vector and streamfunction
import proplot as pplt
import xarray as xr


u = ds.UVEL[0]
v = ds.VVEL[0].rename({'YG':'YC', 'XC':'XG'}).interp_like(u)
m = np.hypot(u, v)

lat, lon = xr.broadcast(u.YC, u.XG)

fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

fontsize = 16

u = u.where(u!=0)
v = v.where(v!=0)
sf = sf.where(sf!=0)

axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
            lonlines=60, latlines=30, lonlabels='b', latlabels='l',
            grid=False, labels=False)

ax = axes[0]
p = ax.contourf(lon, lat, vor.where(vor!=0), cmap='bwr',
                levels=np.linspace(-5e-5, 5e-5, 21))
ax.set_title('relative vorticity', fontsize=fontsize)
ax.set_ylim([-70, 80])
ax.set_xlim([-180, 180])
ax.colorbar(p, loc='b', label='', ticks=5e-6, length=0.987)

ax = axes[1]
skip = 2
p = ax.contourf(lon, lat, sf, levels=31, cmap='jet')
ax.quiver(lon.values[::skip,::skip], lat.values[::skip,::skip],
              u.values[::skip,::skip], v.values[::skip,::skip],
              width=0.0006, headwidth=12., headlength=15.)
              # headwidth=1, headlength=3, width=0.002)
ax.set_title('current vector and inverted streamfunction', fontsize=fontsize)
ax.set_ylim([-70, 80])
ax.set_xlim([-180, 180])
ax.colorbar(p, loc='b', label='', ticks=5e4, length=0.987)
# ax.set_xticklabels([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
#                     fontsize=fontsize)


