# -*- coding: utf-8 -*-
"""
Created on 2020.12.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
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
ax.colorbar(p, loc='b', label='', ticks=5e-6, length=0.985)

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
ax.colorbar(p, loc='b', label='', ticks=5e-6, length=0.985)
# ax.set_xticklabels([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
#                     fontsize=fontsize)

