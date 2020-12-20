# -*- coding: utf-8 -*-
"""
Created on 2020.12.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
from xgrads.xgrads import open_CtlDataset

ds = open_CtlDataset('D:/SOR.ctl')

vor = ds.vor[0].rename('vorticity')
div = ds.div[0].rename('divergence')

#%%
import numpy as np
from xinvert.xinvert import invert_Poisson_animated


sf = invert_Poisson_animated(vor, BCs=['extend', 'periodic'],
                             loop_per_frame=1, max_loop=40)


#%% plot vector
import proplot as pplt
import xarray as xr
import numpy as np

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
