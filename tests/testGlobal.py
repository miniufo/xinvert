# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
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


#%% animate
import cartopy.crs as ccrs
from utils.PlotUtils import animate

ani = animate(sf, projection=ccrs.PlateCarree(central_longitude=180))


#%%
from xmovie.xmovie import Movie, rotating_globe
import proplot as pplt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

fig, axes = pplt.subplots(nrows=1, ncols=1, figsize=(9, 5),
                          proj=ccrs.PlateCarree(central_longitude=180))

fontsize = 16

def plot_func(da, fig, step, vmin, vmax, extend='both'):
    lat, lon = xr.broadcast(da.lat, da.lon)
    
    # ax = axes[0]
    # p = ax.contourf(lon, lat, F[0]*1e5, cmap=plt.cm.seismic, extend='both',
    #             levels=np.linspace(-9, 9, 19), globe=True, colorbar='r')
    # ax.set_title('vorticity', fontsize=fontsize)
    
    ax = axes
    p=ax.contourf(lon, lat, da[step]/1e7, cmap=plt.cm.seismic, extend='both',
                levels=np.linspace(-3, 3, 25), globe=True, colorbar='r')
    ax.set_title('inverted streamfunction', fontsize=fontsize)
    
    axes.format(abc=True, abcloc='l', abcstyle='(a)',
                lonlines=60, latlines=30, coast=True,
                lonlabels='b', latlabels='l')
    
    return axes, p

# plot_func(S, fig, 0)

mov = Movie(sf, plotfunc=rotating_globe)

