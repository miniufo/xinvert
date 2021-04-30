# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
from xgrads.xgrads import open_CtlDataset


ds = open_CtlDataset('D:/Data/Interest/GillModel/MJOTest/Reality.ctl')

geo = ds.h[0, :, :].load()


#%% calculate force
import numpy as np
import xarray as xr
from xinvert.xinvert import Laplacian


force = Laplacian(geo, ['lat', 'lon'], BCx='periodic')

latNew = (geo.lat + geo.lat.diff('lat')[0]/2.0)[:-1]

ny, nx = force.shape

forceHalf = force.interp_like(xr.DataArray(np.zeros((ny-1, nx)),
                                           dims=['lat','lon'],
                                           coords={'lat':latNew,
                                                   'lon':geo.lon}))


#%% invert
from xinvert.xinvert import invert_geostreamfunction


sf = invert_geostreamfunction(forceHalf, dims=['lat','lon'],
                              BCs=['fixed', 'periodic'])


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
p = ax.contourf(force*1e8, cmap='seismic',
                levels=np.linspace(-1, 1, 21))
ax.set_title('Laplacian of geopotential', fontsize=fontsize)
ax.colorbar(p, loc='b', label='', ticks=0.2, length=0.895)

ax = axes[1]
p = ax.contourf(sf, levels=31, cmap='jet')
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

