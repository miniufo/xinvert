# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% test global 1
import xarray as xr
import numpy as np
from xinvert.xinvert import invert_Poisson, cal_flow


ds = xr.open_dataset('./xinvert/Data/Helmholtz_atmos.nc')

vor = ds.vor.rename('vorticity')
div = ds.div.rename('divergence')

iParams = {
    'BCs'      : ['extend', 'periodic'],
    'undef'    : np.nan,
    'mxLoop'   : 5000,
    'tolerance': 1e-11,
    'optArg'   : None,
    'printInfo': True,
    'debug'    : False,
}

vp = invert_Poisson(div, dims=['lat','lon'], iParams=iParams)
sf = invert_Poisson(vor, dims=['lat','lon'], iParams=iParams)

ux, vx = cal_flow(vp, dims=['lat', 'lon'], BCs=iParams['BCs'], vtype='velocitypotential')
us, vs = cal_flow(sf, dims=['lat', 'lon'], BCs=iParams['BCs'], vtype='streamfunction')

uall = us + ux
vall = vs + vx

#%% verification
from xinvert.xinvert import FiniteDiff

fd = FiniteDiff({'X':'lon', 'Y':'lat', 'T':'time'},
                BCs={'X':'periodic', 'Y':'extend'}, fill=0, coords='lat-lon')

div0 = fd.divg((us, vs), ['X', 'Y'])
vor0 = fd.curl( ux, vx , ['X', 'Y'])


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np

u = ds.u.where(ds.u!=0)[0].load()
v = ds.v.where(ds.v!=0)[0].load()
m = np.hypot(u, v)

lat, lon = xr.broadcast(u.lat, u.lon)

fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(10, 10.2), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

fontsize = 14

axes.format(abc='(a)', coast=True,
            lonlines=60, latlines=30, lonlabels='b', latlabels='l',
            grid=False, labels=False)

ax = axes[0]
p = ax.contourf(lon, lat, vor[0]*1e5, cmap='seismic',
                levels=np.linspace(-10, 10, 21))
ax.set_title('relative vorticity (*1e5)', fontsize=fontsize)
ax.colorbar(p, loc='r', label='', ticks=1, length=0.895)

ax = axes[1]
p = ax.contourf(lon, lat, sf[0], levels=31, cmap='jet')
ax.quiver(lon.values, lat.values, u.values, v.values,
              width=0.0007, headwidth=12., headlength=15.)
              # headwidth=1, headlength=3, width=0.002)
ax.set_title('wind field and inverted streamfunction', fontsize=fontsize)
ax.colorbar(p, loc='r', label='', length=0.895)
# ax.set_xticklabels([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
#                     fontsize=fontsize)



#%% test ocean (with topo) 2
import xarray as xr
import numpy as np
from xinvert.xinvert import invert_Poisson

ds = xr.open_dataset('./xinvert/Data/Helmholtz_ocean.nc')

vor = ds.vor[0]

iParams = {
    'BCs'      : ['extend', 'periodic'],
    'undef'    : 0,
    'tolerance': 1e-9,
}

sf = invert_Poisson(vor, dims=['YG','XG'], iParams=iParams)


#%% plot vector and streamfunction
import proplot as pplt
import xarray as xr


u = ds.UVEL[0]
v = ds.VVEL[0].rename({'YG':'YC', 'XC':'XG'}).interp_like(u)
m = np.hypot(u, v)

lat, lon = xr.broadcast(u.YC, u.XG)

fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

fontsize = 16

u = u.where(u!=0)
v = v.where(v!=0)
sf = sf.where(sf!=0)

axes.format(abc='(a)', coast=True,
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




#%% meridional 2D case
import xarray as xr
import numpy as np
from xinvert.xinvert import invert_Poisson, cal_flow, FiniteDiff


ds = xr.open_dataset('./xinvert/Data/ZonalMean.nc')

v = ds.vm
w = ds.wm

fd = FiniteDiff({'Z':'LEV', 'Y':'lat'}, BCs={'Z':'fixed','Y':'fixed'},
                coords='lat-lon')

vor = fd.vort(v=v, w=w, components='i')
div = fd.divg((v, w), ['Y', 'Z'])

iParams = {
    'BCs'      : ['fixed', 'extend'],
    'undef'    : np.nan,
    'mxLoop'   : 5000,
    'tolerance': 1e-13,
}

sf0 = invert_Poisson(vor, dims=['LEV','lat'], coords='z-lat', iParams=iParams)

vs, ws = cal_flow(sf0, dims=['LEV', 'lat'], coords='z-lat', BCs=['fixed', 'extend'])

vor0 = fd.vort(v=vs, w=ws, components='i')
div0 = fd.divg((vs, ws), ['Y', 'Z'])


#%% plot
import proplot as pplt

array = [
    [1, 1, 2, 2],
    [3, 3, 4, 4],
    [0, 5, 5, 0],
]

fig, axes = pplt.subplots(array, figsize=(10.5,8.2), sharex=3, sharey=3)

fontsize = 14

ax = axes[0]
m=ax.contourf(v, levels=np.linspace(-3.4, 3.4, 34), extend='both')
ax.colorbar(m, loc='r', label='', ticks=1)
ax.set_title('original v', fontsize=fontsize)
ax.set_ylabel('')
ax.set_xlabel('')

ax = axes[1]
m=ax.contourf(w, levels=np.linspace(-0.05, 0.05, 20), extend='both')
ax.colorbar(m, loc='r', label='', ticks=0.02)
ax.set_title('original w', fontsize=fontsize)
ax.set_ylabel('')
ax.set_xlabel('')

ax = axes[2]
m=ax.contourf(vs, levels=np.linspace(-3.4, 3.4, 34), extend='both')
ax.colorbar(m, loc='r', label='', ticks=1)
ax.set_title('rotational v', fontsize=fontsize)
ax.set_ylabel('')
ax.set_xlabel('')

ax = axes[3]
m=ax.contourf(ws, levels=np.linspace(-0.05, 0.05, 20), extend='both')
ax.colorbar(m, loc='r', label='', ticks=0.02)
ax.set_title('rotational w', fontsize=fontsize)
ax.set_ylabel('')
ax.set_xlabel('')

ax = axes[4]
m=ax.contourf(sf0, levels=21)
ax.colorbar(m, loc='r', label='')
ax.quiver(vs.lat, vs.LEV, vs, ws*-50, scale=40)
ax.set_title('streamfunction and vector', fontsize=fontsize)
ax.set_ylabel('Pressure (Pa)', fontsize=fontsize-2)
ax.set_xlabel('Latitude', fontsize=fontsize-2)

axes.format(abc='(a)', xlim=[0, 90])





#%% zonal 2D case
import xarray as xr
import numpy as np
from xinvert.xinvert import invert_Poisson, cal_flow, FiniteDiff


ds = xr.open_dataset('./xinvert/Data/atmos3D.nc')
ds['LEV'] = ds['LEV'] * 100 # hPa to Pa

u = ds.U.sel(lat=slice(10, -10)).mean('lat')
w = ds.Omega.sel(lat=slice(10, -10)).mean('lat')

fd = FiniteDiff({'Z':'LEV', 'Y':'lat', 'X':'lon'},
                BCs={'Z':'fixed', 'Y':'fixed', 'X':'periodic'}, coords='lat-lon')

vor = fd.vort(u=u, w=w, components='j')
div = fd.divg((u, w), ['X', 'Z'])

iParams = {
    'BCs'      : ['fixed', 'periodic'],
    'undef'    : np.nan,
    'mxLoop'   : 5000,
    'tolerance': 1e-13,
    'cal_flow' : True,
}

sf0 = invert_Poisson(vor, dims=['LEV','lon'], coords='z-lon', iParams=iParams)

us, ws = cal_flow(sf0, dims=['LEV','lon'], coords='z-lon', BCs=['fixed', 'periodic'])

vor0 = fd.vort(u=us, w=ws, components='j')
div0 = fd.divg((us, ws), ['X', 'Z'])


#%% plot
import proplot as pplt

array = [
    [1, 1, 2, 2],
    [3, 3, 4, 4],
    [0, 5, 5, 0],
]

fig, axes = pplt.subplots(array, figsize=(10.5,8.2), sharex=3, sharey=3)

fontsize = 14

ax = axes[0]
m=ax.contourf(u, levels=np.linspace(-16, 16, 32), extend='both')
ax.colorbar(m, loc='r', label='', ticks=4)
ax.set_title('original u', fontsize=fontsize)
ax.set_ylabel('')
ax.set_xlabel('')

ax = axes[1]
m=ax.contourf(w, levels=np.linspace(-0.06, 0.06, 22), extend='both')
ax.colorbar(m, loc='r', label='', ticks=0.02)
ax.set_title('original w', fontsize=fontsize)
ax.set_ylabel('')
ax.set_xlabel('')

ax = axes[2]
m=ax.contourf(us, levels=np.linspace(-16, 16, 32), extend='both')
ax.colorbar(m, loc='r', label='', ticks=4)
ax.set_title('rotational u', fontsize=fontsize)
ax.set_ylabel('')
ax.set_xlabel('')

ax = axes[3]
m=ax.contourf(ws, levels=np.linspace(-0.06, 0.06, 22), extend='both')
ax.colorbar(m, loc='r', label='', ticks=0.02)
ax.set_title('rotational w', fontsize=fontsize)
ax.set_ylabel('')
ax.set_xlabel('')

ax = axes[4]
m=ax.contourf(sf0, levels=21)
ax.colorbar(m, loc='r', label='')
ax.quiver(us.lon[::5], us.LEV, us[:,::5], ws[:,::5]*-50, scale=250)
ax.set_title('streamfunction and vector', fontsize=fontsize)
ax.set_ylabel('Pressure (Pa)', fontsize=fontsize-2)
ax.set_xlabel('Latitude', fontsize=fontsize-2)

axes.format(abc='(a)', xlim=[0, 360], xticks=np.arange(0, 361, 60))

