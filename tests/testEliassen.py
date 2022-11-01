# -*- coding: utf-8 -*-
"""
Created on 2022.08.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""

#%% prepare the ZonalMean.nc
import xarray as xr
import numpy as np

ds = xr.open_dataset('./xinvert/Data/atmos3D.nc')\
    .isel(lat=np.arange(0, 72))
    # .isel(latitude=np.arange(115, -1, -1))

dset = ds.interp({'LEV':np.linspace(1000, 100, 37)}).astype('f4')
dset['LEV'] = dset['LEV'] * 100 # hPa changed to Pa


Re = 6371200
Rd = 287.04
Cp = 1004.88
P0 = 100000
Om = 7.292e-5

f  = 2 * Om * np.sin(np.deg2rad(dset['lat']))
r  = Re * np.cos(np.deg2rad(dset['lat']))
pi = (dset['LEV'] / P0) ** (Rd / Cp)

u  = dset.U
v  = dset.V
w  = dset.Omega
T  = dset.T
z  = dset.hgt
Th = (T / pi).astype('f4')
Ag = ((u + Om * r) * r).astype('f4')

um = u.mean('lon').rename('um')
vm = v.mean('lon').rename('vm')
wm = w.mean('lon').rename('wm')
Am =Ag.mean('lon').rename('Am')
Tm = T.mean('lon').rename('Tm')
Thm=Th.mean('lon').rename('Thm')
zm = z.mean('lon').rename('zm')

ua = (u - um).rename('ua')
va = (v - vm).rename('va')
wa = (w - wm).rename('wa')
Aa =(Ag - Am).rename('Aa')
Ta = (T - Tm).rename('Ta')
Tha=(Th -Thm).rename('Tha')
za = (z - zm).rename('za')

Aava = (Aa * va).mean('lon').rename('Aava')
Aawa = (Aa * wa).mean('lon').rename('Aawa')
Tava = (Tha* va).mean('lon').rename('Tava')
Tawa = (Tha* wa).mean('lon').rename('Tawa')


#%% calculate A, B, C
from xinvert.xinvert import deriv, FiniteDiff

fd = FiniteDiff(dim_mapping={'Z':'LEV', 'Y':'lat', 'X':'lon'},
                BCs={'Z':['extend', 'extend'],
                     'Y':['extend', 'extend'],
                     'X':['periodic','periodic']},
                coords='lat-lon', fill=0)


deg2m = Re * 3.1416 / 180.0
cosL  = np.cos(np.deg2rad(dset['lat']))
sinL  = np.sin(np.deg2rad(dset['lat']))

# defined on half grid
Ccoef =-deriv(Thm, dim='LEV',
              scheme='backward') * Rd * pi / dset['LEV'] /r
# defined on half grid
Bcoef = deriv(Thm, dim='lat',
              scale=deg2m, scheme='center') * Rd * pi / dset['LEV'] /r
Bcoef2= deriv(Am, dim='LEV',
              scheme='center') * f / r /r
# defined on half grid
Acoef =-deriv(Am, dim='lat',
              scale=deg2m, scheme='backward') * f / r /r
Acoef2=-deriv(Am*Am, dim='lat',
              scale=deg2m, scheme='backward') * sinL / (r**3) /r

EHFC =  fd.divg((Tava, Tawa), dims=['Y', 'Z'])
EAFC = -fd.divg((Aava, Aawa), dims=['Y', 'Z'])

EHFC[:, :1] = 0 # maskout polar forcing to 0
EAFC[:, :1] = 0 # maskout polar forcing to 0

F_EHF = (fd.grad(EHFC, dims='Y') * Rd * pi / dset['LEV']).rename('EHF')
F_AHF =(-fd.grad(EAFC, dims='Z') * f / r).rename('EAF')


F = (Thm - Thm).load().rename('F_ideal')
F[27, 42:50] = 1e-11
F[27, 43] = 1e-11
F[10, 42:50] =-1e-11
F[10, 43] =-1e-11

Acoef = Acoef.rename('Acoef')
Bcoef = Bcoef.rename('Bcoef')
Ccoef = Ccoef.rename('Ccoef')

dsnew = xr.merge([um, vm, wm, Am, Tm, Thm, zm, Aava, Aawa, Tava, Tawa,
                  Acoef, Bcoef, Ccoef, F_EHF, F_AHF, F])

dsnew.to_netcdf('./xinvert/Data/ZonalMean.nc')

#%% plot ABC
import proplot as pplt

fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(9, 7.5), sharex=3, sharey=3)

fontsize = 14

ax = axes[0,0]
tmp = Ccoef.copy()
tmp['LEV'] = tmp['LEV'] / 100
m1 = ax.contour(tmp, levels=25, cmap='jet')
ax.set_title('C', fontsize=fontsize)

ax = axes[1,0]
tmp = Bcoef.copy() * 1e7
tmp['LEV'] = tmp['LEV'] / 100
m2 = ax.contour(tmp, levels=np.linspace(-2, 2, 21), cmap='jet')
ax.set_title('B', fontsize=fontsize)

ax = axes[1,1]
tmp = Bcoef2.copy() * 1e7
tmp['LEV'] = tmp['LEV'] / 100
m3 = ax.contour(tmp, levels=np.linspace(-2, 2, 21), cmap='jet')
ax.set_title('B2', fontsize=fontsize)

ax = axes[2,0]
tmp = Acoef.copy() * 1e8
tmp['LEV'] = tmp['LEV'] / 100
m4 = ax.contour(tmp, levels=np.linspace(0, 2, 21), cmap='jet')
ax.set_title('A', fontsize=fontsize)

ax = axes[2,1]
tmp = Acoef2.copy() * 1e8
tmp['LEV'] = tmp['LEV'] / 100
m5 = ax.contour(tmp, levels=np.linspace(0, 2, 21), cmap='jet')
ax.set_title('A2', fontsize=fontsize)

axes.format(yscale='log', ytickminor=True, xlabel='latitude (deg)',
            ylabel='pressure (hPa)', xticks=[0, 30, 60, 90], xlim=[0, 90],
            yticks=[1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])


#%%
from xinvert.xinvert import invert_Eliassen, invert_Poisson

params = {
    'BCs'      : ['fixed', 'fixed'],
    'mxLoop'   : 600,
    'tolerance': 1e-10,
    'cal_flow' : True,
}

sf, v, w = invert_Eliassen(F_EHF+F_AHF, Acoef, Bcoef, Ccoef, dims=['LEV','lat'], coords='z-lat',
                      debug=True, params=params)


#%% test idealized TC case
import xarray as xr


# load a snapshot as a radius-vertical 2D structure
ds = xr.open_dataset('./xinvert/Data/TC2D.nc')

undef = 9.99e20

A = ds.Aa.where(ds.Aa!=undef).load()
B = ds.Bb.where(ds.Bb!=undef).load()
C = ds.Cc.where(ds.Cc!=undef).load()

A[:,0] = 2 * A[:, 1] - A[:, 2]
# B[:,0] = np.nan
C[:,0] = 2 * C[:, 1] - C[:, 2]

F = (ds.faf.where(ds.faf!=undef) - ds.faf.where(ds.faf!=undef)).load()
F[27, 22] = 1e-11
F[27, 23] =-1e-11
F[10, 22] = 1e-11
F[10, 23] =-1e-11

F = F.where(A)

#%% interpolate
import numpy as np

lat = xr.DataArray(np.linspace(-90, -75.3, 99), dims='lat',
                   coords={'lat':np.linspace(-90, -75.3, 99)})
lev = xr.DataArray(np.linspace(100000, 10000, 73), dims='lev',
                   coords={'lev':np.linspace(100000, 10000, 73)})

A = A.interp({'lev':lev, 'lat':lat})
B = B.interp({'lev':lev, 'lat':lat})
C = C.interp({'lev':lev, 'lat':lat})
F = F.interp({'lev':lev, 'lat':lat})

#%% invert
from xinvert.xinvert import invert_Eliassen

params = {
    'BCs'      : ['fixed', 'fixed'],
    'mxLoop'   : 600,
    'tolerance': 1e-12,
    'optArg'   : 1.4,
    'cal_flow' : True,
}

# use south pole as the center and latlon to fake the cylindrical coordinate
sf, v, w = invert_Eliassen(F, A, B, C, dims=['lev','lat'], coords='z-lat',
                           debug=True, params=params)




#%% test real TC case
import xarray as xr

ds = xr.open_dataset('./xinvert/Data/TC2D.nc')

undef = 9.99e20

A = ds.Aa.where(ds.Aa!=undef).load()
B = ds.Bb.where(ds.Bb!=undef).load()
C = ds.Cc.where(ds.Cc!=undef).load()

F = ds.faf.where(ds.faf!=undef).load()


#%% invert
import numpy as np
from xinvert.xinvert import invert_Eliassen

params = {
    'BCs'      : ['fixed', 'fixed'],
    'mxLoop'   : 600,
    'tolerance': 1e-12,
    'optArg'   : 1.4,
    'cal_flow' : True,
}

# use south pole as the center and latlon to fake the cylindrical coordinate
sf, v, w = invert_Eliassen(F, A, B, C, dims=['lev','lat'], coords='z-lat',
                           debug=False, params=params)

# sf = sf.drop_vars('time').rename('sf')
# sf['time'] = np.arange(len(sf.time))


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr

fig, axes = pplt.subplots(nrows=1, ncols=1, figsize=(7, 5), sharex=3, sharey=3)

fontsize = 14
step = 3

axes.format(yscale='log', ytickminor=True)

ax = axes[0]

tmp = sf.copy()
tmp['lev'] = tmp['lev'] / 100
m1 = ax.contour(tmp, levels=25, colors='k', lw=0.8)
tmp = v.copy()
tmp['lev'] = tmp['lev'] / 100
m2 = ax.contourf(xr.where(np.abs(tmp)<1.5, np.nan, tmp),
                 levels=np.linspace(-4, 4, 17), cmap='bwr', extend='both')
tmp = w.copy().load()
tmp[:, 0] = 0
tmp['lev'] = tmp['lev'] / 100
m3 = ax.contourf(xr.where(np.abs(tmp)<0.2, np.nan, tmp),
                   levels=np.linspace(-0.9, 0.9, 19), cmap='BuGn_r')

ax.set_title('secondary circulation in TC', fontsize=fontsize)
ax.colorbar(m2, loc='b', label='', ticks=1)
ax.set_xlabel('radius (km)')
ax.set_ylabel('pressure (hPa)')
ax.set_xticks(np.linspace(-90, -76.5, 7))
ax.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])
ax.set_xticklabels(np.linspace(0, 1500, 7).astype('int'), fontsize=fontsize-2)
ax.set_yticklabels([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100],
                   fontsize=fontsize-2)





# %%
