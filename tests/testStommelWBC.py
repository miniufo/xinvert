# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr


xnum = 201
ynum = 151
Lx = 1e7 # 10,000 km
Ly = 2 * np.pi * 1e6 # 6249 km
R = 0.0008 # Rayleigh friction (default 0.02)
depth = 200 # fluid depth 200
beta = 1.8e-11 # gradient of f 1e-13
F = 0.3 # 0.1 N/m^2

xdef = xr.DataArray(np.linspace(0, Lx, xnum), dims='xdef',
                    coords={'xdef':np.linspace(0, Lx, xnum)})
ydef = xr.DataArray(np.linspace(0, Ly, ynum), dims='ydef',
                    coords={'ydef':np.linspace(0, Ly, ynum)})

ygrid, xgrid = xr.broadcast(ydef, xdef)

tau_ideal = xr.DataArray(-F * np.cos(np.pi * ygrid / Ly),
                         dims=['ydef','xdef'],
                         coords={'ydef':ydef, 'xdef':xdef})
curl_tau  = xr.DataArray(-F * np.sin(np.pi * ygrid / Ly) * np.pi/Ly,
                         dims=['ydef','xdef'],
                         coords={'ydef':ydef, 'xdef':xdef})


#%% analytical solution for non-rotating case
gamma = F * np.pi / R / Ly
p = np.exp(-np.pi * Lx / Ly)
q = 1

h_a = gamma * (Ly / np.pi)**2 * np.sin(np.pi * ydef / Ly) * (
          np.exp((xdef - Lx)*np.pi/Ly) + np.exp(-xdef*np.pi/Ly) - 1
      )


#%% invert
from xinvert.xinvert import invert_Stommel, invert_StommelMunk, cal_flow

iParams = {
    'BCs'      : ['fixed', 'fixed'],
    'mxLoop'   : 5000,
    'optArg'   : 1.9,
    'tolerance': 1e-12,
}

mParams1 = {'beta': 0   , 'R': R, 'D': depth, 'A':0}
mParams2 = {'beta': beta, 'R': R, 'D': depth, 'A':0}


S1 = invert_Stommel(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                    iParams=iParams, mParams=mParams1)
S2 = invert_Stommel(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                    iParams=iParams, mParams=mParams2)


S11 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                          iParams=iParams, mParams=mParams1)
S22 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                          iParams=iParams, mParams=mParams2)

u1, v1 = cal_flow(S1, dims=['ydef','xdef'], coords='cartesian')
u2, v2 = cal_flow(S2, dims=['ydef','xdef'], coords='cartesian')


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np


array = [
    [1, 2, 2, 3, 3,],
    ]

fig, axes = pplt.subplots(array, figsize=(11,5),
                          sharex=0, sharey=3)

skip = 3
fontsize = 14

ax = axes[0]
ax.plot(tau_ideal[:,0], tau_ideal.ydef)
ax.plot(tau_ideal[:,0]-tau_ideal[:,0], tau_ideal.ydef, color='k', linewidth=0.3)
ax.set_ylabel('y-coordinate (m)', fontsize=fontsize-2)
ax.set_title('wind stress', fontsize=fontsize)

ax = axes[1]
m=ax.contourf(S1/1e6*depth, cmap='rainbow', levels=np.linspace(0, 140, 15))
ax.set_title('$f$-plane Stommel solution', fontsize=fontsize)
p=ax.quiver(xgrid.values[::skip+1,::skip], ygrid.values[::skip+1,::skip],
              u1.values[::skip+1,::skip], v1.values[::skip+1,::skip],
              width=0.0014, headwidth=8., headlength=12., scale=10)
ax.colorbar(m, loc='b')
ax.set_ylim([0, Ly])
ax.set_xlim([0, Lx])
ax.set_xlabel('x-coordinate (m)', fontsize=fontsize-2)
ax.set_ylabel('y-coordinate (m)', fontsize=fontsize-2)

ax = axes[2]
m=ax.contourf(S2/1e6*depth, cmap='rainbow', levels=np.linspace(0, 90, 19))
ax.set_title('$\\beta$-plane Stommel solution', fontsize=fontsize)
ax.quiver(xgrid.values[::skip+1,::skip], ygrid.values[::skip+1,::skip],
              u2.values[::skip+1,::skip], v2.values[::skip+1,::skip],
              width=0.0014, headwidth=8., headlength=12., scale=8)
              # headwidth=1, headlength=3, width=0.002)
ax.colorbar(m, loc='b')
ax.set_ylim([0, Ly])
ax.set_xlim([0, Lx])
ax.set_xlabel('x-coordinate (m)', fontsize=fontsize-2)
ax.set_ylabel('y-coordinate (m)', fontsize=fontsize-2)


axes.format(abc='(a)', grid=False, ylabel='y-coordinate (m)')

#%% real case
import numpy as np
import xarray as xr


ds = xr.open_dataset('./xinvert/Data/SODA_curl.nc')

curl_Jan = ds.curl[0]
curl_Jul = ds.curl[6]

R = 2e-4
depth = 100


#%% invert
from xinvert.xinvert import invert_Stommel, invert_StommelMunk, cal_flow

iParams = {
    'BCs'      : ['extend', 'periodic'],
    'mxLoop'   : 5000,
    'optArg'   : 1,
    'tolerance': 1e-12,
    'undef'    : np.nan,
}

mParams1 = {'R': R, 'D': depth, 'A':5e3}

h1 = invert_Stommel(curl_Jan, dims=['lat','lon'],
                    iParams=iParams, mParams=mParams1)
h2 = invert_Stommel(curl_Jul, dims=['lat','lon'],
                    iParams=iParams, mParams=mParams1)

h11 = invert_StommelMunk(curl_Jan, dims=['lat','lon'],
                    iParams=iParams, mParams=mParams1)
h22 = invert_StommelMunk(curl_Jul, dims=['lat','lon'],
                    iParams=iParams, mParams=mParams1)

u1, v1 = cal_flow(h1, dims=['lat','lon'], BCs=['extend', 'periodic'])
u2, v2 = cal_flow(h2, dims=['lat','lon'], BCs=['extend', 'periodic'])


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np


lat, lon = xr.broadcast(u1.lat, u1.lon)

fig, axes = pplt.subplots(nrows=2, ncols=2, figsize=(11,6.8), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

skip = 3
fontsize = 16

axes.format(abc='(a)', coast=True,
            lonlines=40, latlines=15, lonlabels='b', latlabels='l',
            grid=False, labels=False)

ax = axes[0,0]
p=ax.contourf(curl_Jan, cmap='jet',levels=np.linspace(-7e-7,7e-7,29))
ax.set_title('wind stress curl (January)',
             fontsize=fontsize)
ax.set_ylim([-80, 80])
ax.set_xlim([-180, 180])
ax.colorbar(p, loc='b', label='', length=0.93)

ax = axes[0,1]
p=ax.contourf(h1/1e6*depth, cmap='bwr', levels=np.linspace(-80,80,17))
ax.set_title('Stommel response (January)',
             fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+2], lat.values[::skip,::skip+2],
              u1.values[::skip,::skip+2], v1.values[::skip,::skip+2],
              width=0.001, headwidth=10., headlength=12., scale=40)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-80, 80])
ax.set_xlim([-180, 180])

ax.colorbar(p, loc='b', label='', ticks=20, length=0.93)

ax = axes[1,0]
p=ax.contourf(curl_Jul, cmap='jet',levels=np.linspace(-7e-7,7e-7,29))
ax.set_title('wind stress curl (July)',
             fontsize=fontsize)
ax.set_ylim([-80, 80])
ax.set_xlim([-180, 180])
ax.colorbar(p, loc='b', label='', length=0.93)

ax = axes[1,1]
p=ax.contourf(h2/1e6*depth, cmap='bwr', levels=np.linspace(-100,100,21))
ax.set_title('Stommel response to curl (July)',
             fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+2], lat.values[::skip,::skip+2],
              u2.values[::skip,::skip+2], v2.values[::skip,::skip+2],
              width=0.001, headwidth=10., headlength=12., scale=40)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-80, 80])
ax.set_xlim([-180, 180])
ax.colorbar(p, loc='b', label='', ticks=20, length=0.93)


