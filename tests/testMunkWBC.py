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
R = 0.0001 # Rayleigh friction (default 0.02)
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

# finite difference for derivative
# curl_tau2 = - tau_ideal.differentiate('ydef')


#%% invert
from xinvert.xinvert.core import invert_StommelMunk


h1, u1, v1 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'],
                               BCs=['fixed', 'fixed'],
                               optArg=1, mxLoop=4000,
                               cal_flow=True,
                               coords='cartesian',
                               AH=5e3,
                               beta=0,
                               R=0,
                               depth=depth,
                               undef=np.nan,
                               debug=False)
h2, u2, v2 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'],
                               BCs=['fixed', 'fixed'],
                               optArg=1, mxLoop=4000, tolerance=1e-14,
                               cal_flow=True,
                               coords='cartesian',
                               AH=5e3,
                               beta=beta,
                               R=0,
                               depth=depth,
                               undef=np.nan,
                               debug=False)


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np


array = [
    [1, 2, 2, 3, 3,],
    ]


fig, axes = pplt.subplots(array, figsize=(13,6),
                          sharex=0, sharey=3)

skip = 3
fontsize = 16


ax = axes[0]
ax.plot(tau_ideal[:,0], tau_ideal.ydef)
ax.plot(tau_ideal[:,0]-tau_ideal[:,0], tau_ideal.ydef, color='k', linewidth=0.3)
ax.set_ylabel('y-coordinate (m)', fontsize=fontsize-2)
ax.set_title('wind stress', fontsize=fontsize)

ax = axes[1]
m=ax.contourf(h1/1e6*depth, cmap='rainbow', levels=np.linspace(0, 90, 19))
ax.set_title('$f$-plane Munk solution', fontsize=fontsize)
p=ax.quiver(xgrid.values[::skip+1,::skip], ygrid.values[::skip+1,::skip],
              u1.values[::skip+1,::skip], v1.values[::skip+1,::skip],
              width=0.0014, headwidth=8., headlength=12., scale=10)
ax.colorbar(m, loc='b')
ax.set_ylim([0, Ly])
ax.set_xlim([0, Lx])
ax.set_xlabel('x-coordinate (m)', fontsize=fontsize-2)
ax.set_ylabel('y-coordinate (m)', fontsize=fontsize-2)

ax = axes[2]
m=ax.contourf(h2/1e6*depth, cmap='rainbow', levels=np.linspace(0, 90, 19))
ax.set_title('$\\beta$-plane Munk solution', fontsize=fontsize)
ax.quiver(xgrid.values[::skip+1,::skip], ygrid.values[::skip+1,::skip],
              u2.values[::skip+1,::skip], v2.values[::skip+1,::skip],
              width=0.0014, headwidth=8., headlength=12., scale=8)
              # headwidth=1, headlength=3, width=0.002)
ax.colorbar(m, loc='b')
ax.set_ylim([0, Ly])
ax.set_xlim([0, Lx])
ax.set_xlabel('x-coordinate (m)', fontsize=fontsize-2)
ax.set_ylabel('y-coordinate (m)', fontsize=fontsize-2)

axes.format(abc=True, abcloc='l', abcstyle='(a)', grid=False,
            ylabel='y-coordinate (m)')

#%% real cases
import numpy as np
import xarray as xr
from xgrads.xgrads import open_CtlDataset
from GeoApps.GridUtils import add_latlon_metrics
from GeoApps.DiagnosticMethods import Dynamics

depth = 100
rho0 = 1027
R = 1e-4

ds = open_CtlDataset('D:/Data/SODA/2.2.6/SODA226Clim_1993_2003.ctl')

dset, grid = add_latlon_metrics(ds)

dyn = Dynamics(dset, grid)

tau_real = dyn.curl(dset.taux.where(dset.taux!=dset.undef),
                    dset.tauy.where(dset.taux!=dset.undef))[0].load()


#%% invert
from xinvert.xinvert.core import invert_StommelMunk


h1, u1, v1 = invert_StommelMunk(tau_real, dims=['lat','lon'],
                                BCs=['fixed', 'periodic'],
                                optArg=1, mxLoop=4000,
                                cal_flow=True,
                                depth=depth,
                                R=R,
                                rho=rho0,
                                AH=3e3,
                                undef=np.nan,
                                debug=False)


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np


lat, lon = xr.broadcast(u1.lat, u1.lon)

fig, axes = pplt.subplots(nrows=1, ncols=1, figsize=(10, 6.5), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

skip = 1
fontsize = 16

axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
            lonlines=40, latlines=10, lonlabels='', latlabels='l',
            grid=False, labels=False)

ax = axes[0]
p=ax.contourf(h1/1e6*depth, cmap='bwr', levels=np.linspace(-80,80,33))
ax.set_title('Stommel-Munk solution to wind stress curl',
             fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip], lat.values[::skip,::skip],
              u1.values[::skip,::skip], v1.values[::skip,::skip],
              width=0.0012, headwidth=10., headlength=12., scale=80)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-0, 60])
ax.set_xlim([-80, 30])
ax.set_xlabel([])

ax.colorbar(p, loc='b', label='', ticks=10, length=1)

#%%
lat, lon = xr.broadcast(u1.lat, u1.lon)

fig, axes = pplt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

skip = 3
fontsize = 16

axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
            lonlines=40, latlines=15, lonlabels='b', latlabels='l',
            grid=False, labels=False)

ax = axes[0]
p=ax.contourf(h1/1e6*depth, cmap='bwr', levels=np.linspace(-80,80,33))
ax.set_title('Stommel-Munk solution to wind stress curl',
             fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip], lat.values[::skip,::skip],
              u1.values[::skip,::skip], v1.values[::skip,::skip],
              width=0.0012, headwidth=10., headlength=12., scale=70)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-50, 75])
ax.set_xlim([-180, 180])

ax.colorbar(p, loc='b', label='', ticks=10, length=1)





