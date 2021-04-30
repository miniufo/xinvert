# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xgrads.xgrads import open_CtlDataset

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

# finite difference for derivative
# curl_tau2 = - tau_ideal.differentiate('ydef')

#%% analytical solution for non-rotating case
gamma = F * np.pi / R / Ly
p = np.exp(-np.pi * Lx / Ly)
q = 1

h_a = gamma * (Ly / np.pi)**2 * np.sin(np.pi * ydef / Ly) * (
          np.exp((xdef - Lx)*np.pi/Ly) + np.exp(-xdef*np.pi/Ly) - 1
      )


#%% invert
from xinvert.xinvert.core import invert_Stommel


h1, u1, v1 = invert_Stommel(curl_tau, dims=['ydef','xdef'],
                               BCs=['fixed', 'fixed'],
                               optArg=1.9, mxLoop=3000,
                               cal_flow=True,
                               coords='cartesian',
                               beta=0,
                               R=R,
                               depth=depth,
                               undef=np.nan,
                               debug=False)
h2, u2, v2 = invert_Stommel(curl_tau, dims=['ydef','xdef'],
                               BCs=['fixed', 'fixed'],
                               optArg=1.9, mxLoop=3000, tolerance=1e-14,
                               cal_flow=True,
                               coords='cartesian',
                               beta=beta,
                               R=R,
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
m=ax.contourf(h1/1e6*depth, cmap='rainbow', levels=np.linspace(0, 140, 15))
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
m=ax.contourf(h2/1e6*depth, cmap='rainbow', levels=np.linspace(0, 60, 13))
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


axes.format(abc=True, abcloc='l', abcstyle='(a)', grid=False,
            ylabel='y-coordinate (m)')

#%% real case
import numpy as np
import xarray as xr
from xgrads.xgrads import open_CtlDataset
from GeoApps.GridUtils import add_latlon_metrics
from GeoApps.DiagnosticMethods import Dynamics


ds = open_CtlDataset('D:/Data/SODA/2.2.6/SODA226Clim_1993_2003.ctl')

dset, grid = add_latlon_metrics(ds)

dyn = Dynamics(dset, grid)

tauxJan = dset.taux.where(dset.taux!=dset.undef)[0]
tauyJan = dset.tauy.where(dset.tauy!=dset.undef)[0]

curl_Jan = dyn.curl(tauxJan, tauyJan).load()

tauxJul = dset.taux.where(dset.taux!=dset.undef)[6]
tauyJul = dset.tauy.where(dset.tauy!=dset.undef)[6]

curl_Jul = dyn.curl(tauxJul, tauyJul).load()

lat, lon = xr.broadcast(ds.lat, ds.lon)
R = 2e-4 * (2 - 1*np.cos(np.deg2rad(lat)))
depth = 100


#%% invert
from xinvert.xinvert.core import invert_Stommel


h1, u1, v1 = invert_Stommel(curl_Jan, dims=['lat','lon'],
                               BCs=['fixed', 'periodic'],
                               optArg=1.6, mxLoop=5000,
                               cal_flow=True,
                               R=R,
                               depth=depth,
                               undef=np.nan,
                               debug=False)
h2, u2, v2 = invert_Stommel(curl_Jul, dims=['lat','lon'],
                               BCs=['fixed', 'periodic'],
                               optArg=1.6, mxLoop=5000,
                               cal_flow=True,
                               R=R,
                               depth=depth,
                               undef=np.nan,
                               debug=False)


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np


lat, lon = xr.broadcast(u1.lat, u1.lon)

fig, axes = pplt.subplots(nrows=2, ncols=2, figsize=(16,9), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

skip = 3
fontsize = 16

axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
            lonlines=40, latlines=15, lonlabels='b', latlabels='l',
            grid=False, labels=False)

ax = axes[0,0]
p=ax.contourf(curl_Jan, cmap='jet',levels=np.linspace(-7e-7,7e-7,29))
ax.set_title('wind stress curl (January)',
             fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              tauxJan.values[::skip,::skip+1], tauyJan.values[::skip,::skip+1],
              width=0.001, headwidth=10., headlength=12., scale=20)
ax.set_ylim([-80, 80])
ax.set_xlim([-180, 180])
ax.colorbar(p, loc='b', label='', length=0.93)

ax = axes[0,1]
p=ax.contourf(h1/1e6*depth, cmap='bwr', levels=np.linspace(-40,40,17))
ax.set_title('Stommel response (January)',
             fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+2], lat.values[::skip,::skip+2],
              u1.values[::skip,::skip+2], v1.values[::skip,::skip+2],
              width=0.001, headwidth=10., headlength=12., scale=40)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-80, 80])
ax.set_xlim([-180, 180])

ax.colorbar(p, loc='b', label='', ticks=5, length=0.93)

ax = axes[1,0]
p=ax.contourf(curl_Jul, cmap='jet',levels=np.linspace(-7e-7,7e-7,29))
ax.set_title('wind stress curl (July)',
             fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              tauxJul.values[::skip,::skip+1], tauyJul.values[::skip,::skip+1],
              width=0.001, headwidth=10., headlength=12., scale=20)
ax.set_ylim([-80, 80])
ax.set_xlim([-180, 180])
ax.colorbar(p, loc='b', label='', length=0.93)

ax = axes[1,1]
p=ax.contourf(h2/1e6*depth, cmap='bwr', levels=np.linspace(-60,60,13))
ax.set_title('Stommel response to curl (July)',
             fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+2], lat.values[::skip,::skip+2],
              u2.values[::skip,::skip+2], v2.values[::skip,::skip+2],
              width=0.001, headwidth=10., headlength=12., scale=40)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-80, 80])
ax.set_xlim([-180, 180])
ax.colorbar(p, loc='b', label='', ticks=5, length=0.93)


