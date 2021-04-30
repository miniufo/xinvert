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
R = 0.005 # Rayleigh friction 0.02
depth = 200 # fluid depth 200
beta = 1.8e-11 # gradient of f 1e-13
rho = 1027
F = 0.1 # 0.1 N/m^2 or 1 dyne/cm^2

xdef = xr.DataArray(np.linspace(0, Lx, xnum), dims='xdef',
                    coords={'xdef':np.linspace(0, Lx, xnum)})
ydef = xr.DataArray(np.linspace(0, Ly, ynum), dims='ydef',
                    coords={'ydef':np.linspace(0, Lx, ynum)})

ygrid, xgrid = xr.broadcast(ydef, xdef)

tau_ideal = xr.DataArray(F * np.cos(np.pi * ygrid / Ly) / rho,
                         dims=['ydef','xdef'],
                         coords={'ydef':ydef, 'xdef':xdef})
curl_tau  = xr.DataArray(-F * np.sin(np.pi * ygrid / Ly) / rho * np.pi/Ly,
                         dims=['ydef','xdef'],
                         coords={'ydef':ydef, 'xdef':xdef})

ds = open_CtlDataset('D:/SOR.ctl')

Q = ds.u[0] - ds.u[0]

lat, lon = xr.broadcast(ds.lat, ds.lon)

Q1 = 0.05*np.exp(-((lat-0)**2+(lon-120)**2)/100.0)
Q2 = 0.05*np.exp(-((lat-10)**2+(lon-120)**2)/100.0) \
   - 0.05*np.exp(-((lat+10)**2+(lon-120)**2)/100.0)
Q3 = 0.05*np.exp(-((lat-10)**2+(lon-120)**2)/100.0)


#%% invert
from xinvert.xinvert.core import invert_GillMatsuno


h1, u1, v1 = invert_GillMatsuno(Q1, dims=['lat','lon'],
                                 BCs=['fixed', 'periodic'],
                                 optArg=1.4, mxLoop=600, cal_flow=True,
                                 epsilon=1e-5, Phi=5000,
                                 debug=False)
h2, u2, v2 = invert_GillMatsuno(Q2, dims=['lat','lon'],
                                 BCs=['fixed', 'periodic'],
                                 optArg=1.4, mxLoop=600, cal_flow=True,
                                 epsilon=1e-5, Phi=5000,
                                 debug=False)
h3, u3, v3 = invert_GillMatsuno(Q3, dims=['lat','lon'],
                                 BCs=['fixed', 'periodic'],
                                 optArg=1.4, mxLoop=600, cal_flow=True,
                                 epsilon=1e-5, Phi=5000,
                                 debug=False)


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np
from utils.PlotUtils import maskout_cmap


lat, lon = xr.broadcast(u1.lat, u1.lon)

fig, axes = pplt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

skip = 1
fontsize = 16

axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
            lonlines=40, latlines=15, lonlabels='b', latlabels='l',
            grid=False, labels=False)

cmap = maskout_cmap('bwr', [-0.05, -0.04, -0.03, -0.02, -0.01,  0.,
                   0.01,  0.02,  0.03,  0.04,  0.05], [-0.01, 0.01])

ax = axes[0]
ax.contourf(Q1, cmap=cmap, levels=np.linspace(-0.05, 0.05, 11))
ax.contour(h1, cmap='jet')
ax.set_title('Gill-Matsuno response - type 1 ($\epsilon=10^{-5}, \Phi=5\\times 10^3$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u1.values[::skip,::skip+1], v1.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=250)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-40, 40])
ax.set_xlim([-140, 100])

ax = axes[1]
ax.contourf(Q2, cmap=cmap, levels=np.linspace(-0.05, 0.05, 11))
ax.contour(h2, cmap='jet')
ax.set_title('Gill-Matsuno response - type 2 ($\epsilon=10^{-5}, \Phi=5\\times 10^3$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u2.values[::skip,::skip+1], v2.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=250)
ax.set_ylim([-40, 40])
ax.set_xlim([-140, 100])

ax = axes[2]
p=ax.contourf(Q3, cmap=cmap, levels=np.linspace(-0.05, 0.05, 11))
ax.contour(h3, cmap='jet')
ax.set_title('Gill-Matsuno response - type 3 ($\epsilon=10^{-5}, \Phi=5\\times 10^3$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u3.values[::skip,::skip+1], v3.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=250)
ax.set_ylim([-40, 40])
ax.set_xlim([-140, 100])

fig.colorbar(p, loc='b', label='', ticks=0.01, length=1)






#%% a real case
import numpy as np
import xarray as xr
from xgrads.xgrads import open_CtlDataset


ds1 = open_CtlDataset('D:/Data/Interest/GillModel/MJOTest/Reality.ctl').sel(time='2008-01-25')
ds2 = open_CtlDataset('D:/Data/Interest/GillModel/MJOTest/ObsSim.ctl')

h_ob = ds1.hl
u_ob = ds1.ul
v_ob = ds1.vl

Q = ds2.ol[0].where(np.abs(lat)<60, 0)



#%% invert
from xinvert.xinvert.core import invert_GillMatsuno


h1, u1, v1 = invert_GillMatsuno(Q, dims=['lat','lon'],
                                 BCs=['fixed', 'periodic'],
                                 optArg=1.4, mxLoop=800, cal_wind=True,
                                 epsilon=1e-5, Phi=5000,
                                 debug=False)
h2, u2, v2 = invert_GillMatsuno(Q, dims=['lat','lon'],
                                 BCs=['fixed', 'periodic'],
                                 optArg=1.4, mxLoop=800, cal_wind=True,
                                 epsilon=7e-6, Phi=5000,
                                 debug=False)
h3, u3, v3 = invert_GillMatsuno(Q, dims=['lat','lon'],
                                 BCs=['fixed', 'periodic'],
                                 optArg=1.4, mxLoop=800, cal_wind=True,
                                 epsilon=7e-6, Phi=10000,
                                 debug=False)

#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np
from utils.PlotUtils import maskout_cmap


lat, lon = xr.broadcast(ds1.lat, ds1.lon)

fig, axes = pplt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

skip = 1
fontsize = 16

axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
            lonlines=20, latlines=10, lonlabels='b', latlabels='l',
            grid=False, labels=False)

cmap = maskout_cmap('bwr', [-0.1, -0.09, -0.08, -0.07, -0.06, -0.05,
                            -0.04, -0.03, -0.02, -0.01,  0., 0.01, 0.02,
                            0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                    [-0.01, 0.01])

ax = axes[0,0]
ax.contourf(Q, cmap=cmap, levels=np.linspace(-0.1, 0.1, 21))
ax.contour(h_ob, color='black', linewidth=2,
           levels=[-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25])
ax.set_title('observed mass and wind anomalies', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u_ob.values[::skip,::skip+1]*5, v_ob.values[::skip,::skip+1]*5,
              width=0.0016, headwidth=10., headlength=12., scale=300)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-30, 30])
ax.set_xlim([-160, -20])

ax = axes[0,1]
ax.contourf(Q, cmap=cmap, levels=np.linspace(-0.1, 0.1, 21))
ax.contour(h1, color='black', linewidth=2, levels=11)
ax.set_title('Gill-Matsuno response ($\epsilon=10^{-5}, \Phi=5\\times 10^3$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u1.values[::skip,::skip+1], v1.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=300)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-30, 30])
ax.set_xlim([-160, -20])

ax = axes[1,0]
ax.contourf(Q, cmap=cmap, levels=np.linspace(-0.1, 0.1, 21))
ax.contour(h2, color='black', linewidth=2, levels=11)
ax.set_title('Gill-Matsuno response ($\epsilon=7\\times 10^{-6}, \Phi=5\\times 10^3$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u2.values[::skip,::skip+1], v2.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=300)
ax.set_ylim([-30, 30])
ax.set_xlim([-160, -20])

ax = axes[1,1]
ax.contourf(Q, cmap=cmap, levels=np.linspace(-0.1, 0.1, 21))
ax.contour(h3, color='black', linewidth=2, levels=11)
ax.set_title('Gill-Matsuno response ($\epsilon=7\\times 10^{-6}, \Phi=10^4$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u3.values[::skip,::skip+1], v3.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=300)
ax.set_ylim([-30, 30])
ax.set_xlim([-160, -20])

fig.colorbar(p, loc='b', label='heating Q', ticks=0.01, length=1)
