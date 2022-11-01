# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr


lon = xr.DataArray(np.linspace(0, 360, 144), dims='lon',
                   coords={'lon':np.linspace(0, 360, 144)})
lat = xr.DataArray(np.linspace(-90, 90, 73), dims='lat',
                   coords={'lat':np.linspace(-90, 90, 73)})

lat, lon = xr.broadcast(lat, lon)

Q1 = 0.05*np.exp(-((lat-0)**2+(lon-120)**2)/100.0)
Q2 = 0.05*np.exp(-((lat-10)**2+(lon-120)**2)/100.0) \
   - 0.05*np.exp(-((lat+10)**2+(lon-120)**2)/100.0)
Q3 = 0.05*np.exp(-((lat-10)**2+(lon-120)**2)/100.0)


#%% invert
from xinvert.xinvert import invert_GillMatsuno, cal_flow

iParams = {
    'BCs'      : ['fixed', 'periodic'],
    'mxLoop'   : 2000,
    'tolerance': 1e-8,
    'optArg'   : 1.4,
}

mParams = {
    'epsilon': 1e-5,
    'Phi'    : 5000,
}

h1 = invert_GillMatsuno(Q1, dims=['lat','lon'], iParams=iParams, mParams=mParams)
h2 = invert_GillMatsuno(Q2, dims=['lat','lon'], iParams=iParams, mParams=mParams)
h3 = invert_GillMatsuno(Q3, dims=['lat','lon'], iParams=iParams, mParams=mParams)

u1, v1 = cal_flow(h1, dims=['lat','lon'], BCs=['fixed','periodic'],
                  mParams=mParams, vtype='GillMatsuno')
u2, v2 = cal_flow(h2, dims=['lat','lon'], BCs=['fixed','periodic'],
                  mParams=mParams, vtype='GillMatsuno')
u3, v3 = cal_flow(h3, dims=['lat','lon'], BCs=['fixed','periodic'],
                  mParams=mParams, vtype='GillMatsuno')


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

axes.format(abc='(a)', coast=True,
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


ds = xr.open_dataset('./xinvert/Data/MJO.nc')

h_ob = ds.hl
u_ob = ds.ul
v_ob = ds.vl
Q    = (ds.ol*-0.0015).where(np.abs(ds.lat)<60, 0)



#%% invert
from xinvert.xinvert import invert_GillMatsuno, cal_flow

iParams = {
    'BCs'      : ['fixed', 'periodic'],
    'mxLoop'   : 2000,
    'tolerance': 1e-12,
    'optArg'   : 1.4,
}

mParams1 = {'epsilon': 1e-5, 'Phi': 5000}
mParams2 = {'epsilon': 7e-6, 'Phi': 8000}
mParams3 = {'epsilon': 7e-6, 'Phi': 10000}

h1 = invert_GillMatsuno(Q, dims=['lat','lon'], iParams=iParams, mParams=mParams1)
h2 = invert_GillMatsuno(Q, dims=['lat','lon'], iParams=iParams, mParams=mParams2)
h3 = invert_GillMatsuno(Q, dims=['lat','lon'], iParams=iParams, mParams=mParams3)

u1, v1 = cal_flow(h1, dims=['lat','lon'], BCs=['fixed','periodic'],
                  mParams=mParams1, vtype='GillMatsuno')
u2, v2 = cal_flow(h2, dims=['lat','lon'], BCs=['fixed','periodic'],
                  mParams=mParams2, vtype='GillMatsuno')
u3, v3 = cal_flow(h3, dims=['lat','lon'], BCs=['fixed','periodic'],
                  mParams=mParams3, vtype='GillMatsuno')


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np
from utils.PlotUtils import maskout_cmap


lat, lon = xr.broadcast(ds.lat, ds.lon)

fig, axes = pplt.subplots(nrows=2, ncols=2, figsize=(10.5, 6), sharex=3, sharey=3,
                          proj=pplt.Proj('cyl', lon_0=180))

skip = 1
fontsize = 13

axes.format(abc='(a)', coast=True,
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
ax.set_title('Gill-Matsuno response ($\epsilon=1\\times 10^{-5}, \Phi=5\\times 10^3$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u1.values[::skip,::skip+1], v1.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=300)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([-30, 30])
ax.set_xlim([-160, -20])

ax = axes[1,0]
ax.contourf(Q, cmap=cmap, levels=np.linspace(-0.1, 0.1, 21))
ax.contour(h2, color='black', linewidth=2, levels=11)
ax.set_title('Gill-Matsuno response ($\epsilon=7\\times 10^{-6}, \Phi=8\\times 10^3$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u2.values[::skip,::skip+1], v2.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=300)
ax.set_ylim([-30, 30])
ax.set_xlim([-160, -20])

ax = axes[1,1]
p=ax.contourf(Q, cmap=cmap, levels=np.linspace(-0.1, 0.1, 21))
ax.contour(h3, color='black', linewidth=2, levels=11)
ax.set_title('Gill-Matsuno response ($\epsilon=7\\times 10^{-6}, \Phi=1\\times 10^4$)', fontsize=fontsize)
ax.quiver(lon.values[::skip,::skip+1], lat.values[::skip,::skip+1],
              u3.values[::skip,::skip+1], v3.values[::skip,::skip+1],
              width=0.0016, headwidth=10., headlength=12., scale=300)
ax.set_ylim([-30, 30])
ax.set_xlim([-160, -20])

fig.colorbar(p, loc='b', label='heating Q', ticks=0.01, length=1)

