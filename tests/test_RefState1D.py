# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases for deep ocean circulation (TODO)
# import numpy as np
# import xarray as xr


# ds = xr.open_dataset('./xinvert/Data/barotropicData.nc', chunks={'time':1})

# PV = ds.zeta / ds.h
# r  = 6371200.0 * np.cos(np.deg2rad(ds.lat))
# ctr = PV.mean('lon')
# C0  = (ds.u.mean('lon') + 7.292E-5 * r) * 2 * np.pi * r
# hm = ds.h.mean('lon')

# #%% invert using Stommel-Munk model
# from xinvert.xinvert import invert_RefState1D

# iParams = {
#     'BCs'      : ['fixed'],
#     'mxLoop'   : 5000,
#     'optArg'   : 1.8,
#     'tolerance': 1e-12,
#     'undef'    : np.nan,
#     'debug'    : True,
# }


# mParams = {'C0':C0}

# h1 = invert_RefState1D(ctr, dims=['lat'], icbc=hm,
#                        iParams=iParams, mParams=mParams)


# #%% plot wind and streamfunction
# import proplot as pplt


# fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(11,10),
#                           sharex=3, sharey=3, proj='cyl',
#                           proj_kw={'central_longitude':180})

# skip = 7
# fontsize = 15

# ygrid, xgrid = xr.broadcast(msrc.lat, msrc.lon)

# mag = (u1**2 + v1**2) / 2

# ax = axes[0]
# m = ax.contourf(h1/1e6, levels=np.linspace(-5, 5, 20), cmap='bwr_r')
# ax.contour(msrc, levels=[0.0003, 0.0005, 0.0008], lw=2, color='k')
# ax.colorbar(m, loc='r', label='')
# ax.quiver(xgrid.values[::skip+1,::skip], ygrid.values[::skip+1,::skip],
#           u1.where(mag<6).values[::skip+1,::skip], v1.values[::skip+1,::skip],
#           width=0.0014, headwidth=8., headlength=12., scale=70)
# ax.set_ylabel('longitude', fontsize=fontsize-1)
# ax.set_xlabel('latitude', fontsize=fontsize-1)
# ax.set_title('abyssal circulation (Stommel model)', fontsize=fontsize)

# ax = axes[1]
# m = ax.contourf(h2/1e6, levels=np.linspace(-5, 5, 20), cmap='bwr_r')
# ax.contour(msrc, levels=[0.0003, 0.0005, 0.0008], lw=2, color='k')
# ax.colorbar(m, loc='r', label='')
# ax.quiver(xgrid.values[::skip+1,::skip], ygrid.values[::skip+1,::skip],
#           u2.where(mag<6).values[::skip+1,::skip], v2.values[::skip+1,::skip],
#           width=0.0014, headwidth=8., headlength=12., scale=70)
# ax.set_ylabel('longitude', fontsize=fontsize-1)
# ax.set_xlabel('latitude', fontsize=fontsize-1)
# ax.set_title('abyssal circulation (Stommel-Munk model)', fontsize=fontsize)


# axes.format(abc='(a)', land=True, coast=True, landcolor='gray')



