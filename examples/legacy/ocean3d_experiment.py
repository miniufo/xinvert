# -*- coding: utf-8 -*-
"""
Created on 2022.10.28

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases (TODO)
# import numpy as np
# import xarray as xr
# from xinvert.xinvert import FiniteDiff


# ds = xr.open_dataset('./xinvert/Data/SODA.nc')

# fd = FiniteDiff(dim_mapping={'T':'time', 'Z':'lev', 'Y':'lat', 'X':'lon'},
#                 coords='lat-lon', BCs={'Y':'extend', 'X':'periodic'})

# taux = ds.taux
# tauy = ds.tauy

# eps = 1e-4
# f   = 2 * 7.292e-5 * np.sin(np.deg2rad(ds.lat))
# N2  = 1e-2
# k   = 1e-5

# c1 = eps / (eps**2.0 + f**2.0)
# c2 = f   / (eps**2.0 + f**2.0)

# Fx = taux * c1 + tauy * c2
# Fy = tauy * c1 - taux * c2

# F1  = fd.divg([taux*c1,  tauy*c1], dims=['X', 'Y'])
# F2  = fd.divg([tauy*c2, -taux*c2], dims=['X', 'Y'])

# F = F1 + F2


# #%% invert
# from xinvert.xinvert import invert_3DOcean, cal_flow

# F3D = xr.concat([F, F-F, F-F], dim='lev').transpose('time','lev','lat','lon')
# F3D['lev'] = np.array([0, 100, 10000])

# F3D_interp = F3D.interp(lev=np.linspace(0, 500, 51))


# iParams = {
#     'BCs'      : ['fixed', 'extend', 'periodic'],
#     'mxLoop'   : 500,
#     'optArg'   : 1,
#     'tolerance': 1e-12,
#     'undef'    : np.nan,
#     'debug'    : True,
# }

# mParams = {'epsilon':eps, 'N2':N2, 'k':k}

# h1 = invert_3DOcean(F3D_interp, dims=['lev','lat','lon'],
#                     iParams=iParams, mParams=mParams)

# # u1, v1 = cal_flow(h1, dims=['lat','lon'], BCs=['extend', 'periodic'])



# #%% plot forcings
# import proplot as pplt


# fig, axes = pplt.subplots(nrows=3, ncols=3, figsize=(12,10),
#                           sharex=3, sharey=3)

# fontsize = 16
# step = 0
# levels = np.linspace(-4000, 4000, 17)

# ax = axes[0,0]
# p=ax.contourf((taux*c1)[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('c1 * taux', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')

# ax = axes[1,0]
# p=ax.contourf((tauy*c2)[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('c2 * tauy', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')

# ax = axes[2,0]
# p=ax.contourf(Fx[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('Fx', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')

# ax = axes[0,1]
# p=ax.contourf((tauy*c1)[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('c1 * tauy', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')

# ax = axes[1,1]
# p=ax.contourf((-taux*c2)[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('-c2 * taux', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')

# ax = axes[2,1]
# p=ax.contourf(Fy[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('Fy', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')

# levels = np.linspace(-10, 10, 21)
# ax = axes[0,2]
# p=ax.contourf((F1*1e3)[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('F1', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')

# ax = axes[1,2]
# p=ax.contourf((F2*1e3)[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('F2', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')

# ax = axes[2,2]
# p=ax.contourf((F*1e3)[step], cmap='bwr', levels=levels, extend='both')
# ax.set_title('F = F1 + F2', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='')


# axes.format(abc='(a)')






# #%% plot wind and streamfunction
# import proplot as pplt
# import xarray as xr
# import numpy as np


# lat, lon = xr.broadcast(u1.lat, u1.lon)

# fig, axes = pplt.subplots(nrows=2, ncols=2, figsize=(11,6.8), sharex=3, sharey=3,
#                           proj=pplt.Proj('cyl', lon_0=180))

# skip = 3
# fontsize = 16

# ax = axes[0,0]
# p=ax.contourf(curl_Jan, cmap='jet',levels=np.linspace(-7e-7,7e-7,29))
# ax.set_title('wind stress curl (January)',
#              fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', label='', length=0.93)

# axes.format(abc='(a)', coast=True,
#             lonlines=40, latlines=15, lonlabels='b', latlabels='l',
#             grid=False, labels=False)


