# -*- coding: utf-8 -*-
"""
Created on 2021.09.27

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% global Poisson (TODO)
# import xarray as xr
# import numpy as np
# from xinvert.xinvert.core import invert_Poisson, invert_MultiGrid


# ds = xr.open_dataset('D:/mitgcm.nc')

# vor = ds.vor.where(ds.vor!=0)[0]

# # sf1 = invert_Poisson(vor,
# #                       dims=['YG','XG'], BCs=['fixed', 'periodic'],
# #                       mxLoop=5000, tolerance=1e-13)
# sf2, fs, ss = invert_MultiGrid(invert_Poisson, vor,
#                        dims=['YG','XG'], BCs=['fixed', 'periodic'],
#                       mxLoop=5000, tolerance=1e-13)



#%% plot wind and streamfunction
# import proplot as pplt
# import xarray as xr
# import numpy as np

# u = ds.u.where(ds.u!=0)[0].load()
# v = ds.v.where(ds.v!=0)[0].load()
# m = np.hypot(u, v)

# lat, lon = xr.broadcast(u.lat, u.lon)

# fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(12, 13), sharex=3, sharey=3,
#                           proj=pplt.Proj('cyl', lon_0=180))

# fontsize = 16

# axes.format(abc=True, abcloc='l', abcstyle='(a)', coast=True,
#             lonlines=60, latlines=30, lonlabels='b', latlabels='l',
#             grid=False, labels=False)

# ax = axes[0]
# p = ax.contourf(lon, lat, vor[0], cmap='seismic',
#                 levels=np.linspace(-1e-4, 1e-4, 21))
# ax.set_title('relative vorticity', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='', ticks=1e-5, length=0.895)

# ax = axes[1]
# p = ax.contourf(lon, lat, sf[0], levels=31, cmap='jet')
# ax.quiver(lon.values, lat.values, u.values, v.values,
#               width=0.0007, headwidth=12., headlength=15.)
#               # headwidth=1, headlength=3, width=0.002)
# ax.set_title('wind field and inverted streamfunction', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='', length=0.895)



