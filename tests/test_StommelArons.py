# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases for deep ocean circulation
import numpy as np
import xarray as xr
from xinvert import invert_Stommel, invert_StommelMunk, cal_flow

def test_StommelAron():
    def add_source(msrc, olon, olat, amp, rad):
        lons = msrc.lon
        lats = msrc.lat
        
        msrc += amp * np.exp(-((lats-olat)**2/rad + (lons-olon)**2/rad/4))
    
    
    ds = xr.open_dataset('./Data/SODA_curl.nc')
    
    # mass source/sink over the global ocean
    msrc = (ds.curl-ds.curl)[0].load()
    
    add_source(msrc, 330,  63, -1e-3, 30) # source at the North Atlantic
    add_source(msrc, 350,  67, -1e-3, 30) # source at the North Atlantic
    add_source(msrc, 189, -70, -1e-3, 30) # source at the Ross Sea
    
    # uniform upwelling sink
    totalmsrc = (msrc*np.cos(np.deg2rad(msrc.lat))).sum()
    totalarea = ((msrc-msrc+1)*np.cos(np.deg2rad(msrc.lat))).sum()
    
    msrc = msrc - totalmsrc / totalarea
    
    totalmsrc = (msrc*np.cos(np.deg2rad(msrc.lat))).sum()
    
    # should be close to zero
    print(f'total mass source/sink: {totalmsrc.values}')
    
    # invert using Stommel-Munk model
    iParams = {
        'BCs'      : ['extend', 'periodic'],
        'mxLoop'   : 5000,
        'optArg'   : 1.8,
        'tolerance': 1e-12,
        'undef'    : np.nan,
    }
    
    mParams1 = {'R': 1e-1, 'D': 500}
    mParams2 = {'R': 1e-2, 'D': 500, 'A4':5e3}
    
    h1 = invert_Stommel(msrc, dims=['lat','lon'],
                        iParams=iParams, mParams=mParams1)
    h2 = invert_StommelMunk(msrc, dims=['lat','lon'],
                        iParams=iParams, mParams=mParams2)
    
    u1, v1 = cal_flow(h1, dims=['lat','lon'], BCs=['extend', 'periodic'])
    u2, v2 = cal_flow(h2, dims=['lat','lon'], BCs=['extend', 'periodic'])
    
    assert np.isclose(h2.max(),  20485498.)
    assert np.isclose(h2.min(), -4798475.5)


#%% plot wind and streamfunction
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



