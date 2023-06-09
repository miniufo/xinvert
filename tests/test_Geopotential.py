# -*- coding: utf-8 -*-
"""
Created on 2021.05.14

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
import xarray as xr
import numpy as np
from xinvert import FiniteDiff, invert_Poisson

def test_Geopotential():
    ds = xr.open_dataset('./Data/atmos3D.nc').sel({'LEV':500})
    
    u = ds.U
    v = ds.V
    h = ds.hgt
    
    
    # calculate force
    Rearth = 6371200.0 # consistent with GrADS
    omega = 7.292e-5    # angular speed of the earth rotation
    
    fd = FiniteDiff(dim_mapping={'Y':'lat', 'X':'lon'},
                    BCs={'Y':'reflect', 'X':'periodic'}, coords='lat-lon')
    
    vort = fd.curl(u, v)
    cosF = np.cos(np.deg2rad(u.lat))
    sinF = np.sin(np.deg2rad(u.lat))
    beta = 2.0 * omega * cosF / Rearth
    fcor = 2.0 * omega * sinF
    
    
    # force 2
    tmp2 = fd.grad(u, 'X', BCs='periodic') * u
    frc2 = -fd.grad(tmp2, 'X').load()
    frc2 = xr.where(np.isfinite(frc2), frc2, np.nan)
    
    # force 3
    tmp3 = fd.grad(v, 'X') * u * cosF
    frc3 = -fd.grad(tmp3, 'Y', BCs='extend').load() / cosF
    frc3 = xr.where(np.isfinite(frc3), frc3, np.nan)
    
    # force 4
    tmp4 = fd.grad(u, 'Y', BCs='extend') * v
    frc4 = -fd.grad(tmp4, 'Y').load()
    frc4 = xr.where(np.isfinite(frc4), frc4, np.nan)
    
    # force 5
    tmp5 = fd.grad(v, 'Y', BCs='extend') * v*cosF
    frc5 = -fd.grad(tmp5, 'Y', BCs='extend').load() / cosF
    frc5 = xr.where(np.isfinite(frc5), frc5, np.nan)
    
    # force 9
    frc9 = (fcor * vort).load()
    frc9 = xr.where(np.isfinite(frc9), frc9, np.nan)
    
    # force 10
    frc10 = -(u * beta).load()
    frc10 = xr.where(np.isfinite(frc10), frc10, np.nan)
    
    # force 11
    tmp11 = -u**2 * sinF / Rearth
    frc11 = fd.grad(tmp11, 'Y', BCs='extend').load() / cosF
    frc11 = xr.where(np.isfinite(frc11), frc11, np.nan)
    
    # force 12
    tmp12 = u * v * sinF / cosF / Rearth
    frc12 = fd.grad(tmp12, 'X').load()
    frc12 = xr.where(np.isfinite(frc12), frc12, np.nan)
    
    
    # all forces
    frcAll = frc2 + frc3 + frc4 + frc5 + frc9 + frc10 + frc11 + frc12
    frcAll = xr.where(np.isfinite(frcAll), frcAll, np.nan)
    frcAll = frcAll.fillna(0)
    
    
    # invert
    
    hbc = (h*9.81).load()
    
    iParams = {
        'BCs'      : ['fixed', 'fixed'],
        'mxLoop'   : 5000,
        'tolerance': 1e-11,
    }
    
    zeros = h-h
    
    sf2  = invert_Poisson(frc2 , dims=['lat','lon'], iParams=iParams)
    sf3  = invert_Poisson(frc3 , dims=['lat','lon'], iParams=iParams)
    sf4  = invert_Poisson(frc4 , dims=['lat','lon'], iParams=iParams)
    sf5  = invert_Poisson(frc5 , dims=['lat','lon'], iParams=iParams)
    sf9  = invert_Poisson(frc9 , dims=['lat','lon'], iParams=iParams)
    sf10 = invert_Poisson(frc10, dims=['lat','lon'], iParams=iParams)
    sf11 = invert_Poisson(frc11, dims=['lat','lon'], iParams=iParams)
    sf12 = invert_Poisson(frc12, dims=['lat','lon'], iParams=iParams)
    sfb  = invert_Poisson(zeros, dims=['lat','lon'], icbc=hbc, iParams=iParams)
    sf   = invert_Poisson(frcAll,dims=['lat','lon'], icbc=hbc, iParams=iParams)
    
    assert np.isclose(sf, (sf2+sf3+sf4+sf5+sf9+sf10+sf11+sf12+sfb),
                      rtol=5e-5).all()


#%% plot
# import proplot as pplt

# fig, axes = pplt.subplots(nrows=10, ncols=2, figsize=(10, 21),
#                           sharex=3, sharey=3,
#                           proj=pplt.Proj('cyl', lon_0=180))

# fontsize = 16

# ax = axes[0,0]
# p=ax.contourf(frc2*1e8, cmap='jet', levels=np.linspace(-0.1, 0.1,21))
# ax.set_title('forcing 2', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[1,0]
# p=ax.contourf(frc3*1e8, cmap='jet', levels=np.linspace(-0.1, 0.1,21))
# ax.set_title('forcing 3', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[2,0]
# p=ax.contourf(frc4*1e8, cmap='jet', levels=np.linspace(-0.1, 0.1,21))
# ax.set_title('forcing 4', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[3,0]
# p=ax.contourf(frc5*1e8, cmap='jet', levels=np.linspace(-0.1, 0.1,21))
# ax.set_title('forcing 5', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[4,0]
# p=ax.contourf(frc9*1e8, cmap='jet', levels=np.linspace(-1, 1,21))
# ax.set_title('forcing 9', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[5,0]
# p=ax.contourf(frc10*1e8, cmap='jet', levels=np.linspace(-1, 1,21))
# ax.set_title('forcing 10', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[6,0]
# p=ax.contourf(frc11*1e8, cmap='jet', levels=np.linspace(-0.1, 0.1,21))
# ax.set_title('forcing 11', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[7,0]
# p=ax.contourf(frc12*1e8, cmap='jet', levels=np.linspace(-0.1, 0.1,21))
# ax.set_title('forcing 12', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[8,0]
# p=ax.contourf(frcAll-frcAll, cmap='jet', levels=np.linspace(-0.1, 0.1,21))
# ax.set_title('zero', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[9,0]
# p=ax.contourf(frcAll*1e8, cmap='jet', levels=np.linspace(-1, 1,21))
# ax.set_title('all forcing', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)




# ax = axes[0,1]
# p=ax.contourf(sf2/9.81, cmap='jet', levels=np.linspace(-30, 30,25), extend='both')
# ax.set_title('response 2', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[1,1]
# p=ax.contourf(sf3/9.81, cmap='jet', levels=np.linspace(-30, 30,25), extend='both')
# ax.set_title('response 3', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[2,1]
# p=ax.contourf(sf4/9.81, cmap='jet', levels=np.linspace(-30, 30,25), extend='both')
# ax.set_title('response 4', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[3,1]
# p=ax.contourf(sf5/9.81, cmap='jet', levels=np.linspace(-30, 30,25), extend='both')
# ax.set_title('response 5', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[4,1]
# p=ax.contourf(sf9/9.81, cmap='jet', levels=np.linspace(-2000, 0,21), extend='both')
# ax.set_title('response 9', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[5,1]
# p=ax.contourf(sf10/9.81, cmap='jet', levels=np.linspace(1500, 2300,17), extend='both')
# ax.set_title('response 10', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[6,1]
# p=ax.contourf(sf11/9.81, cmap='jet', levels=np.linspace(0, 46,24), extend='both')
# ax.set_title('response 11', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[7,1]
# p=ax.contourf(sf12/9.81, cmap='jet', levels=np.linspace(-12, 12,25), extend='both')
# ax.set_title('response 12', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[8,1]
# p=ax.contourf(sfb/9.81, cmap='jet', levels=np.linspace(5074, 5100,27), extend='both')
# ax.set_title('response boundary', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# ax = axes[9,1]
# p=ax.contourf(sf/9.81, cmap='jet', levels=np.linspace(4800, 6000,25), extend='both')
# ax.set_title('response all', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='r', length=1)

# axes.format(abc='(a)', coast=True,
#             lonlines=60, latlines=30, lonlabels='b', latlabels='l',
#             grid=False, labels=False)


# fig.colorbar(p, loc='b', label='', ticks=200, cols=(1,2), length=1)

#%%
# import proplot as pplt

# fig, axes = pplt.subplots(nrows=4, ncols=2, figsize=(11, 13.3),
#                           sharex=3, sharey=3,
#                           proj=pplt.Proj('cyl', lon_0=180))

# fontsize = 16

# ax = axes[0,0]
# p=ax.contourf((sf9+sf10+sfb)/9.81, cmap='jet', levels=np.linspace(4900, 5900,21))
# ax.set_title('forcing 9+10+bdy', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', length=1, label='')

# ax = axes[1,0]
# p=ax.contourf((sf9+sf10+sf11+sf12+sfb)/9.81, cmap='jet', levels=np.linspace(4900, 5900,21))
# ax.set_title('forcing 9+10+11+12+bdy', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', length=1, label='')

# ax = axes[2,0]
# p=ax.contourf((sf2+sf3+sf4+sf5+sf9+sf10+sf11+sf12+sfb)/9.81, cmap='jet', levels=np.linspace(4900, 5900,21))
# ax.set_title('forcing  2+3+4+5+9+10+11+12+bdy', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', length=1, label='')

# ax = axes[3,0]
# p=ax.contourf(sf/9.81, cmap='jet', levels=np.linspace(4900, 5900,21))
# ax.set_title('forcing all', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', length=1, label='')


# ax = axes[0,1]
# p=ax.contourf((sf9+sf10+sfb)/9.81-h, cmap='jet', levels=np.linspace(-60, 60,25))
# ax.set_title('9+10+bdy - h', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', length=1, label='')

# ax = axes[1,1]
# p=ax.contourf((sf9+sf10+sf11+sf12+sfb)/9.81-h, cmap='jet', levels=np.linspace(-60, 60,25))
# ax.set_title('9+10+11+12+bdy - h', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', length=1, label='')

# ax = axes[2,1]
# p=ax.contourf((sf2+sf3+sf4+sf5+sf9+sf10+sf11+sf12+sfb)/9.81-h, cmap='jet', levels=np.linspace(-24, 24,25))
# ax.set_title('2+3+4+5+9+10+11+12+bdy - h', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', length=1, label='')

# ax = axes[3,1]
# p=ax.contourf(sf/9.81-h, cmap='jet', levels=np.linspace(-24, 24,25))
# ax.set_title('all - h', fontsize=fontsize)
# ax.set_ylim([-80, 80])
# ax.set_xlim([-180, 180])
# ax.colorbar(p, loc='b', length=1, label='')

# axes.format(abc='(a)', coast=True,
#             lonlines=60, latlines=30, lonlabels='b', latlabels='l',
#             grid=False, labels=False)



