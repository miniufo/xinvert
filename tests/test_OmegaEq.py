# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% test global data from Dr. Yuan Zhao
import xarray as xr
import numpy as np
from xinvert import invert_omega, FiniteDiff

def test_omega_atmos():
    ds = xr.open_dataset('./Data/atmos3D.nc', decode_times=False)
    
    # ds, grid = add_latlon_metrics(dset, dims={'Z':'LEV', 'Y':'lat', 'X':'lon'},
    #                               boundary={'X':'periodic', 'Y':'extend', 'Z':'extend'})
    
    ds['LEV'] = ds['LEV'] * 100
    
    # zonal running smooth of polar grid
    def smooth(v, gridpoint=13, lat=80):
        rolled = v.pad({'lon':(gridpoint,gridpoint)},mode='wrap')\
                  .rolling(lon=gridpoint, center=True, min_periods=1).mean()\
                  .isel(lon=slice(gridpoint, -gridpoint))
        return xr.where(np.abs(v-v+v.lat)>lat, rolled, v)
    
    omega = 7.292e-5
    Rd = 287.04
    Cp = 1004.88
    p   = ds.LEV
    Psfc= ds.psfc
    f   = 2*omega*np.sin(np.deg2rad(ds.lat))
    
    T = smooth(ds.T, lat=80)
    U = smooth(ds.U, lat=80)
    V = smooth(ds.V, lat=80)
    W = smooth(ds.Omega, lat=80)
    
    th   = T * (100000 / p)**(Rd/Cp)
    TH   = th.mean(['lat', 'lon'])
    dTHdp= TH.differentiate('LEV')
    RPiP = (Rd * T / p / TH)
    S    = - RPiP * dTHdp
    
    
    #### calculate forcings using FiniteDiff (which is part of xinvert) ###
    fd = FiniteDiff({'X':'lon', 'Y':'lat', 'Z':'LEV'},
                    BCs={'X':('periodic','periodic'),
                         'Y':('reflect','reflect'),
                         'Z':('extend','extend')}, fill=0, coords='lat-lon')
    
    vor  = fd.curl(U,V).load()
    _, tmp = xr.broadcast(vor, vor.mean('lon'))
    vor[:,0,:] = tmp[:,0,:]
    vor[:,-1,:] = tmp[:,-1,:]
    
    ########## traditional form of forcings ##########
    grdthx, grdthy = fd.grad(th)
    grdvrx, grdvry = fd.grad(vor)
    
    F1 = fd.Laplacian((U * grdthx + V * grdthy) * RPiP)
    F2 = ((U * grdvrx + V * grdvry) * f).differentiate('LEV')
    
    FAll = (F1 + F2)
    FAll = smooth(xr.where(np.isinf(FAll), np.nan, FAll), lat=83)
    
    ########### Q-vector form of forcings ###########
    ux, uy = fd.grad(U)
    vx, vy = fd.grad(V)
    
    Qx = - RPiP * (ux * grdthx + vx * grdthy)
    Qy = - RPiP * (uy * grdthx + vy * grdthy)
    
    FQvec = -2 * fd.divg((Qx, Qy), dims=['X', 'Y'])
    FQvec = smooth(xr.where(np.isinf(FQvec), np.nan, FQvec), gridpoint=17, lat=85)
    
    ##### prepare lower boundary for inversion ######
    p3D = T-T+p
    
    FAll  = FAll.where(p<=Psfc)
    FQvec = FQvec.where(p<=Psfc)
    WBC = xr.where(p3D<=Psfc, 0, W).load()
    
    ##### invert #####
    iParams = {
        'BCs'      : ['fixed', 'fixed', 'periodic'],
        'tolerance': 1e-16,
    }
    
    mParams = {'N2': S}
    
    WQG   = invert_omega(FAll, dims=['LEV', 'lat', 'lon'],
                         iParams=iParams, mParams=mParams)
    WQvec = invert_omega(FQvec, dims=['LEV', 'lat', 'lon'],
                         iParams=iParams, mParams=mParams)
    WQvTp = invert_omega(FQvec, dims=['LEV', 'lat', 'lon'], icbc=WBC,
                         iParams=iParams, mParams=mParams)
    
    assert np.isclose(  WQG.max(),  0.28392872)
    assert np.isclose(  WQG.min(), -0.32804008)
    assert np.isclose(WQvec.max(),  0.11733621)
    assert np.isclose(WQvec.min(), -0.37005570)
    # WBC is non-zero only below the surface, where the forcing is missing.
    # Values of icbc on invalid forcing cells must not enter the stencil and
    # contaminate the valid atmospheric solution.
    xr.testing.assert_allclose(WQvTp, WQvec)

# Historical plotting and external-data experiments are archived in
# examples/legacy/omega_experiments.txt; they are not regression tests.







