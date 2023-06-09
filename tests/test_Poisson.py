# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% test global 1
import xarray as xr
import numpy as np
from xinvert import invert_Poisson, cal_flow, FiniteDiff


def test_poisson_atmos():
    ds = xr.open_dataset('./Data/Helmholtz_atmos.nc')
    
    vor = ds.vor.rename('vorticity')
    div = ds.div.rename('divergence')
    
    iParams = {
        'BCs'      : ['extend', 'periodic'],
        'undef'    : np.nan,
        'mxLoop'   : 5000,
        'tolerance': 1e-11,
    }
    
    vp = invert_Poisson(div, dims=['lat','lon'], iParams=iParams)
    sf = invert_Poisson(vor, dims=['lat','lon'], iParams=iParams)
    
    ux, vx = cal_flow(vp, dims=['lat', 'lon'], BCs=iParams['BCs'], vtype='velocitypotential')
    us, vs = cal_flow(sf, dims=['lat', 'lon'], BCs=iParams['BCs'], vtype='streamfunction')
    
    # verification
    fd = FiniteDiff({'X':'lon', 'Y':'lat', 'T':'time'},
                    BCs={'X':'periodic', 'Y':'extend'}, fill=0, coords='lat-lon')
    
    div0 = fd.divg((us, vs), ['X', 'Y'])
    vor0 = fd.curl( ux, vx , ['X', 'Y'])
    
    assert np.isclose(div0[:,1:-1], 0).all()
    assert np.isclose(vor0[:,1:-1], 0).all()


def test_poisson_ocean():
    ds = xr.open_dataset('./Data/Helmholtz_ocean.nc')
    
    vor = ds.vor[0]
    
    iParams = {
        'BCs'      : ['extend', 'periodic'],
        'undef'    : 0,
        'tolerance': 1e-9,
    }
    
    sf = invert_Poisson(vor, dims=['YG','XG'], iParams=iParams)
    
    us, vs = cal_flow(sf, dims=['YG', 'XG'], BCs=iParams['BCs'], vtype='streamfunction')
    
    # verification
    fd = FiniteDiff({'X':'XG', 'Y':'YG', 'T':'time'},
                    BCs={'X':'periodic', 'Y':'extend'}, fill=0, coords='lat-lon')
    
    div0 = fd.divg((us, vs), ['X', 'Y'])
    
    assert np.isclose(div0[1:-1], 0).all()

