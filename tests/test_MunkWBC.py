# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xinvert import invert_StommelMunk


def test_Munk_ideal():
    xnum = 201
    ynum = 151
    Lx = 1e7 # 10,000 km
    Ly = 2 * np.pi * 1e6 # 6249 km
    R = 0.0001 # Rayleigh friction (default 0.02)
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
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'fixed'],
        'mxLoop'   : 4000,
        'tolerance': 1e-14,
        'optArg'   : 1.0,
        'undef'    : np.nan,
    }
    
    mParams1 = {'A4':5e3, 'beta':beta, 'R':R, 'D':depth}
    mParams2 = {'A4':5e2, 'beta':beta, 'R':R, 'D':depth}
    
    h1 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                            iParams=iParams, mParams=mParams1)
    h2 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                            iParams=iParams, mParams=mParams2)
    
    assert h1.shape == h2.shape == curl_tau.shape
    assert h1.dims  == h2.dims  == curl_tau.dims
    assert np.isclose(h1.max(), 388730.8493746)
    assert np.isclose(h2.max(), 399667.8611556)


def test_Munk_real():
    D = 100
    rho0 = 1027
    R = 1e-4
    
    ds = xr.open_dataset('./Data/SODA_curl.nc')
    curl = ds.curl
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'periodic'],
        'mxLoop'   : 4000,
        'tolerance': 1e-14,
        'optArg'   : 1.0,
        'undef'    : np.nan,
    }
    
    mParams = {'A4':3e3, 'D':D, 'R':R, 'rho0':rho0}
    
    h1 = invert_StommelMunk(curl, dims=['lat','lon'],
                            iParams=iParams, mParams=mParams)
    
    assert h1.shape == curl.shape
    assert h1.dims  == curl.dims
    assert np.isclose(np.abs(h1).max(), 1103877.)
    



