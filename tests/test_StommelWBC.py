# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xinvert import invert_Stommel, invert_StommelMunk, cal_flow


def test_stommel_idealized():
    xnum = 201
    ynum = 151
    Lx = 1e7 # 10,000 km
    Ly = 2 * np.pi * 1e6 # 6249 km
    R = 0.0008 # Rayleigh friction (default 0.02)
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
        'mxLoop'   : 5000,
        'optArg'   : 1.9,
        'tolerance': 1e-12,
    }
    
    mParams1 = {'beta': 0   , 'R': R, 'D': depth}
    mParams2 = {'beta': beta, 'R': R, 'D': depth}
    mParams3 = {'beta': 0   , 'R': R, 'D': depth, 'A4':0}
    mParams4 = {'beta': beta, 'R': R, 'D': depth, 'A4':0}
    
    
    S1 = invert_Stommel(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                        iParams=iParams, mParams=mParams1)
    S2 = invert_Stommel(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                        iParams=iParams, mParams=mParams2)
    S3 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                            iParams=iParams, mParams=mParams3)
    S4 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                            iParams=iParams, mParams=mParams4)
    
    u1, v1 = cal_flow(S1, dims=['ydef','xdef'], coords='cartesian')
    u2, v2 = cal_flow(S2, dims=['ydef','xdef'], coords='cartesian')
    u3, v3 = cal_flow(S3, dims=['ydef','xdef'], coords='cartesian')
    u4, v4 = cal_flow(S4, dims=['ydef','xdef'], coords='cartesian')
    
    
    # analytical solution for non-rotating case
    gamma = F * np.pi / R / Ly
    p = np.exp(-np.pi * Lx / Ly)
    q = 1
    
    h_a = -gamma * (Ly / np.pi)**2 * np.sin(np.pi * ydef / Ly) * (
              np.exp((xdef - Lx)*np.pi/Ly) + np.exp(-xdef*np.pi/Ly) - 1
          )
    
    assert (S1-S3).max() <= 17786.14518303
    assert (S2-S4).max() <= 61902.07682051



def test_stommel_real():
    ds = xr.open_dataset('./Data/SODA_curl.nc')
    
    curl_Jan = ds.curl[0]
    curl_Jul = ds.curl[6]
    
    R = 2e-4
    depth = 100
    
    # invert
    iParams = {
        'BCs'      : ['extend', 'periodic'],
        'mxLoop'   : 5000,
        'optArg'   : 1,
        'tolerance': 1e-12,
        'undef'    : np.nan,
    }
    
    mParams1 = {'R': R, 'D': depth}
    mParams2 = {'R': R, 'D': depth, 'A4':5e3}
    
    h1 = invert_Stommel(curl_Jan, dims=['lat','lon'],
                        iParams=iParams, mParams=mParams1)
    h2 = invert_Stommel(curl_Jul, dims=['lat','lon'],
                        iParams=iParams, mParams=mParams1)
    
    h11 = invert_StommelMunk(curl_Jan, dims=['lat','lon'],
                        iParams=iParams, mParams=mParams2)
    h22 = invert_StommelMunk(curl_Jul, dims=['lat','lon'],
                        iParams=iParams, mParams=mParams2)
    
    u1, v1 = cal_flow(h1, dims=['lat','lon'], BCs=['extend', 'periodic'])
    u2, v2 = cal_flow(h2, dims=['lat','lon'], BCs=['extend', 'periodic'])
    
    assert np.isclose(h1.min(), -437846.34375)
    assert np.isclose(h1.max(),  536152.25000)
    assert np.isclose(h2.min(), -414698.03125)
    assert np.isclose(h2.max(),  826675.93750)
    
    assert np.isclose(h11.min(), -431810.0625)
    assert np.isclose(h11.max(),  432051.2500)
    assert np.isclose(h22.min(), -435084.8125)
    assert np.isclose(h22.max(),  725193.1250)
    

