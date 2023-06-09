# -*- coding: utf-8 -*-
"""
Created on 2021.04.23

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xinvert import invert_Stommel, invert_StommelMunk

def test_Ishida():
    xnum = 251
    ynum = 151
    Lx = 1e7 # 10,000 km
    Ly = 2 * np.pi * 1e6 # 6249 km
    R = 0.0009 # Rayleigh friction (default 0.02)
    depth = 200 # fluid depth 200
    beta = 2.2e-11 # gradient of f 1e-13
    undef = -9999
    
    xdef = xr.DataArray(np.linspace(0, Lx, xnum), dims='xdef',
                        coords={'xdef':np.linspace(0, Lx, xnum)})
    ydef = xr.DataArray(np.linspace(0, Ly, ynum), dims='ydef',
                        coords={'ydef':np.linspace(0, Ly, ynum)})
    
    ygrid, xgrid = xr.broadcast(ydef, xdef)
    
    tau_ideal = xr.DataArray((1.-np.cos(2.*np.pi * ygrid / Ly))/2.,
                             dims=['ydef','xdef'],
                             coords={'ydef':ydef, 'xdef':xdef})
    curl_tau  = xr.DataArray(-np.pi * np.sin(2.*np.pi * ygrid / Ly)/Ly,
                             dims=['ydef','xdef'],
                             coords={'ydef':ydef, 'xdef':xdef})
    
    # add topography
    curl_tau[65:, 100:104] = undef
    curl_tau[:75, 130:134] = undef
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'periodic'],
        'mxLoop'   : 3000,
        'tolerance': 1e-9,
        'optArg'   : 1.4,
        'undef'    : undef,
    }
    
    mParams1 = {'beta':beta, 'R':R   , 'D':depth}
    mParams2 = {'beta':beta, 'R':R*20, 'D':depth}
    mParams3 = {'beta':beta, 'R':R   , 'D':depth, 'A4':0}
    
    h1 = invert_Stommel(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                        iParams=iParams, mParams=mParams1)
    h2 = invert_Stommel(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                        iParams=iParams, mParams=mParams2)
    h3 = invert_StommelMunk(curl_tau, dims=['ydef','xdef'], coords='cartesian',
                        iParams=iParams, mParams=mParams3)
    
    assert (np.abs(h1) <= 5.5e5).all()
    assert (np.abs(h2) <= 2.8e4).all()
    assert (np.abs(h3) <= 5.5e5).all()



