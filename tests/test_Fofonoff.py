# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xinvert import invert_Fofonoff


def test_Fofonoff():
    xc = np.linspace(0, 600000, 301)
    yc = np.linspace(0, 500000, 251)
    
    xdef = xr.DataArray(xc, dims='x', coords={'x':xc})
    ydef = xr.DataArray(yc, dims='y', coords={'y':yc})
    
    F = ydef - xdef # broadcast to 2D field
    
    assert F.shape == (251, 301)
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'fixed'],
        'mxLoop'   : 2000,
        'tolerance': 1e-14,
        'optArg'   : 1.2,
    }
    
    mParams = {
        'f0': 1e-4,
        'beta': 2e-11,
        'c0': 8e-9,
        'c1': 1e-4,
    }
    
    sf = invert_Fofonoff(F, dims=['y', 'x'], coords='cartesian',
                         iParams=iParams, mParams=mParams)
    
    assert sf.dims  == F.dims
    assert sf.shape == F.shape



