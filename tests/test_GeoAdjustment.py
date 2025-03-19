# -*- coding: utf-8 -*-
"""
Created on 2024.10.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xinvert.xinvert import invert_GeoAdjustment

# def test_adjustment():
yc   = 501
ydef = np.linspace(-75, -25, yc)
ydef = xr.DataArray(ydef, dims='lat', coords={'lat':ydef})

R  = 6371200
O  = 7.292e-5
h0 = ydef - ydef + 1500
h0[int(yc/2):] = 1520
g  = 9.81

f   = 2 * O * np.sin(np.deg2rad(ydef))
PV0 = f / h0

# invert
iParams = {
    'BCs'      : ['extend'],
    'mxLoop'   : 100000,
    'tolerance': -1e-11,
    'optArg'   : 1.8,
    'undef'    : -9999,
}

deg2m = R/180*np.pi

h  = invert_GeoAdjustment(h0, dims=['lat'], coords='lat', iParams=iParams)
u  = - h.differentiate('lat') / deg2m * g / f
PV = (f - u.differentiate('lat') / deg2m) / h

