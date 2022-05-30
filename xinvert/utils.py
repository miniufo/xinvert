# -*- coding: utf-8 -*-
"""
Created on 2021.01.03

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import numpy as np


_R_earth = 6371200.0 # consistent with GrADS
_omega = 7.292e-5    # angular speed of the earth rotation
_g = 9.80665         # gravitational acceleration
_undeftmp = -9.99e8  # default undefined value
_deg2m = 2.0 * np.pi * _R_earth / 360.0

_latlon = ['lat', 'latitude' , 'lats', 'yc', 'y',
           'lon', 'longitude', 'long', 'xc', 'x']


def loop_noncore(data, dims=None):
    """
    Loop over the non-core dimensions using generator.
    The non-core dimensions are given outside the list in `dims`.
    
    Parameters
    ----------
    data: xarray.DataArray
        A given multidimensional data.
    dims: list of str
        Core dimensions.  The remaining dimensions are non-core dimension
    
    Returns
    ----------
    re: generator
        iterator
    """
    dimAll = data.dims
    
    dimCore = [] # ordered core dims
    dimNonC = [] # ordered non-core dims
    
    for dim in dimAll:
        if dim in dims:
            dimCore.append(dim)
        else:
            dimNonC.append(dim)
    
    dimLopVars = []
    for dim in dimNonC:
        dimLopVars.append(data[dim].values)
    
    from itertools import product
    for idices in product(*dimLopVars):
        selDict = {}
        for d, i in zip(dimNonC, idices):
            selDict[d] = i
            
            yield selDict
    
    yield {}


