# -*- coding: utf-8 -*-
"""
Created on 2021.01.03

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import numpy as np
import xarray as xr
import numba as nb
# from .numbas import invert_standard_2D, invert_general_2D


"""
Core classes are defined below
"""
R_earth = 6371200.0 # consistent with GrADS
omega_earth = 7.292e-5



def cal_dydx(lat, lon, coords='latlon'):
    if coords == 'latlon':
        xdef = np.deg2rad(lon)
        ydef = np.deg2rad(lat)
        
        dy = _centered_diff(ydef) * R_earth
        dx = _centered_diff(xdef) * R_earth * np.cos(ydef)
        
        return xr.broadcast(dy, dx)
    elif coords == 'cartesian':
        xdef = lon
        ydef = lat
        
        dy = _centered_diff(ydef)
        dx = _centered_diff(xdef)
        
        return xr.broadcast(dy, dx)
    else:
        raise Exception('unsupported coords, should be [\'lat/lon\', \'cartesian\']')
    

def Laplacian(v, dims, BCx, coords='latlon'):
    dy, dx = cal_dydx(v[dims[0]], v[dims[1]], coords=coords)
    
    lp = xr.apply_ufunc(_laplacian_diff, v, dy, dx,
                        dask='allowed',
                        input_core_dims=[dims, dims, dims],
                        output_core_dims=[dims,],
                        vectorize=True)
    return lp.rename('nabla')


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


"""
Core classes are defined below
"""
def _centered_diff(data):
    re = data.copy()
    _centered_difference(data.values, re.values)
    return re
    

@nb.jit(nopython=True)
def _laplacian_diff(v, dy, dx, BCx='fixed'):
    J, I = v.shape
    
    re = np.zeros((J, I))
    
    for j in range(1, J-1):
        if BCx == 'periodic':
            # i == 0
            re[j,0] = (((v[j+1,0]-v[j  , 0])-
                        (v[j  ,0]-v[j-1, 0])) / (dy[j,0]*dy[j,0]) / 4. +
                       ((v[j  ,1]-v[j  , 0])-
                        (v[j  ,0]-v[j  ,-1])) / (dx[j,0]*dx[j,0]) / 4.)
            # i == -1
            re[j,-1] = (((v[j+1,-1]-v[j  ,-1])-
                         (v[j  ,-1]-v[j-1,-1])) / (dy[j,-1]*dy[j,-1]) / 4. +
                        ((v[j  , 0]-v[j  ,-1])-
                         (v[j  ,-1]-v[j  ,-2])) / (dx[j,-1]*dx[j,-1]) / 4.)
        
        # inner loop
        for i in range(1, I-1):
            re[j,i] = (((v[j+1,i  ]-v[j  ,i  ])-
                        (v[j  ,i  ]-v[j-1,i  ])) / (dy[j,i]*dy[j,i]) / 4. +
                       ((v[j  ,i+1]-v[j  ,i  ])-
                        (v[j  ,i  ]-v[j  ,i-1])) / (dx[j,i]*dx[j,i]) / 4.)
    
    return re

@nb.jit(nopython=True)
def _centered_difference(data, re):
    I = len(data)
    
    for i in range(I-1):
        re[i] = data[i+1] - data[i]
    
    re[I-1] = data[I-1] - data[I-2]

