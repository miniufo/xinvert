# -*- coding: utf-8 -*-
"""
Created on 2021.01.03

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numba as nb


def loop_noncore(data, dims=None):
    """Loop over the non-core dimensions using generator.

    The non-core dimensions are given outside the list in `dims`.
    
    Parameters
    ----------
    data: xarray.DataArray
        A given multidimensional data.
    dims: list of str
        Core dimensions.  The remaining dimensions are non-core dimension
    
    Yields
    ------
    dict
        dict indicates the portion of `data` can be extracted
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
    
    if len(dimNonC) >= 1:
        from itertools import product
        for idices in product(*dimLopVars):
            selDict = {}
            for d, i in zip(dimNonC, idices):
                selDict[d] = i
            
            yield selDict
    else:
        yield {}


def smooth9(data, dims=None, times=1, BCx='fixed'):
    """
    Smooth a given gridded data along two dimensions using
    GrADS' 9-point fashion.
    
    Parameters
    ----------
    data: xarray.DataArray
        A given gridded data.
    dims: list of str
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    times: int or list of int
        How many times to repeat such smooth.
    periodic: str or list of str
        Dimension that is periodic e.g., 'lon'.
    
    Returns
    ----------
    re: xarray.DataArray
        Smoothed data.
    """
    if dims is None:
        dims = data.dims
    
    if len(dims) != 2:
        raise Exception('two dimensions are needed')
    
    re = (data - data).load()
    
    for selDict in loop_noncore(data, dims):
        _filter9(data.loc[selDict].values, re.loc[selDict].values,
                 times=times, BCx=BCx)
    
    return re



def coarsen(data, dims=None, smooth_data=True, periodic=None, ratio=2):
    """
    Coarsen a gridded data by a given ratio.
    
    Parameters
    ----------
    data: xarray.DataArray
        A given gridded data.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    smooth_data: bool
        Smooth the data or not before coarsening.
    periodic: str or list of str
        Periodic dimension and its period e.g., 'lon'.
    ratio: int
        Ratio of grid points before and after the coarsening.
        
    Returns
    ----------
    re: xarray.DataArray
        Smoothed data.
    """
    if ratio == 1:
        return data
    
    if dims is None:
        dims = data.dims
    else:
        dims = [dim for dim in dims if dim in data.dims]
    
    # smooth data or not
    if smooth_data:
        if ratio % 2 == 1: # ensure the ratio is an odd number
            window = ratio + 2
        else:
            window = ratio + 1
        
        # for periodic BC
        if periodic is not None:
            if type(periodic) is str:
                periodic = [periodic]
            
            dct = {}
            
            for dim in dims:
                if dim in periodic:
                    dct[dim] = int(window/2)
            
            tmp = pad_periodic(data, pad_widths=dct)
        else:
            tmp = data
        data_smth = smooth(tmp, dims, window=window)
    else:
        data_smth = data
    
    
    dct = {}
    for dim in dims:
        cdef = data[dim].values
        
        cdefInterp = np.linspace(cdef[0], cdef[-1], int(len(cdef)/ratio))
        
        dct[dim] = cdefInterp
    
    re = data_smth.interp(coords=dct)
    
    return re


"""
Helper (private) methods are defined below
"""
@nb.jit(nopython=True, cache=False)
def _filter9(data, re, times=1, BCx='fixed'):
    """
    Smooth using 9 neighbouring points.

    Parameters
    ----------
    data: numpy.ndarray
        Original data.
    re: numpy.ndarray
        Result of smoothed data (output).
    times: int
        How many times the smooth filter is applied.
    BCx: str
        Boundary condition for x-direction, in ['fixed', 'periodic'].
    """
    # print(f"data: {data.shape} | re: {re.shape} | times: {times}")
    J, I = data.shape
    
    w1, w2 = 1, 1
    
    tmp = data.copy()
    
    for l in range(times):
        for j in range(1, J-1):
            if BCx == 'periodic':
                # west boundary (i == 0)
                if np.isnan(tmp[j, 0]):
                    re[j, 0] = np.nan
                else:
                    s = tmp[j, 0]
                    w = 1.0
                    
                    if not np.isnan(tmp[j+1, 0]):
                        s+=tmp[j+1, 0]*w1
                        w+=w1
                    if not np.isnan(tmp[j  , 1]):
                        s+=tmp[j  , 1]*w1
                        w+=w1
                    if not np.isnan(tmp[j-1, 0]):
                        s+=tmp[j-1, 0]*w1
                        w+=w1
                    if not np.isnan(tmp[j  ,-1]):
                        s+=tmp[j  ,-1]*w1
                        w+=w1
                    if not np.isnan(tmp[j+1,-1]):
                        s+=tmp[j+1,-1]*w2
                        w+=w2
                    if not np.isnan(tmp[j-1,-1]):
                        s+=tmp[j-1,-1]*w2
                        w+=w2
                    if not np.isnan(tmp[j+1, 1]):
                        s+=tmp[j+1, 1]*w2
                        w+=w2
                    if not np.isnan(tmp[j-1, 1]):
                        s+=tmp[j-1, 1]*w2
                        w+=w2
                    
                    re[j, 0] = s / w
                
                # east boundary (i == -1)
                if np.isnan(tmp[j, -1]):
                    re[j, -1] = np.nan
                else:
                    s = tmp[j, -1]
                    w = 1.0
                    
                    if not np.isnan(tmp[j+1,-1]):
                        s+=tmp[j+1,-1]*w1
                        w+=w1
                    if not np.isnan(tmp[j  , 0]):
                        s+=tmp[j  , 0]*w1
                        w+=w1
                    if not np.isnan(tmp[j-1,-1]):
                        s+=tmp[j-1,-1]*w1
                        w+=w1
                    if not np.isnan(tmp[j  ,-2]):
                        s+=tmp[j  ,-2]*w1
                        w+=w1
                    if not np.isnan(tmp[j+1,-2]):
                        s+=tmp[j+1,-2]*w2
                        w+=w2
                    if not np.isnan(tmp[j-1,-2]):
                        s+=tmp[j-1,-2]*w2
                        w+=w2
                    if not np.isnan(tmp[j+1, 0]):
                        s+=tmp[j+1, 0]*w2
                        w+=w2
                    if not np.isnan(tmp[j-1, 0]):
                        s+=tmp[j-1, 0]*w2
                        w+=w2
                    
                    re[j, -1] = s / w
            
            # interior
            for i in range(1, I-1):
                if np.isnan(tmp[j, i]):
                    re[j, i] = np.nan
                else:
                    s = tmp[j, i]
                    w = 1.0
                    
                    if not np.isnan(tmp[j+1, i  ]):
                        s+=tmp[j+1, i  ]*w1
                        w+=w1
                    if not np.isnan(tmp[j  , i+1]):
                        s+=tmp[j  , i+1]*w1
                        w+=w1
                    if not np.isnan(tmp[j-1, i  ]):
                        s+=tmp[j-1, i  ]*w1
                        w+=w1
                    if not np.isnan(tmp[j  , i-1]):
                        s+=tmp[j  , i-1]*w1
                        w+=w1
                    if not np.isnan(tmp[j+1, i-1]):
                        s+=tmp[j+1, i-1]*w2
                        w+=w2
                    if not np.isnan(tmp[j-1, i-1]):
                        s+=tmp[j-1, i-1]*w2
                        w+=w2
                    if not np.isnan(tmp[j+1, i+1]):
                        s+=tmp[j+1, i+1]*w2
                        w+=w2
                    if not np.isnan(tmp[j-1, i+1]):
                        s+=tmp[j-1, i+1]*w2
                        w+=w2
                    
                    re[j, i] = s / w
        
        tmp[:, :] = re[:, :]
