# -*- coding: utf-8 -*-
"""
Created on 2021.01.03

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""


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

