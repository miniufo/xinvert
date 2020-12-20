# -*- coding: utf-8 -*-
"""
Created on 2020.12.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import numba as nb
import xarray as xr


"""
Core classes are defined below
"""
_R_earth = 6371200.0 # consistent with GrADS

_latlon = ['lat', 'LAT', 'latitude' , 'LATITUDE' , 'lats', 'LATS',
           'lon', 'LON', 'longitude', 'LONGITUDE', 'long', 'LONG']



def invert_Poisson(force, dims, BCs=['fixed', 'fixed'],
                   undef=np.nan, mxLoop=5000, tolerance=1e-6,
                   printInfo=True, debug=False):
    """
    Inverting Poisson equation of the form \nabla^2 S = F:
    
        d  dS    d  dS
        --(--) + --(--) = F
        dx dx    dy dy
    
    using SOR iteration.
    
    Parameters
    ----------
    force: xarray.DataArray
        Forcing function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    BCs: list
        Boundary conditions for each dimension in dims.
    undef: float
        Undefined value.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    printInfo: boolean
        Flag for printing.
    debug: boolean
        Output debug info.
        
    Returns
    ----------
    S: xarray.DataArray
        Results of the SOR inversion.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    lats = force[dims[0]]
    undeftmp = -9.99e8
    
    if np.isnan(undef):
        forcing = force.fillna(undeftmp)
    elif undef == 0:
        forcing = force.where(force!=0, other=undeftmp)
    else:
        forcing = force.where(force!=undef, other=undeftmp)
        
    
    S =(forcing - forcing).load()
    A = forcing - forcing + 1.0/np.cos(np.deg2rad(lats))
    B = forcing - forcing
    C = forcing - forcing + np.cos(np.deg2rad((lats+lats.shift({'lat':1}))/2.0))
    F =(forcing * np.cos(np.deg2rad(lats))).where(forcing!=undeftmp, undeftmp)
    
    dimAll = F.dims
    
    dimInv = [] # ordered dims for inversion
    dimLop = [] # ordered dims for loop
    for dim in dimAll:
        if dim in dims:
            dimInv.append(dim)
        else:
            dimLop.append(dim)
    
    dim1_var = F[dimInv[1]]
    dim2_var = F[dimInv[0]]
    
    BC2 , BC1  = BCs
    dim1 = len(dim1_var)
    dim2 = len(dim2_var)
    del1 = dim1_var.diff(dimInv[1]).values[0] # assumed uniform
    del2 = dim2_var.diff(dimInv[0]).values[0] # assumed uniform
    
    if dimInv[1] in _latlon:
        del1 = np.deg2rad(del1) * _R_earth # convert lat/lon to m
    if dimInv[0] in _latlon:
        del2 = np.deg2rad(del2) * _R_earth # convert lat/lon to m
    
    ratioQtr = del2 / del1
    ratioSqr = ratioQtr ** 2.0
    ratioQtr /= 4.0
    delD2Sqr = del2 ** 2.0
    flags = np.array([0.0, 1.0, 0.0])
    
    epsilon = np.sin(np.pi/(2.0*dim1+2.0))**2 + np.sin(np.pi/(2.0*dim2+2.0))**2
    optArg  = 2.0 / (1.0 + np.sqrt((2.0 - epsilon) * epsilon))
    
    if debug:
        print('dimLoop: ', dimLop)
        print('dimInvert: ', dimInv)
        print('dim grids:', dim2, dim1)
        print('dim intervals: ', del2, del2)
        print('BCs: ', BC2, BC1)
        print('ratioQtr, Sqr: ', ratioQtr, ratioSqr)
        print('delD2Sqr: ', delD2Sqr)
        print('optArg: ', optArg)
        print('epsilon: ', epsilon)
    
    dimLopVars = []
    for dim in dimLop:
        dimLopVars.append(F[dim].values)
    
    from itertools import product
    for idices in product(*dimLopVars):
        selDict = {}
        for d, i in zip(dimLop, idices):
            selDict[d] = i
            
        invert_slice(S.loc[selDict].values, A.loc[selDict].values,
                     B.loc[selDict].values, C.loc[selDict].values,
                     F.loc[selDict].values,
                     dim2, dim1, del2, del1, BC2, BC1, delD2Sqr,
                     ratioQtr, ratioSqr, optArg, undeftmp, flags,
                     mxLoop=mxLoop, tolerance=tolerance)
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if printInfo:
            if flags[0]:
                print(info + ' {0:4.0f} and tolerance is {1:e} (   overflows!)'
                      .format(flags[2], flags[1]))
            else:
                print(info + ' {0:4.0f} and tolerance is {1:e}'
                      .format(flags[2], flags[1]))
    
    S = S.where(forcing!=undeftmp, other=undef).rename('inverted')
    
    return S


def invert_Poisson_animated(force, BCs=['fixed', 'fixed'], undef=np.nan,
                            loop_per_frame=5, max_loop=100,
                            printInfo=True, debug=False):
    """
    Inverting Poisson equation of the form \nabla^2 S = F:
    
        d  dS    d  dS
        --(--) + --(--) = F
        dx dx    dy dy
    
    using SOR iteration.
    
    Parameters
    ----------
    force: xarray.DataArray
        Forcing function with a single slice (2D field).
    undef: float
        Undefined value.
    printInfo: boolean
        Flag for printing.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    BCs: list
        Boundary conditions for each dimension in dims.
    undef: float
        Undefined value.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    printInfo: boolean
        Flag for printing.
    debug: boolean
        Output debug info.
        
    Returns
    ----------
    S: xarray.DataArray
        Results of the SOR inversion.
    """
    if len(force.dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    lats = force[force.dims[0]]
    undeftmp = -9.99e8
    
    if np.isnan(undef):
        forcing = force.fillna(undeftmp)
    elif undef == 0:
        forcing = force.where(force!=0, other=undeftmp)
    else:
        forcing = force.where(force!=undef, other=undeftmp)
        
    zero = (forcing - forcing).load()
    
    A = zero + 1.0/np.cos(np.deg2rad(lats))
    B = zero.copy()
    C = zero + np.cos(np.deg2rad((lats+lats.shift({'lat':1}))/2.0))
    F =(forcing * np.cos(np.deg2rad(lats))).where(forcing!=undeftmp, undeftmp)
    
    dimAll = F.dims
    
    dim1_var = F[dimAll[1]]
    dim2_var = F[dimAll[0]]
    
    BC2 , BC1  = BCs
    dim1 = len(dim1_var)
    dim2 = len(dim2_var)
    del1 = dim1_var.diff(dimAll[1]).values[0] # assumed uniform
    del2 = dim2_var.diff(dimAll[0]).values[0] # assumed uniform
    
    if dimAll[1] in _latlon:
        del1 = np.deg2rad(del1) * _R_earth # convert lat/lon to m
    if dimAll[0] in _latlon:
        del2 = np.deg2rad(del2) * _R_earth # convert lat/lon to m
    
    ratioQtr = del2 / del1
    ratioSqr = ratioQtr ** 2.0
    ratioQtr /= 4.0
    delD2Sqr = del2 ** 2.0
    flags = np.array([0.0, 1.0, 0.0])
    
    epsilon = np.sin(np.pi/(2.0*dim1+2.0))**2 + np.sin(np.pi/(2.0*dim2+2.0))**2
    optArg  = 2.0 / (1.0 + np.sqrt((2.0 - epsilon) * epsilon))
    
    if debug:
        print('dim grids:', dim2, dim1)
        print('dim intervals: ', del2, del2)
        print('BCs: ', BC2, BC1)
        print('ratioQtr, Sqr: ', ratioQtr, ratioSqr)
        print('delD2Sqr: ', delD2Sqr)
        print('optArg: ', optArg)
        print('epsilon: ', epsilon)
    
    lst = []
    
    snapshot = zero
    loop = 0
    while True:
        invert_slice(snapshot.values, A.values, B.values, C.values, F.values,
                     dim2, dim1, del2, del1, BC2, BC1, delD2Sqr,
                     ratioQtr, ratioSqr, optArg, undeftmp, flags,
                     mxLoop=loop_per_frame)
        
        if printInfo:
            if flags[0]:
                print('loops {0:4.0f} and tolerance is {1:e} (   overflows!)'
                      .format(flags[2], flags[1]))
            else:
                print('loops {0:4.0f} and tolerance is {1:e}'
                      .format(flags[2], flags[1]))
        
        lst.append(snapshot.copy())
        
        if flags[2] < loop_per_frame or loop > max_loop:
            break
        
        loop += 1
    
    return xr.concat(lst, dim='iteration')



"""
Below are the numba functions
"""

@nb.jit(nopython=True)
def invert_slice(S, A, B, C, F,
                    dim2, dim1, del2, del1, BC2, BC1, delD2Sqr,
                    ratioQtr, ratioSqr, optArg, undef, flags,
                    mxLoop=5000, tolerance=1e-6):
    """
    Inverting one slice (2D field that constitute 1st and 2nd dimensions)
    data using SOR iteration.  If F=F['time', 'lat', 'lon'] and we invert
    for the horizontal slice, then 2nd dim is 'lat' and 1st dim is 'lon'.
    
    Parameters
    ----------
    S: numpy.array (output)
        Results of the SOR inversion.
        
    A: numpy.array
        Coefficient for the first dimensional derivative.
    B: numpy.array
        Coefficient for the cross derivatives.
    C: numpy.array
        Coefficient for the second dimensional derivative.
    F: numpy.array
        Forcing function.
    dim2: str
        Name of the second dimension (e.g., Y or lat).
    dim1: str
        Name of the first dimension (e.g., X or lon).
    del2: float
        Increment (interval) in dimension 2 (unit of m, not degree).
    del1: float
        Increment (interval) in dimension 1 (unit of m, not degree).
    BC2: str
        Boundary condition for dimension 2 in ['fixed', 'extend', 'periodic'].
    BC1: str
        Boundary condition for dimension 1 in ['fixed', 'extend', 'periodic'].
    delD2Sqr: float
        Squared increment (interval) in dimension 2 (unit of m^2).
    ratioQtr: float
        Ratio of del2 to del1, divided by 4.
    ratioSqr: float
        Squared Ratio of del2 to del1.
    optArg: float
        Optimal argument 'omega' (relaxation factor between 1 and 2) for SOR.
    undef: float
        Undefined value.
    flags: numpy.array
        Length of 3 array, [0] is flag for overflow, [1] for converge speed and
        [2] for how many loops used for iteration.
    mxLoop: int
        Maximum loop count, larger than this will break the iteration.
    tolerance: float
        Tolerance for iteraction, smaller than this will break the iteraction.

    Returns
    ----------
    S: numpy.array
        Results of the SOR inversion.
    """
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    
    while(True):
        # process boundaries
        if BC2 == 'extend':
            if BC1 == 'periodic':
                for i in range(dim1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
            else:
                for i in range(1, dim1-1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
                for i in range(1, dim2-1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
                
                if  S[ 1, 1] != undef:
                    S[ 0, 0] = S[ 1, 1]
                if  S[ 1,-2] != undef:
                    S[ 0,-1] = S[ 1,-2]
                if  S[-2, 1] != undef:
                    S[-1, 0] = S[-2, 1]
                if  S[-2,-2] != undef:
                    S[-1,-1] = S[-2,-2]
        
        for j in range(1, dim2-1):
            # for the west boundary iteration (i==0)
            if BC1 == 'periodic':
                for i in range(1, dim1-1):
                    cond = (A[j  ,1] != undef and A[j  , 0] != undef and
                            B[j  ,1] != undef and B[j  ,-1] != undef and
                            B[j+1,0] != undef and B[j-1, 0] != undef and
                            C[j+1,0] != undef and C[j  , 0] != undef and
                            F[j  ,0] != undef)
                    
                    if cond:
                        temp = (
                            (
                                A[j,1] * (S[j,1] - S[j, 0])-
                                A[j,0] * (S[j,0] - S[j,-1])
                            ) * ratioSqr + (
                                B[j, 1] * (S[j+1, 1] - S[j-1, 1])-
                                B[j,-1] * (S[j+1,-1] - S[j-1,-1])
                            ) * ratioQtr + (
                                B[j+1,1] * (S[j+1,1] - S[j+1,-1])-
                                B[j-1,0] * (S[j-1,0] - S[j-1,-1])
                            ) * ratioQtr + (
                                C[j+1,0] * (S[j+1,0] - S[j , 0])-
                                C[j  ,0] * (S[j  ,0] - S[j-1,0])
                            )
                        ) - F[j,0] * delD2Sqr
                        
                        temp *= optArg / ((A[j  ,1] + A[j,0]) *ratioSqr +
                                          (C[j+1,0] + C[j,0]))
                        S[j,0] += temp
            
            # inner loop
            for i in range(1, dim1-1):
                cond = (A[j  ,i+1] != undef and A[j  ,  i] != undef and
                        B[j  ,i+1] != undef and B[j  ,i-1] != undef and
                        B[j+1,i  ] != undef and B[j-1,  i] != undef and
                        C[j+1,i  ] != undef and C[j  ,  i] != undef and
                        F[j  ,i  ] != undef)
                
                if cond:
                    temp = (
                        (
                            A[j,i+1] * (S[j,i+1] - S[j,  i])-
                            A[j,i  ] * (S[j,i  ] - S[j,i-1])
                        ) * ratioSqr + (
                            B[j,i+1] * (S[j+1,i+1] - S[j-1,i+1])-
                            B[j,i-1] * (S[j+1,i-1] - S[j-1,i-1])
                        ) * ratioQtr + (
                            B[j+1,i] * (S[j+1,i+1] - S[j+1,i-1])-
                            B[j-1,i] * (S[j-1,i+1] - S[j-1,i-1])
                        ) * ratioQtr + (
                            C[j+1,i] * (S[j+1,i] - S[j  ,i])-
                            C[j  ,i] * (S[j  ,i] - S[j-1,i])
                        )
                    ) - F[j,i] * delD2Sqr
                    
                    temp *= optArg / ((A[j,i+1] + A[j,i]) *ratioSqr +
                                      (C[j+1,i] + C[j,i]))
                    S[j,i] += temp
            
            
            # for the east boundary iteration (i==-1)
            if BC1 == 'periodic':
                for i in range(1, dim1-1):
                    cond = (A[j  , 0] != undef and A[j  ,-1] != undef and
                            B[j  , 0] != undef and B[j  ,-2] != undef and
                            B[j+1,-1] != undef and B[j-1,-1] != undef and
                            C[j+1,-1] != undef and C[j  ,-1] != undef and
                            F[j  ,-1] != undef)
                    
                    if cond:
                        temp = (
                            (
                                A[j, 0] * (S[j, 0] - S[j,-1])-
                                A[j,-1] * (S[j,-1] - S[j,-2])
                            ) * ratioSqr + (
                                B[j, 0] * (S[j+1, 0] - S[j-1, 0])-
                                B[j,-2] * (S[j+1,-2] - S[j-1,-2])
                            ) * ratioQtr + (
                                B[j+1,-1] * (S[j+1,0] - S[j+1,-2])-
                                B[j-1,-1] * (S[j-1,0] - S[j-1,-2])
                            ) * ratioQtr + (
                                C[j+1,-1] * (S[j+1,-1] - S[j , -1])-
                                C[j  ,-1] * (S[j  ,-1] - S[j-1,-1])
                            )
                        ) - F[j,-1] * delD2Sqr
                        
                        temp *= optArg / ((A[j  , 0] + A[j,-1]) *ratioSqr +
                                          (C[j+1,-1] + C[j,-1]))
                        S[j,-1] += temp
        
        norm = absNorm(S, undef)
        
        if np.isnan(norm) or norm > 1e17:
            flags[0] = True
            break
        
        flags[1] = abs(norm - normPrev) / normPrev
        flags[2] = loop
        
        if flags[1] < tolerance or loop >= mxLoop:
            break
        
        normPrev = norm
        loop += 1
        
    return S


@nb.jit(nopython=True)
def absNorm(S, undef):
    norm = 0.0
    
    J, I = S.shape
    count = 0
    for j in range(J):
        for i in range(I):
            if S[j,i] != undef:
                norm += abs(S[j,i])
                count += 1
    
    if count != 0:
        norm /= count
    else:
        norm = np.nan
    
    return norm
    
    