# -*- coding: utf-8 -*-
"""
Created on 2020.12.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
from .numbas import invert_standard_3D, invert_standard_2D, invert_general_2D,\
                    invert_general_bih_2D
from .utils import loop_noncore, _R_earth, _undeftmp, _latlon

"""
Below are the core methods of xinvert
"""
def inv_standard3D(A, B, C, F, S, dims, BCs,
                   mxLoop, tolerance, optArg, printInfo, debug):
    """
    Inverting a 3D volume of elliptic equation in standard form as:
    
        d ┌  dS ┐   d ┌  dS ┐   d ┌  dS ┐
        --│A(--)│ + --│B(--)│ + --│C(--)│ = F
        dz└  dz ┘   dy└  dy ┘   dx└  dx ┘
    
    using SOR iteration. If F = F['time', 'lev', 'lat', 'lon'] and we invert
    for the 3D spatial distribution, then 3rd dim is 'lev', 2nd dim is 'lat'
    and 1st dim is 'lon'.

    Parameters
    ----------
    A: xr.DataArray
        Coefficient A.
    B: xr.DataArray
        Coefficient B.
    C: xr.DataArray
        Coefficient C.
    F: xr.DataArray
        Forcing function F.
    S: xr.DataArray
        Initial guess of the solution (also the output).
    dims: list
        Dimension combination for the inversion e.g., ['lev', 'lat', 'lon'].
        Order is important, should be consistent with the dimensions of F.
    BCs: list
        Boundary conditions for each dimension in dims.
        Order is important, should be consistent with the dimensions of F.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    optArg: float
        Optimal argument for SOR.
    printInfo: boolean
        Flag for printing.
    debug: boolean
        Output debug info.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 3:
        raise Exception('3 dimensions are needed for inversion')
    
    params = __cal_params3D(F[dims[0]], F[dims[1]], F[dims[2]], debug=debug)
    
    if optArg == None:
        optArg = params['optArg']
        
    for selDict in loop_noncore(F, dims):
        invert_standard_3D(S.loc[selDict].values, A.loc[selDict].values,
                           B.loc[selDict].values, C.loc[selDict].values,
                           F.loc[selDict].values,
                           params['gc3' ], params['gc2' ], params['gc1' ],
                           params['del3'], params['del2'], params['del1'],
                           BCs[0], BCs[1], BCs[2], params['del1Sqr'],
                           params['ratio2Sqr'], params['ratio1Sqr'],
                           optArg, _undeftmp, params['flags'],
                           mxLoop=mxLoop, tolerance=tolerance)
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if printInfo:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S


def inv_standard2D(A, B, C, F, S, dims, BCs,
                   mxLoop, tolerance, optArg, printInfo, debug):
    """
    A template for inverting equations in standard form as:
    
        d ┌  dS      dS ┐   d ┌  dS      dS ┐
        --│A(--) + B(--)│ + --│B(--) + C(--)│ = F
        dy└  dy      dx ┘   dx└  dy      dx ┘
    
    using SOR iteration. If F = F['time', 'lat', 'lon'] and we invert
    for the horizontal slice, then 2nd dim is 'lat' and 1st dim is 'lon'.

    Parameters
    ----------
    A: xr.DataArray
        Coefficient A.
    B: xr.DataArray
        Coefficient B.
    C: xr.DataArray
        Coefficient C.
    F: xr.DataArray
        Forcing function F.
    S: xr.DataArray
        Initial guess of the solution (also the output).
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
        Order is important, should be consistent with the order of F.
    BCs: list
        Boundary conditions for each dimension in dims.
        Order is important, should be consistent with the order of F.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    optArg: float
        Optimal argument for SOR.
    printInfo: boolean
        Flag for printing.
    debug: boolean
        Output debug info.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    params = __cal_params(F[dims[0]], F[dims[1]], debug=debug)
    
    if optArg == None:
        optArg = params['optArg']
    
    for selDict in loop_noncore(F, dims):
        invert_standard_2D(S.loc[selDict].values, A.loc[selDict].values,
                           B.loc[selDict].values, C.loc[selDict].values,
                           F.loc[selDict].values,
                           params['gc2' ], params['gc1' ],
                           params['del2'], params['del1'],
                           BCs[0], BCs[1], params['del1Sqr'],
                           params['ratioQtr'], params['ratioSqr'],
                           optArg, _undeftmp, params['flags'],
                           mxLoop=mxLoop, tolerance=tolerance)
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if printInfo:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S


def inv_general2D(A, B, C, D, E, F, G, S, dims, BCs,
                  mxLoop, tolerance, optArg, printInfo, debug):
    """
    A template for inverting a 2D slice of equation in general form as:
    
          d^2S     d^2S     d^2S     dS     dS 
        A ---- + B ---- + C ---- + D -- + E -- + F*S + G = 0
          dy^2     dydx     dx^2     dy     dx 

    Parameters
    ----------
    A: xr.DataArray
        Coefficient A.
    B: xr.DataArray
        Coefficient B.
    C: xr.DataArray
        Coefficient C.
    D: xr.DataArray
        Coefficient D.
    E: xr.DataArray
        Coefficient E.
    F: xr.DataArray
        Forcing function F.
    S: xr.DataArray
        Initial guess of the solution (also the output).
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
        Order is important, should be consistent with the order of F.
    BCs: list
        Boundary conditions for each dimension in dims.
        Order is important, should be consistent with the order of F.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    optArg: float
        Optimal argument for SOR.
    printInfo: boolean
        Flag for printing.
    debug: boolean
        Output debug info.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    params = __cal_params(F[dims[0]], F[dims[1]], debug=debug)
    
    if optArg == None:
        optArg = params['optArg']
    
    for selDict in loop_noncore(F, dims):
        invert_general_2D(S.loc[selDict].values, A.loc[selDict].values,
                          B.loc[selDict].values, C.loc[selDict].values,
                          D.loc[selDict].values, E.loc[selDict].values,
                          F.loc[selDict].values, G.loc[selDict].values,
                          params['gc2' ], params['gc1' ],
                          params['del2'], params['del1'],
                          BCs[0], BCs[1], params['del1Sqr'],
                          params['ratio'], params['ratioQtr'], params['ratioSqr'],
                          optArg, _undeftmp, params['flags'],
                          mxLoop=mxLoop, tolerance=tolerance)
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if printInfo:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S


def inv_general2D_bih(A, B, C, D, E, F, G, H, I, J, S, dims, BCs,
                      mxLoop, tolerance, optArg, printInfo, debug):
    """
    A template for inverting a 2D slice of equation in general form as:
    
      d^4S       d^4S       d^4S     d^2S     d^2S     d^2S     dS     dS 
    A ---- + B -------- + C ---- + D ---- + E ---- + F ---- + G -- + H -- + I*S + J = 0
      dy^4     dy^2dx^2     dx^4     dy^2     dydx     dx^2     dy     dx 

    Parameters
    ----------
    A: xr.DataArray
        Coefficient A.
    B: xr.DataArray
        Coefficient B.
    C: xr.DataArray
        Coefficient C.
    D: xr.DataArray
        Coefficient D.
    E: xr.DataArray
        Coefficient E.
    F: xr.DataArray
        Coefficient F.
    G: xr.DataArray
        Coefficient G.
    H: xr.DataArray
        Coefficient H.
    I: xr.DataArray
        Coefficient I.
    J: xr.DataArray
        Forcing function J.
    S: xr.DataArray
        Initial guess of the solution (also the output).
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
        Order is important, should be consistent with the order of F.
    BCs: list
        Boundary conditions for each dimension in dims.
        Order is important, should be consistent with the order of F.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    optArg: float
        Optimal argument for SOR.
    printInfo: boolean
        Flag for printing.
    debug: boolean
        Output debug info.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    params = __cal_params(F[dims[0]], F[dims[1]], debug=debug)
    
    if optArg == None:
        optArg = 1 # params['optArg'],  1 seems to be safer for 4-order SOR
    
    for selDict in loop_noncore(F, dims):
        invert_general_bih_2D(S.loc[selDict].values, A.loc[selDict].values,
                              B.loc[selDict].values, C.loc[selDict].values,
                              D.loc[selDict].values, E.loc[selDict].values,
                              F.loc[selDict].values, G.loc[selDict].values,
                              H.loc[selDict].values, I.loc[selDict].values,
                              J.loc[selDict].values,
                              params['gc2' ], params['gc1' ],
                              params['del2'], params['del1'],
                              BCs[0], BCs[1],
                              params['del1SSr'], params['del1Tr'], params['del1Sqr'],
                              params['ratio'   ], params['ratioSSr'],
                              params['ratioQtr'], params['ratioSqr'],
                              optArg, _undeftmp, params['flags'],
                              mxLoop=mxLoop, tolerance=tolerance)
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if printInfo:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S


"""
Below are the helper methods of xinvert
"""
def __cal_params3D(dim3_var, dim2_var, dim1_var, debug=False):
    """
    Pre-calculate some parameters needed in SOR for the 3D cases.

    Parameters
    ----------
    dim3_var : xarray.DataArray
        Dimension variable of third dimension (e.g., lev).
    dim2_var : xarray.DataArray
        Dimension variable of second dimension (e.g., lat).
    dim1_var : xarray.DataArray
        Dimension variable of first  dimension (e.g., lon).
    debug : boolean
        Print result for debugging. The default is False.

    Returns
    -------
    re : dict
        Pre-calculated parameters.
    """
    gc3  = len(dim3_var)
    gc2  = len(dim2_var)
    gc1  = len(dim1_var)
    del3 = dim3_var.diff(dim3_var.name).values[0] # assumed uniform
    del2 = dim2_var.diff(dim2_var.name).values[0] # assumed uniform
    del1 = dim1_var.diff(dim1_var.name).values[0] # assumed uniform
    
    if dim2_var.name in _latlon:
        del2 = np.deg2rad(del2) * _R_earth # convert lat/lon to m
    if dim1_var.name in _latlon:
        del1 = np.deg2rad(del1) * _R_earth # convert lat/lon to m
    
    ratio1    = del1 / del2
    ratio2    = del1 / del3
    ratio1Sqr = ratio1 ** 2.0
    ratio2Sqr = ratio2 ** 2.0
    del1Sqr   = del1 ** 2.0
    epsilon   = (np.sin(np.pi/(2.0*gc1+2.0)) **2.0 +
                 np.sin(np.pi/(2.0*gc2+2.0)) **2.0 +
                 np.sin(np.pi/(2.0*gc3+3.0)) **2.0)
    optArg    = 2.0 / (1.0 + np.sqrt((2.0 - epsilon) * epsilon))
    flags     = np.array([0.0, 1.0, 0.0])
    
    if debug:
        print('dim3_var: ', dim3_var)
        print('dim2_var: ', dim2_var)
        print('dim1_var: ', dim1_var)
        print('dim grids:', gc3, gc2, gc1)
        print('dim intervals: ', del3, del2, del1)
        print('ratio1Sqr, 2Sqr: ', ratio1Sqr, ratio2Sqr)
        print('del1Sqr: ', del1Sqr)
        print('optArg: ' , optArg)
    
    # store all and return
    re = {}
    
    re['gc3'      ] = gc3       # grid count in second dimension (e.g., lev)
    re['gc2'      ] = gc2       # grid count in second dimension (e.g., lat)
    re['gc1'      ] = gc1       # grid count in first  dimension (e.g., lon)
    re['del3'     ] = del3      # distance in third  dimension (unit: m or Pa)
    re['del2'     ] = del2      # distance in second dimension (unit: m)
    re['del1'     ] = del1      # distance in first  dimension (unit: m)
    re['ratio1Sqr'] = ratio1Sqr # distance ratio: del1 / del2
    re['ratio2Sqr'] = ratio2Sqr # ratio ** 4
    re['del1Sqr'  ] = del1Sqr   # del1 ** 2
    re['optArg'   ] = optArg    # optimal argument for SOR
    re['flags'    ] = flags     # outputs of the SOR iteration:
                                #   [0] overflow or not
                                #   [1] tolerance
                                #   [2] loop count
    
    return re


def __cal_params(dim2_var, dim1_var, debug=False):
    """
    Pre-calculate some parameters needed in SOR.

    Parameters
    ----------
    dim2_var : xarray.DataArray
        Dimension variable of second dimension (e.g., lat).
    dim1_var : xarray.DataArray
        Dimension variable of first  dimension (e.g., lon).
    debug : boolean
        Print result for debugging. The default is False.

    Returns
    -------
    re : dict
        Pre-calculated parameters.
    """
    gc2  = len(dim2_var)
    gc1  = len(dim1_var)
    del2 = dim2_var.diff(dim2_var.name).values[0] # assumed uniform
    del1 = dim1_var.diff(dim1_var.name).values[0] # assumed uniform
    
    if dim2_var.name in _latlon:
        del2 = np.deg2rad(del2) * _R_earth # convert lat/lon to m
    if dim1_var.name in _latlon:
        del1 = np.deg2rad(del1) * _R_earth # convert lat/lon to m
    
    ratio    = del1 / del2
    ratioSSr = ratio ** 4.0
    ratioSqr = ratio ** 2.0
    ratioQtr = ratio / 4.0
    del1Sqr  = del1 ** 2.0
    del1Tr   = del1 ** 3.0
    del1SSr  = del1 ** 4.0
    epsilon  = np.sin(np.pi/(2.0*gc1+2.0))**2 + np.sin(np.pi/(2.0*gc2+2.0))**2
    optArg   = 2.0 / (1.0 + np.sqrt((2.0 - epsilon) * epsilon))
    flags    = np.array([0.0, 1.0, 0.0])
    
    if debug:
        print('dim2_var: ', dim2_var)
        print('dim1_var: ', dim1_var)
        print('dim grids:', gc2, gc1)
        print('dim intervals: ', del2, del1)
        print('ratio, Qtr, Sqr, SSr: ', ratio, ratioQtr, ratioSqr, ratioSSr)
        print('del1Sqr, delTr, del1SSr: ', del1Sqr, del1Tr, del1SSr)
        print('optArg: ' , optArg)
    
    # store all and return
    re = {}
    
    re['gc2'     ] = gc2       # grid count in second dimension (e.g., lat)
    re['gc1'     ] = gc1       # grid count in first  dimension (e.g., lon)
    re['del2'    ] = del2      # distance in second dimension (unit: m)
    re['del1'    ] = del1      # distance in first  dimension (unit: m)
    re['ratio'   ] = ratio     # distance ratio: del1 / del2
    re['ratioSSr'] = ratioSSr  # ratio ** 4
    re['ratioSqr'] = ratioSqr  # ratio ** 2
    re['ratioQtr'] = ratioQtr  # ratio ** 2 / 4
    re['del1Sqr' ] = del1Sqr   # del1 ** 2
    re['del1Tr'  ] = del1Tr    # del1 ** 3
    re['del1SSr' ] = del1SSr   # del1 ** 4
    re['optArg'  ] = optArg    # optimal argument for SOR
    re['flags'   ] = flags     # outputs of the SOR iteration:
                               #   [0] overflow or not
                               #   [1] tolerance
                               #   [2] loop count
    
    return re

