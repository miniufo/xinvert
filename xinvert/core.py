# -*- coding: utf-8 -*-
"""
Created on 2020.12.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
from .numbas import invert_standard_3D, invert_standard_2D, \
                    invert_general_3D, invert_general_2D, \
                    invert_general_bih_2D, invert_standard_2D_test
from .utils import loop_noncore, _undeftmp

"""
Below are the core methods of xinvert
"""
def inv_standard3D(A, B, C, F, S, dims, params):
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
    params: dict
        Parameters for inversion.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 3:
        raise Exception('3 dimensions are needed for inversion')
        
    for selDict in loop_noncore(F, dims):
        invert_standard_3D(S.loc[selDict].values, A.loc[selDict].values,
                           B.loc[selDict].values, C.loc[selDict].values,
                           F.loc[selDict].values,
                           params['gc3' ], params['gc2' ], params['gc1' ],
                           params['del3'], params['del2'], params['del1'],
                           params['BCs'][0], params['BCs'][1], params['BCs'][2],
                           params['del1Sqr'],
                           params['ratio2Sqr'], params['ratio1Sqr'],
                           params['optArg'], _undeftmp, params['flags'],
                           params['mxLoop'], params['tolerance'])
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if params['printInfo']:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S


def inv_standard2D(A, B, C, F, S, dims, params):
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
    params: dict
        Parameters for inversion.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    for selDict in loop_noncore(F, dims):
        invert_standard_2D(S.loc[selDict].values,
                           A.loc[selDict].values,
                           B.loc[selDict].values, C.loc[selDict].values,
                           F.loc[selDict].values,
                           params['gc2' ], params['gc1' ],
                           params['del2'], params['del1'],
                           params['BCs'][0], params['BCs'][1], params['del1Sqr'],
                           params['ratioQtr'], params['ratioSqr'],
                           params['optArg'], _undeftmp, params['flags'],
                           params['mxLoop'], params['tolerance'])
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if params['printInfo']:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S



def inv_standard2D_test(A, B, C, D, E, F, S, dims, params):
    """
    A template for inverting equations in standard form as:
    
        d ┌  dS      dS ┐   d ┌  dS      dS ┐
        --│A(--) + B(--)│ + --│C(--) + D(--)│ + ES = F
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
    params: dict
        Parameters for inversion.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    for selDict in loop_noncore(F, dims):
        invert_standard_2D_test(S.loc[selDict].values,
                           A.loc[selDict].values,
                           B.loc[selDict].values, C.loc[selDict].values,
                           D.loc[selDict].values, E.loc[selDict].values,
                           F.loc[selDict].values,
                           params['gc2' ], params['gc1' ],
                           params['del2'], params['del1'],
                           params['BCs'][0], params['BCs'][1], params['del1Sqr'],
                           params['ratioQtr'], params['ratioSqr'],
                           params['optArg'], _undeftmp, params['flags'],
                           params['mxLoop'], params['tolerance'])
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if params['printInfo']:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S


def inv_general3D(A, B, C, D, E, F, G, H, S, dims, params):
    """
    A template for inverting a 2D slice of equation in general form as:
    
          d^2S     d^2S     d^2S     dS     dS     dS 
        A ---- + B ---- + C ---- + D -- + E -- + F -- + G*S = H
          dz^2     dy^2     dx^2     dz     dy     dx 

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
        Forcing function H.
    S: xr.DataArray
        Initial guess of the solution (also the output).
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
        Order is important, should be consistent with the order of F.
    params: dict
        Parameters for inversion.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 3:
        raise Exception('3 dimensions are needed for inversion')
    
    for selDict in loop_noncore(F, dims):
        invert_general_3D(S.loc[selDict].values, A.loc[selDict].values,
                          B.loc[selDict].values, C.loc[selDict].values,
                          D.loc[selDict].values, E.loc[selDict].values,
                          F.loc[selDict].values, G.loc[selDict].values,
                          H.loc[selDict].values,
                          params['gc3' ], params['gc2' ], params['gc1' ],
                          params['del3'], params['del2'], params['del1'],
                          params['BCs'][0], params['BCs'][1], params['BCs'][2],
                          params['del1Sqr'], params['ratio2'], params['ratio1'],
                          params['ratio2Sqr'], params['ratio1Sqr'],
                          params['optArg'], _undeftmp, params['flags'],
                          params['mxLoop'], params['tolerance'])
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if params['printInfo']:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S


def inv_general2D(A, B, C, D, E, F, G, S, dims, params):
    """
    A template for inverting a 2D slice of equation in general form as:
    
          d^2S     d^2S     d^2S     dS     dS 
        A ---- + B ---- + C ---- + D -- + E -- + F*S = G
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
    params: dict
        Parameters for inversion.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    for selDict in loop_noncore(F, dims):
        invert_general_2D(S.loc[selDict].values, A.loc[selDict].values,
                          B.loc[selDict].values, C.loc[selDict].values,
                          D.loc[selDict].values, E.loc[selDict].values,
                          F.loc[selDict].values, G.loc[selDict].values,
                          params['gc2' ], params['gc1' ],
                          params['del2'], params['del1'],
                          params['BCs'][0], params['BCs'][1], params['del1Sqr'],
                          params['ratio'], params['ratioQtr'], params['ratioSqr'],
                          params['optArg'], _undeftmp, params['flags'],
                          params['mxLoop'], params['tolerance'])
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if params['printInfo']:
            if params['flags'][0]:
                print(info + ' loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(params['flags'][2], params['flags'][1]))
            else:
                print(info + ' loops {0:4.0f} and tolerance is {1:e}'
                      .format(params['flags'][2], params['flags'][1]))
    
    return S


def inv_general2D_bih(A, B, C, D, E, F, G, H, I, J, S, dims, params):
    """
    A template for inverting a 2D slice of equation in general form as:
    
      d^4S       d^4S       d^4S     d^2S     d^2S     d^2S     dS     dS 
    A ---- + B -------- + C ---- + D ---- + E ---- + F ---- + G -- + H -- + I*S = J
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
    params: dict
        Parameters for inversion.

    Returns
    -------
    S: xr.DataArray
        Solution.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    for selDict in loop_noncore(F, dims):
        invert_general_bih_2D(S.loc[selDict].values, A.loc[selDict].values,
                              B.loc[selDict].values, C.loc[selDict].values,
                              D.loc[selDict].values, E.loc[selDict].values,
                              F.loc[selDict].values, G.loc[selDict].values,
                              H.loc[selDict].values, I.loc[selDict].values,
                              J.loc[selDict].values,
                              params['gc2' ], params['gc1' ],
                              params['del2'], params['del1'],
                              params['BCs'][0], params['BCs'][1],
                              params['del1SSr'], params['del1Tr'], params['del1Sqr'],
                              params['ratio'   ], params['ratioSSr'],
                              params['ratioQtr'], params['ratioSqr'],
                              params['optArg'], _undeftmp, params['flags'],
                              params['mxLoop'], params['tolerance'])
        
        info = str(selDict).replace('numpy.datetime64(', '') \
                           .replace('numpy.timedelta64(', '') \
                           .replace(')', '') \
                           .replace('\'', '') \
                           .replace('.000000000', '')
        
        if params['printInfo']:
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


