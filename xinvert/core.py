# -*- coding: utf-8 -*-
"""
Created on 2020.12.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from .numbas import invert_standard_2D, invert_general_2D, invert_general_bih_2D


"""
Core functions are defined below
"""
_R_earth = 6371200.0 # consistent with GrADS
_omega = 7.292e-5
_undeftmp = -9.99e8

_latlon = ['lat', 'LAT', 'latitude' , 'LATITUDE' , 'lats', 'LATS',
           'lon', 'LON', 'longitude', 'LONGITUDE', 'long', 'LONG']


def invert_Poisson(force, dims, BCs=['fixed', 'fixed'], coords='latlon',
                   undef=np.nan, mxLoop=5000, tolerance=1e-6, optArg=None,
                   cal_flow=False, printInfo=True, debug=False):
    """
    Inverting Poisson equation of the form \nabla^2 S = F:
    
        d  dS    d  dS
        --(--) + --(--) = F
        dy dy    dx dx
    
    using SOR iteration.
    
    Parameters
    ----------
    force: xarray.DataArray
        Forcing function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    BCs: list
        Boundary conditions for each dimension in dims.
    coords: str
        Coordinates in ['latlon', 'cartesian'] are supported.
    undef: float
        Undefined value.
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
    ----------
    S: xarray.DataArray
        Results of the SOR inversion.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')
    
    # properly masking forcing
    if np.isnan(undef):
        forcing = force.fillna(_undeftmp)
    else:
        forcing = force.where(force!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    S    = zero.copy().load() # loaded because dask cannot be modified
    
    ######  calculating the coefficients  ######
    if coords == 'latlon':
        lats = force[dims[0]]
        
        A = zero + np.cos(np.deg2rad((lats+lats.shift({dims[0]:1}))/2.0))
        B = zero
        C = zero + 1.0/np.cos(np.deg2rad(lats))
        F =(forcing * np.cos(np.deg2rad(lats))).where(forcing!=_undeftmp,
                                                      _undeftmp)
    elif coords == 'cartesian':
        A = zero + 1.0
        B = zero
        C = zero + 1.0
        F = forcing.where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    __inv_standard(A, B, C, F, S, dims, BCs,
                  mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    
    if cal_flow:
        u, v = __cal_flow(S, coords)
        
        return S, u, v
    else:
        return S


def invert_Eliassen(force, um, bm, dims, BCs=['fixed', 'fixed'],
                    coords='latlon', f0=None, beta=None, optArg=None,
                    undef=np.nan, mxLoop=5000, tolerance=1e-6,
                    cal_flow=False, printInfo=True, debug=False):
    """
    Inverting meridional overturning streamfunction of the form:
    
        d ┌  dS      dS ┐   d ┌  dS      dS ┐
        --│A(--) + B(--)│ + --│B(--) + C(--)│ = F
        dy└  dy      dx ┘   dx└  dy      dx ┘
    
    using SOR iteration.
    
    Parameters
    ----------
    force: xarray.DataArray
        Forcing function.
    um: xarray.DataArray
        Forcing function.
    bm: xarray.DataArray
        Forcing function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lev'].
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
    
    zero = force - force
    S    = zero.copy().load() # loaded because dask cannot be modified
    
    ######  calculating the coefficients  ######
    if coords == 'latlon':
        lats = force[dims[0]]
        const = _R_earth / 180. * np.pi
        
        f = 2.0 * _omega * np.sin(lats)
        
        A = f * um.differentiate(dims[1]) / const
        B =     bm.differentiate(dims[1])
        C = -   bm.differentiate(dims[0])
        F =(force * np.cos(lats)).where(force!=_undeftmp, _undeftmp)
    elif coords == 'cartesian':
        ydef = force[dims[0]]
        
        f = f0 + beta * ydef
        
        A = f * um.differentiate(dims[1])
        B =     bm.differentiate(dims[1])
        C = -   bm.differentiate(dims[0])
        F = force.where(force!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    __inv_standard(A, B, C, F, S, dims, BCs,
                   mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    S = S.where(force!=_undeftmp, other=undef).rename('inverted')
    
    if cal_flow:
        u, v = __cal_flow(S, coords)
        
        return S, u, v
    else:
        return S


def invert_GillMatsuno(Q, dims, BCs=['fixed', 'fixed'], coords='latlon',
                       Phi=10000, epsilon=7e-6, f0=None, beta=None,
                       undef=np.nan, mxLoop=5000, tolerance=1e-11, optArg=None,
                       cal_flow=False, printInfo=True, debug=False):
    """
    Inverting Gill-Matsuno model of the form:
    
                             dphi
        epsilon * u =   fv - ----
                              dx
    
                             dphi
        epsilon * v = - fu - ----
                              dy
    
                            ┌ du   dv ┐
        epsilon * phi + Phi*│ -- + -- │ = -Q
                            └ dx   dy ┘
    
    given the heating field Q using SOR iteration.
    
    Parameters
    ----------
    Q: xarray.DataArray
        heating function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    BCs: list
        Boundary conditions for each dimension in dims.
    coords: str
        Coordinates in ['latlon', 'cartesian'] are supported.
    Phi: float or xarray.DataArray
        Reference geopotential field.
    epsilon: float or xarray.DataArray
        Rayleigh friction coefficient.
    f0: float
        Coriolis parameter at southern boundary.
    beta: float
        Beta-plane constant = df/dy.
    undef: float
        Undefined value in Q.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    optArg: float
        Optimal argument for SOR.
    cal_flow: boolean
        Calculate associated wind fields or not.
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
    
    if np.isnan(undef):
        forcing = Q.fillna(_undeftmp)
    else:
        forcing = Q.where(Q!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    S    = zero.copy().load() # loaded because dask cannot be modified
    
    ######  calculating the coefficients  ######
    if coords == 'latlon':
        lats = Q[dims[0]]
        cosLat = np.cos(np.deg2rad(lats))
        f = 2.0 * _omega * np.sin(np.deg2rad(lats))
        
        coef1 = epsilon / (epsilon**2. + f**2.)
        coef2 = f       / (epsilon**2. + f**2.)
        const = _R_earth / 180. * np.pi
        
        S = zero.copy().load()
        A = zero + coef1 * Phi * cosLat
        B = zero
        C = zero + coef1 * Phi / cosLat
        D = zero + Phi * (coef1 * cosLat).differentiate(dims[0]) / const
        E = zero - Phi * (coef2 * cosLat).differentiate(dims[0]) / const
        F = zero - epsilon * cosLat
        G = (-forcing * cosLat).where(forcing!=_undeftmp, _undeftmp)
    elif coords == 'cartesian':
        ydef = Q[dims[0]]
        f = f0 + beta * ydef
        
        coef1 = epsilon / (epsilon**2. + f**2.)
        coef2 = f       / (epsilon**2. + f**2.)
        
        S = zero.copy().load()
        A = zero + coef1 * Phi
        B = zero
        C = zero + coef1 * Phi
        D = zero + Phi * (coef1).differentiate(dims[0])
        E = zero - Phi * (coef2).differentiate(dims[0])
        F = zero - epsilon
        G = (-forcing).where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    __inv_general(A, B, C, D, E, F, G, S, dims, BCs,
                  mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    
    if cal_flow:
        if coords == 'latlon':
            u = - coef1 * S.differentiate(dims[1]) / const / cosLat \
                - coef2 * S.differentiate(dims[0]) / const
            v = - coef1 * S.differentiate(dims[0]) / const \
                + coef2 * S.differentiate(dims[1]) / const / cosLat
        elif coords == 'cartesian':
            u = - coef1 * S.differentiate(dims[1]) \
                - coef2 * S.differentiate(dims[0])
            v = - coef1 * S.differentiate(dims[0]) \
                + coef2 * S.differentiate(dims[1])
        else:
            raise Exception('unsupported coords ' + coords +
                            ', should be [latlon, cartesian]')
        
        return S, u, v
    else:
        return S


def invert_Stommel(curl, dims, BCs=['fixed', 'fixed'], coords='latlon',
                      R=5e-5, depth=100, rho=1027, beta=None,
                      undef=np.nan, mxLoop=5000, tolerance=1e-11, optArg=None, 
                      cal_flow=False, printInfo=True, debug=False):
    """
    Inverting Stommel model of the form:
    
         R                          dpsi     curl(tau)
      - --- * \nabla^2 psi - beta * ---- = - ---------
         D                           dx       rho * D
    
    given the wind stress curl field using SOR iteration.
    
    Parameters
    ----------
    curl: xarray.DataArray
        Wind stress curl.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    BCs: list
        Boundary conditions for each dimension in dims.
    coords: str
        Coordinates in ['latlon', 'cartesian'] are supported.
    R: float or xarray.DataArray
        Rayleigh friction coefficient.
    depth: float or xarray.DataArray
        Depth of the ocean.
    rho: float or xarray.DataArray
        Density of the ocean.
    beta: float
        Beta-plane constant = df/dy.
    undef: float
        Undefined value.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    optArg: float
        Optimal argument for SOR.
    cal_flow: boolean
        Calculate associated wind fields or not.
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
    
    if np.isnan(undef):
        forcing = curl.fillna(_undeftmp)
    else:
        forcing = curl.where(curl!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    S    = zero.copy().load() # loaded because dask cannot be modified
    
    ######  calculating the coefficients  ######
    if coords == 'latlon':
        lats = curl[dims[0]]
        cosLat = np.cos(np.deg2rad(lats))
        
        A = zero - R / depth * cosLat
        B = zero
        C = zero - R / depth / cosLat
        D = zero
        E = zero - 2.0 * _omega * cosLat / _R_earth
        F = zero
        G = (forcing / depth / rho * cosLat).where(forcing!=_undeftmp,
                                                    _undeftmp)
    elif coords == 'cartesian':
        A = zero - R / depth
        B = zero
        C = zero - R / depth
        D = zero
        E = zero - beta
        F = zero
        G = (forcing / depth / rho).where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    __inv_general(A, B, C, D, E, F, G, S, dims, BCs,
                  mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    
    if cal_flow:
        u, v = __cal_flow(S, coords)
        
        return S, u, v
    else:
        return S


def invert_StommelMunk(curl, dims, BCs=['fixed', 'fixed'], coords='latlon',
                       AH=1e5, R=5e-5, depth=100, rho=1027, beta=None,
                       undef=np.nan, mxLoop=5000, tolerance=1e-11, optArg=None,
                       cal_flow=False, printInfo=True, debug=False):
    """
    Inverting Stommel-Munk model of the form:
    
                        R                          dpsi     curl(tau)
      AH*nabla^4 psi - --- * \nabla^2 psi - beta * ---- = - ---------
                        D                           dx       rho * D
    
    given the wind stress curl field using SOR iteration.
    
    Parameters
    ----------
    curl: float or xarray.DataArray
        Wind-stress curl (N/m^2).
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    BCs: list
        Boundary conditions for each dimension in dims.
    coords: str
        Coordinates in ['latlon', 'cartesian'] are supported.
    AH: float or xarray.DataArray
        Biharmonic coefficient.
    R: float or xarray.DataArray
        Rayleigh friction coefficient.
    depth: float or xarray.DataArray
        Depth of the ocean.
    rho: float or xarray.DataArray
        Density of the ocean.
    beta: float
        Beta-plane constant = df/dy.
    undef: float
        Undefined value.
    mxLoop: int
        Maximum loop number over which iteration stops.
    tolerance: float
        Tolerance smaller than which iteration stops.
    undef: float
        Undefined value.
    optArg: float
        Optimal argument for SOR.
    cal_flow: boolean
        Calculate associated wind fields or not.
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
    
    if np.isnan(undef):
        forcing = curl.fillna(_undeftmp)
    else:
        forcing = curl.where(curl!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    S    = zero.copy().load() # loaded because dask cannot be modified
    
    ######  calculating the coefficients  ######
    if coords == 'latlon':
        lats = curl[dims[0]]
        cosLat = np.cos(np.deg2rad(lats))
        
        A = zero + AH * cosLat
        B = zero
        C = zero + AH / cosLat
        D = zero - R / depth * cosLat
        E = zero
        F = zero - R / depth / cosLat
        G = zero
        H = zero - 2.0 * _omega * cosLat / _R_earth
        I = zero
        J = (forcing / depth / rho * cosLat).where(forcing!=_undeftmp,
                                                    _undeftmp)
    elif coords == 'cartesian':
        A = zero + AH
        B = zero
        C = zero + AH
        D = zero - R / depth
        E = zero
        F = zero - R / depth
        G = zero
        H = zero - beta
        I = zero
        J = (forcing / depth / rho).where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    __inv_general_bih(A, B, C, D, E, F, G, H, I, J, S, dims, BCs,
                      mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    
    if cal_flow:
        u, v = __cal_flow(S, coords)
        
        return S, u, v
    else:
        return S


def invert_geostreamfunction(lapPhi, dims, BCs=['fixed', 'fixed'],
                             coords='latlon', f0=None, beta=None, optArg=None,
                             undef=np.nan, mxLoop=5000, tolerance=1e-6,
                             cal_flow=False, printInfo=True, debug=False):
    """
    Inverting geostrophic streamfunction of the form:
    
        d    dS    d    dS
        --(f*--) + --(f*--) = F
        dy   dy    dx   dx
    
    using SOR iteration.
    
    Parameters
    ----------
    lapPhi: xarray.DataArray
        Laplacian of geopotential.
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
    
    zero = lapPhi - lapPhi
    S    = zero.copy().load() # loaded because dask cannot be modified
    
    ######  calculating the coefficients  ######
    if coords == 'latlon':
        lats = lapPhi[dims[0]]
        
        latsH = np.deg2rad((lats+lats.shift({dims[0]:1}))/2.0)
        latsG = np.deg2rad(lats)
        fH = 2.0 * _omega * np.sin(latsH)
        fG = 2.0 * _omega * np.sin(latsG)
        
        # regulation for near-zero f
        # fH = fH.where(np.abs(fH)>5.09e-06, other=5.0897425990927414e-06)
        # fG = fG.where(np.abs(fG)>5.09e-06, other=5.0897425990927414e-06)
        
        A = zero + fH * np.cos(latsH)
        B = zero
        C = zero + fG / np.cos(latsG)
        F =(lapPhi * np.cos(latsG)).where(lapPhi!=_undeftmp, _undeftmp)
    elif coords == 'cartesian':
        ydef = lapPhi[dims[0]]
        
        f = f0 + beta * ydef
        
        A = zero + f
        B = zero
        C = zero + f
        F = lapPhi.where(lapPhi!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    __inv_standard(A, B, C, F, S, dims, BCs,
                   mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    S = S.where(lapPhi!=_undeftmp, other=undef).rename('inverted')
    
    if cal_flow:
        u, v = __cal_flow(S, coords)
        
        return S, u, v
    else:
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
    C = zero + np.cos(np.deg2rad((lats+lats.shift({force.dims[0]:1}))/2.0))
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
        loop += 1
        
        invert_slice(snapshot.values, A.values, B.values, C.values, F.values,
                     dim2, dim1, del2, del1, BC2, BC1, delD2Sqr,
                     ratioQtr, ratioSqr, optArg, undeftmp, flags,
                     mxLoop=loop_per_frame)
        
        if printInfo:
            if flags[0]:
                print('loops {0:4.0f} and tolerance is {1:e} (overflows!)'
                      .format(flags[2], flags[1]))
            else:
                print('loops {0:4.0f} and tolerance is {1:e}'
                      .format(flags[2], flags[1]))
        
        lst.append(snapshot.copy())
        
        if flags[2] < loop_per_frame or loop >= max_loop:
            break
    
    re = xr.concat(lst, dim='iteration').rename('inverted')
    re['iteration'] = xr.DataArray(np.arange(loop_per_frame,
                                             loop_per_frame*(max_loop+1),
                                             loop_per_frame),
                                   dims=['iteration'])
    
    return re




"""
Below are the private helper methods
"""
def __inv_standard(A, B, C, F, S, dims, BCs,
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
    
    dimAll = F.dims
    
    dimInv = [] # ordered dims for inversion
    dimLop = [] # ordered dims for loop
    for dim in dimAll:
        if dim in dims:
            dimInv.append(dim)
        else:
            dimLop.append(dim)
    
    dimLopVars = []
    for dim in dimLop:
        dimLopVars.append(F[dim].values)
    
    params = __cal_params(F[dimInv[0]], F[dimInv[1]], debug=debug)
    
    from itertools import product
    for idices in product(*dimLopVars):
        selDict = {}
        for d, i in zip(dimLop, idices):
            selDict[d] = i
        
        if optArg == None:
            optArg = params['optArg']
        
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


def __inv_general(A, B, C, D, E, F, G, S, dims, BCs,
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
    
    dimAll = F.dims
    
    dimInv = [] # ordered dims for inversion
    dimLop = [] # ordered dims for loop
    for dim in dimAll:
        if dim in dims:
            dimInv.append(dim)
        else:
            dimLop.append(dim)
    
    dimLopVars = []
    for dim in dimLop:
        dimLopVars.append(F[dim].values)
    
    params = __cal_params(F[dimInv[0]], F[dimInv[1]], debug=debug)
    
    from itertools import product
    for idices in product(*dimLopVars):
        selDict = {}
        for d, i in zip(dimLop, idices):
            selDict[d] = i
        
        if optArg == None:
            optArg = params['optArg']
        
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


def __inv_general_bih(A, B, C, D, E, F, G, H, I, J, S, dims, BCs,
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
    
    dimAll = F.dims
    
    dimInv = [] # ordered dims for inversion
    dimLop = [] # ordered dims for loop
    for dim in dimAll:
        if dim in dims:
            dimInv.append(dim)
        else:
            dimLop.append(dim)
    
    dimLopVars = []
    for dim in dimLop:
        dimLopVars.append(F[dim].values)
    
    params = __cal_params(F[dimInv[0]], F[dimInv[1]], debug=debug)
    
    from itertools import product
    for idices in product(*dimLopVars):
        selDict = {}
        for d, i in zip(dimLop, idices):
            selDict[d] = i
        
        if optArg == None:
            optArg = 1 # params['optArg'],  1 seems to be safer for 4-order SOR
        
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


def __cal_flow(S, coords):
    """
    Calculate flow using streamfunction.

    Parameters
    ----------
    S : xarray.DataArray
        Streamfunction.
    coords: str
        Coordinates in ['latlon', 'cartesian'] are supported.

    Returns
    -------
    re : tuple
        Flow components.
    """
    dims = S.dims
    
    if coords == 'latlon':
        const = _R_earth / 180. * np.pi
        cosLat = np.cos(np.deg2rad(S[dims[0]]))
        
        u = -S.differentiate(dims[0]) / const
        v =  S.differentiate(dims[1]) / const / cosLat
    elif coords == 'cartesian':
        u = -S.differentiate(dims[0])
        v =  S.differentiate(dims[1])
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    return u, v


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

