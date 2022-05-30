# -*- coding: utf-8 -*-
"""
Created on 2022.04.13

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
from .utils import _R_earth, _omega, _undeftmp, _g
from .core import inv_standard3D, inv_standard2D, inv_general2D,\
                  inv_general2D_bih


"""
Core functions are defined below
"""
def invert_MultiGrid(invert_func, *args, ratio=3, gridNo=3, **kwargs):
    from utils.XarrayUtils import coarsen
    
    ratios = [10, 6, 3, 1]
    
    mxLoop = kwargs['mxLoop'] if 'mxLoop' in kwargs else 5000
    
    loops  = [mxLoop*ratio/10 for ratio in ratios]
    
    if 'dims' not in kwargs:
        raise Exception('kwarg dims= should be provided')
    
    dims = kwargs['dims']
        
    if 'BCs' in kwargs:
        BCs = kwargs['BCs']
    else:
        BCs = ['fixed', 'fixed']
    
    periodics = [dim for dim, BC in zip(dims, BCs) if 'periodic' in BC]
    
    fs = [[coarsen(arg, kwargs['dims'], ratio=ratio,
                   periodic=periodics) for arg in args] for ratio in ratios]
    
    o_guess = fs[0][0] - fs[0][0]
    
    o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
    os = []
    
    for i, (ratio, frs, loop) in enumerate(zip(ratios, fs, loops)):
        kwargs['mxLoop'] = loop
        
        # iteration over coarse grid
        o_guess = invert_func(*frs, **kwargs)
        os.append(o_guess)
        
        print('finish grid of ratio =', ratio, ', loop =', loop)
        
        if ratio == 1:
            break
        
        o_guess = o_guess.interp_like(frs[0])
        o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
    
    return o_guess, fs, os


def invert_Poisson(force, dims, BCs=['fixed', 'fixed'], coords='latlon',
                   undef=np.nan, mxLoop=5000, tolerance=1e-6, optArg=None,
                   cal_flow=False, printInfo=True, debug=False, out=None):
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
    
    if out is None:
        S = zero.copy().load() # loaded because dask cannot be modified
    else:
        S = xr.where(forcing==_undeftmp, out, 0) # applied boundary
    
    ######  calculating the coefficients  ######
    if coords.lower() == 'latlon':
        lats = force[dims[0]]
        
        A = zero + np.cos(np.deg2rad((lats+lats.shift({dims[0]:1}))/2.0))
        B = zero
        C = zero + 1.0/np.cos(np.deg2rad(lats))
        F =(forcing * np.cos(np.deg2rad(lats))).where(forcing!=_undeftmp,
                                                      _undeftmp)
    elif coords.lower() == 'cartesian':
        A = zero + 1.0
        B = zero
        C = zero + 1.0
        F = forcing.where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    inv_standard2D(A, B, C, F, S, dims, BCs, coords,
                  mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    if out is None:
        S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    else:
        S = S.rename('inverted')
    
    if cal_flow:
        u, v = __cal_flow(S, coords)
        
        return S, u, v
    else:
        return S


def invert_Vortex_2D(Q, angM, Gamma, dims, BCs=['fixed', 'fixed'],
                     coords='latlon', f0=1E-4,
                     undef=np.nan, mxLoop=5000, tolerance=1e-6, optArg=None,
                     cal_flow=False, printInfo=True, debug=False, out=None):
    """
    Inverting nonlinear balanced vortex equation of the form:
    
        d   2Λ dΛ    d  Γg dΛ
        --(--- --) + --(-- --)  = 0
        dθ r^3 dθ    dr Qr dr
    
    using SOR iteration.
    
    Parameters
    ----------
    Q: xarray.DataArray
        2D distribution of PV.
    angM: xarray.DataArray
        Initial guess of the unknown angular momentum Λ as the known coefficient.
    Gamma: xarray.DataArray
        A vertical function defined as Γ = Rd/p * (p/p0)^κ = κ * Π/p.
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
        forcing = (Q-Q).fillna(_undeftmp)
    else:
        forcing = (Q-Q).where(Q!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    dim0 = forcing[dims[0]]
    dim1 = forcing[dims[1]]
    
    if out is None:
        S = zero.copy().load() # loaded because dask cannot be modified
    else:
        cond1 = forcing==_undeftmp
        cond2 = dim0.isin([dim0[0], dim0[-1]])
        cond3 = dim1.isin([dim1[0], dim1[-1]])
        # applied boundary
        mask = np.logical_or(cond1, np.logical_or(cond2, cond3))
        S = xr.where(mask, out, 0)
    
    ######  calculating the coefficients  ######
    if coords.lower() == 'latlon':
        lats = dim1
        
        A = zero + np.sin(np.deg2rad(lats))
        B = zero
        C = zero + Gamma * _g / Q / dim1
        F = zero.where(forcing!=_undeftmp, _undeftmp)
    elif coords.lower() == 'cartesian':
        A = zero + 2.0 * angM / dim1**3.0
        B = zero
        C = zero + Gamma * _g / Q / dim1
        F = zero.where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    inv_standard2D(A, B, C, F, S, dims, BCs, coords,
                  mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    if out is None:
        S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    else:
        S = S.rename('inverted')
    
    return S


def invert_QGPV_2D(QGPV, N2, dims, BCs=['fixed', 'fixed'],
                   coords='latlon', f0=1E-4,
                   undef=np.nan, mxLoop=5000, tolerance=1e-6, optArg=None,
                   cal_flow=False, printInfo=True, debug=False, out=None):
    """
    Inverting QG PV equation of the form:
    
        d   f  dS    1 d  dS
        --(--- --) + - --(--) + f = q
        dp N^2 dp    f dy dy
    
    using SOR iteration.  It is slightly changed to:
    
        d  f^2 dS    d  dS
        --(--- --) + --(--) = (q - f)f
        dp N^2 dp    dy dy
    
    Parameters
    ----------
    QGPV: xarray.DataArray
        2D distribution of QGPV.
    N2: xarray.DataArray
        Buoyancy frequency.
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
        forcing = QGPV.fillna(_undeftmp)
    else:
        forcing = QGPV.where(QGPV!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    
    if out is None:
        S = zero.copy().load() # loaded because dask cannot be modified
    else:
        S = xr.where(forcing==_undeftmp, out, 0) # applied boundary
    
    ######  calculating the coefficients  ######
    if coords.lower() == 'latlon':
        # lats = QGPV[dims[0]]
        
        A = zero + f0**2 / N2
        B = zero
        C = zero + 1.0
        F =((forcing-f0)*f0).where(forcing!=_undeftmp, _undeftmp)
    elif coords.lower() == 'cartesian':
        A = zero + f0 / N2
        B = zero
        C = zero + 1.0
        F =((forcing-f0)*f0).where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    inv_standard2D(A, B, C, F, S, dims, BCs, coords,
                  mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    if out is None:
        S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    else:
        S = S.rename('inverted')
    
    return S


def invert_Eliassen(force, Am, Bm, Cm, dims, BCs=['fixed', 'fixed'],
                    coords='latlon', f0=None, beta=None, optArg=None,
                    undef=np.nan, mxLoop=5000, tolerance=1e-6,
                    cal_flow=False, printInfo=True, debug=False, out=None):
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
    Am: xarray.DataArray
        Coefficient for the second direction (e.g., lev or lat).
    Bm: xarray.DataArray
        Forcing function.
    Cm: xarray.DataArray
        Coefficient for the first direction (e.g., lat or lon).
    dims: list
        Dimension combination for the inversion
        e.g., ['lev', 'lat'] or ['lat','lon'].
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
    
    # properly masking forcing
    if np.isnan(undef):
        forcing = force.fillna(_undeftmp)
    else:
        forcing = force.where(force!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    
    if out is None:
        S = zero.copy().load() # loaded because dask cannot be modified
    else:
        S = xr.where(forcing==_undeftmp, out, 0) # applied boundary
    
    ######  calculating the coefficients  ######
    if coords.lower() == 'latlon':
        lats = force[dims[0]]
        
        A = Am
        B = Bm
        C = Cm
        F =(forcing * np.cos(np.deg2rad(lats))).where(forcing!=_undeftmp,
                                                      _undeftmp)
    elif coords.lower() == 'cartesian':
        A = Am
        B = Bm
        C = Cm
        F = forcing.where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    inv_standard2D(A, B, C, F, S, dims, BCs, coords,
                   mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    if out is None:
        S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    else:
        S = S.rename('inverted')
    
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
    if coords.lower() == 'latlon':
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
    elif coords.lower() == 'cartesian':
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
    inv_general2D(A, B, C, D, E, F, G, S, dims, BCs, coords,
                  mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    S = S.where(forcing!=_undeftmp, other=undef).rename('inverted')
    
    if cal_flow:
        if coords.lower() == 'latlon':
            u = - coef1 * S.differentiate(dims[1]) / const / cosLat \
                - coef2 * S.differentiate(dims[0]) / const
            v = - coef1 * S.differentiate(dims[0]) / const \
                + coef2 * S.differentiate(dims[1]) / const / cosLat
        elif coords.lower() == 'cartesian':
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
    if coords.lower() == 'latlon':
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
    elif coords.lower() == 'cartesian':
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
    inv_general2D(A, B, C, D, E, F, G, S, dims, BCs, coords,
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
    if coords.lower() == 'latlon':
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
    elif coords.lower() == 'cartesian':
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
    inv_general2D_bih(A, B, C, D, E, F, G, H, I, J, S, dims, BCs, coords,
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
    if coords.lower() == 'latlon':
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
    elif coords.lower() == 'cartesian':
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
    inv_standard2D(A, B, C, F, S, dims, BCs, coords,
                   mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    S = S.where(lapPhi!=_undeftmp, other=undef).rename('inverted')
    
    if cal_flow:
        u, v = __cal_flow(S, coords)
        
        return S, u, v
    else:
        return S


def invert_Omega_MG(force, S, dims, BCs=['fixed', 'fixed', 'fixed'],
                    coords='latlon', f0=None, beta=None,
                    undef=np.nan, mxLoop=5000, tolerance=1e-6,
                    optArg=None, printInfo=True, debug=False,
                    icbc=None, ratio=4, gridNo=3):
    from utils.XarrayUtils import coarsen
    
    ratios = [10, 6, 3, 1]
    loops  = [mxLoop*ratio/30 for ratio in ratios]
    
    fs = [coarsen(force, dims, ratio=ratio) for ratio in ratios]
    Ss = [coarsen(S    , dims, ratio=ratio) for ratio in ratios]
    
    o_guess = fs[0] - fs[0]
    o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
    os = []
    
    for i, (ratio, frc, stab, loop) in enumerate(zip(ratios, fs, Ss, loops)):
        # iteration over coarse grid
        o_guess = invert_OmegaEquation(frc, stab, dims=dims, BCs=BCs, f0=f0,
                                       coords=coords, beta=beta, undef=undef,
                                       mxLoop=loop, tolerance=tolerance,
                                       optArg=optArg, debug=debug, icbc=o_guess,
                                       printInfo=printInfo)
        os.append(o_guess)
        
        print('finish grid of ratio =', ratio, ', loop =', loop)
        
        if ratio == 1:
            break
        
        o_guess = o_guess.interp_like(fs[i+1])
        o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
        
        # # iteration over fine grid from the coarse initial guess
        # o = invert_OmegaEquation(force, S, dims=dims, BCs=BCs, coords=coords,
        #                          f0=f0, beta=beta, undef=undef, mxLoop=mxLoop/50,
        #                          tolerance=tolerance, optArg=optArg, debug=debug,
        #                          printInfo=printInfo, icbc=o_guess)
    
    return o_guess, fs, os



def invert_OmegaEquation(force, S, dims, BCs=['fixed', 'fixed', 'fixed'],
                         coords='latlon', f0=None, beta=None,
                         undef=np.nan, mxLoop=5000, tolerance=1e-6,
                         optArg=None, printInfo=True, debug=False,
                         icbc=None):
    """
    Inverting Omega equation of the form:
    
               f^2 d ┌ dw ┐   d ┌ dw ┐   d ┌ dw ┐   F
        L(w) = --- --│(--)│ + --│(--)│ + --│(--)│ = -
                S  dz└ dz ┘   dy└ dy ┘   dx└ dx ┘   S
    
    using SOR iteration.  It is slightly changed to:
    
                d ┌    dw ┐   d ┌  dw ┐   d ┌  dw ┐
        L(w) =  --│f^2(--)│ + --│S(--)│ + --│S(--)│ = F
                dz└    dz ┘   dy└  dy ┘   dx└  dx ┘
    
    Parameters
    ----------
    force: xarray.DataArray
        Forcing function.
    S: xarray.DataArray
        Stratification averaged over the domain.
    dims: list
        Dimension combination for the inversion e.g., ['lev', 'lat', 'lon'].
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
    icbc: xarray.DataArray
        Initial and boundary conditions for the result.
        
    Returns
    ----------
    S: xarray.DataArray
        Results of the SOR inversion.
    """
    if len(dims) != 3:
        raise Exception('3 dimensions are needed for inversion')
    
    if not np.isfinite(S[1:]).all():
        raise Exception('inifinite stratification coefficient A')
    
    if np.isnan(S[1:]).any():
        raise Exception('nan in coefficient A')
    
    if (S[1:]<=0).any():
        raise Exception('unstable stratification in coefficient A')
    
    force, S = xr.broadcast(force, S)
    
    # properly masking forcing
    if np.isnan(undef):
        forcing = force.fillna(_undeftmp)
    else:
        forcing = force.where(force!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    
    if icbc is None:
        omega = zero.copy().load() # loaded because dask cannot be modified
    else:
        omega = icbc
    
    ######  calculating the coefficients  ######
    if coords.lower() == 'latlon':
        lats = force[dims[1]]
        
        latsH = np.deg2rad((lats+lats.shift({dims[1]:1}))/2.0)
        latsG = np.deg2rad(lats)
        f = 2.0 * _omega * np.sin(latsG)
        
        A = zero + f**2 * np.cos(latsG)
        B = zero + S*np.cos(latsH)
        C = zero + S/np.cos(latsG)
        F =(forcing * np.cos(latsG)).where(forcing!=_undeftmp, _undeftmp)
        
    elif coords.lower() == 'cartesian':
        ydef = force[dims[1]]
        
        f = f0 + beta * ydef
        # dTHdz = np.abs((theta.shift({dims[0]:1}) - theta) / \
        #                ( levs.shift({dims[0]:1}) - levs ))
        
        A = zero + f**2
        B = zero + S
        C = zero + S
        F = forcing.where(forcing!=_undeftmp, _undeftmp)
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be [latlon, cartesian]')
    
    # inversion
    inv_standard3D(A, B, C, F, omega, dims, BCs, coords,
                   mxLoop, tolerance, optArg, printInfo, debug)
    
    # properly de-masking
    omega = omega.where(forcing!=_undeftmp, other=undef).rename('inverted')
    
    return omega


# def invert_Poisson_animated(force, BCs=['fixed', 'fixed'], undef=np.nan,
#                             loop_per_frame=5, max_loop=100,
#                             printInfo=True, debug=False):
#     """
#     Inverting Poisson equation of the form \nabla^2 S = F:
    
#         d  dS    d  dS
#         --(--) + --(--) = F
#         dx dx    dy dy
    
#     using SOR iteration.
    
#     Parameters
#     ----------
#     force: xarray.DataArray
#         Forcing function with a single slice (2D field).
#     undef: float
#         Undefined value.
#     printInfo: boolean
#         Flag for printing.
#     dims: list
#         Dimension combination for the inversion e.g., ['lat', 'lon'].
#     BCs: list
#         Boundary conditions for each dimension in dims.
#     undef: float
#         Undefined value.
#     mxLoop: int
#         Maximum loop number over which iteration stops.
#     tolerance: float
#         Tolerance smaller than which iteration stops.
#     printInfo: boolean
#         Flag for printing.
#     debug: boolean
#         Output debug info.
        
#     Returns
#     ----------
#     S: xarray.DataArray
#         Results of the SOR inversion.
#     """
#     if len(force.dims) != 2:
#         raise Exception('2 dimensions are needed for inversion')
    
#     lats = force[force.dims[0]]
#     undeftmp = -9.99e8
    
#     if np.isnan(undef):
#         forcing = force.fillna(undeftmp)
#     elif undef == 0:
#         forcing = force.where(force!=0, other=undeftmp)
#     else:
#         forcing = force.where(force!=undef, other=undeftmp)
        
#     zero = (forcing - forcing).load()
    
#     A = zero + 1.0/np.cos(np.deg2rad(lats))
#     B = zero.copy()
#     C = zero + np.cos(np.deg2rad((lats+lats.shift({force.dims[0]:1}))/2.0))
#     F =(forcing * np.cos(np.deg2rad(lats))).where(forcing!=undeftmp, undeftmp)
    
#     dimAll = F.dims
    
#     dim1_var = F[dimAll[1]]
#     dim2_var = F[dimAll[0]]
    
#     BC2 , BC1  = BCs
#     dim1 = len(dim1_var)
#     dim2 = len(dim2_var)
#     del1 = dim1_var.diff(dimAll[1]).values[0] # assumed uniform
#     del2 = dim2_var.diff(dimAll[0]).values[0] # assumed uniform
    
#     if dimAll[1] in _latlon:
#         del1 = np.deg2rad(del1) * _R_earth # convert lat/lon to m
#     if dimAll[0] in _latlon:
#         del2 = np.deg2rad(del2) * _R_earth # convert lat/lon to m
    
#     ratioQtr = del2 / del1
#     ratioSqr = ratioQtr ** 2.0
#     ratioQtr /= 4.0
#     delD2Sqr = del2 ** 2.0
#     flags = np.array([0.0, 1.0, 0.0])
    
#     epsilon = np.sin(np.pi/(2.0*dim1+2.0))**2 + np.sin(np.pi/(2.0*dim2+2.0))**2
#     optArg  = 2.0 / (1.0 + np.sqrt((2.0 - epsilon) * epsilon))
    
#     if debug:
#         print('dim grids:', dim2, dim1)
#         print('dim intervals: ', del2, del2)
#         print('BCs: ', BC2, BC1)
#         print('ratioQtr, Sqr: ', ratioQtr, ratioSqr)
#         print('delD2Sqr: ', delD2Sqr)
#         print('optArg: ', optArg)
#         print('epsilon: ', epsilon)
    
#     lst = []
    
#     snapshot = zero
#     loop = 0
#     while True:
#         loop += 1
        
#         invert_slice(snapshot.values, A.values, B.values, C.values, F.values,
#                      dim2, dim1, del2, del1, BC2, BC1, delD2Sqr,
#                      ratioQtr, ratioSqr, optArg, undeftmp, flags,
#                      mxLoop=loop_per_frame)
        
#         if printInfo:
#             if flags[0]:
#                 print('loops {0:4.0f} and tolerance is {1:e} (overflows!)'
#                       .format(flags[2], flags[1]))
#             else:
#                 print('loops {0:4.0f} and tolerance is {1:e}'
#                       .format(flags[2], flags[1]))
        
#         lst.append(snapshot.copy())
        
#         if flags[2] < loop_per_frame or loop >= max_loop:
#             break
    
#     re = xr.concat(lst, dim='iteration').rename('inverted')
#     re['iteration'] = xr.DataArray(np.arange(loop_per_frame,
#                                              loop_per_frame*(max_loop+1),
#                                              loop_per_frame),
#                                    dims=['iteration'])
    
#     return re



"""
Below are the helper methods of these applications
"""
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

def __mask_vars(forcing, out):
    # properly mask forcing with _undeftmp
    if np.isnan(undef):
        forcing = force.fillna(_undeftmp)
    else:
        forcing = force.where(force!=undef, other=_undeftmp)
    
    zero = forcing - forcing
    
    if out is None:
        S = zero.copy().load() # loaded because dask cannot be modified
    else:
        S = xr.where(forcing==_undeftmp, out, 0) # applied boundary
