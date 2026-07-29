# -*- coding: utf-8 -*-
"""
Created on 2020.12.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
import sys
from .numbas import invert_standard_3D, invert_standard_2D, invert_standard_1D,\
                    invert_general_3D, invert_general_2D, \
                    invert_general_bih_2D, invert_standard_2D_test
from .utils import loop_noncore


# default undefined value
_undeftmp = -9.99e8

"""
Below are the core methods of xinvert
"""
def inv_standard3D(A, B, C, F, S, dims, iParams):
    r"""Inverting a 3D volume of elliptic equation in a standard form.

    .. math::

        \frac{1}{\partial z}\left(A\frac{\partial \omega}{\partial z}\right)+
        \frac{1}{\partial y}\left(B\frac{\partial \omega}{\partial y}\right)+
        \frac{1}{\partial x}\left(C\frac{\partial \omega}{\partial x}\right)=F
    
    Invert this equation using SOR iteration. If F = F['time', 'lev', 'lat',
    'lon'] and we invert for the 3D spatial distribution, then 3rd dim is 'lev',
    2nd dim is 'lat' and 1st dim is 'lon'.

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
    iParams: dict
        Parameters for inversion.

    Returns
    -------
    xarray.DataArray
        Solution :math:`\psi`.
    """
    if len(dims) != 3:
        raise Exception('3 dimensions are needed for inversion')

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc3'], iParams['gc2'], iParams['gc1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['BCs'][2],
        iParams['del1Sqr'], iParams['ratio2Sqr'], iParams['ratio1Sqr'],
    ]
    _kernel_ = _make_kernel(invert_standard_3D, grid_args, iParams)
    
    re = xr.apply_ufunc(
        _kernel_, S, A, B, C, F, info,
        input_core_dims=[dims, dims, dims, dims, dims, []],
        output_core_dims=[dims],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
        vectorize=True,
        output_dtypes=[S.dtype],
    )
    
    return re


def inv_standard2D(A, B, C, F, S, dims, iParams):
    r"""Inverting equations in 2D standard form.

    .. math::

        \frac{1}{\partial y}\left(
        A\frac{\partial \psi}{\partial y} + 
        B\frac{\partial \psi}{\partial x} \right) +
        \frac{1}{\partial x}\left(
        B\frac{\partial \psi}{\partial y} +
        C\frac{\partial \psi}{\partial x} \right) = F
    
    Invert this equation using SOR iteration. If F = F['time', 'lat', 'lon'] then
    for the horizontal slice, the 2nd dim is 'lat' and 1st dim is 'lon'.

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
    iParams: dict
        Parameters for inversion.

    Returns
    -------
    xarray.DataArray
        Solution :math:`\psi`.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc2'], iParams['gc1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['del1Sqr'],
        iParams['ratioQtr'], iParams['ratioSqr'],
    ]
    _kernel_ = _make_kernel(invert_standard_2D, grid_args, iParams)
    
    re = xr.apply_ufunc(
        _kernel_, S, A, B, C, F, info,
        input_core_dims=[dims, dims, dims, dims, dims, []],
        output_core_dims=[dims],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
        vectorize=True,
        output_dtypes=[S.dtype],
    )
    
    return re



def inv_standard2D_test(A, B, C, D, E, F, S, dims, iParams):
    r"""Inverting equations in 2D standard form (test only).

    .. math::

        \frac{1}{\partial y}\left(
        A\frac{\partial \psi}{\partial y} + 
        B\frac{\partial \psi}{\partial x} \right) +
        \frac{1}{\partial x}\left(
        B\frac{\partial \psi}{\partial y} +
        C\frac{\partial \psi}{\partial x} \right) + E\psi= F
    
    Invert this equation using SOR iteration. If F = F['time', 'lat', 'lon'], then
    for the horizontal slice, the 2nd dim is 'lat' and 1st dim is 'lon'.

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
    iParams: dict
        Parameters for inversion.

    Returns
    -------
    xarray.DataArray
        Solution :math:`\psi`.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc2'], iParams['gc1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['del1Sqr'],
        iParams['ratioQtr'], iParams['ratioSqr'],
    ]
    _kernel_ = _make_kernel(invert_standard_2D_test, grid_args, iParams)
    
    re = xr.apply_ufunc(
        _kernel_, S, A, B, C, D, E, F, info,
        input_core_dims=[dims, dims, dims, dims, dims, dims, dims, []],
        output_core_dims=[dims],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
        vectorize=True,
        output_dtypes=[S.dtype],
    )
    
    return re


def inv_standard1D(A, B, F, S, dims, iParams):
    r"""Inverting equations in 1D standard form.

    .. math::

        \frac{1}{\partial x}\left(
        A\frac{\partial \psi}{\partial x} + B\psi= F
    
    Invert this equation using SOR iteration. If F = F['time', 'lat'], then
    for the meridional series, the 1st dim is 'lat' .

    Parameters
    ----------
    A: xr.DataArray
        Coefficient A.
    B: xr.DataArray
        Coefficient B.
    F: xr.DataArray
        Forcing function F.
    S: xr.DataArray
        Initial guess of the solution (also the output).
    dims: list or str
        Dimension combination for the inversion e.g., ['lat'].
    iParams: dict
        Parameters for inversion.

    Returns
    -------
    xarray.DataArray
        Solution :math:`\psi`.
    """
    if len(dims) != 1:
        raise Exception('1 dimensions are needed for inversion')

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc1'], iParams['BCs'][0], iParams['del1Sqr'],
    ]
    _kernel_ = _make_kernel(invert_standard_1D, grid_args, iParams)
    
    re = xr.apply_ufunc(
        _kernel_, S, A, B, F, info,
        input_core_dims=[dims, dims, dims, dims, []],
        output_core_dims=[dims],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
        vectorize=True,
        output_dtypes=[S.dtype],
    )
    
    return re


def inv_general3D(A, B, C, D, E, F, G, H, S, dims, iParams):
    r"""Inverting a 3D volume of elliptic equation in the general form.

    .. math::

        A \frac{\partial^2 \psi}{\partial z^2} +
        B \frac{\partial^2 \psi}{\partial y^2} +
        C \frac{\partial^2 \psi}{\partial x^2} +
        D \frac{\partial \psi}{\partial z} +
        E \frac{\partial \psi}{\partial y} +
        F \frac{\partial \psi}{\partial x} + G \psi = H
    
    Invert this equation using SOR iteration. If F = F['time', 'lev', 'lat',
    'lon'], then for the 3D volume, the 3rd dim is 'lev' and 1st dim is 'lon'.

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
    iParams: dict
        Parameters for inversion.

    Returns
    -------
    xarray.DataArray
        Solution :math:`\psi`.
    """
    if len(dims) != 3:
        raise Exception('3 dimensions are needed for inversion')

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc3'], iParams['gc2'], iParams['gc1'], iParams['del1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['BCs'][2],
        iParams['del1Sqr'], iParams['ratio2'], iParams['ratio1'],
        iParams['ratio2Sqr'], iParams['ratio1Sqr'],
    ]
    _kernel_ = _make_kernel(invert_general_3D, grid_args, iParams)
    
    re = xr.apply_ufunc(
        _kernel_, S, A, B, C, D, E, F, G, H, info,
        input_core_dims=[dims, dims, dims, dims, dims,
                         dims, dims, dims, dims, []],
        output_core_dims=[dims],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
        vectorize=True,
        output_dtypes=[S.dtype],
    )
    
    return re


def inv_general2D(A, B, C, D, E, F, G, S, dims, iParams):
    r"""Inverting a 2D slice of elliptic equation in general form.

    .. math::

        A \frac{\partial^2 \psi}{\partial y^2} +
        B \frac{\partial^2 \psi}{\partial y \partial x} +
        C \frac{\partial^2 \psi}{\partial x^2} +
        D \frac{\partial \psi}{\partial y} +
        E \frac{\partial \psi}{\partial x} + F \psi = G
    
    Invert this equation using SOR iteration. If F = F['time', 'lat', 'lon'], then
    for the horizontal slice, the 2nd dim is 'lat' and 1st dim is 'lon'.

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
    iParams: dict
        Parameters for inversion.

    Returns
    -------
    xarray.DataArray
        Solution :math:`\psi`.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc2'], iParams['gc1'], iParams['del1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['del1Sqr'],
        iParams['ratio'], iParams['ratioQtr'], iParams['ratioSqr'],
    ]
    _kernel_ = _make_kernel(invert_general_2D, grid_args, iParams)
    
    re = xr.apply_ufunc(
        _kernel_, S, A, B, C, D, E, F, G, info,
        input_core_dims=[dims, dims, dims, dims, dims,
                         dims, dims, dims, []],
        output_core_dims=[dims],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
        vectorize=True,
        output_dtypes=[S.dtype],
    )
    
    return re


def inv_general2D_bih(A, B, C, D, E, F, G, H, I, J, S, dims, iParams):
    r"""Inverting a 2D slice of elliptic equation in the general form.

    .. math::

        A \frac{\partial^4 \psi}{\partial y^4} +
        B \frac{\partial^4 \psi}{\partial y^2 \partial x^2} +
        C \frac{\partial^4 \psi}{\partial x^4} +
        D \frac{\partial^2 \psi}{\partial y^2} +
        E \frac{\partial^2 \psi}{\partial y \partial x} +
        F \frac{\partial^2 \psi}{\partial x^2} +
        G \frac{\partial \psi}{\partial y} +
        H \frac{\partial \psi}{\partial x} + I \psi = J
    
    Invert this equation using SOR iteration. If F = F['time', 'lat', 'lon'], then
    for the horizontal slice, the 2nd dim is 'lat' and 1st dim is 'lon'.

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
    iParams: dict
        Parameters for inversion.

    Returns
    -------
    xarray.DataArray
        Solution :math:`\psi`.
    """
    if len(dims) != 2:
        raise Exception('2 dimensions are needed for inversion')

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc2'], iParams['gc1'],
        iParams['BCs'][0], iParams['BCs'][1],
        iParams['del1SSr'], iParams['del1Tr'], iParams['del1Sqr'],
        iParams['ratio'], iParams['ratioSSr'],
        iParams['ratioQtr'], iParams['ratioSqr'],
    ]
    _kernel_ = _make_kernel(invert_general_bih_2D, grid_args, iParams)
    
    re = xr.apply_ufunc(
        _kernel_, S, A, B, C, D, E, F, G, H, I, J, info,
        input_core_dims=[dims, dims, dims, dims, dims, dims,
                         dims, dims, dims, dims, dims, []],
        output_core_dims=[dims],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True},
        vectorize=True,
        output_dtypes=[S.dtype],
    )
    
    return re


"""
Below are the helper methods of xinvert
"""
def _get_info(F, dims):
    info = []
    for selDict in loop_noncore(F, dims):
        parts = []
        for k, v in selDict.items():
            if isinstance(v, (np.datetime64, np.timedelta64)):
                s = str(v).split('.')[0] if '.' in str(v) else str(v)
            elif isinstance(v, (np.floating, np.integer)):
                s = str(v.item())
            else:
                s = str(v)
            parts.append(f'{k}: {s}')
        info.append('{' + ', '.join(parts) + '}' if parts else '{}')
    ncdims = list(selDict.keys()) # non-core dimensions
    
    if ncdims == []:
        info = '{}'
    else:
        info = xr.DataArray(np.array(info), dims=ncdims, coords={dim: F[dim] for dim in ncdims})
    
    return info, ncdims


def _make_kernel(numba_func, grid_args, iParams):
    """Create a kernel function for xr.apply_ufunc that wraps a numba SOR function.

    The numba function is called as:
        numba_func(o, *coeffs, info, *grid_args,
                   optArg, _undeftmp, flags, mxLoop, tolerance)

    Parameters
    ----------
    numba_func : callable
        The numba-compiled inversion function.
    grid_args : list
        Pre-extracted grid/BC parameters from iParams, passed between
        the info array and optArg in the numba function call.
    iParams : dict
        Inversion parameters.

    Returns
    -------
    callable
        A kernel function suitable for xr.apply_ufunc.
    """
    def _kernel_(s, *args):
        *coeffs, info = args
        s.setflags(write=1)
        o = s
        flags = np.array([0, 0, 0], dtype='float64')

        numba_func(o, *coeffs, info, *grid_args,
                   iParams['optArg'], _undeftmp, flags,
                   iParams['mxLoop'], iParams['tolerance'])

        if iParams['printInfo']:
            msg = f'{info} loops {flags[2]:4.0f}, tolerance is {flags[1]:e}'
            if flags[0]:
                msg = msg + ' (overflow!)'
            print(msg, file=sys.stderr, flush=True)

        return o

    return _kernel_

