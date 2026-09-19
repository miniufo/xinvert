# -*- coding: utf-8 -*-
"""
Utility module of xinvert: data preprocessing helpers.

Provides ``loop_noncore`` for iterating over non-core dimensions,
``smooth9`` for 9-point smoothing, and ``coarsen`` for block-averaging
xarray DataArrays with flexible boundary handling.
"""
import numpy as np
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
    -------
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


def pad_periodic(data, pad_widths):
    """Circularly (wrap) pad the given periodic dimensions.

    Data values are wrapped around (``np.pad`` style ``mode='wrap'``),
    while coordinate labels are extended *linearly* so the index stays
    monotonic (``xarray``'s own ``pad`` wraps the coords too, which would
    break downstream ``.interp``).

    Parameters
    ----------
    data: xarray.DataArray
        A given gridded data.
    pad_widths: dict {dim: int}
        Number of cells to pad on **each** side of each dimension,
        e.g. ``{'lon': 2}``.

    Returns
    -------
    re: xarray.DataArray
        Padded data (a shallow copy; original is untouched).
    """
    if not pad_widths:
        return data

    new = data.pad(pad_width={d: (n, n) for d, n in pad_widths.items()},
                   mode='wrap')

    # rebuild padded coordinates by linear extrapolation (monotonic)
    for d, n in pad_widths.items():
        c = np.asarray(data[d].values)
        if c.size < 2:
            continue
        step = c[1] - c[0]
        extended = np.concatenate([c[0] - step * np.arange(n, 0, -1),
                                   c,
                                   c[-1] + step * np.arange(1, n + 1)])
        new = new.assign_coords({d: (d, extended)})

    return new


def smooth(data, dims=None, window=3):
    """Boxcar-smooth along the given dimensions (centered rolling mean).

    For periodic dimensions, call :func:`pad_periodic` beforehand (as
    :func:`coarsen` does) so the mean wraps around; the padded rows are
    never sampled by :func:`coarsen`'s interpolation, which only queries
    the original coordinate range.

    Parameters
    ----------
    data: xarray.DataArray
        A given gridded data.
    dims: list of str or None
        Dimensions to smooth along; None means all dimensions.
    window: int
        Width of the smoothing window (odd values give a symmetric stencil).

    Returns
    -------
    re: xarray.DataArray
        Smoothed data.
    """
    if dims is None:
        dims = list(data.dims)

    re = data
    for d in dims:
        re = re.rolling({d: int(window)}, center=True, min_periods=1).mean()

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
        Periodic dimension(s) e.g., ['lon']; circularly padded before
        smoothing.
    ratio: int
        Ratio of grid points before and after the coarsening.
        
    Returns
    -------
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
