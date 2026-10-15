"""
Utility module of xinvert: data preprocessing helpers.

Provides ``loop_noncore`` for iterating over non-core dimensions,
``smooth9`` for 9-point smoothing, and ``coarsen`` for block-averaging
xarray DataArrays with flexible boundary handling.
"""
import warnings

import numba as nb
import numpy as np
import xarray as xr


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
        raise ValueError('two dimensions are needed')

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



def invert_MultiGrid(invert_func, *args, ratio=3, gridNo=3, **kwargs):
    r"""Use an experimental multi-grid driver for inversion.

    .. warning::

        This function is experimental and is not part of xinvert's stable
        public API.  Its parameters and numerical workflow may change, and
        the current implementation has not yet been validated for production
        research results.

    All the `invert_xxx` function can be solved using multi-grid method here.

    Parameters
    ----------
    app_name: {'Poisson', 'PV2D', 'GillMatsuno', 'Eliassen', 'geostrophic', 'StommelMunk', 'RefState', 'Omega'}
        Application name as a suffix of `invert_xxx`.
    F: xarray.DataArray
        A forcing function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: str, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe initial condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Model parameters.  None for the Poisson model.
    iParams: dict, optional
        Iteration parameters.
    ratio: int, optional
        Ratio of multi grids.
    gridNo: int, optional
        Number of multi grids.

    Returns
    -------
    xarray.DataArray
        The result, in which an extra dimension called `iter` will be added.
    """
    warnings.warn(
        'invert_MultiGrid() is experimental and not part of the stable API; '
        'its parameters and numerical workflow may change',
        UserWarning,
        stacklevel=2,
    )


    ratios = [10, 6, 3, 1]

    mxLoop = kwargs['mxLoop'] if 'mxLoop' in kwargs else 5000

    loops  = [mxLoop*ratio/10 for ratio in ratios]

    if 'dims' not in kwargs:
        raise ValueError('kwarg dims= should be provided')

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


def _invert_omega_MG(force, S, dims, BCs=None,
                    coords='latlon', f0=None, beta=None,
                    undef=np.nan, mxLoop=5000, tolerance=1e-6,
                    optArg=None, printInfo=True, debug=False,
                    icbc=None, ratio=4, gridNo=3):

    from .apps import invert_omega

    if BCs is None:
        BCs = ['fixed', 'fixed', 'fixed']

    ratios = [10, 6, 3, 1]
    loops  = [mxLoop*ratio/30 for ratio in ratios]

    fs = [coarsen(force, dims, ratio=ratio) for ratio in ratios]
    Ss = [coarsen(S    , dims, ratio=ratio) for ratio in ratios]

    o_guess = fs[0] - fs[0]
    o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
    os = []

    for i, (ratio, frc, stab, loop) in enumerate(zip(ratios, fs, Ss, loops)):
        # iteration over coarse grid
        o_guess = invert_omega(frc, stab, dims=dims, BCs=BCs, f0=f0,
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
        # o = invert_omegaEquation(force, S, dims=dims, BCs=BCs, coords=coords,
        #                          f0=f0, beta=beta, undef=undef, mxLoop=mxLoop/50,
        #                          tolerance=tolerance, optArg=optArg, debug=debug,
        #                          printInfo=printInfo, icbc=o_guess)

    return o_guess, fs, os


def cal_flow(S, dims, coords='lat-lon', BCs=None,
             vtype='streamfunction', mParams=None):
    r"""Calculate flow vector using streamfunction or velocity potential.

    Parameters
    ----------
    S: xarray.DataArray
        Streamfunction or velocity potential.
    dims: list of str
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'z-lat', 'z-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    BCs: list of str, optional
        One boundary condition per dimension, applied to both endpoints,
        e.g., ['fixed', 'periodic'].  Use ``FiniteDiff`` directly for
        different boundary conditions at the two endpoints.
    vtype: {'streamfunction', 'velocitypotential', 'GillMatsuno'}, optional
        Type of the given variable, which determines the returns.

    Returns
    -------
    tuple
        Flow vector components.
    """
    from .apps import __update, default_mParams
    from .core import _normalize_scalar_bcs, _validate_inversion_dims

    if BCs is None:
        BCs = ['fixed', 'fixed']

    dims = _validate_inversion_dims(S, dims, 2)
    BCs = _normalize_scalar_bcs(
        BCs, 2, parameter='BCs',
        valid=('fixed', 'extend', 'reflect', 'periodic'))

    if not isinstance(vtype, str):
        raise ValueError(f'vtype must be a string, got {type(vtype).__name__}')
    if not isinstance(coords, str):
        raise ValueError(f'coords must be a string, got {type(coords).__name__}')
    vtype = vtype.strip().lower()
    coords = coords.strip().lower()
    if vtype not in ['streamfunction', 'velocitypotential', 'gillmatsuno']:
        raise ValueError('unsupported vtype: ' + vtype + ', should be one of:\n'+
                        "['streamfunction', 'velocitypotential', 'gillmatsuno']")

    if vtype != 'gillmatsuno': # Poisson case
        if vtype == 'streamfunction':
            sf = True
        else:
            sf = False

        from .finitediffs import FiniteDiff

        if coords == 'lat-lon':
            dmap = {'Y': dims[0], 'X': dims[1]}

            fd = FiniteDiff(dmap, {'Y': (BCs[0], BCs[0]),
                                   'X': (BCs[1], BCs[1])}, coords='lat-lon')

            grdy, grdx = fd.grad(S, ['Y', 'X'])

            if sf:
                return -grdy, grdx
            else:
                return grdx, grdy

        elif coords == 'z-lat':
            dmap = {'Z': dims[0], 'Y': dims[1]}

            fd = FiniteDiff(dmap, {'Z': (BCs[0], BCs[0]),
                                   'Y': (BCs[1], BCs[1])}, coords='lat-lon')

            grdz, grdy = fd.grad(S, ['Z', 'Y'])

            cos = np.cos(np.deg2rad(S[dims[1]]))

            grdz, grdy = grdz / cos, grdy / cos

            grdy = grdy.where(np.abs(grdy[dims[1]])!=90, other=0)

            if sf:
                return -grdz, grdy
            else:
                return grdy, grdz

        elif coords == 'z-lon':
            dmap = {'Z': dims[0], 'X': dims[1]}

            fd = FiniteDiff(dmap, {'Z': (BCs[0], BCs[0]),
                                   'X': (BCs[1], BCs[1])}, coords='lat-lon')

            grdz, grdx = fd.grad(S, ['Z', 'X'])

            if sf:
                return grdz, -grdx
            else:
                return grdx, grdz

        elif coords == 'cartesian':
            dmap = {'Y': dims[0], 'X': dims[1]}

            fd = FiniteDiff(dmap, {'Y': (BCs[0], BCs[0]),
                                   'X': (BCs[1], BCs[1])}, coords=coords)

            grdy, grdx = fd.grad(S, ['Y', 'X'])

            if sf:
                return -grdy, grdx
            else:
                return grdx, grdy

        else:
            raise ValueError('unsupported coords ' + coords +
                            ', should be [lat-lon, z-lat, z-lon, cartesian]')

    else: # GillMatsuno case
        mParams = __update(default_mParams, mParams,
                           ['f0', 'beta', 'epsilon', 'Phi', 'Omega', 'Rearth'])

        eps    = mParams['epsilon']
        f0     = mParams['f0']
        beta   = mParams['beta']
        Omega  = mParams['Omega']
        Rearth = mParams['Rearth']

        if coords.lower() == 'lat-lon':
            lats = np.deg2rad(S[dims[0]])
            cosLat = np.cos(lats)
            sinLat = np.sin(lats)

            f = 2.0 * Omega * sinLat
            deg2m = np.deg2rad(1.0) * Rearth

            coef1 = eps / (eps**2.0 + f**2.0)
            coef2 = f   / (eps**2.0 + f**2.0)

            c1 = - coef1 * S.differentiate(dims[1]) / deg2m / cosLat \
                 - coef2 * S.differentiate(dims[0]) / deg2m
            c2 = - coef1 * S.differentiate(dims[0]) / deg2m \
                 + coef2 * S.differentiate(dims[1]) / deg2m / cosLat
        elif coords.lower() == 'cartesian':
            ydef = S[dims[0]]
            f = f0 + beta * ydef

            coef1 = eps / (eps**2.0 + f**2.0)
            coef2 = f   / (eps**2.0 + f**2.0)

            c1 = - coef1 * S.differentiate(dims[1]) \
                 - coef2 * S.differentiate(dims[0])
            c2 = - coef1 * S.differentiate(dims[0]) \
                 + coef2 * S.differentiate(dims[1])
        else:
            raise ValueError('unsupported coords ' + coords +
                            ', should be [lat-lon, cartesian]')

        return c1, c2



"""
Below are the helper methods of these applications
"""
