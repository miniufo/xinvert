# -*- coding: utf-8 -*-
"""
Core module of xinvert: SOR iteration solvers for elliptic PDEs.

Implements the low-level ``inv_standard3D`` / ``inv_general3D`` /
``inv_general2D_bih`` solvers that dispatch to the numba-jitted kernels in
:mod:`xinvert.cpus`, plus iteration-loop, convergence-check, and
animation helpers used by the high-level wrappers in :mod:`xinvert.apps`.
"""
import numpy as np
import xarray as xr
import sys
import threading
from .cpus import invert_standard_3D, invert_standard_2D, invert_standard_1D,\
                    invert_general_3D, invert_general_2D, \
                    invert_general_bih_2D, invert_standard_2D_test
from .utils import loop_noncore

# Diagnostic printInfo output: live, thread-safe, distribution-safe.
#
# With dask='parallelized' the per-time-step inversions run concurrently
# in dask worker threads (the numba kernels release the GIL via
# nogil=True), and each call prints its line as soon as it finishes.
# Two pitfalls are handled in :func:`_print_live`:
#
# 1. Line splicing: ipykernel's OutStream buffers writes and flushes on
#    newline, so a thread split across two ``write`` calls (as ``print``
#    does: msg, then '\n') can be spliced by a concurrent thread into a
#    corrupted line.  We therefore emit each line as ONE atomic write.
#
# 2. Output misattribution: ipykernel binds each NEW thread to the parent
#    header (i.e. the cell) that spawned it.  dask reuses its worker pool
#    across cells, so on a re-run those threads still carry the previous
#    cell's parent header and their output is attributed to a finished
#    cell -- the current cell shows nothing.  Before writing we drop this
#    thread's stale registration so the header falls back to ipykernel's
#    global, which is always the *currently executing* cell.
#
# dask.distributed serialisation: the ``_kernel_`` closure only ever
# references this module-level *function* (transferred by reference by
# cloudpickle), never a lock or buffer, so the graph stays picklable and
# worker processes print to their own stdout (visible in worker logs).
def _print_live(msg):
    """Print one diagnostic line atomically and immediately.

    Safe to call from any thread; only touches ipykernel internals when
    they exist (no-op under a plain Python interpreter).
    """
    out = sys.stdout
    ident = threading.get_ident()
    for attr in ('_thread_to_parent', '_thread_to_parent_header'):
        reg = getattr(out, attr, None)
        if reg is not None:
            try:
                reg.pop(ident, None)
            except Exception:
                pass
    out.write(msg + '\n')   # single write => atomic line
    out.flush()

# ---------------------------------------------------------------------------
# GPU kernel registry: maps CPU numba kernel → GPU equivalent
# Populated lazily; if numba.cuda is unavailable the dict stays empty and
# only the CPU path is functional.
# ---------------------------------------------------------------------------
_gpu_kernel_map = {}
try:
    from .gpus import invert_standard_2D_gpu
    _gpu_kernel_map[invert_standard_2D] = invert_standard_2D_gpu
except Exception:
    pass  # CUDA not available or GPU module not yet implemented


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


def _make_kernel(kernel_func, grid_args, iParams):
    """Create a kernel function for xr.apply_ufunc that wraps an SOR function.

    Dispatches to a CPU (numba) or GPU (cuda) kernel based on
    ``iParams['architect']`` (default ``'cpu'``).

    Both CPU and GPU kernels share the same call signature::

        func(o, *coeffs, info, *grid_args,
             optArg, _undeftmp, flags, mxLoop, tolerance)

    Parameters
    ----------
    kernel_func : callable
        The numba-compiled CPU inversion function (e.g. ``invert_standard_2D``).
        When ``architect == 'gpu'`` this is used as a lookup key to find the
        GPU equivalent in :data:`_gpu_kernel_map`.
    grid_args : list
        Pre-extracted grid/BC parameters from iParams, passed between
        the info array and optArg in the kernel call.
    iParams : dict
        Inversion parameters (must contain ``'architect'``).

    Returns
    -------
    callable
        A kernel function suitable for ``xr.apply_ufunc``.
    """
    architect = iParams.get('architect', 'cpu')

    if architect == 'cpu':
        func = kernel_func
    elif architect == 'gpu':
        func = _gpu_kernel_map.get(kernel_func)
        if func is None:
            raise NotImplementedError(
                f"GPU kernel not implemented for '{kernel_func.__name__}', "
                f"available: {[k.__name__ for k in _gpu_kernel_map]}")
        # Inject per-call GPU block config from iParams (None = auto/env default)
        from functools import partial
        func = partial(func,
                       block_2d=iParams.get('gpu_block2d'))
    else:
        raise ValueError(
            f"unsupported architect '{architect}', should be 'cpu' or 'gpu'")

    def _kernel_(s, *args):
        *coeffs, info = args
        s.setflags(write=1)
        o = s
        flags = np.array([0, 0, 0], dtype='float64')

        func(o, *coeffs, info, *grid_args,
             iParams['optArg'], _undeftmp, flags,
             iParams['mxLoop'], iParams['tolerance'])

        if iParams['printInfo']:
            msg = f'{info} loops {flags[2]:4.0f}, tolerance is {flags[1]:e}'
            if flags[0]:
                msg = msg + ' (overflow!)'
            # module-level function => transferred by reference by
            # cloudpickle, keeping the closure picklable
            _print_live(msg)

        return o

    return _kernel_

