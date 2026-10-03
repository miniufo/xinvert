"""
Core module of xinvert: SOR iteration solvers for elliptic PDEs.

Implements the low-level ``inv_standard3D`` / ``inv_general3D`` /
``inv_general2D_bih`` solvers that dispatch to the numba-jitted kernels in
:mod:`xinvert.cpus`, plus iteration-loop, convergence-check, and
animation helpers used by the high-level wrappers in :mod:`xinvert.apps`.
"""
import numbers
import sys
import threading
import warnings

import numpy as np
import xarray as xr

from .cpus import (
    invert_general_2D,
    invert_general_3D,
    invert_general_bih_2D,
    invert_standard_1D,
    invert_standard_2D,
    invert_standard_2D_full,
    invert_standard_3D,
)
from .utils import loop_noncore


def _validate_inversion_dims(F, dims, expected):
    """Return a validated dimension list for a DataArray inversion."""
    if not isinstance(F, xr.DataArray):
        raise ValueError(
            f'forcing must be an xarray.DataArray, got {type(F).__name__}')
    if isinstance(dims, str):
        if expected == 1:
            dims = [dims]
        else:
            raise ValueError(
                f'dims must contain {expected} dimension names, not one string')
    elif isinstance(dims, (list, tuple)):
        dims = list(dims)
    else:
        raise ValueError(
            f'dims must be a list or tuple of {expected} strings, '
            f'got {type(dims).__name__}')

    if len(dims) != expected:
        raise ValueError(
            f'{expected} dimensions are needed for inversion, got {len(dims)}')
    if not all(isinstance(dim, str) for dim in dims):
        raise ValueError('every entry in dims must be a string')
    if len(set(dims)) != len(dims):
        raise ValueError(f'dims contains duplicate dimensions: {dims}')

    missing = [dim for dim in dims if dim not in F.dims]
    if missing:
        raise ValueError(
            f'inversion dimensions {missing} are not present in forcing; '
            f'available dimensions are {list(F.dims)}')
    return dims


def _validate_solver_controls(iParams):
    """Validate inexpensive scalar controls before dispatching a solver."""
    mx_loop = iParams.get('mxLoop')
    if (isinstance(mx_loop, (bool, np.bool_)) or
            not isinstance(mx_loop, numbers.Integral) or mx_loop <= 0):
        raise ValueError(
            f"iParams['mxLoop'] must be a positive integer, got {mx_loop!r}")

    tolerance = iParams.get('tolerance')
    if (isinstance(tolerance, (bool, np.bool_)) or
            not isinstance(tolerance, numbers.Real) or
            not np.isfinite(tolerance)):
        raise ValueError(
            f"iParams['tolerance'] must be a finite real number, "
            f"got {tolerance!r}")

    undef = iParams.get('undef')
    if (isinstance(undef, (bool, np.bool_)) or
            not isinstance(undef, numbers.Real)):
        raise ValueError(
            f"iParams['undef'] must be a real scalar (NaN is allowed), "
            f"got {undef!r}")


def _normalize_scalar_bcs(bcs, dim_len, *, parameter='BCs', valid=None):
    """Return normalized BCs for APIs using one scalar BC per dimension."""
    if not isinstance(bcs, (list, tuple)):
        raise ValueError(
            f"{parameter} must be a list or tuple containing one string "
            "per inversion dimension; endpoint dictionaries are supported "
            "by FiniteDiff only"
        )
    if len(bcs) != dim_len:
        raise ValueError(
            f"{parameter} has {len(bcs)} entries, but this operation "
            f"uses {dim_len} dimensions"
        )

    normalized = []
    for axis, bc in enumerate(bcs):
        if not isinstance(bc, str):
            choices = f' in {valid}' if valid is not None else ''
            raise ValueError(
                f"{parameter}[{axis}] must be a string{choices}, "
                f"got {type(bc).__name__}; endpoint pairs or dictionaries "
                "are supported by FiniteDiff only"
            )
        value = bc.strip().lower()
        if valid is not None and value not in valid:
            raise ValueError(
                f"{parameter}[{axis}]={bc!r} is invalid; "
                f"expected one of {valid}"
            )
        normalized.append(value)
    return normalized


def _validate_inversion_bcs(iParams, dim_len):
    """Validate scalar-per-dimension BCs and warn for pending kernels."""
    bcs = iParams.get('BCs')
    marker = iParams.get('_validated_BCs')
    if (isinstance(bcs, (list, tuple)) and
            marker == (dim_len, tuple(bcs))):
        return

    valid = ('fixed', 'extend', 'periodic')
    normalized = _normalize_scalar_bcs(
        bcs, dim_len, parameter="iParams['BCs']", valid=valid)

    iParams['BCs'] = normalized
    iParams['_validated_BCs'] = (dim_len, tuple(normalized))

    if dim_len == 3 and normalized[0] != 'fixed':
        warnings.warn(
            f"BCs[0]={normalized[0]!r} for the z dimension is accepted but "
            "not implemented by the 3-D SOR kernels; the requested z-boundary "
            "update will not be applied",
            RuntimeWarning,
            stacklevel=3,
        )

    y_index = dim_len - 2
    if dim_len >= 2 and normalized[y_index] == 'periodic':
        warnings.warn(
            f"BCs[{y_index}]='periodic' for the y dimension is accepted but "
            "not implemented by the SOR kernels; only periodic x (the last "
            "dimension) is currently implemented",
            RuntimeWarning,
            stacklevel=3,
        )

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
_ensure_cuda = None    # gpus.ensure_context, if the GPU module is available
_gpu_import_error = None
try:
    from .gpus import ensure_context as _ensure_cuda
    from .gpus import (
        invert_general_2D_gpu,
        invert_general_3D_gpu,
        invert_general_bih_2D_gpu,
        invert_standard_1D_gpu,
        invert_standard_2D_full_gpu,
        invert_standard_2D_gpu,
        invert_standard_3D_gpu,
    )
    _gpu_kernel_map[invert_standard_2D] = invert_standard_2D_gpu
    _gpu_kernel_map[invert_standard_2D_full] = invert_standard_2D_full_gpu
    _gpu_kernel_map[invert_standard_1D] = invert_standard_1D_gpu
    _gpu_kernel_map[invert_general_2D] = invert_general_2D_gpu
    _gpu_kernel_map[invert_standard_3D] = invert_standard_3D_gpu
    _gpu_kernel_map[invert_general_3D] = invert_general_3D_gpu
    _gpu_kernel_map[invert_general_bih_2D] = invert_general_bih_2D_gpu
except ImportError as exc:
    # Keep CPU-only installations importable, but retain the real cause so a
    # user who explicitly requests the GPU backend receives an actionable
    # error.  Other exceptions are deliberately not swallowed: they indicate
    # a bug in the GPU module rather than a missing optional dependency.
    _gpu_import_error = exc


# default undefined value
_undeftmp = -9.99e8

"""
Below are the core methods of xinvert
"""
def inv_standard3D(A, B, C, F, S, dims, iParams):
    r"""Inverting a 3D volume of elliptic equation in a standard form.

    .. math::

        \frac{\partial}{\partial z}\left(A\frac{\partial \omega}{\partial z}\right)+
        \frac{\partial}{\partial y}\left(B\frac{\partial \omega}{\partial y}\right)+
        \frac{\partial}{\partial x}\left(C\frac{\partial \omega}{\partial x}\right)=F
    
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
    dims = _validate_inversion_dims(F, dims, 3)
    _validate_inversion_bcs(iParams, 3)
    _validate_solver_controls(iParams)

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc3'], iParams['gc2'], iParams['gc1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['BCs'][2],
        iParams['del1Sqr'], iParams['ratio2Sqr'], iParams['ratio1Sqr'],
    ]
    _kernel_ = _make_kernel(invert_standard_3D, grid_args, iParams)
    
    re = _apply_solver(
        _kernel_, [S, A, B, C, F, info],
        [dims, dims, dims, dims, dims, []], dims, S, iParams)
    
    return re


def inv_standard2D(A, B, C, F, S, dims, iParams):
    r"""Inverting equations in 2D standard form.

    .. math::

        \frac{\partial}{\partial y}\left(
        A\frac{\partial \psi}{\partial y} + 
        B\frac{\partial \psi}{\partial x} \right) +
        \frac{\partial}{\partial x}\left(
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
    dims = _validate_inversion_dims(F, dims, 2)
    _validate_inversion_bcs(iParams, 2)
    _validate_solver_controls(iParams)

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc2'], iParams['gc1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['del1Sqr'],
        iParams['ratioQtr'], iParams['ratioSqr'],
    ]
    _kernel_ = _make_kernel(invert_standard_2D, grid_args, iParams)
    
    re = _apply_solver(
        _kernel_, [S, A, B, C, F, info],
        [dims, dims, dims, dims, dims, []], dims, S, iParams)
    
    return re



def inv_standard2D_full(A, B, C, D, E, F, S, dims, iParams):
    r"""Inverting equations in 2D standard form (test only).

    .. math::

        \frac{\partial}{\partial y}\left(
        A\frac{\partial \psi}{\partial y} + 
        B\frac{\partial \psi}{\partial x} \right) +
        \frac{\partial}{\partial x}\left(
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
    dims = _validate_inversion_dims(F, dims, 2)
    _validate_inversion_bcs(iParams, 2)
    _validate_solver_controls(iParams)

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc2'], iParams['gc1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['del1Sqr'],
        iParams['ratioQtr'], iParams['ratioSqr'],
    ]
    _kernel_ = _make_kernel(invert_standard_2D_full, grid_args, iParams)
    
    re = _apply_solver(
        _kernel_, [S, A, B, C, D, E, F, info],
        [dims, dims, dims, dims, dims, dims, dims, []], dims, S, iParams)
    
    return re


def inv_standard1D(A, B, F, S, dims, iParams):
    r"""Inverting equations in 1D standard form.

    .. math::

        \frac{\partial}{\partial x}\left(
        A\frac{\partial \psi}{\partial x} + B\psi\right)= F
    
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
    dims = _validate_inversion_dims(F, dims, 1)
    _validate_inversion_bcs(iParams, 1)
    _validate_solver_controls(iParams)

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc1'], iParams['BCs'][0], iParams['del1Sqr'],
    ]
    _kernel_ = _make_kernel(invert_standard_1D, grid_args, iParams)
    
    re = _apply_solver(
        _kernel_, [S, A, B, F, info],
        [dims, dims, dims, dims, []], dims, S, iParams)
    
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
    dims = _validate_inversion_dims(H, dims, 3)
    _validate_inversion_bcs(iParams, 3)
    _validate_solver_controls(iParams)

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc3'], iParams['gc2'], iParams['gc1'], iParams['del1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['BCs'][2],
        iParams['del1Sqr'], iParams['ratio2'], iParams['ratio1'],
        iParams['ratio2Sqr'], iParams['ratio1Sqr'],
    ]
    _kernel_ = _make_kernel(invert_general_3D, grid_args, iParams)
    
    re = _apply_solver(
        _kernel_, [S, A, B, C, D, E, F, G, H, info],
        [dims, dims, dims, dims, dims, dims, dims, dims, dims, []],
        dims, S, iParams)
    
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
    dims = _validate_inversion_dims(G, dims, 2)
    _validate_inversion_bcs(iParams, 2)
    _validate_solver_controls(iParams)

    # get info for print and non-core dimensions
    info, ncdims = _get_info(F, dims)
    
    grid_args = [
        iParams['gc2'], iParams['gc1'], iParams['del1'],
        iParams['BCs'][0], iParams['BCs'][1], iParams['del1Sqr'],
        iParams['ratio'], iParams['ratioQtr'], iParams['ratioSqr'],
    ]
    _kernel_ = _make_kernel(invert_general_2D, grid_args, iParams)
    
    re = _apply_solver(
        _kernel_, [S, A, B, C, D, E, F, G, info],
        [dims, dims, dims, dims, dims, dims, dims, dims, []],
        dims, S, iParams)
    
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
    dims = _validate_inversion_dims(J, dims, 2)
    _validate_inversion_bcs(iParams, 2)
    _validate_solver_controls(iParams)

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
    
    re = _apply_solver(
        _kernel_, [S, A, B, C, D, E, F, G, H, I, J, info],
        [dims, dims, dims, dims, dims, dims,
         dims, dims, dims, dims, dims, []], dims, S, iParams)
    
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
    # case-insensitive (and whitespace-tolerant): 'GPU', ' GPU ' etc. all work
    architect = str(iParams.get('architect', 'cpu')).strip().lower()
    convergence = _convergence_code(iParams.get('convergence', 'norm'))

    if architect == 'cpu':
        func = kernel_func
    elif architect == 'gpu':
        warnings.warn(
            "xinvert's GPU backend is experimental; numerical kernels, "
            "configuration options, and performance characteristics may "
            "change before it is declared stable",
            UserWarning,
            stacklevel=3,
        )
        func = _gpu_kernel_map.get(kernel_func)
        if func is None:
            if _gpu_import_error is not None:
                raise ImportError(
                    "GPU backend is unavailable. Install the optional GPU "
                    "dependencies with 'pip install xinvert[gpu]' and "
                    "ensure that a compatible NVIDIA driver and CUDA "
                    "runtime are available."
                ) from _gpu_import_error
            raise NotImplementedError(
                f"GPU kernel not implemented for '{kernel_func.__name__}', "
                f"available: {[k.__name__ for k in _gpu_kernel_map]}")
        # Inject per-call GPU block config from iParams (None = auto default)
        from functools import partial
        func = partial(func,
                       block_2d=iParams.get('gpu_block2d'))
        # Initialise the CUDA context in the main thread: numba-cuda manages
        # contexts thread-locally, and dask worker threads would otherwise
        # crash with CUDA_ERROR_NOT_INITIALIZED on their first launch.
        if _ensure_cuda is not None:
            _ensure_cuda()
    else:
        raise ValueError(
            f"unsupported architect '{architect}', should be 'cpu' or 'gpu' "
            f"(case-insensitive)")

    def _kernel_(s, *args):
        *coeffs, info = args
        s.setflags(write=1)
        o = s
        flags = np.array([0, 0, 0], dtype='float64')

        func(o, *coeffs, info, *grid_args,
             iParams['optArg'], _undeftmp, flags,
             iParams['mxLoop'], iParams['tolerance'], convergence)

        if iParams['printInfo']:
            metric = 'residual' if convergence == 1 else 'norm change'
            msg = (f'{info} loops {flags[2]:4.0f}, {metric} is '
                   f'{flags[1]:e}')
            if flags[0]:
                msg = msg + ' (overflow!)'
            # module-level function => transferred by reference by
            # cloudpickle, keeping the closure picklable
            _print_live(msg)

        if iParams.get('return_diagnostics', False):
            return o, flags
        return o

    return _kernel_


def _apply_solver(kernel, inputs, input_core_dims, solution_dims, solution,
                  iParams):
    """Apply one vectorized solver and optionally expose per-slice status."""
    if not iParams.get('return_diagnostics', False):
        return xr.apply_ufunc(
            kernel, *inputs,
            input_core_dims=input_core_dims,
            output_core_dims=[solution_dims],
            dask='parallelized',
            dask_gufunc_kwargs={'allow_rechunk': True},
            vectorize=True,
            output_dtypes=[solution.dtype],
        )

    result, raw = xr.apply_ufunc(
        kernel, *inputs,
        input_core_dims=input_core_dims,
        output_core_dims=[solution_dims, ['diagnostic']],
        dask='parallelized',
        dask_gufunc_kwargs={
            'allow_rechunk': True,
            'output_sizes': {'diagnostic': 3},
        },
        vectorize=True,
        output_dtypes=[solution.dtype, np.float64],
    )
    return result, _diagnostics_from_flags(raw, iParams)


def _diagnostics_from_flags(raw, iParams):
    """Convert the kernel flag vector into a labelled diagnostics Dataset."""
    overflow = raw.isel(diagnostic=0, drop=True).astype(bool)
    error = raw.isel(diagnostic=1, drop=True)
    iterations = raw.isel(diagnostic=2, drop=True).astype(np.int64)
    tolerance = float(iParams['tolerance'])
    converged = (~overflow) & np.isfinite(error) & (tolerance > 0.0) & (
        error < tolerance)

    reason = xr.full_like(iterations, 'max_iterations', dtype=object)
    reason = xr.where(converged, 'converged', reason)
    reason = xr.where(overflow, 'overflow', reason)

    diagnostics = xr.Dataset({
        'converged': converged,
        'iterations': iterations,
        'error': error,
        'stop_reason': reason,
    })
    diagnostics.attrs.update({
        'convergence': ('residual' if _convergence_code(
            iParams.get('convergence', 'norm')) == 1 else 'norm'),
        'tolerance': tolerance,
        'max_iterations': int(iParams['mxLoop']),
    })
    return diagnostics


def _convergence_code(value):
    """Validate a public convergence-mode value and return its kernel code."""
    mode = str(value).strip().lower().replace('-', '_')
    aliases = {
        'norm': 0,
        'solution_norm': 0,
        'legacy': 0,
        'residual': 1,
        'preconditioned_residual': 1,
    }
    if mode not in aliases:
        raise ValueError(
            f"unsupported convergence mode '{value}', should be 'norm' or "
            "'residual' (case-insensitive)"
        )
    return aliases[mode]
