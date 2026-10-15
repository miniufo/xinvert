"""Streaming iteration frames for CPU, CPU-mc and resident GPU backends."""
import numpy as np
import xarray as xr

from .core import (
    _validate_inversion_bcs,
    _validate_inversion_dims,
    _validate_solver_controls,
)
from .utils import loop_noncore


def animate_iteration(app_name, F, dims, coords='lat-lon', icbc=None,
                      mParams=None, iParams=None,
                      loop_per_frame=5, max_frames=30, *, output, complevels=0):
    r"""Synchronously write iteration frames to a NetCDF file.

    Parameters
    ----------
    app_name : str
        Physical application name, such as Poisson, Stommel or omega.
    F : xarray.DataArray
        One spatial problem. A non-spatial dimension may have length one.
        Lazy inputs are materialized once during setup.
    dims : list of str
        Spatial dimensions, ordered as expected by the inversion.
    coords : str, optional
        Coordinate system used by the selected physical application.
    icbc : xarray.DataArray or None
        Initial guess and fixed boundary values.
    mParams, iParams : dict or None
        Model and solver controls. Architecture and dtype are retained;
        mxLoop, tolerance and printInfo are overridden:
        each frame always advances loop_per_frame additional sweeps.
        A positive diag_interval explicitly enables scalar diagnostic history.
    loop_per_frame, max_frames : int
        Positive sweep count per frame and number of frames to write.
    output : str or os.PathLike
        Required keyword-only destination. Existing files are never replaced.
    complevels : int, optional
        Zlib compression level for the solution variable: 0 disables
        compression; 1--9 enable lossless zlib compression. Passed as
        the NetCDF variable's standard complevel setting.

    Returns
    -------
    xarray.Dataset or None
        Returns diagnostics only when iParams['diag_interval'] is positive.
        Otherwise returns None. Writes an ``inverted`` variable with dimensions
        ``(iter, *input_dimensions)``.
        Reopen with xarray.open_dataset for lazy analysis.

    Notes
    -----
    No Dask frame graph or temporary files are created. CPU and CPU-mc reuse
    the current solution and prepared coefficients. GPU keeps the solution,
    coefficients and workspace on device for the entire call; only one frame
    is downloaded at a time. Writing and flushing finish before advancing.
    Live frame buffers and the HDF5 variable cache have bounded size,
    independent of max_frames. Optional scalar diagnostic history grows with
    the number of checkpoints; it never contains spatial frames. The full
    spatial working set must still fit in host/device memory; other processes
    can still exhaust system memory.
    On failure the file is closed and completed_frames identifies valid
    frames; a partially written trailing frame must be ignored.
    """
    import numbers
    from pathlib import Path

    from . import apps as _apps

    if isinstance(complevels, bool) or not isinstance(complevels, numbers.Integral):
        raise TypeError('complevels must be an integer between 0 and 9')
    if not 0 <= complevels <= 9:
        raise ValueError('complevels must be an integer between 0 and 9')
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f'Animation output already exists: {destination}')

    for name, value in [('loop_per_frame', loop_per_frame),
                        ('max_frames', max_frames)]:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f'{name} must be a positive integer')
        if value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if 'iter' in F.dims or 'iter' in F.coords:
        raise ValueError("'iter' is reserved for animation frames")
    len_nc = 0

    for sel in loop_noncore(F, dims):
        len_nc = len_nc + 1

    if len_nc != 1:
        raise ValueError('For 2D case, only 2D slice  F is allowed;\n'+
                        'For 3D case, only 3D volume F is allowed.')

    app_name = app_name.lower()
    if isinstance(dims, str):
        dimLen = 1
    elif isinstance(dims, (list, tuple)):
        dimLen = len(dims)
    else:
        raise TypeError(
            f'dims must be a string, list, or tuple, got {type(dims).__name__}')
    dims = _validate_inversion_dims(F, dims, dimLen)

    iParams = _apps.__update(_apps.default_iParams, iParams)
    _validate_inversion_bcs(iParams, dimLen)
    _validate_solver_controls(iParams)
    if iParams['internal_bcs'].strip().lower() == 'neumann':
        raise NotImplementedError(
            "animate_iteration does not yet support internal_bcs='neumann'")
    try:
        import netCDF4
    except ImportError as exc:
        raise ImportError(
            "Animation output requires netCDF4; install xinvert[io]") from exc

    if   app_name == 'poisson':
        coef_func = _apps.__coeffs_Poisson
        invt_func = _apps.inv_standard2D
        validMPs  = ['g', 'Omega', 'Rearth']

    elif app_name == 'pv2d':
        coef_func = _apps.__coeffs_PV2D
        invt_func = _apps.inv_standard2D
        validMPs  = ['f0', 'beta', 'N2', 'g', 'Omega', 'Rearth']

    elif app_name == 'geostrophic':
        coef_func = _apps.__coeffs_geostrophic
        invt_func = _apps.inv_standard2D
        validMPs  = ['f0', 'beta', 'g', 'Omega', 'Rearth']

    elif app_name == 'gillmatsuno':
        coef_func = _apps.__coeffs_GillMatsuno
        invt_func = _apps.inv_general2D
        validMPs  = ['f0', 'beta', 'epsilon', 'Phi', 'g', 'Omega', 'Rearth']

    elif app_name == 'eliassen':
        coef_func = _apps.__coeffs_Eliassen
        invt_func = _apps.inv_standard2D
        validMPs  = ['A', 'B', 'C', 'g', 'Omega', 'Rearth']

    elif app_name == 'stommel':
        coef_func = _apps.__coeffs_Stommel
        invt_func = _apps.inv_general2D
        validMPs  = ['beta', 'R', 'D', 'rho0', 'g', 'Omega', 'Rearth']

    elif app_name == 'stommelmunk':
        coef_func = _apps.__coeffs_StommelMunk
        invt_func = _apps.inv_general2D_bih
        validMPs  = ['A4', 'beta', 'R', 'D', 'rho0', 'g', 'Omega', 'Rearth']

    elif app_name == 'refstate':
        coef_func = _apps.__coeffs_RefState
        invt_func = _apps.inv_standard2D
        validMPs  = ['Ang0', 'Gamma', 'g', 'Omega', 'Rearth']

    elif app_name == 'brethertonhaidvogel':
        coef_func = _apps.__coeffs_Bretherton
        invt_func = _apps.inv_standard2D_full
        validMPs  = ['f0', 'beta', 'D', 'lambda', 'g', 'Omega', 'Rearth']

    elif app_name == 'fofonoff':
        coef_func = _apps.__coeffs_Fofonoff
        invt_func = _apps.inv_standard2D_full
        validMPs  = ['c0', 'c1', 'f0', 'beta', 'g', 'Omega', 'Rearth']

    elif app_name == 'omega':
        coef_func = _apps.__coeffs_omega
        invt_func = _apps.inv_standard3D
        validMPs  = ['f0', 'beta', 'N2', 'g', 'Omega', 'Rearth']

    elif app_name == '3docean':
        coef_func = _apps.__coeffs_3DOcean
        invt_func = _apps.inv_general3D
        validMPs  = ['f0', 'beta', 'N2', 'epsilon', 'k', 'g', 'Omega', 'Rearth']
    else:
        raise ValueError('unsupported problem: '+app_name+', should be one of:\n'+
                        "'Poisson'\n'PV2D'\n'GillMatsuno'\n'Eliassen'\n"+
                        "'geostrophic'\n'StommelMunk'\n'RefState'\n'omega'")

    mParams = _apps.__update(_apps.default_mParams, mParams, validMPs)

    ######  1. calculating the coefficients  ######
    maskF, initS, coeffs = coef_func(F, dims, coords, mParams, iParams, icbc)

    ######  2. calculating the parameters  ######
    if dimLen == 2:
        ps = _apps.__cal_params2D(maskF[dims[0]], maskF[dims[1]], coords,
                            BCs=iParams['BCs'], Rearth=mParams['Rearth'])
    elif dimLen == 3:
        ps = _apps.__cal_params3D(maskF[dims[0]], maskF[dims[1]], maskF[dims[2]], coords,
                            BCs=iParams['BCs'], Rearth=mParams['Rearth'])
    else:
        raise ValueError('dimension length should be one of [2, 3]')

    iParams = _apps.__update(ps, iParams)

    opt = iParams.get('optArg')
    if opt is not None and not (1.0 <= opt <= 2.0):
        raise ValueError(
            f"iParams['optArg']={opt} is out of range, "
            f"should be between 1 and 2 for SOR convergence"
        )

    if iParams['debug']:
        _apps.__print_params(iParams)

    from contextlib import ExitStack

    from . import core

    kernels = {
        'inv_standard2D': core.invert_standard_2D,
        'inv_standard2D_full': core.invert_standard_2D_full,
        'inv_general2D': core.invert_general_2D,
        'inv_general2D_bih': core.invert_general_bih_2D,
        'inv_standard3D': core.invert_standard_3D,
        'inv_general3D': core.invert_general_3D,
    }
    kernel = kernels[invt_func.__name__]
    frame_params = dict(iParams, mxLoop=int(loop_per_frame), tolerance=0.,
                        diag_interval=0, printInfo=False)
    grid_args, settings = _animation_kernel_settings(kernel, frame_params)
    # Validate backend availability before creating the destination.
    core._make_kernel(kernel, grid_args, frame_params)
    architect = str(iParams['architect']).strip().lower()
    state = np.array(_spatial_array(initS, initS, dims), copy=True, order='C')
    arrays = tuple(_spatial_array(coef, initS, dims)
                   for coef in (*coeffs, maskF))
    invalid = _spatial_array(maskF, initS, dims) == core._undeftmp
    has_invalid = bool(invalid.any())
    masked = (np.empty_like(state) if icbc is None and has_invalid
              and architect != 'gpu' else None)

    total_loops = int(loop_per_frame)*int(max_frames)
    diagnostic_interval = max(int(iParams.get('diag_interval', 0)), 0)
    diagnostic_count = ((total_loops + diagnostic_interval - 1)//diagnostic_interval
                        if diagnostic_interval else 0)
    history = (np.full(3 + 3*diagnostic_count, np.nan, dtype=np.float64)
               if diagnostic_interval else None)
    diagnostic_index = 0
    completed_loops = 0
    batch_dims = [dim for dim in initS.dims if dim not in dims]
    batch_coords = {name: coord for name, coord in initS.coords.items()
                    if set(coord.dims).issubset(batch_dims)}
    next_diagnostic = min(diagnostic_interval, total_loops) if history is not None else total_loops

    with ExitStack() as resources:
        if architect == 'gpu':
            from . import gpus

            resources.enter_context(gpus._get_gpu_sem())
            prepare = getattr(gpus, '_prepare_' + kernel.__name__[7:] + '_gpu')
            import inspect

            names = list(inspect.signature(kernel.py_func).parameters)
            values = dict(zip(names[1:names.index('info')], arrays))
            values.update(settings, S=state, info='', optArg=iParams['optArg'],
                          undef=core._undeftmp)
            if 'block_2d' in inspect.signature(prepare).parameters:
                values['block_2d'] = iParams.get('gpu_block2d')
            session = prepare(**values)
            resources.callback(session.close)
        else:
            function = kernel if architect == 'cpu' else core._cpu_mc_kernel_map[kernel]

        writer = resources.enter_context(_AnimationWriter(
            destination, initS, dims, state.dtype, netCDF4,
            architect, loop_per_frame, max_frames, int(complevels)))
        del coeffs, maskF, initS
        for frame in range(max_frames):
            frame_end = (frame + 1)*loop_per_frame
            while completed_loops < frame_end:
                target = min(frame_end, next_diagnostic)
                steps = target - completed_loops
                flags = np.zeros(3, dtype=np.float64)
                if architect == 'gpu':
                    overflow, error, loops = session.advance(steps, 0., flags, steps)
                else:
                    function(state, *arrays, '', *grid_args, iParams['optArg'],
                             core._undeftmp, flags, steps, 0., steps)
                    overflow, error, loops = flags
                if overflow or not np.isfinite(error):
                    raise RuntimeError(f'Animation solver overflow at frame {frame + 1}')
                if int(loops) != steps:
                    raise RuntimeError('Animation solver did not complete the requested sweeps')
                completed_loops += steps
                if history is not None and completed_loops == next_diagnostic:
                    if architect == 'gpu':
                        session.record_diagnostics(
                            history[3:], diagnostic_index, error, total_loops,
                            diagnostic_interval, core._undeftmp)
                    else:
                        from .cpus import _record_solution_diagnostics

                        _record_solution_diagnostics(
                            history, diagnostic_index, error, state, arrays[-1],
                            core._undeftmp, total_loops, diagnostic_interval)
                    diagnostic_index += 1
                    next_diagnostic = min(
                        next_diagnostic + diagnostic_interval, total_loops)
            if architect == 'gpu':
                session.copy_to_host(state)
            if icbc is None and has_invalid:
                if masked is None:
                    state[invalid] = iParams['undef']  # host copy only on GPU
                    data = state
                else:
                    np.copyto(masked, state)
                    masked[invalid] = iParams['undef']
                    data = masked
            else:
                data = state
            writer.write(frame, frame_end, data)
        writer.complete()

    if history is not None:
        history[:3] = 0., error, completed_loops
        diagnostics = core._diagnostics_from_flags(
            xr.DataArray(history.reshape((1,)*len(batch_dims) + (history.size,)),
                         dims=[*batch_dims, 'diagnostic'], coords=batch_coords),
            dict(frame_params, mxLoop=total_loops, diag_interval=diagnostic_interval))
        diagnostics.attrs.update(
            output=str(destination), completed_frames=int(max_frames),
            status='complete', mode='fixed_work')
        return diagnostics
    return None


def _spatial_array(field, template, dims):
    """Materialize one broadcast spatial field in contiguous solver order.

    Parameters
    ----------
    field, template : xarray.DataArray
        Coefficient/forcing or initial state and the complete problem template.
    dims : list of str
        Core spatial dimensions; all other dimensions must be singleton.

    Returns
    -------
    numpy.ndarray
        Contiguous spatial array in template dtype. No frame history is kept.
    """
    extra = [dim for dim in template.dims if dim not in dims]
    broadcast = xr.broadcast(field, template)[0].transpose(*dims, *extra)
    values = np.asarray(broadcast.values, dtype=template.dtype)
    return np.ascontiguousarray(values.reshape(tuple(template.sizes[d] for d in dims)))


def _animation_kernel_settings(kernel, params):
    """Map high-level grid controls to the low-level kernel argument order.

    Parameters
    ----------
    kernel : numba dispatcher
        CPU reference function defining the shared backend signature.
    params : dict
        Effective grid, boundary and iteration controls.

    Returns
    -------
    tuple of (list, dict)
        Ordered positional grid arguments and equivalent keyword settings.
    """
    import inspect

    names = list(inspect.signature(kernel.py_func).parameters)
    grid_names = names[names.index('info') + 1:names.index('optArg')]
    aliases = {'xc': 'gc1', 'yc': 'gc2', 'zc': 'gc3',
               'delx': 'del1', 'delxSqr': 'del1Sqr',
               'delxTr': 'del1Tr', 'delxSSr': 'del1SSr'}
    boundary = {'BCx': -1, 'BCy': -2, 'BCz': -3}
    settings = {name: params['BCs'][boundary[name]] if name in boundary
                else params[aliases.get(name, name)] for name in grid_names}
    return [settings[name] for name in grid_names], settings


class _AnimationWriter:
    """Append frames synchronously to one exclusively created NetCDF4 file.

    Parameters
    ----------
    output : pathlib.Path
        Destination, which must not exist.
    template : xarray.DataArray
        Input dimensions, sizes and coordinate metadata.
    dims : list of str
        Spatial solver order.
    dtype : numpy.dtype
        Solution storage precision.
    netcdf : module
        Imported netCDF4 module.
    architect : str
        Backend metadata.
    interval, count : int
        Sweeps per frame and requested frame count.
    complevels : int
        Validated compression level (0 disables zlib, 1--9 enable it).
    """
    def __init__(self, output, template, dims, dtype, netcdf, architect,
                 interval, count, complevels=0):
        self._file = netcdf.Dataset(str(output), 'w', clobber=False, format='NETCDF4')
        self._completed = 0
        extra = [d for d in template.dims if d not in dims]
        self._shape = tuple(template.sizes[d] for d in [*dims, *extra])
        self._axes = tuple([*dims, *extra].index(d) for d in template.dims)
        try:
            dataset = self._file
            dataset.setncatts({
                'architect': architect, 'loop_per_frame': interval,
                'requested_frames': count, 'completed_frames': 0,
                'status': 'running'})
            dataset.createDimension('iter', None)
            for dim in template.dims:
                dataset.createDimension(dim, template.sizes[dim])
            for name, coordinate in template.coords.items():
                variable = xr.conventions.encode_cf_variable(coordinate.variable, name=name)
                values = np.asarray(variable.values)
                datatype = str if values.dtype.kind in 'OU' else values.dtype
                attributes = dict(variable.attrs)
                fill = attributes.pop('_FillValue', None)
                target = dataset.createVariable(
                    name, datatype, coordinate.dims, fill_value=fill)
                target.setncatts(attributes)
                target[...] = values
            self._iteration = dataset.createVariable('iter', 'i8', ('iter',))
            self._iteration.long_name = 'Completed SOR sweeps'
            self._iteration.set_var_chunk_cache(
                size=64*1024, nelems=101, preemption=0.75)
            chunk_extent = 256 if len(dims) == 2 else 32
            chunks = (1, *[min(template.sizes[d], chunk_extent) for d in template.dims])
            self._data = dataset.createVariable(
                'inverted', dtype, ('iter', *template.dims),
                fill_value=np.nan, chunksizes=chunks,
                compression='zlib' if complevels > 0 else None,
                complevel=complevels)
            auxiliary = [name for name in template.coords if name not in template.dims]
            if auxiliary:
                self._data.coordinates = ' '.join(auxiliary)
            self._data.set_var_chunk_cache(size=4*1024**2, nelems=1009, preemption=0.75)
            dataset.sync()
        except BaseException as exc:
            try:
                self._file.status = 'failed'
                self._file.failure_type = type(exc).__name__
                self._file.sync()
            except (OSError, RuntimeError):
                pass  # Preserve the original failure if even metadata cannot be flushed.
            finally:
                self._file.close()
            raise

    def __enter__(self):
        """Return this writer; file creation occurs during construction."""
        return self

    def write(self, index, iteration, values):
        """Write and flush one frame before the solver advances again.

        Parameters
        ----------
        index, iteration : int
            Zero-based frame index and cumulative completed sweep count.
        values : numpy.ndarray
            Current spatial state, consumed synchronously without retaining it.

        Returns
        -------
        None
            Updates completed_frames only after the frame data are flushed.
        """
        self._data[index] = values.reshape(self._shape).transpose(self._axes)
        self._iteration[index] = iteration
        self._file.sync()
        self._completed = index + 1
        self._file.completed_frames = self._completed
        self._file.sync()

    def complete(self):
        """Mark completion after all requested frames have been flushed."""
        self._file.status = 'complete'
        self._file.sync()

    def __exit__(self, exception_type, exception, traceback):
        """Close the file on success or failure, preserving completed frames."""
        try:
            if exception_type is not None:
                self._file.status = 'failed'
                self._file.failure_type = exception_type.__name__
                self._file.completed_frames = self._completed
                self._file.sync()
        except (OSError, RuntimeError):
            if exception_type is None:
                raise
        finally:
            try:
                self._file.close()
            except (OSError, RuntimeError):
                if exception_type is None:
                    raise

