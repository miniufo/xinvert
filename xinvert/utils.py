"""
Shared utilities: preprocessing, flow diagnostics, animation and thread controls.

Provides ``loop_noncore`` for iterating over non-core dimensions,
``smooth9`` for 9-point smoothing, and ``coarsen`` for block-averaging
xarray DataArrays with flexible boundary handling.
"""
import numbers
import warnings
from contextlib import contextmanager

import numba as nb
import numpy as np
import xarray as xr
from numba import config, get_num_threads, set_num_threads

# Inherit the executing thread's mask; do not override worker configuration.
DEFAULT_THREADS = None


def validate_threads(threads):
    """Accept a thread count or None (inherit the executing thread's mask)."""
    if threads is not None and (
        isinstance(threads, bool)
        or not isinstance(threads, numbers.Integral)
        or not 1 <= threads <= config.NUMBA_NUM_THREADS
    ):
        raise ValueError(
            "threads must be None or a positive integer within Numba's "
            f"thread limit ({config.NUMBA_NUM_THREADS})"
        )


@contextmanager
def numba_threads(architect, threads=DEFAULT_THREADS):
    """Temporarily select CPU-mc threads inside the executing worker thread."""
    if architect != 'cpu-mc':
        yield
        return
    validate_threads(threads)
    previous = get_num_threads()
    try:
        if threads is not None:
            set_num_threads(int(threads))
        yield
    finally:
        set_num_threads(previous)




class _AnimationRenderer:
    """Process-local file handle and plotting state; no open file crosses IPC."""

    def __init__(self, settings, fig=None, axes=None):
        self.settings = settings
        self.dataset = xr.open_dataset(settings['source'])
        self.fig, self.axes = fig, axes
        self.original_size = fig.get_size_inches().copy() if fig is not None else None
        self.original_dpi = fig.dpi if fig is not None else None

    def render(self, index):
        import matplotlib as mpl
        from matplotlib.axes import Axes
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        settings = self.settings
        frame = self.dataset.isel({settings['framedim']: index})
        figure, axes = self.fig, self.axes
        owned = figure is None
        try:
            if owned:
                if settings['figfunc'] is not None:
                    figure, axes = settings['figfunc'](**settings['fig_kwargs'])
                    if not isinstance(figure, Figure):
                        raise TypeError('figfunc must return (Figure, axes)')
                else:
                    figure = Figure(figsize=settings['figsize'], dpi=settings['dpi'])
                    FigureCanvasAgg(figure)
                    axes = figure.add_subplot(111, **settings['subplot_kw'])
            flat = [axes] if isinstance(axes, Axes) else np.asarray(axes, dtype=object).ravel()
            if not len(flat) or any(not isinstance(ax, Axes) or ax.figure is not figure for ax in flat):
                raise ValueError('figfunc axes must belong to its Figure')
            if settings['plotfunc'] is None:
                field = frame[settings['plot_variable']].squeeze(drop=True)
                if field.ndim != 2:
                    raise ValueError('default plot needs a 2-D frame; provide plotfunc')
                artist = field.plot.pcolormesh(ax=axes, **settings['plot_kwargs'])
                if settings['plot_kwargs'].get('norm') is None:
                    low, high = artist.get_clim()
                    settings['plot_kwargs'].setdefault('vmin', low)
                    settings['plot_kwargs'].setdefault('vmax', high)
            else:
                settings['plotfunc'](frame, axes, **settings['plot_kwargs'])
            name = settings['folder'] / f'frame_{index:08d}.png'
            # PNG is lossless. Low compression trades temporary disk space for
            # rendering throughput; fixed canvas size is required for encoding.
            with mpl.rc_context({'savefig.bbox': None}):
                figure.savefig(name, format='png', dpi=settings['dpi'],
                               bbox_inches=None, pil_kwargs={'compress_level': 1})
            return index
        finally:
            if owned and figure is not None:
                from matplotlib import pyplot as plt
                plt.close(figure)
                figure.clear()

    def close(self):
        self.dataset.close()
        if self.fig is not None:
            self.fig.set_size_inches(self.original_size, forward=False)
            self.fig.set_dpi(self.original_dpi)


_ANIMATION_RENDERER = None


def _init_animation_worker(payload):
    import atexit
    import importlib
    import warnings

    import cloudpickle
    import matplotlib

    matplotlib.use('Agg', force=True)
    global _ANIMATION_RENDERER
    settings = cloudpickle.loads(payload)
    # Replay the plotting-library modules the parent had already imported.  A
    # spawn worker starts from stock Matplotlib, and the captured style depends
    # on side effects that a bare ``import ultraplot`` does not produce:
    # Ultraplot 2 lazy-loads its submodules, and the colormaps it registers (its
    # default ``image.cmap`` is 'Fire') as well as its rc validators (relative
    # font sizes such as 'med-large') only arrive with them.  Without the replay
    # the worker rejects a style value or cannot resolve a colormap, and a dead
    # worker takes the whole process pool down with it.
    for name in settings.get('ultraplot_modules', ()):
        try:
            importlib.import_module(name)
        except ImportError:  # pragma: no cover - module layout may move
            pass
    # Anything an older or partial install still rejects may cost the frame
    # style but never the worker; feeding the whole mapping to rcParams in one
    # call used to abort the pool on the first unusable value.
    skipped = []
    for key, value in settings['rc_params'].items():
        try:
            matplotlib.rcParams[key] = value
        except (KeyError, ValueError):
            skipped.append(key)
    if skipped:
        warnings.warn('animation worker kept its own style for '
                      + ', '.join(sorted(skipped)), RuntimeWarning, stacklevel=2)
    _ANIMATION_RENDERER = _AnimationRenderer(settings)
    atexit.register(_ANIMATION_RENDERER.close)


def _render_animation_worker(index):
    return _ANIMATION_RENDERER.render(index)


def _render_animation_parallel(settings, count, workers):
    """Bound outstanding work; return no pixel arrays through process queues."""
    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    import cloudpickle

    payload = cloudpickle.dumps(settings)
    indices = iter(range(1, count))
    with ProcessPoolExecutor(max_workers=min(workers, count-1),
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_animation_worker,
                             initargs=(payload,)) as pool:
        pending = set()
        try:
            for _ in range(min(2*workers, count-1)):
                pending.add(pool.submit(_render_animation_worker, next(indices)))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    index = next(indices, None)
                    if index is not None:
                        pending.add(pool.submit(_render_animation_worker, index))
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def _encode_animation_frames(folder, target, count, fps, writer, executable):
    """Encode ordered PNGs only after all render workers have exited."""
    from PIL import Image

    size = None
    for index in range(count):
        with Image.open(folder / f'frame_{index:08d}.png') as image:
            if size is None:
                size = image.size
            elif image.size != size:
                raise ValueError('frame sizes differ; use a fixed figure size and layout')
    if writer == 'ffmpeg':
        import subprocess

        args = [executable, '-v', 'error', '-y', '-threads', '1',
                '-filter_threads', '1', '-framerate', str(fps), '-start_number', '0',
                '-i', str(folder / 'frame_%08d.png'), '-frames:v', str(count)]
        if target.suffix == '.gif':
            args += ['-filter_complex_threads', '1', '-filter_complex',
                     'split[a][b];[a]palettegen=stats_mode=single[p];' +
                     '[b][p]paletteuse=new=1', '-loop', '0']
        else:
            args += ['-c:v', 'libx264', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                     '-pix_fmt', 'yuv420p']
        args += ['-threads', '1', str(target)]
        result = subprocess.run(args, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode:
            raise RuntimeError('FFmpeg encoding failed: ' + result.stderr.strip())
    else:
        frames = []
        try:
            for index in range(count):
                with Image.open(folder / f'frame_{index:08d}.png') as image:
                    frames.append(image.convert('RGB'))
            frames[0].save(target, format='GIF', save_all=True,
                           append_images=frames[1:], duration=int(1000/fps), loop=0)
        finally:
            for image in frames:
                image.close()


def create_animation(input_file, output_file, plotfunc=None, *, framedim=None, fps=6, figsize=(8, 5), dpi=100,
                    plot_kwargs=None, subplot_kw=None, ffmpeg=None,
                    overwrite=False, writer="auto", fig=None, axes=None,
                    workers=1, figfunc=None, fig_kwargs=None,
                    remove_frames=True, temp_dir=None):
    """Create a GIF or MP4 from NetCDF frames.

    Parameters
    ----------
    input_file, output_file : str or pathlib.Path
        Input NetCDF file and output path ending in .gif or .mp4.
        Existing output is preserved unless overwrite=True. Returns only
        after encoding succeeds; failures leave no partial output file.
    plotfunc : callable or None
        Called once per frame as ``plotfunc(frame, ax, **plot_kwargs)``.
        ``frame`` is a Dataset selected at one index along framedim, with all
        variables available and read only when accessed. Static variables
        without framedim are retained. Select variables inside plotfunc. ``ax``
        is a fresh Matplotlib Axes by default. With fig/axes, the same supplied
        axes object is passed on every call. For full-frame drawing use
        figfunc, which creates fresh axes for each frame.
        Draw on this figure; the callback return value is ignored. Functions,
        callable objects, closures and functools.partial are accepted.
        None draws a two-dimensional pcolormesh, selecting 'inverted' or the
        only data variable. Default color limits are fixed from the first
        frame; use plot_kwargs to choose explicit vmin/vmax.
    framedim : str or None
        Frame dimension. None selects 'iter', then 'time'; otherwise specify
        it explicitly. Iteration files use only completed_frames records.
    fps : positive number
        Frames per second (default 6). GIF timing has 0.01-second resolution.
    figsize : pair of float
        Figure width and height in inches (default (8, 5)).
    dpi : positive number
        Figure resolution (default 100). MP4 pixel dimensions are made even.
    plot_kwargs : mapping or None
        Keyword arguments forwarded to plotfunc, or to DataArray.plot.pcolormesh
        with the default plot. The caller's mapping is not modified.
    subplot_kw : mapping or None
        Arguments for Figure.add_subplot, e.g. {'projection': cartopy_crs}.
    ffmpeg : str or pathlib.Path or None
        Optional executable path. None uses system FFmpeg, falling back to
        the executable provided by imageio-ffmpeg.
    overwrite : bool
        Whether to replace an existing output after successful encoding.
    writer : {'auto', 'ffmpeg', 'pillow'}
        Case-insensitive. Auto prefers FFmpeg (system or imageio-ffmpeg),
        falling back to Pillow for GIF if no usable FFmpeg is found.
        Pillow does not support MP4 and caches rendered frames in memory.
        An explicitly supplied ffmpeg path must be usable; it does not
        silently fall back. Encoding errors are propagated without retrying.
    fig : matplotlib.figure.Figure or None
        Optional preconfigured Figure, including an UltraPlot Figure.
        Requires plotfunc. Its axes, formatting and colorbars are preserved;
        the figure is not cleared or closed. figsize is ignored in this mode.
    axes : Axes, axes collection or None
        Axes object passed unchanged to plotfunc when fig is supplied.
        None uses the figure's sole axes or a tuple of all existing axes.
        All axes must belong to fig. With supplied axes, subplot_kw is unused.

    workers : positive int
        Rendering processes (default 1). Each process opens its own NetCDF
        handle and writes distinct numbered PNGs. Encoding remains serial.
        Existing fig/axes require workers=1. Outstanding jobs are bounded.
    figfunc : callable or None
        Optional factory called as ``figfunc(**fig_kwargs)`` for each frame,
        returning (figure, axes). Use for independent UltraPlot/Cartopy plots
        with parallel rendering; requires plotfunc. Factory figures are closed
        after saving. It cannot be combined with existing fig/axes.
    fig_kwargs : mapping or None
        Keyword arguments passed to figfunc.
    remove_frames : bool
        True (default) deletes the frame directory, including after failures.
        False retains it and returns (animation_path, frame_directory).
        On failure the directory path is reported in a warning.
    temp_dir : str or pathlib.Path or None
        Parent directory for a unique frame subdirectory. None uses the output
        directory. Choose a fast local disk with space for all lossless PNGs.

    Returns
    -------
    pathlib.Path or tuple of pathlib.Path
        Absolute animation path. With remove_frames=False, returns
        (animation_path, frame_directory).

    Notes
    -----
    Each frame is read independently and saved as a lossless PNG in a unique
    temporary directory. Rendering completes before serial GIF/MP4 encoding.
    PNGs are deleted by default on success or failure. FFmpeg uses one encoder
    thread; Pillow still caches decoded frames during GIF assembly.
    For workers>1, callbacks and factories are serialized with cloudpickle.
    Do not capture a live Figure/Axes/artist or an open Dataset in their
    closures; create them in figfunc instead. In standalone scripts protect
    the call with ``if __name__ == '__main__':`` (spawn multiprocessing).
    This function does not run an inversion or interpolate between frames.

    Examples
    --------
    >>> create_animation('iteration.nc', 'iteration.gif',
    ...                 plot_kwargs={'vmin': -1e6, 'vmax': 1e6})
    >>> def draw(frame, ax, scale=1):
    ...     (frame.inverted * scale).plot(ax=ax, vmin=-1, vmax=1)
    >>> create_animation('iteration.nc', 'iteration.mp4', draw,
    ...                 plot_kwargs={'scale': 1e-6})
    """
    import os
    import shutil
    import tempfile
    from collections.abc import Mapping
    from pathlib import Path

    if plotfunc is not None and not callable(plotfunc):
        raise TypeError('plotfunc must be callable or None')
    options = {}
    for name, value in [('plot_kwargs', plot_kwargs), ('subplot_kw', subplot_kw),
                        ('fig_kwargs', fig_kwargs)]:
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f'{name} must be a mapping or None')
        options[name] = dict(value or {})
    for name, value in [('fps', fps), ('dpi', dpi)]:
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be a positive finite number')
    if isinstance(workers, bool) or not isinstance(workers, numbers.Integral) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if not isinstance(remove_frames, bool):
        raise TypeError('remove_frames must be boolean')
    if figfunc is not None and not callable(figfunc):
        raise TypeError('figfunc must be callable or None')
    if figfunc is not None and (fig is not None or axes is not None or plotfunc is None):
        raise ValueError('figfunc requires plotfunc and cannot be combined with fig/axes')
    if workers > 1 and (fig is not None or axes is not None):
        raise ValueError('parallel rendering cannot share fig/axes; use figfunc instead')
    if not isinstance(overwrite, bool):
        raise TypeError('overwrite must be boolean')
    source, target = Path(input_file).expanduser().resolve(), Path(output_file).expanduser().resolve()
    suffix = target.suffix.lower()
    if suffix not in ('.gif', '.mp4'):
        raise ValueError('output_file must end in .gif or .mp4')
    if source == target:
        raise ValueError('input and output must be different files')
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    if not isinstance(writer, str) or writer.strip().lower() not in ('auto', 'ffmpeg', 'pillow'):
        raise ValueError("writer must be 'auto', 'ffmpeg' or 'pillow'")
    writer = writer.strip().lower()
    if writer == 'pillow' and suffix != '.gif':
        raise ValueError('Pillow supports GIF only; MP4 requires FFmpeg')

    # Keep plotting imports lazy: ordinary inversions do not initialize a GUI.
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    owned_figure = fig is None
    if owned_figure:
        if axes is not None:
            raise ValueError('axes requires fig')
    else:
        if not isinstance(fig, Figure):
            raise TypeError('fig must be a Matplotlib or UltraPlot Figure')
        if plotfunc is None:
            raise ValueError('a supplied fig requires plotfunc')
        if axes is None:
            axes = fig.axes[0] if len(fig.axes) == 1 else tuple(fig.axes)
        flat_axes = [axes] if isinstance(axes, Axes) else np.asarray(axes, dtype=object).ravel()
        if not len(flat_axes) or any(not isinstance(ax, Axes) or ax.figure is not fig for ax in flat_axes):
            raise ValueError('axes must be nonempty and belong to fig')

    executable = None
    if writer != 'pillow':
        # Check availability before rendering; never reinterpret an encoding
        # error or a callback failure as a missing FFmpeg installation.
        import subprocess

        candidates = []
        if ffmpeg is not None:
            candidate = shutil.which(os.fspath(ffmpeg))
            if candidate is None:
                raise FileNotFoundError(f'FFmpeg executable not found: {ffmpeg}')
            candidates.append(candidate)
        else:
            candidate = shutil.which('ffmpeg')
            if candidate:
                candidates.append(candidate)
            # Only inspect the bundled fallback when the system candidate fails.
        for candidate in candidates:
            try:
                subprocess.run([candidate, '-version'], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                executable = candidate
                break
            except (OSError, subprocess.SubprocessError):
                pass
        if executable is None and ffmpeg is None:
            try:
                import imageio_ffmpeg
                candidate = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run([candidate, '-version'], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                executable = candidate
            except (ImportError, OSError, RuntimeError, subprocess.SubprocessError):
                pass
        if executable is None:
            if ffmpeg is not None or writer == 'ffmpeg' or suffix == '.mp4':
                raise RuntimeError('FFmpeg is unavailable; MP4 requires FFmpeg. '
                                   "For GIF use writer='pillow', or install imageio-ffmpeg.")
            warnings.warn('FFmpeg is unavailable; using Pillow for GIF. '
                          'Pillow keeps rendered frames in memory.', UserWarning, stacklevel=2)
            writer = 'pillow'
        else:
            writer = 'ffmpeg'

    with xr.open_dataset(source) as dataset:
        if framedim is None:
            framedim = next((dim for dim in ('iter', 'time') if dim in dataset.dims), None)
        if framedim not in dataset.dims:
            raise ValueError('specify framedim: expected an iter or time dimension')
        count = dataset.sizes[framedim]
        if framedim == 'iter' and 'completed_frames' in dataset.attrs:
            completed = dataset.attrs['completed_frames']
            if (isinstance(completed, (bool, np.bool_))
                    or not isinstance(completed, numbers.Integral)
                    or not 0 <= completed <= count):
                raise ValueError('invalid completed_frames in input file')
            count = int(completed)
        if count == 0:
            raise ValueError('input contains no completed frames')
        plot_variable = None
        if plotfunc is None:
            if 'inverted' in dataset:
                plot_variable = 'inverted'
            elif len(dataset.data_vars) == 1:
                plot_variable = next(iter(dataset.data_vars))
            else:
                raise ValueError('provide plotfunc for a multi-variable file')
            if framedim not in dataset[plot_variable].dims:
                raise ValueError('default plotted variable must contain the frame dimension')
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = target.parent if temp_dir is None else Path(temp_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='.xinvert-animation-', dir=parent))
    settings = dict(source=source, plot_variable=plot_variable, framedim=framedim,
                    folder=folder, figsize=figsize, dpi=dpi,
                    plotfunc=plotfunc, figfunc=figfunc,
                    **options)
    renderer = None
    try:
        renderer = _AnimationRenderer(settings, fig, axes)
        # Render frame zero once to validate the callback/factory and establish
        # default normalization before copying settings to spawn workers.
        renderer.render(0)
        # Spawn starts with fresh Matplotlib defaults. Copy the parent's style
        # after the first plot/factory has initialized its plotting library.
        import sys

        import matplotlib as mpl

        # Ultraplot 2 lazy-loads, so record the submodules this process has
        # already imported; a spawn worker replays them to rebuild the same
        # style (see _init_animation_worker).
        settings['ultraplot_modules'] = sorted(
            name for name in sys.modules if name.partition('.')[0] == 'ultraplot')
        settings['rc_params'] = {key: mpl.rcParams[key] for key in mpl.rcParams
                                 if key not in ('backend', 'backend_fallback', 'interactive')}
        renderer.close()
        renderer = None
        if count > 1:
            if workers > 1:
                _render_animation_parallel(settings, count, int(workers))
            else:
                renderer = _AnimationRenderer(settings, fig, axes)
                for index in range(1, count):
                    renderer.render(index)
                renderer.close()
                renderer = None
        # Encoded output stays on the destination filesystem for atomic
        # publication, even when PNGs are on a different temporary disk.
        with tempfile.TemporaryDirectory(prefix='.xinvert-encode-', dir=target.parent) as encoding:
            temporary = Path(encoding) / ('movie' + suffix)
            _encode_animation_frames(folder, temporary, count, fps, writer, executable)
            if overwrite:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
    except BaseException:
        if not remove_frames:
            warnings.warn(f'Animation failed; rendered frames retained at {folder}',
                          UserWarning, stacklevel=2)
        raise
    finally:
        if renderer is not None:
            renderer.close()
        if remove_frames:
            shutil.rmtree(folder)
    return target if remove_frames else (target, folder)




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
