"""NetCDF-to-animation integration and callback/resource contracts."""
import functools
import shutil
import subprocess

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from xinvert import create_animation
from xinvert.utils import create_animation as utility


@pytest.fixture
def nc(tmp_path):
    values = np.arange(4*3*4, dtype=float).reshape(4, 3, 4)
    ds = xr.Dataset({'inverted': (('iter', 'y', 'x'), values),
                     'other': (('iter', 'y', 'x'), -values)},
                    coords={'iter': [0, 20, 40, 60], 'y': [0., 1., 2.],
                            'x': [0., 1., 2., 3.]},
                    attrs={'completed_frames': 3})
    path = tmp_path / 'frames with spaces.nc'
    ds.to_netcdf(path)
    return path


def read_video(path, ffmpeg):
    """Decode the actual result, independent of the Matplotlib writer."""
    return subprocess.run(
        [ffmpeg, '-v', 'error', '-threads', '1', '-i', str(path),
         '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-'],
        check=True, capture_output=True).stdout


@pytest.mark.parametrize('suffix', ['gif', 'mp4'])
@pytest.mark.parametrize('bundled', [False, True])
def test_default_real_encoding(nc, tmp_path, suffix, bundled):
    import imageio_ffmpeg
    if bundled:
        # The conda package is a wrapper only - the binary rides along with the
        # PyPI wheels - so get_ffmpeg_exe() raises for a conda-forge install.
        # That is a missing capability of the environment, not a defect.
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError:
            pytest.skip('imageio-ffmpeg installed without a bundled binary')
    else:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            pytest.skip('system FFmpeg unavailable')
    settings = {'vmin': 0., 'vmax': 47., 'cmap': 'viridis'}
    saved = settings.copy()
    result = create_animation(nc, tmp_path / ('output with spaces.'+suffix),
                             figsize=(3, 2), dpi=60, fps=5,
                             plot_kwargs=settings, ffmpeg=ffmpeg)
    assert result.is_absolute() and result.exists()
    assert settings == saved
    raw = read_video(result, ffmpeg)
    assert len(raw) == 3*180*120*3
    if suffix == 'gif':
        with Image.open(result) as image:
            assert image.n_frames == 3
            assert image.info['loop'] == 0
    assert not list(tmp_path.glob('.xinvert-animation-*'))
    assert create_animation is utility


def test_callback_partial_receives_loaded_dataset_and_kwargs(nc, tmp_path):
    seen = []
    def draw(frame, ax, factor, color):
        assert isinstance(frame, xr.Dataset) and set(frame.dims) == {'y', 'x'}
        assert {'inverted', 'other'} == set(frame)
        assert isinstance(frame.inverted.data, np.ndarray)
        assert len(ax.figure.axes) == 1  # no colorbars left over from previous frame
        seen.append((int(frame['iter']), float(frame.inverted[0, 0]), factor, color))
        (frame.inverted*factor).plot(ax=ax, vmin=0, vmax=100, cmap=color)
    result = create_animation(nc, tmp_path/'custom.gif',
                             functools.partial(draw, factor=2),
                             plot_kwargs={'color': 'plasma'}, figsize=(3, 2), dpi=60)
    assert result.is_file()
    assert seen == [(0, 0., 2, 'plasma'), (20, 12., 2, 'plasma'), (40, 24., 2, 'plasma')]


def test_dataset_callback_can_draw_multiple_panels(nc, tmp_path):
    seen = []
    def draw(frame, ax, offset=1):
        assert isinstance(frame, xr.Dataset)
        assert 'iter' not in frame.dims
        assert {'inverted', 'other'} == set(frame)
        seen.append(int(frame['iter']))
        fig = ax.figure
        fig.clear()
        for axis, name in zip(fig.subplots(1, 2), ['inverted', 'other']):
            (frame[name]+offset).plot(ax=axis, vmin=-50, vmax=50)
    create_animation(nc, tmp_path/'panels.mp4', draw,
                    plot_kwargs={'offset': 2}, figsize=(4, 2), dpi=60)
    assert seen == [0, 20, 40]


def test_time_dimension_and_automatic_single_variable(nc, tmp_path):
    with xr.open_dataset(nc) as source:
        ds = source[['inverted']].rename({'iter': 'time', 'inverted': 'temperature'}).load()
    ds.to_netcdf(tmp_path/'time.nc')
    out = create_animation(tmp_path/'time.nc', tmp_path/'time.gif', figsize=(3, 2), dpi=60)
    with Image.open(out) as image:
        assert image.n_frames == 4  # completed_frames applies only to iteration files


def test_callback_failure_preserves_output_and_closes_input(nc, tmp_path):
    target = tmp_path/'protected.gif'
    target.write_bytes(b'original')
    seen = []
    def fail(frame, ax):
        seen.append(int(frame['iter']))
        if len(seen) == 2:
            raise RuntimeError('plot failed')
        frame.inverted.plot(ax=ax)
    with pytest.raises(RuntimeError, match='plot failed'):
        create_animation(nc, target, fail, overwrite=True, figsize=(3, 2), dpi=60)
    assert target.read_bytes() == b'original'
    assert not list(tmp_path.glob('.xinvert-animation-*'))
    # Windows also requires all file handles to be closed before renaming.
    nc.rename(tmp_path/'closed.nc')


def test_no_overwrite_and_missing_encoder(nc, tmp_path):
    target = tmp_path/'existing.mp4'
    target.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        create_animation(nc, target)
    with pytest.raises(FileNotFoundError, match='FFmpeg'):
        create_animation(nc, tmp_path/'missing.mp4', ffmpeg='no-such-xinvert-ffmpeg')
    assert target.read_bytes() == b'original'


@pytest.mark.parametrize('kwargs, message', [
    ({'fps': 0}, 'fps'), ({'dpi': float('nan')}, 'dpi'),
    ({'plotfunc': 'draw'}, 'plotfunc'), ({'plot_kwargs': []}, 'mapping'),
    ({'framedim': 'missing'}, 'framedim'), ({'variable': 'missing'}, 'variable'),
])
def test_invalid_inputs(nc, tmp_path, kwargs, message):
    with pytest.raises((ValueError, TypeError), match=message):
        create_animation(nc, tmp_path/'invalid.gif', **kwargs)
    assert not (tmp_path/'invalid.gif').exists()


def test_unsupported_output(nc, tmp_path):
    with pytest.raises(ValueError, match='gif or .mp4'):
        create_animation(nc, tmp_path/'bad.png')


def test_empty_completed_file(nc, tmp_path):
    with xr.open_dataset(nc) as source:
        ds = source.load()
    ds.attrs['completed_frames'] = 0
    ds.to_netcdf(tmp_path/'empty.nc')
    with pytest.raises(ValueError, match='no completed frames'):
        create_animation(tmp_path/'empty.nc', tmp_path/'empty.gif')


def test_first_frame_callback_error_is_preserved(nc, tmp_path, monkeypatch):
    from matplotlib.animation import FFMpegWriter
    def no_encoder(*args, **kwargs):
        raise AssertionError('encoder started before validating the first plot')
    monkeypatch.setattr(FFMpegWriter, 'setup', no_encoder)
    def fail(frame, ax):
        raise RuntimeError('first plot failed')
    with pytest.raises(RuntimeError, match='first plot failed'):
        create_animation(nc, tmp_path/'first.gif', fail)
    assert not (tmp_path/'first.gif').exists()
    assert not list(tmp_path.glob('.xinvert-animation-*'))


def test_automatic_bundled_fallback_and_normalization(nc, tmp_path, monkeypatch):
    import matplotlib as mpl
    from matplotlib import colors
    original = mpl.rcParams['animation.ffmpeg_path']
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    target = create_animation(nc, tmp_path/'fallback.gif', figsize=(3, 2), dpi=60,
                             plot_kwargs={'norm': colors.Normalize(0, 47)})
    assert mpl.rcParams['animation.ffmpeg_path'] == original
    with Image.open(target) as image:
        assert image.n_frames == 3


@pytest.mark.parametrize('writer', ['pillow', 'PILLOW', 'PiLlOw'])
def test_explicit_pillow_does_not_probe_ffmpeg(nc, tmp_path, monkeypatch, writer):
    def forbidden(*args, **kwargs):
        raise AssertionError('Pillow must not probe or invoke FFmpeg')
    monkeypatch.setattr(shutil, 'which', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    out = create_animation(nc, tmp_path/'pillow.gif', writer=writer,
                          figsize=(3, 2), dpi=60, fps=5,
                          plot_kwargs={'vmin': 0, 'vmax': 47})
    with Image.open(out) as image:
        assert image.n_frames == 3
        assert image.info['duration'] == 200


def missing_ffmpeg(monkeypatch):
    import imageio_ffmpeg
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    def missing():
        raise RuntimeError('FFmpeg unavailable in test')
    monkeypatch.setattr(imageio_ffmpeg, 'get_ffmpeg_exe', missing)


def test_auto_pillow_fallback(nc, tmp_path, monkeypatch):
    missing_ffmpeg(monkeypatch)
    with pytest.warns(UserWarning, match='using Pillow'):
        out = create_animation(nc, tmp_path/'auto.gif', figsize=(3, 2), dpi=60)
    with Image.open(out) as image:
        assert image.n_frames == 3


@pytest.mark.parametrize('kwargs, suffix', [({'writer': 'ffmpeg'}, 'gif'), ({}, 'mp4')])
def test_missing_ffmpeg_required_output_fails(nc, tmp_path, monkeypatch, kwargs, suffix):
    missing_ffmpeg(monkeypatch)
    with pytest.raises(RuntimeError, match='FFmpeg is unavailable'):
        create_animation(nc, tmp_path/('missing.'+suffix), **kwargs)
    assert not (tmp_path/('missing.'+suffix)).exists()


def test_pillow_rejects_mp4_and_invalid_writer(nc, tmp_path):
    with pytest.raises(ValueError, match='GIF only'):
        create_animation(nc, tmp_path/'bad.mp4', writer='pillow')
    with pytest.raises(ValueError, match='writer must be'):
        create_animation(nc, tmp_path/'bad.gif', writer='other')


@pytest.mark.parametrize('writer', ['pillow', 'ffmpeg'])
def test_external_figure_keeps_layout_and_updates_existing_artists(nc, tmp_path, writer):
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(4, 2), dpi=80)
    meshes = [ax.imshow(np.zeros((3, 4)), vmin=0, vmax=47) for ax in axs]
    for ax, mesh in zip(axs, meshes):
        fig.colorbar(mesh, ax=ax)
        ax.set_xlabel('kept label')
    fig.canvas.draw()  # settle Matplotlib colorbar aspect before comparing layout
    original_axes = tuple(fig.axes)
    positions = [ax.get_position().bounds for ax in fig.axes]
    original_size = fig.get_size_inches().copy()
    seen = []
    def draw(frame, axes, scale):
        assert axes is axs
        assert tuple(fig.axes) == original_axes
        for mesh in meshes:
            mesh.set_data(frame.inverted.values*scale)
        seen.append(int(frame['iter']))
    try:
        create_animation(nc, tmp_path/(writer+'.gif'), draw,
                        fig=fig, axes=axs, writer=writer, dpi=60,
                        plot_kwargs={'scale': 1})
        assert seen == [0, 20, 40]
        assert plt.fignum_exists(fig.number)
        assert tuple(fig.axes) == original_axes
        assert all(ax.get_xlabel() == 'kept label' for ax in axs)
        np.testing.assert_allclose([ax.get_position().bounds for ax in fig.axes], positions)
        np.testing.assert_array_equal(fig.get_size_inches(), original_size)
        assert fig.dpi == 80
        np.testing.assert_array_equal(meshes[0].get_array(), np.arange(24, 36).reshape(3, 4))
    finally:
        plt.close(fig)


def test_external_figure_survives_callback_failure(nc, tmp_path):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title('preserved')
    def fail(frame, axes):
        raise RuntimeError('custom plot failed')
    try:
        with pytest.raises(RuntimeError, match='custom plot failed'):
            create_animation(nc, tmp_path/'failed.gif', fail, fig=fig, writer='pillow')
        assert fig.axes == [ax] and ax.get_title() == 'preserved'
        assert plt.fignum_exists(fig.number)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('suffix, writer', [('gif', 'pillow'), ('mp4', 'ffmpeg')])
def test_ultraplot_actual_encoding(nc, tmp_path, suffix, writer):
    uplt = pytest.importorskip('ultraplot')
    fig, axs = uplt.subplots(figsize=(4, 3))
    mesh = axs[0].pcolormesh(np.zeros((3, 4)), vmin=0, vmax=47, cmap='viridis')
    axs[0].format(xlabel='longitude', ylabel='latitude', title='UltraPlot')
    fig.colorbar(mesh, loc='r')
    seen = []
    def draw(frame, axes):
        assert axes is axs
        assert hasattr(axes[0], 'format')
        mesh.set_array(frame.inverted.values.ravel())
        seen.append(int(frame['iter']))
    try:
        out = create_animation(nc, tmp_path/('ultraplot.'+suffix), draw,
                              fig=fig, axes=axs,
                              writer=writer, dpi=60)
        assert out.stat().st_size > 0
        assert seen == [0, 20, 40]
        assert axs[0] in fig.axes and mesh in axs[0].collections
        if suffix == 'gif':
            with Image.open(out) as image:
                assert image.n_frames == 3
    finally:
        uplt.close(fig)


def test_retained_numbered_frames_and_unique_directories(nc, tmp_path):
    outputs = []
    for name in ['a', 'b']:
        result, directory = create_animation(
            nc, tmp_path/(name+'.gif'), writer='pillow', remove_frames=False,
            temp_dir=tmp_path/'scratch', figsize=(3, 2), dpi=60)
        assert result.exists() and directory.parent == tmp_path/'scratch'
        assert [p.name for p in sorted(directory.iterdir())] == [
            'frame_00000000.png', 'frame_00000001.png', 'frame_00000002.png']
        outputs.append(directory)
    assert outputs[0] != outputs[1]


def test_parallel_rendering_uses_processes_and_preserves_frame_order(nc, tmp_path):
    import os
    markers = tmp_path/'markers'
    markers.mkdir()
    def draw(frame, ax, marker_dir):
        import os
        import time
        from pathlib import Path
        index = int(frame['iter'])
        (Path(marker_dir)/f'{index}.txt').write_text(str(os.getpid()))
        time.sleep(.3)
        ax.imshow(frame.inverted.values, vmin=0, vmax=47, cmap='viridis')
    serial, serial_frames = create_animation(
        nc, tmp_path/'serial.gif', draw,
        plot_kwargs={'marker_dir': markers}, writer='pillow',
        remove_frames=False, figsize=(3, 2), dpi=60)
    result, frames = create_animation(
        nc, tmp_path/'parallel.gif', draw,
        plot_kwargs={'marker_dir': markers}, writer='pillow', workers=2,
        remove_frames=False, figsize=(3, 2), dpi=60)
    pids = {int(p.read_text()) for p in markers.glob('*.txt')}
    assert os.getpid() in pids and len(pids) >= 2
    for first, second in zip(sorted(serial_frames.glob('*.png')), sorted(frames.glob('*.png'))):
        with Image.open(first) as a, Image.open(second) as b:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with Image.open(result) as image:
        assert image.n_frames == 3
    assert serial.exists()


def test_parallel_factory_is_local_and_creates_independent_figures(nc, tmp_path):
    def factory(width):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        figure = Figure(figsize=(width, 2))
        FigureCanvasAgg(figure)
        return figure, figure.subplots(1, 2)
    def draw(frame, axes, scale):
        for axis in axes:
            axis.imshow(frame.inverted.values*scale, vmin=0, vmax=47)
    out = create_animation(nc, tmp_path/'factory.mp4', draw,
                          figfunc=factory, fig_kwargs={'width': 4},
                          plot_kwargs={'scale': 1}, workers=2, dpi=60)
    assert out.exists()
    assert not list(tmp_path.glob('.xinvert-animation-*'))


def test_parallel_error_cleanup_and_explicit_retention(nc, tmp_path):
    def fail(frame, ax):
        if int(frame['iter']) == 20:
            raise RuntimeError('worker plot failed')
        ax.imshow(frame.inverted.values, vmin=0, vmax=47)
    target = tmp_path/'protected.gif'
    target.write_bytes(b'original')
    with pytest.raises(RuntimeError, match='worker plot failed'):
        create_animation(nc, target, fail, workers=2, overwrite=True, writer='pillow')
    assert target.read_bytes() == b'original'
    assert not list(tmp_path.glob('.xinvert-animation-*'))
    with pytest.warns(UserWarning, match='retained at'), pytest.raises(RuntimeError):
        create_animation(nc, target, fail, workers=2, overwrite=True,
                        writer='pillow', remove_frames=False)
    assert len(list(tmp_path.glob('.xinvert-animation-*'))) == 1
    assert target.read_bytes() == b'original'


def test_parallel_rejects_shared_figure(nc, tmp_path):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match='cannot share fig/axes'):
            create_animation(nc, tmp_path/'bad.gif', lambda frame, ax: None,
                            workers=2, fig=fig, axes=ax)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
def test_invalid_workers(nc, tmp_path, workers):
    with pytest.raises(ValueError, match='workers'):
        create_animation(nc, tmp_path/'bad.gif', workers=workers)


def test_parallel_ultraplot_factory(nc, tmp_path):
    pytest.importorskip('ultraplot')
    def factory():
        import ultraplot as uplt
        fig, axs = uplt.subplots(figsize=(4, 3))
        axs[0].format(xlabel='longitude', ylabel='latitude')
        return fig, axs[0]
    def draw(frame, ax):
        ax.pcolormesh(frame.inverted, vmin=0, vmax=47, cmap='viridis', colorbar='r')
    out = create_animation(nc, tmp_path/'parallel-ultraplot.mp4', draw,
                          figfunc=factory, workers=2, dpi=60)
    assert out.stat().st_size > 0
    assert not list(tmp_path.glob('.xinvert-animation-*'))


@pytest.mark.parametrize('workers', [1, 2])
def test_multivariable_frame_keeps_static_data_without_loading_all(nc, tmp_path, monkeypatch, workers):
    with xr.open_dataset(nc) as source:
        ds = source.load()
    ds['topography'] = xr.DataArray(np.arange(12).reshape(3, 4), dims=('y', 'x'))
    path = tmp_path/'many.nc'
    ds.to_netcdf(path)
    original_load = xr.Dataset.load
    def checked_load(self, *args, **kwargs):
        if {'inverted', 'other', 'topography'} <= set(self):
            raise AssertionError('the whole multi-variable frame was loaded eagerly')
        return original_load(self, *args, **kwargs)
    monkeypatch.setattr(xr.Dataset, 'load', checked_load)
    def draw(frame, ax):
        assert isinstance(frame, xr.Dataset)
        assert {'inverted', 'other', 'topography'} == set(frame)
        assert frame.inverted.dims == frame.other.dims == frame.topography.dims == ('y', 'x')
        np.testing.assert_array_equal(frame.topography, np.arange(12).reshape(3, 4))
        np.testing.assert_array_equal(frame.other, -frame.inverted)
        ax.imshow(frame.inverted.values, vmin=0, vmax=47)
    out = create_animation(path, tmp_path/'many.mp4', draw, workers=workers,
                          figsize=(3, 2), dpi=60)
    assert out.exists()
