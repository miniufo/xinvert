"""Synchronous frame output and persistent GPU workspaces."""
import numpy as np
import pytest
import xarray as xr

from xinvert import animate_iteration, apps, invert_Poisson


def _forcing():
    x = np.linspace(0., 1., 10)
    values = np.sin(np.pi*x[:, None])*np.sin(np.pi*x[None, :])
    return xr.DataArray(values, dims=['y', 'x'], coords={'y': x, 'x': x})


def test_streaming_frames_match_full_solve(architect, tmp_path):
    forcing = _forcing()
    params = {'architect': architect, 'dtype': 'float64', 'optArg': 1.2,
              'BCs': ['fixed', 'fixed'], 'diag_interval': 1, 'tolerance': 2.}
    output = tmp_path / 'animation.nc'
    result = animate_iteration(
        'Poisson', forcing, ['y', 'x'], coords='cartesian',
        iParams=params, loop_per_frame=2, max_frames=3, output=output)
    assert isinstance(result, xr.Dataset)
    assert result.iterations.item() == 6
    assert result.iteration.values.tolist() == list(range(1, 7))
    assert result.attrs['completed_frames'] == 3
    assert result.attrs['mode'] == 'fixed_work'
    with xr.open_dataset(output) as saved:
        assert saved.inverted.dims == ('iter', 'y', 'x')
        assert saved.iter.values.tolist() == [2, 4, 6]
        assert saved.attrs['completed_frames'] == 3
        assert saved.attrs['status'] == 'complete'
        assert saved.inverted.dtype == np.float64
        assert not np.array_equal(saved.inverted[0], saved.inverted[-1])
        expected = invert_Poisson(
            forcing, ['y', 'x'], coords='cartesian',
            iParams={**params, 'mxLoop': 6, 'tolerance': 0.,
                     'diag_interval': 0, 'printInfo': False})
        xr.testing.assert_allclose(
            saved.inverted.isel(iter=-1, drop=True), expected)
    assert set(tmp_path.iterdir()) == {output}


def test_output_is_not_overwritten(tmp_path):
    output = tmp_path / 'existing.nc'
    output.touch()
    with pytest.raises(FileExistsError):
        animate_iteration('Poisson', _forcing(), ['y', 'x'], output=output)
    assert output.stat().st_size == 0


def test_output_is_required():
    with pytest.raises(TypeError, match='output'):
        animate_iteration('Poisson', _forcing(), ['y', 'x'])


@pytest.mark.parametrize('name,value,exception', [
    ('loop_per_frame', 0, ValueError), ('max_frames', -1, ValueError),
    ('loop_per_frame', 1.5, TypeError), ('max_frames', True, TypeError),
])
def test_invalid_frame_controls(name, value, exception, tmp_path):
    with pytest.raises(exception, match='positive integer'):
        animate_iteration('Poisson', _forcing(), ['y', 'x'],
                          output=tmp_path/'frames.nc', **{name: value})


def test_input_can_be_lazy_and_transposed(architect, tmp_path):
    forcing = (_forcing()*xr.DataArray(np.linspace(1., 2., 10), dims=['x']))
    forcing = forcing.transpose('x', 'y').chunk({'y': 5, 'x': 5})
    forcing = forcing.expand_dims(time=[np.datetime64('2000-01-01')])
    output = tmp_path/'frames.nc'
    animate_iteration(
        'Poisson', forcing, ['y', 'x'], coords='cartesian',
        iParams={'architect': architect, 'dtype': 'float32'},
        loop_per_frame=2, max_frames=2, output=output)
    with xr.open_dataset(output) as saved:
        assert saved.inverted.dims == ('iter', 'time', 'x', 'y')
        assert saved.inverted.dtype == np.float32
        np.testing.assert_array_equal(
            saved.time.values.astype('datetime64[ns]'),
            forcing.time.values.astype('datetime64[ns]'))
        assert np.isfinite(saved.inverted).all()
        expected = invert_Poisson(
            forcing, ['y', 'x'], coords='cartesian',
            iParams={'architect': architect, 'dtype': 'float32',
                     'mxLoop': 4, 'tolerance': 0., 'printInfo': False})
        np.testing.assert_allclose(
            saved.inverted.isel(iter=-1).transpose(*expected.dims).values,
            expected.values, rtol=1e-6, atol=1e-7)


def test_failed_write_preserves_completed_frames(architect, tmp_path, monkeypatch):
    from xinvert.animate import _AnimationWriter

    write = _AnimationWriter.write

    def fail_on_second(self, index, iteration, values):
        if index == 1:
            raise OSError('injected write failure')
        return write(self, index, iteration, values)

    monkeypatch.setattr(_AnimationWriter, 'write', fail_on_second)
    output = tmp_path/'frames.nc'
    with pytest.raises(OSError, match='injected'):
        animate_iteration(
            'Poisson', _forcing(), ['y', 'x'], coords='cartesian',
            iParams={'architect': architect}, max_frames=3, output=output)
    with xr.open_dataset(output) as saved:
        assert saved.attrs['status'] == 'failed'
        assert saved.attrs['completed_frames'] == 1
        assert saved.sizes['iter'] == 1
        assert np.isfinite(saved.inverted).all()


def test_gpu_uploads_spatial_arrays_only_once(tmp_path, monkeypatch):
    from tests.gpu_utils import gpu_available
    if not gpu_available():
        pytest.skip('CUDA unavailable')
    from xinvert import gpus

    uploads = []
    downloads = []
    upload = gpus.cuda.to_device
    download = gpus._GpuSorSession.copy_to_host

    def observed(array, *args, **kwargs):
        if isinstance(array, np.ndarray) and array.ndim == 2:
            uploads.append(array.shape)
        return upload(array, *args, **kwargs)

    monkeypatch.setattr(gpus.cuda, 'to_device', observed)
    def observed_download(self, destination):
        downloads.append(destination.shape)
        return download(self, destination)

    monkeypatch.setattr(gpus._GpuSorSession, 'copy_to_host', observed_download)
    diagnostics = animate_iteration(
        'Poisson', _forcing(), ['y', 'x'], coords='cartesian',
        iParams={'architect': 'gpu', 'diag_interval': 1},
        loop_per_frame=2, max_frames=5,
        output=tmp_path/'gpu.nc')
    assert uploads == [(10, 10)]*5  # S, A, B, C, forcing; not per frame
    assert downloads == [(10, 10)]*5  # not at every diagnostic checkpoint
    assert diagnostics.sizes['iteration'] == 10


@pytest.mark.parametrize('name,model', [
    ('Eliassen', {'A': 1., 'B': 0.1, 'C': 1.}),
    ('BrethertonHaidvogel', {'D': 100.}),
    ('StommelMunk', {'A4': 1., 'D': 100.}),
    ('omega', {'N2': 0.01, 'f0': 0.1}),
    ('Stommel', {'D': 100.}),
    ('3DOcean', {'N2': 0.01, 'epsilon': 1e-4, 'k': 1e-5}),
])
def test_streaming_dispatch(architect, name, model, tmp_path):
    forcing = _forcing()
    if name in ('3DOcean', 'omega'):
        forcing = xr.zeros_like(forcing).expand_dims(z=np.linspace(0., 1., 9))
    dims = list(forcing.dims)
    params = {'architect': architect, 'dtype': 'float64', 'optArg': 1.,
              'BCs': ['fixed']*len(dims), 'printInfo': False}
    output = tmp_path/'frames.nc'
    animate_iteration(
        name, forcing, dims, coords='cartesian', mParams=model,
        iParams=params, loop_per_frame=2, max_frames=2, output=output)
    expected = getattr(apps, 'invert_' + name)(
        forcing, dims, coords='cartesian', mParams=model,
        iParams={**params, 'mxLoop': 4, 'tolerance': 0.})
    with xr.open_dataset(output) as saved:
        xr.testing.assert_allclose(
            saved.inverted.isel(iter=-1, drop=True), expected)


def test_cpu_reuses_one_output_buffer(tmp_path, monkeypatch):
    from xinvert.animate import _AnimationWriter

    buffers = set()
    write = _AnimationWriter.write

    def observed(self, index, iteration, values):
        buffers.add(values.__array_interface__['data'][0])
        return write(self, index, iteration, values)

    monkeypatch.setattr(_AnimationWriter, 'write', observed)
    forcing = _forcing()
    forcing[4, 4] = np.nan
    animate_iteration(
        'Poisson', forcing, ['y', 'x'], coords='cartesian',
        loop_per_frame=1, max_frames=20, output=tmp_path/'frames.nc')
    assert len(buffers) == 1
    with xr.open_dataset(tmp_path/'frames.nc') as saved:
        assert saved.inverted[:, 4, 4].isnull().all()


def test_streaming_mask_does_not_change_internal_state(architect, tmp_path):
    forcing = _forcing()
    forcing[4, 4] = np.nan
    output = tmp_path/'masked.nc'
    params = {'architect': architect, 'dtype': 'float64', 'optArg': 1.}
    animate_iteration(
        'Poisson', forcing, ['y', 'x'], coords='cartesian',
        iParams=params, loop_per_frame=2, max_frames=3, output=output)
    expected = invert_Poisson(
        forcing, ['y', 'x'], coords='cartesian',
        iParams={**params, 'mxLoop': 6, 'tolerance': 0., 'printInfo': False})
    with xr.open_dataset(output) as saved:
        assert saved.inverted[:, 4, 4].isnull().all()
        xr.testing.assert_allclose(
            saved.inverted.isel(iter=-1, drop=True), expected)


def test_auxiliary_coordinates_are_preserved(tmp_path):
    forcing = _forcing().assign_coords(experiment='control')
    forcing = forcing.assign_coords(latitude=forcing.y + 20.)
    output = tmp_path/'coordinates.nc'
    animate_iteration(
        'Poisson', forcing, ['y', 'x'], coords='cartesian',
        loop_per_frame=1, max_frames=1, output=output)
    with xr.open_dataset(output) as saved:
        assert saved.inverted.experiment.item() == 'control'
        xr.testing.assert_equal(saved.latitude, forcing.latitude)


@pytest.mark.parametrize('interval,expected', [
    (3, [3, 6, 9, 12, 14]), (5, [5, 10, 14]), (100, [14]),
])
def test_diagnostics_are_independent_of_frame_interval(
        architect, interval, expected, tmp_path):
    output = tmp_path/'diagnosed.nc'
    diagnostics = animate_iteration(
        'Poisson', _forcing(), ['y', 'x'], coords='cartesian',
        iParams={'architect': architect, 'dtype': 'float64',
                 'diag_interval': interval},
        loop_per_frame=7, max_frames=2, output=output)
    assert diagnostics.iteration.values.tolist() == expected
    assert diagnostics.iterations.item() == 14
    assert diagnostics.attrs['diag_interval'] == interval
    assert diagnostics.attrs['tolerance'] == 0.
    assert diagnostics.stop_reason.item() == 'max_iterations'
    assert not diagnostics.converged.item()
    assert np.isfinite(diagnostics.error).all()
    with xr.open_dataset(output) as saved:
        assert saved.iter.values.tolist() == [7, 14]
        final = saved.inverted.isel(iter=-1)
        assert diagnostics.solution_min.isel(iteration=-1).item() == float(final.min())
        assert diagnostics.solution_max.isel(iteration=-1).item() == float(final.max())
        assert saved.attrs['completed_frames'] == 2


def test_default_has_no_diagnostic_return(tmp_path):
    assert animate_iteration(
        'Poisson', _forcing(), ['y', 'x'], coords='cartesian',
        loop_per_frame=1, max_frames=1, output=tmp_path/'frames.nc') is None


def test_diagnostics_retain_single_time_coordinate(architect, tmp_path):
    forcing = _forcing().expand_dims(time=[np.datetime64('2000-01-01', 'ns')])
    diagnostics = animate_iteration(
        'Poisson', forcing, ['y', 'x'], coords='cartesian',
        iParams={'architect': architect, 'diag_interval': 3},
        loop_per_frame=2, max_frames=2, output=tmp_path/'time.nc')
    assert diagnostics.error.dims == ('time', 'iteration')
    xr.testing.assert_equal(diagnostics.time, forcing.time)
    assert diagnostics.iteration.values.tolist() == [3, 4]


@pytest.mark.parametrize('level', [0, 1, 4, 9])
def test_zlib_compression_is_lossless(architect, level, tmp_path):
    import netCDF4

    output = tmp_path/'compressed.nc'
    animate_iteration(
        'Poisson', _forcing(), ['y', 'x'], coords='cartesian',
        iParams={'architect': architect, 'dtype': 'float64'},
        loop_per_frame=2, max_frames=2, output=output, complevels=level)
    with netCDF4.Dataset(output) as dataset:
        filters = dataset.variables['inverted'].filters()
        assert filters['zlib'] == (level > 0)
        assert filters['complevel'] == level
    expected = invert_Poisson(
        _forcing(), ['y', 'x'], coords='cartesian',
        iParams={'architect': architect, 'dtype': 'float64',
                 'mxLoop': 4, 'tolerance': 0., 'printInfo': False})
    with xr.open_dataset(output) as saved:
        xr.testing.assert_allclose(
            saved.inverted.isel(iter=-1, drop=True), expected)


@pytest.mark.parametrize('level,exception', [
    (-1, ValueError), (10, ValueError), (1.5, TypeError),
    (True, TypeError), ('4', TypeError),
])
def test_invalid_compression_level_creates_no_file(level, exception, tmp_path):
    output = tmp_path/'invalid.nc'
    with pytest.raises(exception, match='complevels'):
        animate_iteration('Poisson', _forcing(), ['y', 'x'],
                          output=output, complevels=level)
    assert not output.exists()
