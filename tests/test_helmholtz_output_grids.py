"""Center input/output and physically labelled face output."""
import numpy as np
import pytest
import xarray as xr

from xinvert.helmholtz import decompose, reconstruct
from xinvert.projection import project_vector


def inputs(periodic):
    y, x = np.arange(10.), np.arange(14.)
    u = xr.DataArray(np.sin(x[None, :]*.4)*np.cos(y[:, None]*.3),
                     dims=["y", "x"], coords={'y': y, 'x': x})
    v = u.roll(x=2)*.7
    wet = xr.ones_like(u, dtype=bool)
    wet[[0, -1], :] = False
    if not periodic:
        wet[:, [0, -1]] = False
    wet[4:6, 7] = False
    return u, v, wet


def faces(u, v, wet, periodic):
    east = wet & (wet.roll(x=-1, roll_coords=False) if periodic else wet.shift(x=-1, fill_value=False))
    north = wet & wet.shift(y=-1, fill_value=False)
    nxt = u.roll(x=-1, roll_coords=False) if periodic else u.shift(x=-1)
    return (.5*(u+nxt)).where(east, 0), (.5*(v+v.shift(y=-1))).where(north, 0), east, north


def average(a, valid, axis, periodic):
    out = np.zeros(a.shape)
    for j, i in np.ndindex(a.shape):
        indices = [(j, i)]
        if axis == 1 and (periodic or i > 0):
            indices.append((j, (i-1) % a.shape[1]))
        if axis == 0 and j > 0:
            indices.append((j-1, i))
        values = [a[y, x] for y, x in indices if valid[y, x]]
        if values:
            out[j, i] = sum(values)/len(values)
    return out


@pytest.mark.parametrize("periodic", [False, True])
def test_center_adapter_matches_native_projection(periodic):
    u, v, wet = inputs(periodic)
    saved = u.copy(deep=True)
    uf, vf, east, north = faces(u, v, wet, periodic)
    opts = {'BCs': ["fixed", "periodic" if periodic else "fixed"], 'mxLoop': 6000, 'tolerance': 1e-11, 'printInfo': False, 'diag_interval': 20}
    direct, _ = project_vector(uf, vf, wet, ["y", "x"], coords="cartesian",
                               periodic_x=periodic, iParams=opts)
    result, diag = decompose(u, v, ["y", "x"], coords="cartesian", method="forget",
                             mask=wet, iParams=opts, return_potentials=True, return_diagnostics=True)
    mapping = {'udiv': "x_gradient", 'vdiv': "y_gradient", 'urot': "x_solenoidal", 'vrot': "y_solenoidal"}
    for name, ref in mapping.items():
        valid, axis = (east, 1) if name.startswith("u") else (north, 0)
        expected = average(direct[ref].values, valid.values, axis, periodic)
        np.testing.assert_allclose(result[name], np.where(wet, expected, np.nan), rtol=1e-10, atol=1e-11)
        assert result[name].dims == u.dims
        xr.testing.assert_equal(result[name].coords.to_dataset(), u.coords.to_dataset())
    assert "streamfunction" not in result
    xr.testing.assert_equal(u, saved)
    vel = reconstruct(result)
    err = np.sqrt(((vel.u-u)**2+(vel.v-v)**2).where(wet).sum()/(u*u+v*v).where(wet).sum())
    np.testing.assert_allclose(diag.vel_error, err, rtol=1e-13)
    from xinvert.finitediffs import FiniteDiff
    valid = wet & wet.shift(y=1, fill_value=False) & wet.shift(y=-1, fill_value=False)
    for off in (-1, 1):
        valid = valid & (wet.roll(x=off, roll_coords=False) if periodic else wet.shift(x=off, fill_value=False))
    fd = FiniteDiff({"Y": "y", "X": "x"}, BCs={"Y": "extend", "X": "periodic" if periodic else "extend"}, coords="cartesian")
    div = fd.divg((result.urot.fillna(0), result.vrot.fillna(0)), ["X", "Y"]).where(valid)
    np.testing.assert_allclose(diag.div_error, np.sqrt((div**2).mean()))
    assert diag.attrs["grid"] == "arakawa-a"


@pytest.mark.parametrize("periodic", [False, True])
def test_face_output_uses_physical_coordinates(periodic):
    u, v, wet = inputs(periodic)
    uf, vf, _, _ = faces(u, v, wet, periodic)
    opts = {'BCs': ["fixed", "periodic" if periodic else "fixed"], 'mxLoop': 6000, 'tolerance': 1e-11, 'printInfo': False}
    direct = project_vector(uf, vf, wet, ["y", "x"], coords="cartesian",
                             periodic_x=periodic, iParams=opts)
    result = decompose(uf, vf, ["y", "x"], coords="cartesian", method="forget",
                        grid="arakawa-c", mask=wet, iParams=opts, return_potentials=True)
    assert result.udiv.dims == ("y", "x_face")
    assert result.vdiv.dims == ("y_face", "x")
    assert result.velocity_potential.dims == ("y", "x")
    np.testing.assert_array_equal(result.x_face, u.x.values+.5)
    np.testing.assert_array_equal(result.y_face, u.y.values+.5)
    np.testing.assert_array_equal(result.udiv, direct.x_gradient)
    np.testing.assert_array_equal(result.vrot, direct.y_solenoidal)
    vel = reconstruct(result)
    assert vel.u.dims == ("y", "x_face") and vel.v.dims == ("y_face", "x")
    np.testing.assert_allclose(vel.u.values, (direct.x_gradient+direct.x_solenoidal).values, equal_nan=True)
    assert result.udiv.attrs["location"] == "east_face"


def test_sor_dispatch_is_unchanged():
    u, v, wet = inputs(False)
    result = decompose(u, v, ["y", "x"], coords="cartesian", method="sor", mask=wet,
                       iParams={'mxLoop': 3000, 'tolerance': 1e-9, 'printInfo': False})
    assert result.udiv.dims == u.dims
    assert "input_interpolation" not in result.attrs


@pytest.mark.parametrize("grid", ["arakawa-a", "arakawa-c"])
def test_forget_preserves_input_axis_order(grid):
    u, v, wet = inputs(True)
    if grid == "arakawa-c":
        u, v, _, _ = faces(u, v, wet, True)
    opts = {"BCs": ["fixed", "periodic"], "mxLoop": 6000,
            "tolerance": 1e-11, "printInfo": False}
    result = decompose(u.transpose("x", "y"), v.transpose("x", "y"),
                       ["y", "x"], coords="cartesian", method="forget",
                       grid=grid, mask=wet, iParams=opts, return_potentials=True)
    assert result.velocity_potential.dims == ("x", "y")
    assert result.udiv.dims == (("x_face", "y") if grid == "arakawa-c" else ("x", "y"))
    assert result.vdiv.dims == (("x", "y_face") if grid == "arakawa-c" else ("x", "y"))


@pytest.mark.parametrize("grid", ["arakawa-a", "arakawa-c"])
def test_forget_gpu_adapter_matches_cpu(grid):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA required")
    u, v, wet = inputs(True)
    if grid == "arakawa-c":
        u, v, _, _ = faces(u, v, wet, True)
    opts = {"BCs": ["fixed", "periodic"], "mxLoop": 6000,
            "tolerance": 1e-11, "printInfo": False}
    cpu = decompose(u, v, ["y", "x"], coords="cartesian", method="forget",
                    grid=grid, mask=wet, iParams=opts)
    gpu = decompose(u, v, ["y", "x"], coords="cartesian", method="forget",
                    grid=grid, mask=wet, iParams={**opts, "architect": "gpu"})
    for name in cpu:
        assert cpu[name].dims == gpu[name].dims
        np.testing.assert_allclose(cpu[name], gpu[name], rtol=1e-6, atol=1e-9, equal_nan=True)
