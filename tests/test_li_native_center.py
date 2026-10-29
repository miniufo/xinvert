"""Native C-grid differentiation followed by same-grid velocity interpolation."""
import numba
import numpy as np
import pytest
import xarray as xr

from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import decompose, reconstruct, streamfunction_potential

from .test_helmholtz_closed import fields, problem


def center_reference(op, face):
    """Independent scalar face-map gather, including one-sided coasts."""
    answer = np.zeros((2, *op.shape))
    nx = op.shape[1]
    for j, i in np.ndindex(op.shape):
        if not op.active[j, i]:
            continue
        east = [op.ui[j, i]]
        if i > 0 or op.periodic[1]:
            east.append(op.ui[j, (i - 1) % nx])
        north = [op.vi[j, i]]
        if j > 0:
            north.append(op.vi[j - 1, i])
        for axis, candidates in enumerate([east, north]):
            ids = [int(k) for k in candidates if k >= 0]
            if ids:
                answer[axis, j, i] = sum(face[k] for k in ids) / len(ids)
    return answer


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_closed_native_components_against_independent_csr(periodic, descending):
    op, matrix, _, _ = problem(periodic, descending)
    z = np.random.default_rng(981).normal(size=op.size)
    zr = z.copy()
    zr[op.npsi:] = 0
    zd = z.copy()
    zd[:op.npsi] = 0
    expected = np.concatenate([
        center_reference(op, matrix @ zr), center_reference(op, matrix @ zd),
    ])
    actual = li._native_velocity_to_centers(op, z, components=True)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    target = np.empty_like(actual)
    assert li._native_velocity_to_centers(op, z, components=True, out=target) is target
    np.testing.assert_array_equal(target, actual)
    total = li._native_velocity_to_centers(op, z)
    np.testing.assert_array_equal(total, actual[:2] + actual[2:])


@pytest.mark.parametrize("periodic", [False, True])
def test_isolated_cell_and_thread_independence(periodic):
    wet = np.zeros((9, 12), bool)
    wet[1:4, 1:5] = True
    wet[7, 9] = True
    op = li._ClosedGrid(wet, .3, np.linspace(.2, .4, 9), np.linspace(.2, .4, 8), periodic)
    z = np.random.default_rng(45).normal(size=op.size)
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = li._native_velocity_to_centers(op, z, components=True)
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        two = li._native_velocity_to_centers(op, z, components=True)
    finally:
        numba.set_num_threads(old)
    np.testing.assert_array_equal(one, two)
    np.testing.assert_array_equal(one[:, 7, 9], 0)
    np.testing.assert_array_equal(one[:, ~wet], 0)


@pytest.mark.parametrize("py,px", [(False, False), (False, True), (True, True)])
def test_general_native_operator_mask_and_periodicity(py, px):
    wet = np.ones((9, 12), bool)
    wet[3:5, 6] = False
    op = li._Operator(wet, -.7, np.linspace(-.4, -.2, 9),
                      ["periodic" if py else "open", "periodic" if px else "open"],
                      np.dtype("float64"))
    z = np.random.default_rng(29).normal(size=op.size)
    zr = z.copy()
    zr[op.npsi:] = 0
    zd = z.copy()
    zd[:op.npsi] = 0
    expected = []
    for zz in [zr, zd]:
        packed = op.forward(zz)
        out = np.zeros((2, *wet.shape))
        for axis, shape, valid in [(1, op.u_shape, op.uwet), (0, op.v_shape, op.vwet)]:
            face = (packed[:op.nu] if axis == 1 else packed[op.nu:]).reshape(shape)
            for j, i in np.ndindex(wet.shape):
                if not wet[j, i]:
                    continue
                neighbours = [(j, i)]
                if axis == 1 and (px or i > 0):
                    neighbours.append((j, (i - 1) % wet.shape[1]))
                if axis == 0 and (py or j > 0):
                    neighbours.append(((j - 1) % wet.shape[0], i))
                values = [face[y, x] for y, x in neighbours
                          if y < shape[0] and x < shape[1] and valid[y, x]]
                if values:
                    out[1-axis, j, i] = sum(values) / len(values)
        expected.extend(out)
    actual = li._native_velocity_to_centers(op, z, components=True)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("wavelength", [32, 16, 8, 4])
def test_vertex_averaging_is_not_used_to_generate_rotational_velocity(wavelength):
    n = 64
    op = li._Operator(np.ones((n, n), bool), 1., np.ones(n),
                      ["periodic", "periodic"], np.dtype("float64"))
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    q = 2*np.pi/wavelength
    psi = np.sin(q*xx)*np.sin(q*yy)
    z = np.r_[psi.ravel(), np.zeros(n*n)]
    native = li._native_velocity_to_centers(op, z)
    old = li._returned_velocity(op, op.potentials_to_centers(z))
    gain = np.sum(old*native)/np.sum(native*native)
    np.testing.assert_allclose(gain, np.cos(q/2)**2, rtol=1e-13)
    assert np.linalg.norm(native) > np.linalg.norm(old)


@pytest.mark.parametrize("closed", [False, True])
def test_public_components_coordinates_diagnostics_and_potential_compatibility(monkeypatch, closed):
    u, v, wet = fields()
    opts = {'BCs': ["closed" if closed else "open", "periodic"], 'mxLoop': 2000, 'tolerance': 1e-10, 'on_nonconvergence': "raise"}
    captured = []
    original = li._native_velocity_to_centers

    def observe(op, z, **kwargs):
        result = original(op, z, **kwargs)
        captured.append(result.copy())
        return result

    monkeypatch.setattr(li, "_native_velocity_to_centers", observe)
    result, diagnostic = decompose(u, v, mask=wet, iParams=opts,
                                   return_potentials=True, return_diagnostics=True)
    assert len(captured) == 1
    names = ["urot", "vrot", "udiv", "vdiv"]
    for j, name in enumerate(names):
        xr.testing.assert_equal(result[name].coords.to_dataset(), u.coords.to_dataset())
        np.testing.assert_allclose(result[name], np.where(wet, captured[0][j], np.nan))
        assert result[name].attrs["location"] == "cell_center"
    velocity = reconstruct(result)
    np.testing.assert_allclose(velocity.u, result.urot + result.udiv)
    np.testing.assert_allclose(velocity.v, result.vrot + result.vdiv)
    active = wet.values
    err = np.sqrt(np.sum(((velocity.u-u)**2+(velocity.v-v)**2).values[active])
                  / np.sum((u*u+v*v).values[active]))
    np.testing.assert_allclose(diagnostic.vel_error, err, rtol=2e-14)
    pots = streamfunction_potential(u, v, mask=wet, iParams=opts)
    np.testing.assert_array_equal(result.streamfunction, pots.streamfunction)
    np.testing.assert_array_equal(result.velocity_potential, pots.velocity_potential)
    assert "native C-grid" in result.attrs["reconstruction"]


def test_batch_axis_order_and_all_dry():
    u, v, wet = fields()
    uu = xr.concat([u, u*.5], dim="time").transpose("lon", "time", "lat")
    vv = xr.concat([v, v*.5], dim="time").transpose("lon", "time", "lat")
    opts = {'BCs': ["closed", "periodic"], 'mxLoop': 2000, 'tolerance': 1e-10}
    result = decompose(uu, vv, dims=["lat", "lon"], mask=wet, iParams=opts)
    for a in result.data_vars.values():
        assert a.dims == uu.dims
        np.testing.assert_allclose(a.isel(time=1), a.isel(time=0)*.5, atol=1e-12)
    dry, d = decompose(u, v, mask=xr.zeros_like(wet), iParams=opts, return_diagnostics=True)
    assert all(bool(a.isnull().all()) for a in dry.data_vars.values())
    assert bool(d.converged)


def test_gpu_public_native_velocity_matches_cpu():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA required")
    u, v, wet = fields()
    opts = {'BCs': ["closed", "periodic"], 'mxLoop': 3000, 'tolerance': 1e-11, 'on_nonconvergence': "raise"}
    cpu, cd = decompose(u, v, mask=wet, iParams=opts, return_diagnostics=True)
    gpu, gd = decompose(u, v, mask=wet, iParams={**opts, "architect": "gpu"},
                        return_diagnostics=True)
    assert bool(cd.converged) and bool(gd.converged)
    for name in cpu:
        np.testing.assert_allclose(cpu[name], gpu[name], rtol=2e-6, atol=1e-8, equal_nan=True)
    np.testing.assert_allclose(cd.vel_error, gd.vel_error,
                               rtol=1e-7, atol=1e-10)
