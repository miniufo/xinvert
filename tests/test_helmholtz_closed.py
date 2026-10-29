"""Independent sparse oracles and API contracts for compact closed Li."""

import numba
import numpy as np
import pytest
import xarray as xr
from scipy.sparse import csr_matrix, diags

from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import decompose, reconstruct, streamfunction_potential


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = numba.get_num_threads()
    numba.set_num_threads(min(2, previous))
    yield
    numba.set_num_threads(previous)


def problem(px=True, descending=False):
    wet = np.ones((9, 14), bool)
    wet[3:5, 4:6] = False
    wet[6, 10] = False
    lat = np.linspace(-65, 65, 9)
    if descending:
        lat = lat[::-1]
    sy = 1 / (lat[1] - lat[0])
    sx = 1 / (np.cos(np.deg2rad(lat)) * 2)
    sxn = 1 / (np.cos(np.deg2rad(0.5 * (lat[:-1] + lat[1:]))) * 2)
    wc = np.cos(np.deg2rad(lat))
    wn = np.cos(np.deg2rad(0.5 * (lat[:-1] + lat[1:])))
    op = li._ClosedGrid(wet, sy, sx, sxn, px, wc, wn)
    rows = []
    cols = []
    values = []
    weights = np.empty(op.data_size)
    for j, i in np.ndindex(wet.shape):
        east = (i + 1) % op.vm.shape[1]
        if op.ui[j, i] >= 0:
            k = op.ui[j, i]
            indices = [
                op.vm[j, east],
                op.vm[j + 1, east],
                op.npsi + op.cm[j, i],
                op.npsi + op.cm[j, (i + 1) % wet.shape[1]],
            ]
            coeff = [sy, -sy, -sx[j], sx[j]]
            weights[k] = wc[j]
            rows.extend([k] * 4)
            cols.extend(indices)
            values.extend(coeff)
        if op.vi[j, i] >= 0:
            k = op.vi[j, i]
            indices = [
                op.vm[j + 1, i],
                op.vm[j + 1, east],
                op.npsi + op.cm[j, i],
                op.npsi + op.cm[j + 1, i],
            ]
            coeff = [-sxn[j], sxn[j], -sy, sy]
            weights[k] = wn[j]
            rows.extend([k] * 4)
            cols.extend(indices)
            values.extend(coeff)
    A = csr_matrix((values, (rows, cols)), shape=(op.data_size, op.size))
    A.eliminate_zeros()
    return op, A, weights, lat


@pytest.mark.parametrize("px", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_compiled_operator_and_preconditioner(px, descending):
    op, A, w, _ = problem(px, descending)
    rng = np.random.default_rng(3)
    x = rng.normal(size=op.size)
    y = rng.normal(size=op.data_size)
    np.testing.assert_allclose(op.forward(x), A @ x, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(op.adjoint(y), A.T @ y, rtol=2e-14, atol=2e-14)
    L = 0.7
    alpha = 1e-5
    mult = np.r_[op.counts, np.ones(op.nchi)]
    B = diags(np.sqrt(w)) @ A * L
    diagonal = np.asarray(B.power(2).sum(axis=0)).ravel() + alpha * mult
    scale = 1 / np.sqrt(diagonal)
    B = B @ diags(scale)
    p = np.sqrt(alpha * mult) * scale
    op.prepare(L, alpha)
    np.testing.assert_allclose(op.scale, scale, rtol=1e-14)
    augmented = np.r_[y, rng.normal(size=op.size)]
    np.testing.assert_allclose(
        op.matvec(x), np.r_[B @ x, p * x], rtol=3e-14, atol=3e-14
    )
    np.testing.assert_allclose(
        op.rmatvec(augmented),
        B.T @ y + p * augmented[op.data_size :],
        rtol=3e-14,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        np.dot(op.matvec(x), augmented),
        np.dot(x, op.rmatvec(augmented)),
        rtol=1e-13,
        atol=1e-13,
    )


def test_independent_coastal_constants():
    op, _, _, _ = problem(True)
    north = np.unique(op.vm[0])
    south = np.unique(op.vm[-1])
    island1 = np.unique(
        op.vm[3:6, 4:7][[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]]
    )
    island2 = np.unique(op.vm[6:8, 10:12])
    assert len(north) == len(south) == len(island1) == len(island2) == 1
    assert len({int(north[0]), int(south[0]), int(island1[0]), int(island2[0])}) == 4


def test_objective_gradient_includes_weights_and_multiplicity():
    op, A, w, _ = problem()
    rng = np.random.default_rng(4)
    x = rng.normal(size=op.size)
    data = rng.normal(size=op.data_size)
    direction = rng.normal(size=op.size)
    direction /= np.linalg.norm(direction)
    alpha = 0.03
    residual = A @ x - data
    grad = op.add_regularization_gradient(
        op.adjoint(op.weight_faces(residual)), x, alpha
    )

    def objective(z):
        r = A @ z - data
        return 0.5 * np.dot(w * r, r) + 0.5 * alpha * op.regularization_norm(z)

    delta = 1e-4
    numeric = (objective(x + delta * direction) - objective(x - delta * direction)) / (
        2 * delta
    )
    np.testing.assert_allclose(numeric, np.dot(grad, direction), rtol=1e-7, atol=1e-8)


def test_float32_sampling_uses_float64_face_average():
    op, _, _, _ = problem()
    u = np.random.default_rng(2).normal(size=op.shape).astype("float32")
    v = u * 0.17
    packed = op.sample_velocity(u, v)
    for j, i in np.ndindex(op.shape):
        k = op.ui[j, i]
        if k >= 0:
            assert packed[k] == 0.5 * (
                float(u[j, i]) + float(u[j, (i + 1) % op.shape[1]])
            )
        k = op.vi[j, i]
        if k >= 0:
            assert packed[k] == 0.5 * (float(v[j, i]) + float(v[j + 1, i]))


def test_public_float32_retention_does_not_reduce_computation_precision():
    u, v, mask = fields()
    u = u.astype("float32")
    v = v.astype("float32")
    opts = {"BCs": ["closed", "periodic"], "mxLoop": 500, "tolerance": 1e-9}
    a, ad = streamfunction_potential(
        u, v, mask=mask, iParams=opts, return_diagnostics=True
    )
    b, bd = streamfunction_potential(
        u.astype("float64"),
        v.astype("float64"),
        mask=mask,
        iParams=opts,
        return_diagnostics=True,
    )
    for name in ("streamfunction", "velocity_potential"):
        np.testing.assert_array_equal(a[name].values, b[name].values)
        assert a[name].dtype == np.dtype("float64")
    for name in ad:
        np.testing.assert_array_equal(ad[name].values, bd[name].values)


def fields():
    rng = np.random.default_rng(5)
    u = xr.DataArray(
        rng.normal(size=(10, 18)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-70, 70, 10), "lon": np.arange(18) * 20.0},
    )
    v = u.roll(lon=3) * 0.3
    mask = xr.ones_like(u, dtype=bool)
    mask[3:6, 5:8] = False
    return u, v, mask


@pytest.mark.parametrize("solver", ["lsmr", "lsqr"])
def test_public_closed_parallel_batch_initial(solver):
    u, v, mask = fields()
    saved = u.copy(deep=True)
    opts = {
        "BCs": ["closed", "periodic"],
        "solver": solver,
        "tolerance": 1e-10,
        "mxLoop": 1500,
        "on_nonconvergence": "raise",
    }
    result, diag = streamfunction_potential(
        u, v, mask=mask, iParams=opts, return_diagnostics=True
    )
    parallel, pdiag = streamfunction_potential(
        u,
        v,
        mask=mask,
        iParams=dict(opts, architect="cpu-mc", threads=2),
        return_diagnostics=True,
    )
    for name in ["streamfunction", "velocity_potential"]:
        xr.testing.assert_equal(result[name], parallel[name])
    assert bool(diag.converged) and bool(pdiag.converged)
    assert set(diag) == {"vel_error", "div_error", "converged", "iterations"}
    warm = streamfunction_potential(u, v, mask=mask, iParams=opts, initial=result)
    np.testing.assert_allclose(
        warm.streamfunction, result.streamfunction, rtol=3e-4, atol=0.05, equal_nan=True
    )
    batch = streamfunction_potential(
        xr.concat([u, u * 0.5], dim="time").transpose("lon", "time", "lat"),
        xr.concat([v, v * 0.5], dim="time").transpose("lon", "time", "lat"),
        dims=["lat", "lon"],
        mask=mask,
        iParams=opts,
    )
    assert batch.streamfunction.dims == ("lon", "time", "lat")
    xr.testing.assert_equal(u, saved)
    velocity = decompose(u, v, mask=mask, iParams=opts)
    assert all(
        name in velocity
        for name in [
            "urot",
            "vrot",
            "udiv",
            "vdiv",
        ]
    )
    assert reconstruct(result).u.shape == u.shape


def test_thread_state_restored_on_error_and_invalid_modes():
    u, v, mask = fields()
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        with pytest.raises(RuntimeError, match="stopped"):
            streamfunction_potential(
                u,
                v,
                mask=mask,
                iParams={
                    "BCs": ["closed", "periodic"],
                    "architect": "cpu-mc",
                    "threads": 2,
                    "mxLoop": 1,
                    "on_nonconvergence": "raise",
                },
            )
        assert numba.get_num_threads() == 1
    finally:
        numba.set_num_threads(previous)
    with pytest.raises(NotImplementedError):
        streamfunction_potential(u, v, iParams={"architect": "cpu-mc"})
    with pytest.raises(ValueError, match="closed Li"):
        streamfunction_potential(u, v, iParams={"BCs": ["open", "closed"]})
    with pytest.raises(ValueError, match="threads"):
        streamfunction_potential(
            u, v, iParams={"BCs": ["closed", "periodic"], "threads": 0}
        )


def test_all_dry_and_isolated_cells():
    u, v, mask = fields()
    dry, diag = streamfunction_potential(
        u,
        v,
        mask=xr.zeros_like(mask),
        iParams={"BCs": ["closed", "closed"]},
        return_diagnostics=True,
    )
    assert bool(diag.converged) and bool(dry.streamfunction.isnull().all())
    isolated = xr.zeros_like(mask)
    isolated[4, 7] = True
    field, diag = streamfunction_potential(
        u,
        v,
        mask=isolated,
        iParams={"BCs": ["closed", "closed"]},
        return_diagnostics=True,
    )
    assert bool(diag.converged) and np.isnan(diag.div_error)
    assert float(field.velocity_potential[4, 7]) == 0


@pytest.mark.parametrize("px", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_fused_returned_velocity_matches_masked_numpy(px, descending, missing):
    op, _, _, _ = problem(px, descending)
    rng = np.random.default_rng(23)
    potentials = [rng.normal(size=op.shape)[:, ::-1] for _ in range(2)]
    if missing:
        for a in potentials:
            a[~op.active] = np.nan
    derivatives = []
    for axis in (0, 1):
        edge = op.active & np.roll(op.active, -1, axis=axis)
        if not op.periodic[axis]:
            end = [slice(None), slice(None)]
            end[axis] = -1
            edge[tuple(end)] = False
        count = np.maximum(edge.astype(np.uint8) + np.roll(edge, 1, axis=axis), 1)
        scale = op.sy if axis == 0 else op.sx
        components = []
        for a in potentials:
            flux = (np.roll(a, -1, axis=axis) - a) * edge
            components.append((flux + np.roll(flux, 1, axis=axis)) / count * scale)
        derivatives.append(components)
    expected = np.stack(
        (-derivatives[0][0] + derivatives[1][1], derivatives[1][0] + derivatives[0][1])
    )
    actual = li._returned_velocity(op, potentials)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
