"""Precision contracts: inferred inputs, explicit override and typed Li vectors."""

import numba
import numpy as np
import pytest
import xarray as xr

from xinvert import record_iteration, invert_Poisson
from xinvert.apps import _solve_dtype
from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import decompose

from .test_helmholtz_closed import problem
from .test_helmholtz_output_grids import inputs


@pytest.fixture(autouse=True)
def threads():
    old = numba.get_num_threads()
    numba.set_num_threads(min(old, 2))
    yield
    numba.set_num_threads(old)


@pytest.mark.parametrize(
    "input_dtype,expected",
    [
        ("float16", "float64"),
        ("int32", "float64"),
        ("float32", "float32"),
        ("float64", "float64"),
    ],
)
@pytest.mark.parametrize("setting", [None, "float32", "float64"])
@pytest.mark.parametrize("architect", ["cpu", "cpu-mc"])
def test_scalar_precision(input_dtype, expected, setting, architect):
    f = xr.DataArray(
        np.ones((8, 10), dtype=input_dtype),
        dims=["y", "x"],
        coords={"y": np.arange(8.0), "x": np.arange(10.0)},
    )
    result = invert_Poisson(
        f,
        ["y", "x"],
        coords="cartesian",
        icbc=xr.zeros_like(f, dtype="float64"),
        iParams={
            "dtype": setting,
            "architect": architect,
            "threads": 2,
            "printInfo": False,
            "mxLoop": 1000,
            "tolerance": 1e-5,
        },
    )
    assert result.dtype == np.dtype(setting or expected)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("bad", ["complex64", "complex128", "U3"])
def test_reject_nonreal(bad):
    with pytest.raises(TypeError, match="real numeric"):
        _solve_dtype({"dtype": "float32"}, np.dtype(bad))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("method", ["sor", "li2006"])
def test_helmholtz_precision(dtype, method):
    u, v, wet = inputs(False)
    u, v = u.astype(dtype), v.astype(dtype)
    opts = {
        "BCs": ["fixed", "fixed"] if method == "li2006" else ["fixed", "fixed"],
        "tolerance": 1e-5,
        "mxLoop": 4000,
        "printInfo": False,
        "threads": 2,
    }
    r, d = decompose(
        u,
        v,
        ["y", "x"],
        coords="cartesian",
        method=method,
        mask=wet,
        iParams=opts,
        return_potentials=True,
        return_diagnostics=True,
    )
    assert all(a.dtype == np.dtype(dtype) for a in r.data_vars.values())
    assert bool(d.converged)


@pytest.mark.parametrize("method", ["sor", "li2006"])
def test_mixed_and_override(method):
    u, v, wet = inputs(False)
    opts = {
        "BCs": ["fixed", "fixed"] if method == "li2006" else ["fixed", "fixed"],
        "tolerance": 1e-5,
        "mxLoop": 4000,
        "printInfo": False,
        "threads": 2,
    }
    for explicit, expected in [(None, "float64"), ("float32", "float32")]:
        r = decompose(
            u.astype("float32"),
            v.astype("float64"),
            ["y", "x"],
            coords="cartesian",
            method=method,
            mask=wet,
            iParams={**opts, "dtype": explicit},
        )
        assert all(a.dtype == np.dtype(expected) for a in r.data_vars.values())


def test_lazy_inference():
    f = xr.DataArray(np.ones((8, 10), dtype="float32"), dims=["y", "x"]).chunk(
        {"y": 8, "x": 10}
    )
    r = invert_Poisson(f, ["y", "x"], coords="cartesian", iParams={"printInfo": False})
    assert r.chunks is not None and r.dtype == np.dtype("float32")
    assert r.compute().dtype == np.dtype("float32")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_animation_precision(dtype, tmp_path):
    f = xr.DataArray(
        np.ones((8, 10), dtype=dtype),
        dims=["y", "x"],
        coords={"y": np.arange(8.0), "x": np.arange(10.0)},
    )
    path = tmp_path / "frames.nc"
    record_iteration(
        "Poisson",
        f,
        ["y", "x"],
        coords="cartesian",
        iter_per_frame=2,
        max_frames=2,
        output=path,
    )
    with xr.open_dataset(path) as ds:
        assert ds.inverted.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("solver", ["lsmr", "lsqr"])
def test_closed_operator_dense_oracle(dtype, solver):
    ref, A, weights, _ = problem(False)
    op = li._ClosedGrid(
        ref.active,
        ref.sy,
        ref.sx[:, 0],
        ref.sx_north,
        False,
        ref.wu**2,
        ref.wv**2,
        dtype=dtype,
    )
    rng = np.random.default_rng(19)
    data = rng.normal(size=op.data_size).astype(dtype)
    alpha = 0.03
    z, status = li._solve_closed_grid(
        op,
        data,
        1.0,
        alpha,
        solver=solver,
        tolerance=2e-6 if dtype == "float32" else 1e-11,
        maxiter=3000,
    )
    M = np.r_[ref.counts, np.ones(ref.nchi)]
    dense = np.vstack(
        [np.sqrt(weights)[:, None] * A.toarray(), np.diag(np.sqrt(alpha * M))]
    )
    rhs = np.r_[np.sqrt(weights) * data, np.zeros(op.size)]
    exact = np.linalg.lstsq(dense, rhs, rcond=None)[0]
    assert z.dtype == np.dtype(dtype) and status["converged"]
    np.testing.assert_allclose(
        z,
        exact,
        rtol=2e-3 if dtype == "float32" else 2e-7,
        atol=2e-4 if dtype == "float32" else 2e-8,
    )
    op.prepare(1.0, alpha)
    assert op.scale.dtype == op.penalty.dtype == np.dtype(dtype)
    assert all(a.dtype == np.dtype(dtype) for a in op.coeff)
    assert op.matvec(z).dtype == op.rmatvec(op.matvec(z)).dtype == np.dtype(dtype)


@pytest.mark.skipif(
    __import__("os").environ.get("XINVERT_TEST_LI_GPU") != "1",
    reason="requires CUDA and CuPy",
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gpu_precision_and_dense_oracle(dtype):
    from xinvert.helmholtz._li2006_gpu import _GPUOperator, solve_closed_gpu

    ref, A, weights, _ = problem(False)
    op = li._ClosedGrid(
        ref.active,
        ref.sy,
        ref.sx[:, 0],
        ref.sx_north,
        False,
        ref.wu**2,
        ref.wv**2,
        dtype=dtype,
    )
    op.prepare(1.0, 0.03)
    gpu = _GPUOperator(op)
    cp = gpu.cp
    rng = np.random.default_rng(13)
    z = rng.normal(size=op.size).astype(dtype)
    data = rng.normal(size=op.data_size).astype(dtype)
    result = gpu.matvec(cp.asarray(z))
    assert result.dtype == np.dtype(dtype)
    assert gpu.rmatvec(result).dtype == np.dtype(dtype)
    assert gpu.scale.dtype == gpu.penalty.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        cp.asnumpy(result),
        op.matvec(z),
        rtol=3e-6 if dtype == "float32" else 2e-13,
        atol=3e-6 if dtype == "float32" else 2e-13,
    )
    del gpu, result
    actual, status = solve_closed_gpu(
        op,
        data,
        1.0,
        0.03,
        maxiter=3000,
        tolerance=2e-6 if dtype == "float32" else 1e-11,
        conlim=1e8,
    )
    M = np.r_[ref.counts, np.ones(ref.nchi)]
    dense = np.vstack(
        [np.sqrt(weights)[:, None] * A.toarray(), np.diag(np.sqrt(0.03 * M))]
    )
    rhs = np.r_[np.sqrt(weights) * data, np.zeros(op.size)]
    exact = np.linalg.lstsq(dense, rhs, rcond=None)[0]
    assert actual.dtype == np.dtype(dtype) and status["converged"]
    np.testing.assert_allclose(
        actual,
        exact,
        rtol=2e-3 if dtype == "float32" else 2e-7,
        atol=2e-4 if dtype == "float32" else 2e-8,
    )


@pytest.mark.parametrize("backend", ["cpu", "cpu-mc"])
def test_float32_closed_backend(backend):
    u, v, wet = inputs(False)
    r, d = decompose(
        u.astype("float32"),
        v.astype("float32"),
        ["y", "x"],
        coords="cartesian",
        mask=wet,
        iParams={
            "BCs": ["fixed", "fixed"],
            "architect": backend,
            "threads": 2,
            "tolerance": 1e-5,
            "mxLoop": 3000,
        },
        return_diagnostics=True,
    )
    assert all(a.dtype == np.dtype("float32") for a in r.data_vars.values())
    assert bool(d.converged)


@pytest.mark.parametrize("solver", ["lsmr", "lsqr"])
def test_open_float32_solver_vectors(monkeypatch, solver):
    original = li._Operator.forward
    seen = []

    def observe(self, x):
        seen.append(x.dtype)
        return original(self, x)

    monkeypatch.setattr(li._Operator, "forward", observe)
    u, v, wet = inputs(False)
    psi, chi = li.helmholtz(
        u.astype("float32"),
        v.astype("float32"),
        ["y", "x"],
        coords="cartesian",
        mask=wet,
        iParams={"BCs": ["open", "open"], "solver": solver, "tolerance": 1e-5, "mxLoop": 3000},
    )
    assert psi.dtype == chi.dtype == np.dtype("float32")
    assert seen and all(d == np.dtype("float32") for d in seen)


@pytest.mark.skipif(
    __import__("os").environ.get("XINVERT_TEST_LI_GPU") != "1",
    reason="requires CUDA and CuPy",
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_public_gpu_infers_precision(dtype):
    u, v, wet = inputs(False)
    u, v = u.astype(dtype), v.astype(dtype)
    opts = {
        "BCs": ["fixed", "fixed"],
        "tolerance": 1e-5 if dtype == "float32" else 1e-9,
        "mxLoop": 3000,
        "threads": 2,
    }
    cpu = decompose(u, v, ["y", "x"], coords="cartesian", mask=wet, iParams=opts)
    gpu, d = decompose(
        u,
        v,
        ["y", "x"],
        coords="cartesian",
        mask=wet,
        iParams={**opts, "architect": "gpu"},
        return_diagnostics=True,
    )
    assert bool(d.converged)
    for name in gpu:
        assert gpu[name].dtype == np.dtype(dtype)
        np.testing.assert_allclose(
            gpu[name],
            cpu[name],
            atol=3e-3 if dtype == "float32" else 2e-6,
            rtol=3e-3 if dtype == "float32" else 2e-6,
            equal_nan=True,
        )


@pytest.mark.skipif(
    __import__("os").environ.get("XINVERT_TEST_LI_GPU") != "1",
    reason="requires CUDA and CuPy",
)
@pytest.mark.parametrize(
    "input_dtype,setting,expected",
    [
        ("float32", None, "float32"),
        ("float16", None, "float64"),
        ("float64", "float32", "float32"),
    ],
)
def test_scalar_gpu_precision(input_dtype, setting, expected):
    f = xr.DataArray(
        np.ones((8, 10), dtype=input_dtype),
        dims=["y", "x"],
        coords={"y": np.arange(8.0), "x": np.arange(10.0)},
    )
    r = invert_Poisson(
        f,
        ["y", "x"],
        coords="cartesian",
        iParams={
            "architect": "gpu",
            "dtype": setting,
            "printInfo": False,
            "mxLoop": 1000,
            "tolerance": 1e-5,
        },
    )
    assert r.dtype == np.dtype(expected)
    assert np.isfinite(r).all()
