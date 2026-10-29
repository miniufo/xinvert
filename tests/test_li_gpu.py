"""CUDA equivalence against the independent CPU/CSR operator oracle."""

import numba
import numpy as np
import pytest
import xarray as xr

from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import streamfunction_potential
from xinvert.helmholtz._li2006_gpu import _GPUOperator

from .test_helmholtz_closed import fields, problem

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() == 0:
    pytest.skip("CUDA device required", allow_module_level=True)


@pytest.fixture(autouse=True)
def small_cpu_pool():
    saved = numba.get_num_threads()
    numba.set_num_threads(min(2, saved))
    yield
    numba.set_num_threads(saved)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 1e-5])
def test_augmented_operator_and_adjoint(periodic, descending, alpha):
    op, _, _, _ = problem(periodic, descending)
    op.prepare(0.7, alpha)
    gpu = _GPUOperator(op)
    rng = np.random.default_rng(3)
    x = rng.normal(size=op.size)
    y = rng.normal(size=op.size + op.data_size)
    ax = cp.asnumpy(gpu.matvec(cp.asarray(x)))
    aty = cp.asnumpy(gpu.rmatvec(cp.asarray(y)))
    np.testing.assert_allclose(ax, op.matvec(x), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(aty, op.rmatvec(y), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.dot(ax, y), np.dot(x, aty), rtol=1e-12, atol=1e-12)


def test_large_coastal_group_uses_multiple_reduction_blocks():
    wet = np.ones((6, 800), bool)
    op = li._ClosedGrid(wet, 1.0, np.ones(6), np.ones(5), True)
    op.prepare(0.7, 1e-5)
    gpu = _GPUOperator(op)
    assert len(gpu.parts) > len(gpu.ids)
    y = np.random.default_rng(12).normal(size=op.size + op.data_size)
    np.testing.assert_allclose(
        cp.asnumpy(gpu.rmatvec(cp.asarray(y))), op.rmatvec(y), rtol=2e-12, atol=2e-12
    )


def test_64_bit_maps_and_memory_guard():
    op, _, _, _ = problem()
    for name in ["vm", "cm", "ui", "vi", "vertices", "coast_ids"]:
        setattr(op, name, getattr(op, name).astype(np.int64))
    op.prepare(0.7, 1e-5)
    gpu = _GPUOperator(op)
    x = np.random.default_rng(4).normal(size=op.size)
    y = np.random.default_rng(5).normal(size=op.size + op.data_size)
    np.testing.assert_allclose(
        cp.asnumpy(gpu.matvec(cp.asarray(x))), op.matvec(x), rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        cp.asnumpy(gpu.rmatvec(cp.asarray(y))), op.rmatvec(y), rtol=2e-13, atol=2e-13
    )
    op.size = 10**12
    with pytest.raises(MemoryError, match="precision is not changed"):
        _GPUOperator(op)


def test_public_solution_diagnostics_and_warm_start():
    u, v, mask = fields()
    opts = {
        "BCs": ["closed", "periodic"],
        "tolerance": 1e-11,
        "mxLoop": 3000,
        "on_nonconvergence": "raise",
    }
    cpu, cd = streamfunction_potential(
        u, v, mask=mask, iParams=opts, return_diagnostics=True
    )
    gpu, gd = streamfunction_potential(
        u, v, mask=mask, iParams={**opts, "architect": "gpu"}, return_diagnostics=True
    )
    assert bool(cd.converged) and bool(gd.converged)
    for name in cpu:
        np.testing.assert_allclose(gpu[name], cpu[name], rtol=2e-6, atol=1e-3)
    for name in ["vel_error", "div_error"]:
        np.testing.assert_allclose(gd[name], cd[name], rtol=1e-7, atol=1e-10)
    warm = streamfunction_potential(
        u, v, mask=mask, initial=cpu, iParams={**opts, "architect": "gpu"}
    )
    for name in warm:
        np.testing.assert_allclose(warm[name], cpu[name], rtol=2e-6, atol=1e-3)


def test_zero_and_isolated():
    u, v, mask = fields()
    mask = xr.zeros_like(mask)
    mask[4, 7] = True
    result, diag = streamfunction_potential(
        u * 0,
        v * 0,
        mask=mask,
        iParams={"BCs": ["closed", "closed"], "architect": "gpu"},
        return_diagnostics=True,
    )
    assert bool(diag.converged)
    assert float(result.velocity_potential[4, 7]) == 0.0


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("steps", [1, 7, 1000])
@pytest.mark.parametrize("overwrite", [False, True])
def test_fused_lsmr_matches_cupy_stopping_and_solution(warm, steps, overwrite):
    from cupyx.scipy.sparse.linalg import LinearOperator, lsmr

    from xinvert.helmholtz._li2006_gpu import _lsmr

    rng = np.random.default_rng(73)
    a = cp.asarray(rng.normal(size=(57, 31)))
    b = cp.asarray(rng.normal(size=57))
    initial = cp.asarray(rng.normal(size=31)) if warm else None
    operator = LinearOperator(
        a.shape, matvec=lambda x: a @ x, rmatvec=lambda x: a.T @ x, dtype=cp.float64
    )
    opts = {"x0": initial, "atol": 1e-12, "btol": 1e-12, "maxiter": steps}
    expected = lsmr(operator, b, **opts)
    actual = _lsmr(operator, b, overwrite_b=overwrite, **opts)
    assert actual[1:3] == expected[1:3]
    np.testing.assert_allclose(
        cp.asnumpy(actual[0]), cp.asnumpy(expected[0]), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(actual[3:], expected[3:], rtol=2e-11, atol=2e-12)


def test_fused_lsmr_zero_rhs_and_exact_warm_start():
    from cupyx.scipy.sparse.linalg import LinearOperator

    from xinvert.helmholtz._li2006_gpu import _lsmr

    operator = LinearOperator(
        (4, 4), matvec=lambda x: x.copy(), rmatvec=lambda x: x.copy(), dtype=cp.float64
    )
    for b, x0 in [(cp.zeros(4), None), (cp.ones(4), cp.ones(4))]:
        result = _lsmr(operator, b, x0=x0)
        assert result[1:3] == (0, 0)
        np.testing.assert_array_equal(cp.asnumpy(result[0]), cp.asnumpy(b))
