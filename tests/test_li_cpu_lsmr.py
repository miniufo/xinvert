"""Independent SciPy oracle and stable reductions for closed CPU LSMR."""

import numba
import numpy as np
import pytest
from scipy.linalg import norm
from scipy.sparse.linalg import lsmr

from xinvert.helmholtz import _li2006 as li


@pytest.fixture(autouse=True)
def restore_mask():
    saved = numba.get_num_threads()
    numba.set_num_threads(min(4, saved))
    yield
    numba.set_num_threads(saved)


class DenseClosed:
    def __init__(self, a):
        self.a = a
        self.size = a.shape[1]
        self.data_size = a.shape[0] - a.shape[1]

    def matvec(self, x, out=None):
        if out is None:
            return self.a @ x
        np.matmul(self.a, x, out=out)
        return out

    def rmatvec(self, x, out=None):
        if out is None:
            return self.a.T @ x
        np.matmul(self.a.T, x, out=out)
        return out


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("steps", [7, 300])
def test_lsmr_matches_independent_scipy(warm, overwrite, steps):
    rng = np.random.default_rng(92)
    a = rng.normal(size=(57, 31))
    b = rng.normal(size=57)
    saved = b.copy()
    x0 = rng.normal(size=31) if warm else None
    opts = {"atol": 1e-12, "btol": 1e-12, "conlim": 1e12, "maxiter": steps, "x0": x0}
    reference = lsmr(a, b, **opts)
    actual = li._lsmr_closed_cpu(DenseClosed(a), b, overwrite_b=overwrite, **opts)
    assert actual[1] == reference[1]
    if steps == 7:
        assert actual[2] == reference[2]
    np.testing.assert_allclose(actual[0], reference[0], rtol=2e-10, atol=2e-11)
    # Residual and normal-residual estimates agree; condition estimates can
    # depend on small rounding differences in the bidiagonalization trajectory.
    np.testing.assert_allclose(actual[3], reference[3], rtol=2e-10, atol=2e-11)
    if steps == 7:
        np.testing.assert_allclose(actual[4], reference[4], rtol=2e-8, atol=2e-11)
    else:
        # Changed summation order can move the final stop by one iteration.
        # Check the same normal-residual criterion and an independent gradient.
        assert actual[4] <= opts["atol"] * actual[5] * actual[3]
        residual = saved - a @ actual[0]
        gradient = np.linalg.norm(a.T @ residual)
        assert gradient <= 5 * opts["atol"] * actual[5] * np.linalg.norm(residual)
    if not overwrite:
        np.testing.assert_array_equal(saved, b)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(20),
        np.empty(0),
        np.array([1e308, -1e308]),
        np.array([1e-308, -1e-308]),
        np.full(100, 1e-160),
        np.full(100, 1e100),
        np.full(100, 1e-100),
        np.array([5e-324, 5e-324]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.linspace(-2.0, 3.0, 24001),
    ],
)
def test_scaled_norm_is_stable_and_thread_independent(x):
    parts = np.empty(((len(x) + li._CPU_BLOCK - 1) // li._CPU_BLOCK, 2))
    numba.set_num_threads(1)
    one = li._cpu_norm(x, parts)
    numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
    many = li._cpu_norm(x, parts)
    np.testing.assert_equal(one, many)
    expected = norm(x, check_finite=False)
    np.testing.assert_allclose(one, expected, rtol=3e-15, atol=5e-324, equal_nan=True)


def test_zero_rhs_exact_initial_and_zero_iteration():
    op = DenseClosed(np.eye(5))
    for b, x0 in [(np.zeros(5), None), (np.ones(5), np.ones(5))]:
        result = li._lsmr_closed_cpu(op, b, x0=x0)
        assert result[1:3] == (0, 0)
        np.testing.assert_array_equal(result[0], b)
    result = li._lsmr_closed_cpu(op, np.ones(5), maxiter=0)
    assert result[2] == 0
    np.testing.assert_array_equal(result[0], np.zeros(5))


def test_closed_operator_supports_owned_scratch_and_preserves_default():
    from .test_helmholtz_closed import problem

    op, _, _, _ = problem()
    op.prepare(0.7, 1e-5)
    rng = np.random.default_rng(8)
    x = rng.normal(size=op.size)
    y = rng.normal(size=op.size + op.data_size)
    forward = op.matvec(x)
    scratch = np.empty(op.size + op.data_size)
    assert op.matvec(x, out=scratch) is scratch
    np.testing.assert_array_equal(scratch, forward)
    transpose = op.rmatvec(y)
    view = scratch[: op.size]
    assert op.rmatvec(y, out=view) is view
    np.testing.assert_array_equal(view, transpose)
    # A default result is independently owned and remains unchanged.
    np.testing.assert_array_equal(forward, op.matvec(x))


@pytest.mark.parametrize("threads", [None, 1])
def test_fused_kernels_obey_inherited_or_explicit_thread_mask(monkeypatch, threads):
    from xinvert.helmholtz import streamfunction_potential

    from .test_helmholtz_closed import fields

    external = min(2, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(external)
    expected = external if threads is None else threads
    observed = []
    for name in ("_cpu_axpy_parts", "_cpu_update_parts"):
        original = getattr(li, name)

        def spy(*args, _name=name, _original=original):
            observed.append((_name, numba.get_num_threads()))
            return _original(*args)

        monkeypatch.setattr(li, name, spy)
    u, v, wet = fields()
    streamfunction_potential(
        u,
        v,
        mask=wet,
        iParams={
            "BCs": ["closed", "periodic"],
            "architect": "cpu-mc",
            "threads": threads,
            "mxLoop": 3,
            "on_nonconvergence": "ignore",
        },
    )
    assert {name for name, _ in observed} == {"_cpu_axpy_parts", "_cpu_update_parts"}
    assert {count for _, count in observed} == {expected}
    assert numba.get_num_threads() == external


@pytest.mark.parametrize(
    "diagonal", [np.arange(1.0, 6.0), np.array([0.0, 1.0, 2.0, 3.0, 0.0])]
)
def test_consistent_and_rank_deficient_minimum_norm(diagonal):
    a = np.diag(diagonal)
    b = a @ np.arange(1.0, 6.0)
    actual = li._lsmr_closed_cpu(DenseClosed(a), b, atol=1e-12, btol=1e-12)
    expected = np.linalg.lstsq(a, b, rcond=None)[0]
    assert actual[1] in (1, 4)
    np.testing.assert_allclose(actual[0], expected, rtol=1e-10, atol=1e-11)


def test_condition_limit_preserves_scipy_stop_reason():
    rng = np.random.default_rng(19)
    a = rng.normal(size=(57, 31))
    b = rng.normal(size=57)
    opts = {"atol": 0.0, "btol": 0.0, "conlim": 1.0, "maxiter": 20}
    expected = lsmr(a, b, **opts)
    actual = li._lsmr_closed_cpu(DenseClosed(a), b, **opts)
    assert actual[1] == expected[1] == 3
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)


def test_unregularized_closed_mask_against_csr():
    from scipy.sparse import diags, vstack

    from .test_helmholtz_closed import problem

    op, a, weights, _ = problem()
    op.prepare(0.7, 0.0)
    weighted = diags(np.sqrt(weights)) @ a * 0.7 @ diags(op.scale)
    augmented = vstack([weighted, diags(np.zeros(op.size))]).tocsr()
    rhs = np.random.default_rng(24).normal(size=augmented.shape[0])
    rhs[op.data_size :] = 0.0
    opts = {"atol": 1e-11, "btol": 1e-11, "maxiter": 3000, "conlim": 1e12}
    expected = lsmr(augmented, rhs, **opts)
    actual = li._lsmr_closed_cpu(op, rhs, **opts)
    assert actual[1] in (1, 2, 4, 5)
    np.testing.assert_allclose(
        augmented @ actual[0], augmented @ expected[0], rtol=1e-7, atol=1e-8
    )


@pytest.mark.parametrize("solver", ["lsmr", "lsqr"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("penalty", [1e-5, 0.03])
def test_closed_solve_matches_independent_dense_least_squares(
    solver, periodic, descending, penalty
):
    """Validate the physical objective, not just matching solver stop flags."""
    from .test_helmholtz_closed import problem

    op, matrix, weights, _ = problem(periodic, descending)
    data = np.random.default_rng(29).normal(size=op.data_size)
    length = 0.7
    physical_alpha = penalty / length**2
    multiplicity = np.r_[op.counts, np.ones(op.nchi)]
    # Independent CSR oracle and LAPACK solve; no matrix-free kernels here.
    augmented = np.vstack((
        np.sqrt(weights)[:, None] * matrix.toarray(),
        np.diag(np.sqrt(physical_alpha * multiplicity)),
    ))
    rhs = np.r_[np.sqrt(weights) * data, np.zeros(op.size)]
    expected = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    actual, status = li._solve_closed_grid(
        op, data, length, penalty, solver=solver,
        tolerance=1e-12, conlim=0, maxiter=5000,
    )
    assert status["converged"]
    residual = augmented @ actual - rhs
    reference = augmented @ expected - rhs
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(
        np.linalg.norm(residual), np.linalg.norm(reference), rtol=1e-12,
    )
    assert np.linalg.norm(augmented.T @ residual) / np.linalg.norm(rhs) < 1e-9
